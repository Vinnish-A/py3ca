from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .types import AnalysisOptions, PreparedSampleRef, Sample
from .utils import top_n_indices


def load_manifest(data_dir: str) -> pd.DataFrame:
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        required = {"sample_id", "study_id", "path"}
        missing = required.difference(set(manifest.columns))
        if missing:
            raise ValueError(f"manifest.csv missing columns: {sorted(missing)}")
        manifest["path"] = manifest["path"].apply(
            lambda p: p if os.path.isabs(p) else os.path.join(data_dir, p)
        )
        return manifest

    rows = []
    for item in sorted(os.listdir(data_dir)):
        if item == "manifest.csv":
            continue
        path = os.path.join(data_dir, item)
        if os.path.isdir(path) or os.path.isfile(path):
            sample_id = os.path.splitext(os.path.basename(path))[0]
            rows.append({"sample_id": sample_id, "study_id": sample_id, "path": path})
    return pd.DataFrame(rows)


def find_samples(data_dir: str) -> list[Sample]:
    manifest = load_manifest(data_dir)
    return [
        Sample(
            sample_id=str(row.sample_id),
            study_id=str(row.study_id),
            path=str(row.path),
        )
        for row in manifest.itertuples(index=False)
    ]


def _scanpy():
    import scanpy as sc

    return sc


def read_sample(path: str) -> ad.AnnData:
    if os.path.isdir(path):
        return _scanpy().read_10x_mtx(path, var_names="gene_symbols", make_unique=True)
    lower = path.lower()
    if lower.endswith(".h5ad"):
        return ad.read_h5ad(path)
    if lower.endswith(".h5"):
        return _scanpy().read_10x_h5(path)
    if lower.endswith(".loom"):
        return _scanpy().read_loom(path)
    return _scanpy().read(path)


def read_sample_data(sample: Union[Sample, str]) -> ad.AnnData:
    if isinstance(sample, str):
        return read_sample(sample)
    sample.validate()
    if sample.adata is not None:
        return sample.adata.copy()
    if sample.path is None:
        raise ValueError("Sample requires either path or adata")
    return read_sample(sample.path)


def log2_cpm_div10(X):
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X = X / 10.0
        X.data = np.log2(X.data + 1.0)
        return X
    return np.log2((X / 10.0) + 1.0)


def top_genes_by_mean(adata: ad.AnnData, n_top: int) -> ad.AnnData:
    mean_vals = np.asarray(adata.X.mean(axis=0)).ravel()
    if mean_vals.size <= n_top:
        return adata
    top_idx = top_n_indices(mean_vals, n_top)
    return adata[:, top_idx].copy()


def normalize_log(adata: ad.AnnData) -> ad.AnnData:
    X = adata.X
    totals = np.asarray(X.sum(axis=1)).ravel()
    scale = np.zeros_like(totals, dtype=float)
    nonzero = totals > 0
    scale[nonzero] = 1e6 / totals[nonzero]
    if sp.issparse(X):
        adata.X = sp.diags(scale).dot(X).tocsr()
    else:
        adata.X = X * scale[:, None]
    adata.X = log2_cpm_div10(adata.X)
    return adata


def _sample_baseline_part(sample: Sample, n_top: int) -> tuple[str, pd.Series, pd.Series]:
    adata = read_sample_data(sample)
    adata = top_genes_by_mean(adata, n_top)
    adata = normalize_log(adata)
    gene_names = adata.var_names
    gene_sums = pd.Series(np.asarray(adata.X.sum(axis=0)).ravel(), index=gene_names)
    gene_counts = pd.Series(adata.n_obs, index=gene_names)
    return sample.study_id, gene_sums, gene_counts


def estimate_study_baselines(
    samples: Sequence[Sample],
    options: Optional[AnalysisOptions] = None,
    n_top: Optional[int] = None,
) -> Dict[str, pd.Series]:
    if options is None:
        options = AnalysisOptions()
    top_genes = options.top_genes if n_top is None else n_top
    sums: Dict[str, pd.Series] = {}
    counts: Dict[str, pd.Series] = {}
    for sample in samples:
        study_id, gene_sums, gene_counts = _sample_baseline_part(sample, top_genes)
        if study_id not in sums:
            sums[study_id] = gene_sums
            counts[study_id] = gene_counts
        else:
            sums[study_id] = sums[study_id].add(gene_sums, fill_value=0.0)
            counts[study_id] = counts[study_id].add(gene_counts, fill_value=0.0)
    return {study: sums[study] / counts[study] for study in sums}


def collect_study_means(manifest: pd.DataFrame, n_top: int) -> Dict[str, pd.Series]:
    samples = [
        Sample(sample_id=str(row.sample_id), study_id=str(row.study_id), path=str(row.path))
        for row in manifest.itertuples(index=False)
    ]
    return estimate_study_baselines(samples, n_top=n_top)


def center_and_clip(adata: ad.AnnData, study_mean: pd.Series) -> ad.AnnData:
    gene_means = study_mean.reindex(adata.var_names).fillna(0.0).to_numpy()
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data = X.data - gene_means[X.indices]
        X.data[X.data < 0] = 0
        X.eliminate_zeros()
        adata.X = X
    else:
        X = X - gene_means
        X[X < 0] = 0
        adata.X = X
    return adata


def prepare_sample(
    sample: Union[Sample, str],
    study_mean: pd.Series,
    options: Optional[AnalysisOptions] = None,
    normalized_sample: Optional[PreparedSampleRef] = None,
    n_top: Optional[int] = None,
) -> ad.AnnData:
    if options is None:
        options = AnalysisOptions()
    top_genes = options.top_genes if n_top is None else n_top
    if normalized_sample is not None and normalized_sample.adata is not None:
        adata = normalized_sample.adata.copy()
    elif normalized_sample is not None and normalized_sample.path is not None:
        adata = ad.read_h5ad(normalized_sample.path)
    else:
        adata = read_sample_data(sample)
        adata = top_genes_by_mean(adata, top_genes)
        adata = normalize_log(adata)
    adata = center_and_clip(adata, study_mean)
    return adata


def preprocess_sample(
    path: str,
    study_mean: pd.Series,
    n_top: int,
) -> Tuple[str, ad.AnnData]:
    adata = prepare_sample(path, study_mean, n_top=n_top)
    return path, adata
