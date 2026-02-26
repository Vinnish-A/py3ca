from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


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


def read_sample(path: str) -> sc.AnnData:
    if os.path.isdir(path):
        return sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True)
    lower = path.lower()
    if lower.endswith(".h5ad"):
        return sc.read_h5ad(path)
    if lower.endswith(".h5"):
        return sc.read_10x_h5(path)
    if lower.endswith(".loom"):
        return sc.read_loom(path)
    return sc.read(path)


def log2_cpm_div10(X):
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X = X / 10.0
        X.data = np.log2(X.data + 1.0)
        return X
    return np.log2((X / 10.0) + 1.0)


def top_genes_by_mean(adata: sc.AnnData, n_top: int) -> sc.AnnData:
    mean_vals = np.asarray(adata.X.mean(axis=0)).ravel()
    if mean_vals.size <= n_top:
        return adata
    top_idx = np.argsort(mean_vals)[::-1][:n_top]
    return adata[:, top_idx].copy()


def normalize_log(adata: sc.AnnData) -> sc.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e6)
    adata.X = log2_cpm_div10(adata.X)
    return adata


def collect_study_means(manifest: pd.DataFrame, n_top: int) -> Dict[str, pd.Series]:
    sums: Dict[str, pd.Series] = {}
    counts: Dict[str, pd.Series] = {}
    for row in manifest.itertuples(index=False):
        path = str(row.path)
        study_id = str(row.study_id)
        adata = read_sample(path)
        adata = top_genes_by_mean(adata, n_top)
        adata = normalize_log(adata)
        gene_names = adata.var_names
        gene_sums = pd.Series(np.asarray(adata.X.sum(axis=0)).ravel(), index=gene_names)
        gene_counts = pd.Series(adata.n_obs, index=gene_names)
        if study_id not in sums:
            sums[study_id] = gene_sums
            counts[study_id] = gene_counts
        else:
            sums[study_id] = sums[study_id].add(gene_sums, fill_value=0.0)
            counts[study_id] = counts[study_id].add(gene_counts, fill_value=0.0)
    return {study: sums[study] / counts[study] for study in sums}


def center_and_clip(adata: sc.AnnData, study_mean: pd.Series) -> sc.AnnData:
    gene_means = study_mean.reindex(adata.var_names).fillna(0.0).to_numpy()
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X - gene_means
    X[X < 0] = 0
    adata.X = X
    return adata


def preprocess_sample(
    path: str,
    study_mean: pd.Series,
    n_top: int,
) -> Tuple[str, sc.AnnData]:
    adata = read_sample(path)
    adata = top_genes_by_mean(adata, n_top)
    adata = normalize_log(adata)
    adata = center_and_clip(adata, study_mean)
    return path, adata
