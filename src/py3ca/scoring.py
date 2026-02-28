from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals)
    n = pvals.size
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    sorted_p = pvals[order]
    qvals = sorted_p * n / ranks
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    out = np.empty_like(qvals)
    out[order] = np.clip(qvals, 0, 1)
    return out


def score_mps(adata, mp_list: Dict[str, List[str]]) -> pd.DataFrame:
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    gene_index = {g: i for i, g in enumerate(adata.var_names)}

    scores = []
    for mp_name, genes in mp_list.items():
        idx = [gene_index[g] for g in genes if g in gene_index]
        if len(idx) == 0:
            continue
        data = X[:, idx]
        n = data.shape[1]
        mean = data.mean(axis=1)
        std = data.std(axis=1, ddof=1)
        denom = std / np.sqrt(n)
        denom[denom == 0] = np.inf
        t_stat = mean / denom
        pvals = 2 * stats.t.sf(np.abs(t_stat), df=n - 1)
        fdr = bh_fdr(pvals)
        scores.append(
            pd.DataFrame(
                {
                    "cell_id": adata.obs_names,
                    "meta_program": mp_name,
                    "t_stat": t_stat,
                    "p_value": pvals,
                    "fdr": fdr,
                }
            )
        )

    if not scores:
        return pd.DataFrame(columns=["cell_id", "meta_program", "t_stat", "p_value", "fdr"])

    return pd.concat(scores, ignore_index=True)


def score_sample_programs(
    sample_id: str,
    study_id: str,
    cell_ids: Sequence[str],
    program_ids: Sequence[str],
    k: int,
    loadings: np.ndarray,
) -> pd.DataFrame:
    if loadings.ndim != 2:
        raise ValueError("loadings must be a 2D matrix")
    if loadings.shape[0] != len(cell_ids):
        raise ValueError("Number of rows in loadings must match number of cells")
    if loadings.shape[1] != len(program_ids):
        raise ValueError("Number of columns in loadings must match number of programs")

    component = np.arange(1, len(program_ids) + 1, dtype=int)
    rows: List[pd.DataFrame] = []
    for col_idx, program_id in enumerate(program_ids):
        rows.append(
            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "sample_id": sample_id,
                    "study_id": study_id,
                    "program_id": program_id,
                    "k": k,
                    "component": int(component[col_idx]),
                    "program_score": loadings[:, col_idx],
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cell_id",
                "sample_id",
                "study_id",
                "program_id",
                "k",
                "component",
                "program_score",
            ]
        )

    return pd.concat(rows, ignore_index=True)
