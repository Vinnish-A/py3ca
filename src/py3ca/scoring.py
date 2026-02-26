from __future__ import annotations

from typing import Dict, List

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
