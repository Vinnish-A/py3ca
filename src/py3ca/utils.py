from __future__ import annotations

import re

import numpy as np


def top_n_indices(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values)
    if n <= 0:
        return np.array([], dtype=int)
    if values.size <= n:
        return np.argsort(values)[::-1]
    idx = np.argpartition(values, -n)[-n:]
    return idx[np.argsort(values[idx])[::-1]]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"
