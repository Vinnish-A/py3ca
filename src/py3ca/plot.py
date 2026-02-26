from __future__ import annotations

import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir in sys.path:
    sys.path.remove(_curr_dir)

import argparse
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa.intersection(sb)) / len(sa.union(sb))


def _load_programs(robust_csv: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    df = pd.read_csv(robust_csv)
    prog_genes: Dict[str, List[str]] = {}
    prog_sample: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        prog_genes.setdefault(row.program_id, []).append(row.gene)
        prog_sample[row.program_id] = str(row.sample_id)
    return prog_genes, prog_sample


def _load_mps(mp_csv: str) -> Dict[str, List[str]]:
    df = pd.read_csv(mp_csv)
    mp_genes: Dict[str, List[str]] = {}
    for row in df.itertuples(index=False):
        mp_genes.setdefault(row.meta_program, []).append(row.gene)
    return mp_genes


def _sample_mp_matrix(
    prog_genes: Dict[str, List[str]],
    prog_sample: Dict[str, str],
    mp_genes: Dict[str, List[str]],
    agg: str = "max",
) -> pd.DataFrame:
    samples = sorted(set(prog_sample.values()))
    mps = sorted(mp_genes.keys())
    mat = np.zeros((len(samples), len(mps)), dtype=float)

    programs_by_sample: Dict[str, List[str]] = {}
    for pid, sid in prog_sample.items():
        programs_by_sample.setdefault(sid, []).append(pid)

    for i, sid in enumerate(samples):
        pids = programs_by_sample.get(sid, [])
        for j, mp in enumerate(mps):
            overlaps = [
                _jaccard(prog_genes[pid], mp_genes[mp])
                for pid in pids
            ]
            if not overlaps:
                val = 0.0
            elif agg == "mean":
                val = float(np.mean(overlaps))
            else:
                val = float(np.max(overlaps))
            mat[i, j] = val

    return pd.DataFrame(mat, index=samples, columns=mps)


def _sample_sample_matrix(
    prog_genes: Dict[str, List[str]],
    prog_sample: Dict[str, str],
    agg: str = "max",
) -> pd.DataFrame:
    samples = sorted(set(prog_sample.values()))
    programs_by_sample: Dict[str, List[str]] = {}
    for pid, sid in prog_sample.items():
        programs_by_sample.setdefault(sid, []).append(pid)

    mat = np.zeros((len(samples), len(samples)), dtype=float)
    for i, s1 in enumerate(samples):
        p1 = programs_by_sample.get(s1, [])
        for j, s2 in enumerate(samples):
            p2 = programs_by_sample.get(s2, [])
            if not p1 or not p2:
                val = 0.0
            else:
                overlaps = [
                    _jaccard(prog_genes[a], prog_genes[b])
                    for a in p1
                    for b in p2
                ]
                if agg == "mean":
                    val = float(np.mean(overlaps))
                else:
                    val = float(np.max(overlaps))
            mat[i, j] = val

    return pd.DataFrame(mat, index=samples, columns=samples)


def plot_heatmap(
    matrix: pd.DataFrame,
    out_path: str,
    title: str,
    cmap: str = "viridis",
) -> None:
    plt.figure(figsize=(1 + 0.4 * matrix.shape[1], 1 + 0.4 * matrix.shape[0]))
    sns.heatmap(matrix, cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Jaccard heatmaps from robust programs")
    parser.add_argument("--robust-csv", required=True, help="Path to programs_robust.csv")
    parser.add_argument("--mp-csv", help="Path to meta_programs.csv (required for sample-mp mode)")
    parser.add_argument("--mode", choices=["sample-mp", "sample-sample"], default="sample-mp")
    parser.add_argument("--agg", choices=["max", "mean"], default="max")
    parser.add_argument("--out", required=True, help="Output image path (png/pdf)")
    parser.add_argument("--title", default="Jaccard heatmap")
    args = parser.parse_args()

    prog_genes, prog_sample = _load_programs(args.robust_csv)

    if args.mode == "sample-mp":
        if not args.mp_csv:
            raise SystemExit("--mp-csv is required for sample-mp mode")
        mp_genes = _load_mps(args.mp_csv)
        matrix = _sample_mp_matrix(prog_genes, prog_sample, mp_genes, agg=args.agg)
        plot_heatmap(matrix, args.out, args.title)
        return

    matrix = _sample_sample_matrix(prog_genes, prog_sample, agg=args.agg)
    plot_heatmap(matrix, args.out, args.title)


if __name__ == "__main__":
    main()
