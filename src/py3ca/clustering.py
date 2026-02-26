from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .program_types import Program


def overlap_count(a: set, b: set) -> int:
    return len(a.intersection(b))


def _gene_weight(program: Program, gene: str) -> float:
    idx = program.gene_index.get(gene)
    if idx is None:
        return 0.0
    return float(program.weights[idx])


def _seed_pair(
    programs: List[Program],
    min_intersect: int,
    min_group_size: int,
    allow_weak_seeds: bool,
) -> Optional[Tuple[int, int]]:
    n = len(programs)
    if n < 2:
        return None

    pairs: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = overlap_count(programs[i].gene_set, programs[j].gene_set)
            if ov >= min_intersect:
                pairs.append((ov, i, j))

    if not pairs:
        return None

    pairs.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    for _, i, j in pairs:
        if allow_weak_seeds:
            return i, j
        p_i = programs[i]
        p_j = programs[j]
        support = 0
        for k in range(n):
            if k in (i, j):
                continue
            ov_i = overlap_count(p_i.gene_set, programs[k].gene_set)
            ov_j = overlap_count(p_j.gene_set, programs[k].gene_set)
            if min(ov_i, ov_j) >= min_intersect:
                support += 1
        if support > min_group_size:
            return i, j

    return None


def _init_mp_genes(p1: Program, p2: Program, top_n: int) -> List[str]:
    intersect = list(p1.gene_set.intersection(p2.gene_set))
    if len(intersect) >= top_n:
        return intersect[:top_n]

    weights: Dict[str, float] = {}
    for g in set(p1.genes).union(p2.genes):
        weights[g] = max(_gene_weight(p1, g), _gene_weight(p2, g))

    remaining = [g for g in sorted(weights, key=lambda g: (-weights[g], g)) if g not in intersect]
    needed = top_n - len(intersect)
    return intersect + remaining[:needed]


def _update_mp_genes(cluster_programs: List[Program], top_n: int) -> List[str]:
    freq = Counter()
    for p in cluster_programs:
        freq.update(p.genes)

    if len(freq) <= top_n:
        return [g for g, _ in freq.most_common(top_n)]

    freqs_sorted = sorted(freq.values(), reverse=True)
    cutoff = freqs_sorted[top_n - 1]
    high_genes = [g for g, c in freq.items() if c > cutoff]
    border_genes = [g for g, c in freq.items() if c == cutoff]

    if border_genes:
        weights: Dict[str, float] = {}
        for g in border_genes:
            weights[g] = max(_gene_weight(p, g) for p in cluster_programs)
        border_genes = sorted(border_genes, key=lambda g: (-weights[g], g))

    genes_mp = high_genes + border_genes
    return genes_mp[:top_n]


def greedy_cluster(
    programs: List[Program],
    min_intersect_initial: int = 15,
    min_intersect_cluster: int = 15,
    min_group_size: int = 5,
    top_n: int = 100,
    allow_weak_seeds: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    if not programs:
        return {}, {}

    remaining = programs.copy()
    mp_list: Dict[str, List[str]] = {}
    cluster_list: Dict[str, List[str]] = {}
    cluster_idx = 1

    while True:
        seed = _seed_pair(remaining, min_intersect_initial, min_group_size, allow_weak_seeds)
        if seed is None:
            break

        i, j = seed
        p1 = remaining[i]
        p2 = remaining[j]

        curr_cluster = [p1, p2]
        genes_mp = _init_mp_genes(p1, p2, top_n)

        remaining = [p for k, p in enumerate(remaining) if k not in (i, j)]

        while True:
            if not remaining:
                break
            target_set = set(genes_mp)
            overlaps = [overlap_count(target_set, p.gene_set) for p in remaining]
            best_idx = int(np.argmax(overlaps))
            best_val = int(overlaps[best_idx])
            if best_val < min_intersect_cluster:
                break
            candidate = remaining.pop(best_idx)
            curr_cluster.append(candidate)
            genes_mp = _update_mp_genes(curr_cluster, top_n)

        cluster_list[f"Cluster_{cluster_idx}"] = [p.program_id for p in curr_cluster]
        mp_list[f"MP_{cluster_idx}"] = genes_mp
        cluster_idx += 1

        remaining_ids = set(p.program_id for p in curr_cluster)
        remaining = [p for p in remaining if p.program_id not in remaining_ids]

    return cluster_list, mp_list
