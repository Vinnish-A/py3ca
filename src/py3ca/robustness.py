from __future__ import annotations

from typing import Dict, List

import numpy as np

from .program_types import Program


def overlap_count(a: set, b: set) -> int:
    return len(a.intersection(b))


def select_robust_programs(
    programs: List[Program],
    intra_min: int = 50,
    inter_min: int = 15,
    redundancy_min: int = 15,
) -> List[Program]:
    if not programs:
        return []

    by_sample: Dict[str, List[Program]] = {}
    for p in programs:
        by_sample.setdefault(p.sample_id, []).append(p)

    # Within-sample robustness
    intra_kept: List[Program] = []
    for sample_id, plist in by_sample.items():
        sets = [p.gene_set for p in plist]
        overlaps = np.zeros((len(plist), len(plist)), dtype=int)
        for i in range(len(plist)):
            for j in range(len(plist)):
                if i == j:
                    continue
                overlaps[i, j] = overlap_count(sets[i], sets[j])
        max_other = overlaps.max(axis=1) if overlaps.size else np.array([])
        for idx, keep in enumerate(max_other >= intra_min):
            if keep:
                intra_kept.append(plist[idx])

    if not intra_kept:
        return []

    # Across-sample robustness
    sets_all = [p.gene_set for p in intra_kept]
    max_external: Dict[str, int] = {}
    for i, p in enumerate(intra_kept):
        best = 0
        for j, q in enumerate(intra_kept):
            if p.sample_id == q.sample_id:
                continue
            best = max(best, overlap_count(sets_all[i], sets_all[j]))
        max_external[p.program_id] = best

    inter_kept = [p for p in intra_kept if max_external[p.program_id] >= inter_min]
    if not inter_kept:
        return []

    # Non-redundant selection (within sample)
    by_sample = {}
    for p in inter_kept:
        by_sample.setdefault(p.sample_id, []).append(p)

    selected: List[Program] = []
    for sample_id, plist in by_sample.items():
        ranked = sorted(plist, key=lambda p: max_external[p.program_id], reverse=True)
        kept: List[Program] = []
        for p in ranked:
            if not kept:
                kept.append(p)
                continue
            if all(overlap_count(p.gene_set, q.gene_set) < redundancy_min for q in kept):
                kept.append(p)
        selected.extend(kept)

    return selected
