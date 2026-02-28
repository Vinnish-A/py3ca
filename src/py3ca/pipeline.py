from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from .clustering import greedy_cluster
from .io import collect_study_means, load_manifest, preprocess_sample
from .robustness import select_robust_programs
from .scoring import score_mps, score_sample_programs
from .program_types import Program


def run_nmf_for_sample(
    adata,
    sample_id: str,
    study_id: str,
    k_min: int,
    k_max: int,
    random_state: int = 0,
) -> Tuple[List[Program], pd.DataFrame]:
    programs: List[Program] = []
    program_score_frames: List[pd.DataFrame] = []
    gene_names = adata.var_names.to_list()
    gene_index = {g: i for i, g in enumerate(gene_names)}
    X = adata.X
    cell_ids = adata.obs_names.to_list()

    for k in range(k_min, k_max + 1):
        model = NMF(
            n_components=k,
            init="nndsvd",
            random_state=random_state,
            max_iter=1000,
        )
        W = model.fit_transform(X)
        H = model.components_
        program_ids_for_k: List[str] = []
        for comp_idx, weights in enumerate(H):
            top_idx = np.argsort(weights)[::-1][:100]
            genes = [gene_names[i] for i in top_idx]
            program_id = f"{sample_id}.k{k}.c{comp_idx + 1}"
            program_ids_for_k.append(program_id)
            programs.append(
                Program(
                    program_id=program_id,
                    sample_id=sample_id,
                    study_id=study_id,
                    k=k,
                    component=comp_idx + 1,
                    genes=genes,
                    weights=weights,
                    gene_index=gene_index,
                )
            )
        program_score_frames.append(
            score_sample_programs(
                sample_id=sample_id,
                study_id=study_id,
                cell_ids=cell_ids,
                program_ids=program_ids_for_k,
                k=k,
                loadings=W,
            )
        )
    if program_score_frames:
        program_scores = pd.concat(program_score_frames, ignore_index=True)
    else:
        program_scores = pd.DataFrame(
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
    return programs, program_scores


def run_pipeline(
    data_dir: str,
    out_dir: str,
    top_genes: int = 7000,
    k_min: int = 4,
    k_max: int = 9,
    min_intersect_initial: int = 15,
    min_intersect_cluster: int = 15,
    min_group_size: int = 5,
    allow_weak_seeds: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    manifest = load_manifest(data_dir)
    if manifest.empty:
        raise SystemExit("No samples found in data directory.")

    study_means = collect_study_means(manifest, n_top=top_genes)

    all_programs: List[Program] = []
    preprocessed_cache: Dict[str, object] = {}
    components_by_sample: Dict[str, List[str]] = {}
    program_scores_by_sample: Dict[str, pd.DataFrame] = {}

    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        study_id = str(row.study_id)
        path = str(row.path)
        _, adata = preprocess_sample(path, study_means[study_id], top_genes)
        preprocessed_cache[sample_id] = adata
        programs, program_scores = run_nmf_for_sample(
            adata,
            sample_id=sample_id,
            study_id=study_id,
            k_min=k_min,
            k_max=k_max,
        )
        all_programs.extend(programs)
        components_by_sample[sample_id] = [p.program_id for p in programs]
        program_scores_by_sample[sample_id] = program_scores

    programs_df = pd.DataFrame(
        [
            {
                "program_id": p.program_id,
                "sample_id": p.sample_id,
                "study_id": p.study_id,
                "k": p.k,
                "component": p.component,
                "gene_rank": i + 1,
                "gene": g,
            }
            for p in all_programs
            for i, g in enumerate(p.genes)
        ]
    )
    programs_df.to_csv(os.path.join(out_dir, "programs_all.csv"), index=False)

    components_rows = [
        {"sample_id": p.sample_id, "program_id": p.program_id, "k": p.k, "component": p.component}
        for p in all_programs
    ]
    pd.DataFrame(components_rows).to_csv(os.path.join(out_dir, "components_by_sample.csv"), index=False)
    with open(os.path.join(out_dir, "components_by_sample.json"), "w", encoding="utf-8") as f:
        json.dump(components_by_sample, f, indent=2)

    robust_programs = select_robust_programs(all_programs)
    robust_df = pd.DataFrame(
        [
            {
                "program_id": p.program_id,
                "sample_id": p.sample_id,
                "study_id": p.study_id,
                "k": p.k,
                "component": p.component,
                "gene_rank": i + 1,
                "gene": g,
            }
            for p in robust_programs
            for i, g in enumerate(p.genes)
        ]
    )
    robust_df.to_csv(os.path.join(out_dir, "programs_robust.csv"), index=False)

    cluster_list, mp_list = greedy_cluster(
        robust_programs,
        min_intersect_initial=min_intersect_initial,
        min_intersect_cluster=min_intersect_cluster,
        min_group_size=min_group_size,
        allow_weak_seeds=allow_weak_seeds,
    )
    if not mp_list and robust_programs:
        print("No clusters formed with current thresholds; retrying with allow_weak_seeds=True and min_group_size=1.")
        cluster_list, mp_list = greedy_cluster(
            robust_programs,
            min_intersect_initial=min_intersect_initial,
            min_intersect_cluster=min_intersect_cluster,
            min_group_size=1,
            allow_weak_seeds=True,
        )
    with open(os.path.join(out_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_list, f, indent=2)
    with open(os.path.join(out_dir, "meta_programs.json"), "w", encoding="utf-8") as f:
        json.dump(mp_list, f, indent=2)

    # Map MP -> sample -> program_ids
    program_to_sample = {p.program_id: p.sample_id for p in robust_programs}
    mp_sample_map: Dict[str, Dict[str, List[str]]] = {}
    mp_sample_rows: List[Dict[str, str]] = []
    for cluster_name, program_ids in cluster_list.items():
        mp_name = cluster_name.replace("Cluster_", "MP_")
        for pid in program_ids:
            sample_id = program_to_sample.get(pid, "")
            if not sample_id:
                continue
            mp_sample_map.setdefault(mp_name, {}).setdefault(sample_id, []).append(pid)
            mp_sample_rows.append(
                {"meta_program": mp_name, "sample_id": sample_id, "program_id": pid}
            )

    with open(os.path.join(out_dir, "meta_programs_by_sample.json"), "w", encoding="utf-8") as f:
        json.dump(mp_sample_map, f, indent=2)
    pd.DataFrame(mp_sample_rows).to_csv(
        os.path.join(out_dir, "meta_programs_by_sample.csv"), index=False
    )

    mp_rows = [
        {"meta_program": mp, "gene_rank": i + 1, "gene": g}
        for mp, genes in mp_list.items()
        for i, g in enumerate(genes)
    ]
    pd.DataFrame(mp_rows).to_csv(os.path.join(out_dir, "meta_programs.csv"), index=False)

    scores_dir = os.path.join(out_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    for sample_id, adata in preprocessed_cache.items():
        scores = score_mps(adata, mp_list)
        scores.to_csv(os.path.join(scores_dir, f"{sample_id}_mp_scores.csv"), index=False)

    program_scores_dir = os.path.join(out_dir, "program_scores")
    os.makedirs(program_scores_dir, exist_ok=True)
    for sample_id, program_scores in program_scores_by_sample.items():
        program_scores.to_csv(
            os.path.join(program_scores_dir, f"{sample_id}_program_scores.csv"),
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="3CA NMF pipeline (Python)")
    parser.add_argument("--data-dir", required=True, help="Directory containing samples or manifest.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--top-genes", type=int, default=5000)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--min-intersect-initial", type=int, default=15)
    parser.add_argument("--min-intersect-cluster", type=int, default=15)
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--allow-weak-seeds", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        top_genes=args.top_genes,
        k_min=args.k_min,
        k_max=args.k_max,
        min_intersect_initial=args.min_intersect_initial,
        min_intersect_cluster=args.min_intersect_cluster,
        min_group_size=args.min_group_size,
        allow_weak_seeds=args.allow_weak_seeds,
    )


if __name__ == "__main__":
    main()
