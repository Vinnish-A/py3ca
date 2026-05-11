from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import anndata as ad
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(*args, **kwargs):
        yield

from .clustering import greedy_cluster
from .io import (
    center_and_clip,
    find_samples,
    normalize_log,
    prepare_sample,
    read_sample_data,
    top_genes_by_mean,
)
from .parallel import effective_options, map_tasks, stage_in_memory_samples_if_needed
from .program_types import Program
from .robustness import select_robust_programs
from .scoring import score_cells_by_meta_programs, score_cells_by_sample_programs
from .types import AnalysisOptions, CohortAnalysis, PreparedSampleRef, Sample, SampleAnalysis
from .utils import safe_filename, top_n_indices


def _empty_program_scores() -> pd.DataFrame:
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


def discover_sample_programs(
    adata,
    sample_id: str,
    study_id: str,
    k_min: int,
    k_max: int,
    random_state: int = 0,
) -> Tuple[List[Program], pd.DataFrame]:
    from sklearn.decomposition import NMF

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
            top_idx = top_n_indices(weights, 100)
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
            score_cells_by_sample_programs(
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
        program_scores = _empty_program_scores()
    return programs, program_scores


run_nmf_for_sample = discover_sample_programs


def _cache_ref(
    sample_id: str,
    study_id: str,
    adata,
    options: AnalysisOptions,
    suffix: str,
) -> Optional[PreparedSampleRef]:
    if options.cache_mode == "none":
        return PreparedSampleRef(sample_id=sample_id, study_id=study_id, adata=adata)
    if options.cache_mode == "memory":
        return PreparedSampleRef(sample_id=sample_id, study_id=study_id, adata=adata)
    if options.cache_dir is None:
        raise ValueError("cache_dir is required when cache_mode='disk'")
    os.makedirs(options.cache_dir, exist_ok=True)
    path = os.path.join(options.cache_dir, f"{safe_filename(sample_id)}_{suffix}.h5ad")
    adata.write_h5ad(path)
    return PreparedSampleRef(sample_id=sample_id, study_id=study_id, path=path)


def _read_ref(ref: PreparedSampleRef):
    if ref.adata is not None:
        return ref.adata.copy()
    if ref.path is None:
        raise ValueError("PreparedSampleRef requires either path or adata")
    return ad.read_h5ad(ref.path)


def _baseline_worker(task: tuple[Sample, AnalysisOptions]) -> tuple[str, str, pd.Series, pd.Series, Optional[PreparedSampleRef]]:
    sample, options = task
    with threadpool_limits(limits=options.threads_per_worker):
        adata = read_sample_data(sample)
        adata = top_genes_by_mean(adata, options.top_genes)
        adata = normalize_log(adata)
        gene_sums = pd.Series(np.asarray(adata.X.sum(axis=0)).ravel(), index=adata.var_names)
        gene_counts = pd.Series(adata.n_obs, index=adata.var_names)
        ref = _cache_ref(sample.sample_id, sample.study_id, adata, options, "normalized")
        return sample.sample_id, sample.study_id, gene_sums, gene_counts, ref


def _estimate_study_baselines_with_cache(
    samples: Sequence[Sample],
    options: AnalysisOptions,
) -> tuple[Dict[str, pd.Series], Dict[str, PreparedSampleRef]]:
    tasks = [(sample, options) for sample in samples]
    parts = map_tasks(_baseline_worker, tasks, options)
    sums: Dict[str, pd.Series] = {}
    counts: Dict[str, pd.Series] = {}
    normalized_refs: Dict[str, PreparedSampleRef] = {}
    for sample_id, study_id, gene_sums, gene_counts, ref in parts:
        if study_id not in sums:
            sums[study_id] = gene_sums
            counts[study_id] = gene_counts
        else:
            sums[study_id] = sums[study_id].add(gene_sums, fill_value=0.0)
            counts[study_id] = counts[study_id].add(gene_counts, fill_value=0.0)
        if ref is not None:
            normalized_refs[sample_id] = ref
    study_means = {study: sums[study] / counts[study] for study in sums}
    return study_means, normalized_refs


def estimate_study_baselines(
    samples: Sequence[Sample],
    options: Optional[AnalysisOptions] = None,
) -> Dict[str, pd.Series]:
    options = effective_options(options or AnalysisOptions())
    created_cache_dir: Optional[str] = None
    if options.cache_mode == "disk" and options.cache_dir is None:
        created_cache_dir = tempfile.mkdtemp(prefix="py3ca_cache_")
        options = replace(options, cache_dir=created_cache_dir)
    try:
        study_means, _ = _estimate_study_baselines_with_cache(samples, options)
        return study_means
    finally:
        if created_cache_dir is not None and os.path.isdir(created_cache_dir):
            shutil.rmtree(created_cache_dir, ignore_errors=True)


def _prepare_and_cache_sample(
    sample: Sample,
    study_mean: pd.Series,
    options: AnalysisOptions,
    normalized_ref: Optional[PreparedSampleRef] = None,
):
    if normalized_ref is not None:
        adata = _read_ref(normalized_ref)
        adata = center_and_clip(adata, study_mean)
    else:
        adata = prepare_sample(sample, study_mean, options)
    prepared_ref = _cache_ref(sample.sample_id, sample.study_id, adata, options, "prepared")
    return adata, prepared_ref


def analyze_one_sample(
    sample: Sample,
    study_mean: pd.Series,
    options: AnalysisOptions,
    normalized_ref: Optional[PreparedSampleRef] = None,
) -> SampleAnalysis:
    with threadpool_limits(limits=options.threads_per_worker):
        adata, prepared_ref = _prepare_and_cache_sample(sample, study_mean, options, normalized_ref)
        programs, program_scores = discover_sample_programs(
            adata,
            sample_id=sample.sample_id,
            study_id=sample.study_id,
            k_min=options.k_min,
            k_max=options.k_max,
            random_state=options.random_state,
        )
    return SampleAnalysis(
        sample_id=sample.sample_id,
        study_id=sample.study_id,
        programs=programs,
        program_scores=program_scores,
        prepared_sample=prepared_ref,
    )


def _analyze_one_sample_worker(
    task: tuple[Sample, pd.Series, AnalysisOptions, Optional[PreparedSampleRef]],
) -> SampleAnalysis:
    sample, study_mean, options, normalized_ref = task
    return analyze_one_sample(sample, study_mean, options, normalized_ref)


def _score_meta_programs_worker(
    task: tuple[PreparedSampleRef, Dict[str, List[str]], AnalysisOptions],
) -> tuple[str, pd.DataFrame]:
    ref, meta_programs, options = task
    with threadpool_limits(limits=options.threads_per_worker):
        adata = _read_ref(ref)
        return ref.sample_id, score_cells_by_meta_programs(adata, meta_programs)


def _programs_to_frame(programs: Sequence[Program]) -> pd.DataFrame:
    return pd.DataFrame(
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
            for p in programs
            for i, g in enumerate(p.genes)
        ]
    )


def _components_to_frame(programs: Sequence[Program]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": p.sample_id, "program_id": p.program_id, "k": p.k, "component": p.component}
            for p in programs
        ]
    )


def analyze_cohort(
    samples: Sequence[Sample],
    options: Optional[AnalysisOptions] = None,
) -> CohortAnalysis:
    if not samples:
        raise SystemExit("No samples found in data directory.")

    options = effective_options(options or AnalysisOptions())
    for sample in samples:
        sample.validate()

    created_cache_dir: Optional[str] = None
    created_stage_dir: Optional[str] = None
    if options.cache_mode == "disk" and options.cache_dir is None:
        created_cache_dir = tempfile.mkdtemp(prefix="py3ca_cache_")
        options = replace(options, cache_dir=created_cache_dir)
    elif options.cache_mode == "disk":
        os.makedirs(str(options.cache_dir), exist_ok=True)

    if options.cache_dir is None and options.backend == "process" and options.workers > 1:
        created_stage_dir = tempfile.mkdtemp(prefix="py3ca_stage_")
    staged_dir = options.cache_dir or created_stage_dir or tempfile.gettempdir()
    samples_for_workers = stage_in_memory_samples_if_needed(samples, options, staged_dir)

    try:
        study_means, normalized_refs = _estimate_study_baselines_with_cache(samples_for_workers, options)
        sample_tasks = [
            (
                sample,
                study_means[sample.study_id],
                options,
                normalized_refs.get(sample.sample_id),
            )
            for sample in samples_for_workers
        ]
        sample_results = map_tasks(_analyze_one_sample_worker, sample_tasks, options)

        all_programs: List[Program] = []
        components_by_sample_map: Dict[str, List[str]] = {}
        program_scores_by_sample: Dict[str, pd.DataFrame] = {}
        prepared_refs: Dict[str, PreparedSampleRef] = {}
        for result in sample_results:
            all_programs.extend(result.programs)
            components_by_sample_map[result.sample_id] = [p.program_id for p in result.programs]
            program_scores_by_sample[result.sample_id] = result.program_scores
            if result.prepared_sample is not None:
                prepared_refs[result.sample_id] = result.prepared_sample

        programs_df = _programs_to_frame(all_programs)
        components_df = _components_to_frame(all_programs)

        robust_programs = select_robust_programs(all_programs)
        robust_df = _programs_to_frame(robust_programs)

        cluster_list, mp_list = greedy_cluster(
            robust_programs,
            min_intersect_initial=options.min_intersect_initial,
            min_intersect_cluster=options.min_intersect_cluster,
            min_group_size=options.min_group_size,
            allow_weak_seeds=options.allow_weak_seeds,
        )
        fallback_used = False
        if not mp_list and robust_programs:
            fallback_used = True
            print(
                "No clusters formed with current thresholds; retrying with "
                "allow_weak_seeds=True and min_group_size=1."
            )
            cluster_list, mp_list = greedy_cluster(
                robust_programs,
                min_intersect_initial=options.min_intersect_initial,
                min_intersect_cluster=options.min_intersect_cluster,
                min_group_size=1,
                allow_weak_seeds=True,
            )

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
        mp_sample_df = pd.DataFrame(mp_sample_rows)

        score_tasks = [(ref, mp_list, options) for ref in prepared_refs.values()]
        mp_scores_by_sample = dict(map_tasks(_score_meta_programs_worker, score_tasks, options))

        diagnostics: Dict[str, Any] = {
            "n_samples": len(samples),
            "n_programs": len(all_programs),
            "n_robust_programs": len(robust_programs),
            "fallback_clustering_used": fallback_used,
        }
        if options.cache_dir is not None:
            diagnostics["cache_dir"] = options.cache_dir

        return CohortAnalysis(
            programs=programs_df,
            robust_programs=robust_df,
            components_by_sample=components_df,
            clusters=cluster_list,
            meta_programs=mp_list,
            meta_programs_by_sample=mp_sample_df,
            mp_scores=mp_scores_by_sample,
            program_scores=program_scores_by_sample,
            options=options,
            diagnostics=diagnostics,
            components_by_sample_map=components_by_sample_map,
            meta_programs_by_sample_map=mp_sample_map,
        )
    finally:
        if created_cache_dir is not None and os.path.isdir(created_cache_dir):
            shutil.rmtree(created_cache_dir, ignore_errors=True)
        if created_stage_dir is not None and os.path.isdir(created_stage_dir):
            shutil.rmtree(created_stage_dir, ignore_errors=True)


def analyze_files(
    data_dir: str,
    options: Optional[AnalysisOptions] = None,
    out_dir: Optional[str] = None,
) -> CohortAnalysis:
    samples = find_samples(data_dir)
    result = analyze_cohort(samples, options)
    if out_dir is not None:
        save_analysis(result, out_dir)
    return result


def save_analysis(result: CohortAnalysis, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    result.programs.to_csv(os.path.join(out_dir, "programs_all.csv"), index=False)
    result.robust_programs.to_csv(os.path.join(out_dir, "programs_robust.csv"), index=False)
    result.components_by_sample.to_csv(os.path.join(out_dir, "components_by_sample.csv"), index=False)

    with open(os.path.join(out_dir, "components_by_sample.json"), "w", encoding="utf-8") as f:
        json.dump(result.components_by_sample_map, f, indent=2)
    with open(os.path.join(out_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(result.clusters, f, indent=2)
    with open(os.path.join(out_dir, "meta_programs.json"), "w", encoding="utf-8") as f:
        json.dump(result.meta_programs, f, indent=2)
    with open(os.path.join(out_dir, "meta_programs_by_sample.json"), "w", encoding="utf-8") as f:
        json.dump(result.meta_programs_by_sample_map, f, indent=2)

    result.meta_programs_by_sample.to_csv(
        os.path.join(out_dir, "meta_programs_by_sample.csv"), index=False
    )
    mp_rows = [
        {"meta_program": mp, "gene_rank": i + 1, "gene": g}
        for mp, genes in result.meta_programs.items()
        for i, g in enumerate(genes)
    ]
    pd.DataFrame(mp_rows).to_csv(os.path.join(out_dir, "meta_programs.csv"), index=False)

    scores_dir = os.path.join(out_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    for sample_id, scores in result.mp_scores.items():
        scores.to_csv(os.path.join(scores_dir, f"{sample_id}_mp_scores.csv"), index=False)

    program_scores_dir = os.path.join(out_dir, "program_scores")
    os.makedirs(program_scores_dir, exist_ok=True)
    for sample_id, program_scores in result.program_scores.items():
        program_scores.to_csv(
            os.path.join(program_scores_dir, f"{sample_id}_program_scores.csv"),
            index=False,
        )


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
    workers: int = 1,
    backend: str = "process",
    threads_per_worker: int = 1,
    cache_mode: str = "disk",
    cache_dir: Optional[str] = None,
    random_state: int = 0,
) -> None:
    options = AnalysisOptions(
        top_genes=top_genes,
        k_min=k_min,
        k_max=k_max,
        min_intersect_initial=min_intersect_initial,
        min_intersect_cluster=min_intersect_cluster,
        min_group_size=min_group_size,
        allow_weak_seeds=allow_weak_seeds,
        random_state=random_state,
        workers=workers,
        backend=backend,  # type: ignore[arg-type]
        threads_per_worker=threads_per_worker,
        cache_mode=cache_mode,  # type: ignore[arg-type]
        cache_dir=cache_dir,
    )
    analyze_files(data_dir, options=options, out_dir=out_dir)
    return None


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=["serial", "thread", "process"], default="process")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--cache-mode", choices=["none", "memory", "disk"], default="disk")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0] == "analyze":
        argv = argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

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
        workers=args.workers,
        backend=args.backend,
        threads_per_worker=args.threads_per_worker,
        cache_mode=args.cache_mode,
        cache_dir=args.cache_dir,
        random_state=args.random_state,
    )
