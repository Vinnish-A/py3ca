from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from typing import Callable, Sequence, TypeVar

from .types import AnalysisOptions, Sample
from .utils import safe_filename

T = TypeVar("T")
R = TypeVar("R")


def effective_options(options: AnalysisOptions) -> AnalysisOptions:
    workers = max(1, int(options.workers or 1))
    backend = options.backend
    if workers <= 1:
        backend = "serial"
    return replace(options, workers=workers, backend=backend)


def map_tasks(fn: Callable[[T], R], tasks: Sequence[T], options: AnalysisOptions) -> list[R]:
    options = effective_options(options)
    if options.backend == "serial" or options.workers <= 1:
        return [fn(task) for task in tasks]

    executor_cls = ThreadPoolExecutor if options.backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=options.workers) as executor:
        return list(executor.map(fn, tasks))


def map_samples(
    fn: Callable[[Sample], R],
    samples: Sequence[Sample],
    options: AnalysisOptions,
) -> list[R]:
    return map_tasks(fn, samples, options)


def stage_in_memory_samples_if_needed(
    samples: Sequence[Sample],
    options: AnalysisOptions,
    directory: str,
) -> list[Sample]:
    options = effective_options(options)
    if options.backend != "process" or options.workers <= 1:
        return list(samples)

    os.makedirs(directory, exist_ok=True)
    staged: list[Sample] = []
    for sample in samples:
        sample.validate()
        if sample.adata is None:
            staged.append(sample)
            continue
        path = sample.path
        if path is None:
            path = os.path.join(directory, f"{safe_filename(sample.sample_id)}_raw.h5ad")
            sample.adata.write_h5ad(path)
        staged.append(
            Sample(
                sample_id=sample.sample_id,
                study_id=sample.study_id,
                path=path,
                adata=None,
            )
        )
    return staged
