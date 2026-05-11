"""Project analysis types plus a stdlib ``types`` proxy.

The proxy keeps this file from breaking Python startup if the package directory
is accidentally placed on ``sys.path`` ahead of the standard library.
"""

from __future__ import annotations

import importlib.util
import os
import sysconfig

_MODULE_NAME = __name__


def _load_stdlib_types():
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if not stdlib_path:
        return None
    types_path = os.path.join(stdlib_path, "types.py")
    spec = importlib.util.spec_from_file_location("types", types_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_types = _load_stdlib_types()
if _stdlib_types is not None:
    for _name, _value in _stdlib_types.__dict__.items():
        if _name in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"}:
            continue
        globals()[_name] = _value
else:
    raise ImportError("Failed to load stdlib 'types' module")


if _MODULE_NAME != "types":
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Literal, Optional

    import pandas as pd

    from .program_types import Program

    @dataclass
    class Sample:
        sample_id: str
        study_id: str
        path: Optional[str] = None
        adata: Any = None

        def validate(self) -> None:
            if self.path is None and self.adata is None:
                raise ValueError("Sample requires either path or adata")

    @dataclass
    class AnalysisOptions:
        top_genes: int = 5000
        k_min: int = 4
        k_max: int = 12
        min_intersect_initial: int = 15
        min_intersect_cluster: int = 15
        min_group_size: int = 5
        allow_weak_seeds: bool = False
        random_state: int = 0
        workers: int = 1
        backend: Literal["serial", "process", "thread"] = "process"
        threads_per_worker: int = 1
        cache_mode: Literal["none", "memory", "disk"] = "disk"
        cache_dir: Optional[str] = None

    @dataclass
    class PreparedSampleRef:
        sample_id: str
        study_id: str
        path: Optional[str] = None
        adata: Any = None

    @dataclass
    class SampleAnalysis:
        sample_id: str
        study_id: str
        programs: List[Program]
        program_scores: pd.DataFrame
        prepared_sample: Optional[PreparedSampleRef] = None
        diagnostics: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class CohortAnalysis:
        programs: pd.DataFrame
        robust_programs: pd.DataFrame
        components_by_sample: pd.DataFrame
        clusters: Dict[str, List[str]]
        meta_programs: Dict[str, List[str]]
        meta_programs_by_sample: pd.DataFrame
        mp_scores: Dict[str, pd.DataFrame]
        program_scores: Dict[str, pd.DataFrame]
        options: AnalysisOptions
        diagnostics: Dict[str, Any]
        components_by_sample_map: Dict[str, List[str]] = field(default_factory=dict)
        meta_programs_by_sample_map: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
