"""py3ca package."""

from .pipeline import (
    analyze_cohort,
    analyze_files,
    discover_sample_programs,
    estimate_study_baselines,
    prepare_sample,
    run_pipeline,
    save_analysis,
)
from .scoring import score_cells_by_meta_programs, score_cells_by_sample_programs
from .types import AnalysisOptions, CohortAnalysis, PreparedSampleRef, Sample, SampleAnalysis

try:
    from .plot import plot_heatmap
except ImportError as _plot_import_error:

    def plot_heatmap(*args, _error=_plot_import_error, **kwargs):
        raise ImportError("plot_heatmap requires optional plotting dependencies") from _error

__all__ = [
    "AnalysisOptions",
    "CohortAnalysis",
    "PreparedSampleRef",
    "Sample",
    "SampleAnalysis",
    "analyze_cohort",
    "analyze_files",
    "discover_sample_programs",
    "estimate_study_baselines",
    "plot_heatmap",
    "prepare_sample",
    "run_pipeline",
    "save_analysis",
    "score_cells_by_meta_programs",
    "score_cells_by_sample_programs",
]
