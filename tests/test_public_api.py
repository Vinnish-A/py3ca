from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _adata(sample_id: str, offset: int) -> ad.AnnData:
    matrix = np.arange(offset, offset + 60, dtype=float).reshape(4, 15)
    adata = ad.AnnData(sp.csr_matrix(matrix))
    adata.obs_names = [f"{sample_id}_c{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    return adata


def _samples(py3ca):
    return [
        py3ca.Sample("S1", "StudyA", adata=_adata("S1", 1)),
        py3ca.Sample("S2", "StudyA", adata=_adata("S2", 2)),
    ]


def _options(py3ca, **kwargs):
    values = {
        "top_genes": 10,
        "k_min": 2,
        "k_max": 2,
        "workers": 1,
        "backend": "serial",
        "cache_mode": "memory",
    }
    values.update(kwargs)
    return py3ca.AnalysisOptions(**values)


def test_analyze_cohort_accepts_in_memory_anndata_and_save_analysis(tmp_path):
    import py3ca

    result = py3ca.analyze_cohort(_samples(py3ca), _options(py3ca))
    assert result.programs.shape == (40, 7)
    assert sorted(result.program_scores) == ["S1", "S2"]

    py3ca.save_analysis(result, tmp_path)
    expected = [
        "programs_all.csv",
        "programs_robust.csv",
        "components_by_sample.csv",
        "components_by_sample.json",
        "clusters.json",
        "meta_programs.json",
        "meta_programs.csv",
        "meta_programs_by_sample.json",
        "meta_programs_by_sample.csv",
        "scores/S1_mp_scores.csv",
        "scores/S2_mp_scores.csv",
        "program_scores/S1_program_scores.csv",
        "program_scores/S2_program_scores.csv",
    ]
    assert not [path for path in expected if not (tmp_path / path).exists()]


def test_workers_one_and_two_keep_program_ids_and_shapes(tmp_path):
    import py3ca

    serial = py3ca.analyze_cohort(_samples(py3ca), _options(py3ca))
    process = py3ca.analyze_cohort(
        _samples(py3ca),
        _options(py3ca, workers=2, backend="process", cache_mode="disk", cache_dir=str(tmp_path)),
    )

    assert serial.components_by_sample["program_id"].tolist() == process.components_by_sample[
        "program_id"
    ].tolist()
    assert serial.programs.shape == process.programs.shape


def test_run_pipeline_returns_none_and_writes_outputs(tmp_path):
    import py3ca

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    s1_path = data_dir / "s1.h5ad"
    s2_path = data_dir / "s2.h5ad"
    _adata("S1", 1).write_h5ad(s1_path)
    _adata("S2", 2).write_h5ad(s2_path)
    pd.DataFrame(
        [
            {"sample_id": "S1", "study_id": "StudyA", "path": os.path.basename(s1_path)},
            {"sample_id": "S2", "study_id": "StudyA", "path": os.path.basename(s2_path)},
        ]
    ).to_csv(data_dir / "manifest.csv", index=False)

    value = py3ca.run_pipeline(
        str(data_dir),
        str(out_dir),
        top_genes=10,
        k_min=2,
        k_max=2,
        cache_mode="memory",
    )

    assert value is None
    assert (out_dir / "programs_all.csv").exists()
    assert (out_dir / "program_scores" / "S1_program_scores.csv").exists()
