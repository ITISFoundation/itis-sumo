"""E1 SuMo model export/import: model-store unit tests + real-Dakota round-trip
(T12-T14, T17bc).

The round-trip test is a genuine (unmocked) Dakota run: it is the empirical
test SPEC.md R2 calls for, resolving whether Dakota's `import_model` needs the
literal export-time training file path, or just a same-content file present at
import time. It builds a surrogate, exports it, then re-imports it from a
different run_dir (simulating a separate session) and asserts the reimported
surrogate's predictions match the freshly-built one.

The model store persists a copy of the real training data by default (kept
for reference/debuggability). The placeholder-fallback test is the empirical
basis for R8/R9: it shows that even though the stored copy is not technically
required -- Dakota's reloaded surrogate ignores the `import_build_points_file`
row values entirely, matching only on the header's descriptor set (against the
Dakota study's own `variables` block, not the archive) -- a missing stored
copy can be safely papered over with a header-only placeholder instead of
crashing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itis_sumo.core.sumo_model_store import (
    get_models_dir,
    load_model_metadata,
    stage_model_for_import,
    store_exported_model,
)
from itis_sumo.evaluate.funs_evaluate import (
    evaluate_sumo,
    export_sumo_model,
    import_sumo_model,
)
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor

SEED = 42

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    """Every test gets its own models dir, never the real cwd/sumo_models."""
    monkeypatch.setenv("ITIS_SUMO_MODELS_DIR", str(tmp_path / "models"))


def _make_training_and_eval_files(run_dir: Path):
    """Synthetic 2-input/1-output problem, mirroring examples/headless_smoke.py."""
    rng = np.random.default_rng(SEED)
    n = 25
    length = rng.uniform(0.0, 1.0, size=n)
    width = rng.uniform(0.0, 1.0, size=n)
    stress = 3.0 * length + 2.0 * width**2
    train_raw = pd.DataFrame({"length": length, "width": width, "stress": stress})

    preprocessor = DataPreprocessor()
    preprocessor.setup_variables(["length", "width"], ["stress"])
    preprocessor.fit(train_raw)
    train_processed = preprocessor.transform(train_raw)
    training_file = run_dir / "df_processed_jobs.dat"
    train_processed.to_csv(training_file, sep=" ", index=False)

    eval_raw = pd.DataFrame({"length": [0.25, 0.75], "width": [0.25, 0.75]})
    eval_processed = preprocessor.transform(eval_raw)
    eval_file = run_dir / "eval_samples.dat"
    eval_processed.to_csv(eval_file, sep=" ", index=False)

    return training_file, eval_file


def test_get_models_dir_honors_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom_models"
    monkeypatch.setenv("ITIS_SUMO_MODELS_DIR", str(override))
    assert get_models_dir() == override
    assert override.is_dir()


def test_load_model_metadata_unknown_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No SuMo model found"):
        load_model_metadata("does-not-exist")


def test_store_exported_model_no_archive_files_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No exported model files"):
        store_exported_model(
            run_dir=run_dir,
            training_file=run_dir / "missing.dat",
            surrogate_conf_block="whatever",
            input_descriptors=["x1"],
            output_descriptor="y1",
            export_prefix="export",
        )


def test_stage_model_for_import_unknown_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No SuMo model found"):
        stage_model_for_import("does-not-exist", tmp_path / "run")


def test_export_then_import_round_trip_predictions_match(tmp_path):
    """Empirical resolution of R2: import from a copied (not original-path)
    training file, in a fresh run_dir, reproduces the export-time predictions.
    """
    export_run_dir = tmp_path / "export_run"
    export_run_dir.mkdir()
    training_file, eval_file = _make_training_and_eval_files(export_run_dir)

    export_results, sumo_model_id = export_sumo_model(
        export_run_dir,
        training_file,
        eval_file,
        ["x1", "x2"],
        "y1",
    )
    assert len(export_results["y1_hat"]) == 2

    metadata = load_model_metadata(sumo_model_id)
    assert metadata.input_descriptors == ["x1", "x2"]
    assert metadata.output_descriptor == "y1"
    assert metadata.export_format == "text_archive"
    assert "export_model" in metadata.surrogate_conf_block

    # Simulate a separate session: fresh run_dir, no access to export_run_dir.
    import_run_dir = tmp_path / "import_run"
    import_run_dir.mkdir()
    import_results = import_sumo_model(
        import_run_dir,
        sumo_model_id,
        eval_file,
        ["x1", "x2"],
        "y1",
    )

    assert np.allclose(import_results["y1_hat"], export_results["y1_hat"], atol=1e-6)


def test_export_persists_real_training_data(tmp_path):
    """The model store keeps a copy of the real training data alongside the
    metadata sidecar and Dakota's own archive files, for reference/
    debuggability -- not just the archive."""
    export_run_dir = tmp_path / "export_run"
    export_run_dir.mkdir()
    training_file, eval_file = _make_training_and_eval_files(export_run_dir)

    _, sumo_model_id = export_sumo_model(
        export_run_dir, training_file, eval_file, ["x1", "x2"], "y1"
    )

    stored_files = sorted(f.name for f in get_models_dir().glob(f"{sumo_model_id}.*"))
    assert stored_files == [
        f"{sumo_model_id}.metadata.json",
        f"{sumo_model_id}.processed_training.dat",
        f"{sumo_model_id}.y1.sps",
    ]
    stored_training_file = get_models_dir() / f"{sumo_model_id}.processed_training.dat"
    assert stored_training_file.read_text() == training_file.read_text()


def test_stage_model_for_import_stages_real_training_data(tmp_path):
    """The staged points file is a copy of the real, stored training data
    (not a placeholder) when that stored copy is present."""
    export_run_dir = tmp_path / "export_run"
    export_run_dir.mkdir()
    training_file, eval_file = _make_training_and_eval_files(export_run_dir)

    _, sumo_model_id = export_sumo_model(
        export_run_dir, training_file, eval_file, ["x1", "x2"], "y1"
    )

    import_run_dir = tmp_path / "import_run"
    _, staged_file = stage_model_for_import(sumo_model_id, import_run_dir)

    assert staged_file.read_text() == training_file.read_text()


def test_stage_model_for_import_falls_back_to_placeholder_when_missing(
    tmp_path, caplog
):
    """R9: if the stored training-data copy is missing (e.g. a legacy model,
    or the file was deleted out-of-band), staging falls back to a
    header-only placeholder instead of crashing, with a loud warning --
    verified safe because Dakota never reads the points file's row values
    back on import."""
    export_run_dir = tmp_path / "export_run"
    export_run_dir.mkdir()
    training_file, eval_file = _make_training_and_eval_files(export_run_dir)

    _, sumo_model_id = export_sumo_model(
        export_run_dir, training_file, eval_file, ["x1", "x2"], "y1"
    )
    (get_models_dir() / f"{sumo_model_id}.processed_training.dat").unlink()

    import_run_dir = tmp_path / "import_run"
    with caplog.at_level("WARNING"):
        _, staged_file = stage_model_for_import(sumo_model_id, import_run_dir)

    assert staged_file.read_text().splitlines() == ["x1 x2 y1"]
    assert any(
        "no stored training-data file" in record.message for record in caplog.records
    )


def test_import_sumo_model_binary_archive_round_trip(tmp_path):
    """R10: `binary_archive` is documented as re-importable (unlike
    `algebraic_file`/`algebraic_console`) -- verify it actually round-trips."""
    export_run_dir = tmp_path / "export_run"
    export_run_dir.mkdir()
    training_file, eval_file = _make_training_and_eval_files(export_run_dir)

    export_results, sumo_model_id = export_sumo_model(
        export_run_dir,
        training_file,
        eval_file,
        ["x1", "x2"],
        "y1",
        export_format="binary_archive",
    )

    stored_files = sorted(f.name for f in get_models_dir().glob(f"{sumo_model_id}.*"))
    assert stored_files == [
        f"{sumo_model_id}.metadata.json",
        f"{sumo_model_id}.processed_training.dat",
        f"{sumo_model_id}.y1.bsps",
    ]

    import_run_dir = tmp_path / "import_run"
    import_run_dir.mkdir()
    import_results = import_sumo_model(
        import_run_dir, sumo_model_id, eval_file, ["x1", "x2"], "y1"
    )
    assert np.allclose(import_results["y1_hat"], export_results["y1_hat"], atol=1e-6)


def test_import_sumo_model_unknown_id_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _, eval_file = _make_training_and_eval_files(run_dir)
    with pytest.raises(FileNotFoundError, match="No SuMo model found"):
        import_sumo_model(run_dir, "does-not-exist", eval_file, ["x1", "x2"], "y1")


def test_import_sumo_model_mismatched_descriptors_raises(tmp_path):
    export_run_dir = tmp_path / "export_run"
    export_run_dir.mkdir()
    training_file, eval_file = _make_training_and_eval_files(export_run_dir)

    _, sumo_model_id = export_sumo_model(
        export_run_dir, training_file, eval_file, ["x1", "x2"], "y1"
    )

    import_run_dir = tmp_path / "import_run"
    import_run_dir.mkdir()
    with pytest.raises(ValueError, match="was exported with inputs"):
        import_sumo_model(import_run_dir, sumo_model_id, eval_file, ["x2", "x1"], "y1")


def test_export_predictions_match_plain_evaluate_sumo(tmp_path):
    """Exporting must not change the surrogate's own predictions vs. the
    non-exporting evaluate_sumo path (export is a side effect, not a
    different fit)."""
    run_dir_a = tmp_path / "plain"
    run_dir_a.mkdir()
    training_file, eval_file = _make_training_and_eval_files(run_dir_a)
    plain_results = evaluate_sumo(
        run_dir_a, training_file, eval_file, ["x1", "x2"], "y1"
    )

    run_dir_b = tmp_path / "exported"
    run_dir_b.mkdir()
    export_results, _ = export_sumo_model(
        run_dir_b, training_file, eval_file, ["x1", "x2"], "y1"
    )

    assert np.allclose(export_results["y1_hat"], plain_results["y1_hat"], atol=1e-6)
