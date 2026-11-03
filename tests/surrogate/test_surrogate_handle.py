"""T57rc SuMo handle: fit -> catalog methods -> save/load/predict (V27fq, V24af).

Round-trip tests are real (unmocked) Dakota runs mirroring
`tests/test_sumo_model_store.py`; the failure-lifetime and loaded-query guard
tests are engine-free (unit tier in the same file for cohesion).
"""

import numpy as np
import pandas as pd
import pytest

from itis_sumo import SuMo, fit, load
from itis_sumo.errors import SumoEngineError, SumoInputError
from itis_sumo.surrogate._session import SumoSession
from itis_sumo.tabular.preprocessor import DataPreprocessor

SEED = 42

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ITIS_SUMO_MODELS_DIR", str(tmp_path / "models"))


def _samples(n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    length = rng.uniform(0.05, 1.0, size=n)
    width = rng.uniform(0.05, 1.0, size=n)
    stress = 3.0 * length + 2.0 * width**2
    return pd.DataFrame({"length": length, "width": width, "stress": stress})


VARIABLES = ["length", "width"]
RESPONSE = "stress"


def test_fit_returns_a_live_handle_with_the_catalog():
    model = fit(_samples(), VARIABLES, RESPONSE)
    assert isinstance(model, SuMo)
    for name in ("cross_validate", "along_axes", "grid", "sobol", "predict", "save"):
        assert callable(getattr(model, name))
    model.close()


def test_fit_predict_matches_cross_validate_shape():
    model = fit(_samples(), VARIABLES, RESPONSE)
    fresh = _samples(6)
    result = model.predict(fresh[VARIABLES].assign(**{RESPONSE: 0.0}))
    assert result.response == RESPONSE
    assert len(result.values) == 6
    assert result.sumo_model_id is None  # a fitted handle, not a loaded one
    model.close()


def test_save_load_predict_round_trip():
    model = fit(_samples(), VARIABLES, RESPONSE)
    fresh = _samples(6)
    probe = fresh[VARIABLES].assign(**{RESPONSE: 0.0})
    fitted = model.predict(probe)
    sumo_model_id = model.save()
    model.close()

    reopened = load(sumo_model_id)
    loaded = reopened.predict(probe)
    # V22rs: both in original units; R2/R9: the archive reproduces predictions.
    assert loaded.sumo_model_id == sumo_model_id
    np.testing.assert_allclose(loaded.values, fitted.values, rtol=1e-6, atol=1e-6)
    reopened.close()


def test_saved_sidecar_persists_preprocessing_config():
    model = fit(_samples(), VARIABLES, RESPONSE)
    sumo_model_id = model.save()
    model.close()
    from itis_sumo.engine.sumo_model_store import load_model_metadata

    metadata = load_model_metadata(sumo_model_id)
    assert metadata.preprocessor_config is not None  # T26eq closes the E1 gap


def test_legacy_sidecar_without_config_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ITIS_SUMO_MODELS_DIR", str(tmp_path / "models"))
    from itis_sumo.engine.sumo_model_store import get_models_dir

    models_dir = get_models_dir()
    sidecar = models_dir / "legacyid.metadata.json"
    sidecar.write_text(
        '{"sumo_model_id":"legacyid","surrogate_conf_block":"x",'
        '"input_descriptors":["x1","x2"],"output_descriptor":"y1",'
        '"export_format":"text_archive","preprocessor_config":null}'
    )
    with pytest.raises(SumoInputError, match="preprocessing persistence"):
        load("legacyid")


def test_loaded_handle_rejects_retraining_queries():
    model = fit(_samples(), VARIABLES, RESPONSE)
    sumo_model_id = model.save()
    model.close()
    reopened = load(sumo_model_id)
    with pytest.raises(SumoInputError, match="loaded model"):
        reopened.cross_validate(folds=2, seed=SEED)
    reopened.close()


def test_predict_handles_a_wide_dynamic_range_response():
    rng = np.random.default_rng(SEED)
    n = 30
    length = rng.uniform(1.0, 100.0, size=n)
    stress = 3.0 * np.sqrt(length)
    samples = pd.DataFrame({"length": length, "stress": stress})
    model = fit(samples, ["length"], "stress", preprocessing=None)
    probe = samples[["length", "stress"]].head(4)
    result = model.predict(probe)
    assert result.values and all(v > 0 for v in result.values)
    model.close()


def test_close_keeps_a_workspace_run_dir(tmp_path):
    workspace = tmp_path / "ws"
    model = fit(_samples(), VARIABLES, RESPONSE, workspace=workspace)
    run_dir = model._run_dir
    assert run_dir is not None and run_dir.is_dir()
    model.close()
    model.close()  # idempotent
    # A workspace run dir is intentionally kept (caller asked to keep files).
    assert run_dir.is_dir()


def test_close_discards_a_temp_run_dir_on_success():
    model = fit(_samples(), VARIABLES, RESPONSE)
    run_dir = model._run_dir
    assert run_dir is not None and run_dir.is_dir()
    model.close()
    assert not run_dir.exists()


def test_engine_failure_preserves_run_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ITIS_SUMO_MODELS_DIR", str(tmp_path / "models"))
    import itis_sumo.surrogate._session as session_module

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic Dakota abort")

    monkeypatch.setattr(session_module, "evaluate_sumo", _boom)
    model = fit(_samples(), VARIABLES, RESPONSE)
    with pytest.raises(SumoEngineError):
        model.predict(_samples(3).assign(**{RESPONSE: 0.0}))
    run_dir = model._run_dir
    assert run_dir is not None  # V24af: evidence survives on failure
    model.close()
    assert run_dir.is_dir()  # preserved


def test_sumo_is_sumosession_alias():
    assert SuMo is SumoSession


def test_preprocessor_config_round_trip():
    pre = DataPreprocessor()
    pre.setup_variables(["a", "b"], ["y"])
    pre.setup_log_transform(input_log_vars=["a"], output_log_vars=[])
    data = pd.DataFrame(
        {
            "a": np.linspace(1.0, 10.0, 8),
            "b": np.linspace(0.0, 1.0, 8),
            "y": np.arange(8.0),
        }
    )
    pre.fit(data)
    clone = DataPreprocessor.from_config(pre.to_config())
    a = pre.transform(data)
    b = clone.transform(data)
    pd.testing.assert_frame_equal(a, b)
