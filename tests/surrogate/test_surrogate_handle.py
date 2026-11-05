"""T57rc SuMo handle: fit -> catalog methods -> save/load/predict (V27fq, V24af).

Round-trip tests are real (unmocked) Dakota runs mirroring
`tests/test_sumo_model_store.py`; the failure-lifetime and loaded-query guard
tests are engine-free (unit tier in the same file for cohesion).
"""

import numpy as np
import pandas as pd
import pytest

from itis_sumo import SuMo, fit, load
from itis_sumo.engine.evaluation import _UQ_METRICS
from itis_sumo.errors import SumoEngineError, SumoInputError
from itis_sumo.surrogate._session import SumoSession
from itis_sumo.tabular.preprocessor import DataPreprocessor
from itis_sumo.types import DistributionSpec, PreprocessingSpec, VariableSpec

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


# ------------------------------------------------- uq_convergence_curve (T67kb)
#
# The sweep ASSEMBLY contract, with the per-draw refit+propagate step replaced by
# a recorder: what these pin is sizes, subset semantics, seed derivation,
# aggregation, and the warn-and-continue lifetime rule — never Dakota. The
# numbers the curve CONVERGES TO are the analytical suite's job
# (tests/engine/test_metamodeling_analytical.py, T67kb's acceptance clause).

CURVE_DISTRIBUTIONS = {
    "length": DistributionSpec("uniform", minimum=0.1, maximum=1.0),
    "width": DistributionSpec("uniform", minimum=0.2, maximum=0.9),
}


def _record_propagation(monkeypatch, fail=None):
    """Swap the per-draw refit+propagate step for a recorder of its arguments.

    `fail(size, draw)` answers True for the (subset size, per-size attempt index)
    pairs that raise — the synthetic stand-in for "Dakota could not train this
    subset", which the handle must turn into a point warning, not a run failure.
    A draw's successful metrics are the token map {token: float(its seed)}, so a
    test can check aggregation against the recorded seeds without hard-coding
    random numbers.
    """
    calls: list = []
    attempts: dict[int, int] = {}

    def fake(self, rows, distributions, *, num_samples, n_histograms, seed):
        size = len(rows)
        draw = attempts.get(size, 0)
        attempts[size] = draw + 1
        if fail is None or not fail(size, draw):
            calls.append(
                {
                    "size": size,
                    "rows": list(rows),
                    "seed": seed,
                    "num_samples": num_samples,
                    "n_histograms": n_histograms,
                }
            )
            return {token: float(seed) for token in _UQ_METRICS}
        raise SumoEngineError(f"synthetic abort at n_samples={size}")

    monkeypatch.setattr(SumoSession, "_uq_propagation_metrics", fake)
    return calls


def test_uq_convergence_curve_single_draw_point_shape(monkeypatch):
    calls = _record_propagation(monkeypatch)
    model = fit(_samples(), VARIABLES, RESPONSE)
    curve = model.uq_convergence_curve(
        distributions=CURVE_DISTRIBUTIONS,
        min_samples=5,
        max_points=4,
        num_samples=137,
        n_histograms=3,
        seed=11,
    )
    assert [point["n_samples"] for point in curve] == [5, 11, 18, 25]
    assert [call["size"] for call in calls] == [5, 11, 18, 25]
    for point, call in zip(curve, calls):
        # the deterministic single-draw shape: first `size` rows on the base seed
        assert call["rows"] == list(range(call["size"]))
        assert call["seed"] == 11
        assert call["num_samples"] == 137 and call["n_histograms"] == 3
        assert set(point["metrics"]) == set(_UQ_METRICS)
        assert point["metrics"] == dict.fromkeys(_UQ_METRICS, 11.0)
        assert all(spread == 0.0 for spread in point["metrics_std"].values())
        assert point["n_bootstrap"] == 1
        assert point["warnings"] == []
        assert point["num_samples"] == 137 and point["n_histograms"] == 3
        assert point["seed"] == 11
        assert point["distributions"].keys() == CURVE_DISTRIBUTIONS.keys(), (
            "the spec is echoed so a saved curve states what it was propagated against"
        )
    model.close()


def test_uq_convergence_curve_bootstrap_aggregates_distinct_subsets(monkeypatch):
    calls = _record_propagation(monkeypatch)
    model = fit(_samples(), VARIABLES, RESPONSE)
    curve = model.uq_convergence_curve(
        distributions=CURVE_DISTRIBUTIONS,
        min_samples=5,
        max_points=2,
        n_bootstrap=3,
        seed=11,
    )
    assert [point["n_samples"] for point in curve] == [5, 25]

    first, last = curve
    seeds = [call["seed"] for call in calls if call["size"] == 5]
    subsets = [call["rows"] for call in calls if call["size"] == 5]
    assert len(seeds) == 3
    assert first["n_bootstrap"] == 3
    for subset in subsets:
        assert len(subset) == len(set(subset)) == 5, (
            "distinct rows — never with replacement"
        )
        assert set(subset) <= set(range(25))
    assert len({frozenset(subset) for subset in subsets}) == 3, "distinct SUBSETS"
    assert len(set(seeds)) == 3, "one draw's randomness is not another draw's"
    for token in _UQ_METRICS:
        assert first["draws_by_metric"][token] == pytest.approx(seeds)
        assert first["metrics"][token] == pytest.approx(float(np.mean(seeds)))
        assert first["metrics_std"][token] == pytest.approx(float(np.std(seeds)))

    # the full-sample point runs once: there is only one subset of that size
    assert last["n_bootstrap"] == 1
    assert [call["rows"] for call in calls if call["size"] == 25] == [list(range(25))]
    assert all(spread == 0.0 for spread in last["metrics_std"].values())
    model.close()


def test_uq_convergence_curve_points_ignore_the_sweep_shape(monkeypatch):
    """V25sd: a point's rows and seeds derive from (base, size, draw) alone, so
    changing `max_points` cannot renumber the points that stayed the same."""
    model = fit(_samples(), VARIABLES, RESPONSE)

    calls_short = _record_propagation(monkeypatch)
    model.uq_convergence_curve(
        distributions=CURVE_DISTRIBUTIONS,
        min_samples=5,
        max_points=2,
        n_bootstrap=2,
        seed=11,
    )
    calls_long = _record_propagation(monkeypatch)
    model.uq_convergence_curve(
        distributions=CURVE_DISTRIBUTIONS,
        min_samples=5,
        max_points=4,
        n_bootstrap=2,
        seed=11,
    )
    short_at_5 = [(c["rows"], c["seed"]) for c in calls_short if c["size"] == 5]
    long_at_5 = [(c["rows"], c["seed"]) for c in calls_long if c["size"] == 5]
    assert short_at_5 == long_at_5
    model.close()


def test_uq_convergence_curve_warns_and_continues_per_draw(monkeypatch):
    calls = _record_propagation(
        monkeypatch, fail=lambda size, draw: size == 18 and draw == 1
    )
    model = fit(_samples(), VARIABLES, RESPONSE)
    curve = model.uq_convergence_curve(
        distributions=CURVE_DISTRIBUTIONS,
        min_samples=5,
        max_points=4,
        n_bootstrap=2,
        seed=11,
    )
    by_size = {point["n_samples"]: point for point in curve}

    survivor = [call for call in calls if call["size"] == 18]
    assert len(survivor) == 1, "the failed draw propagated nothing"
    point = by_size[18]
    assert len(point["warnings"]) == 1
    assert "n_samples=18 draw 1" in point["warnings"][0]
    assert point["n_bootstrap"] == 1, "the point reports the draws that PRODUCED it"
    assert point["metrics"] == dict.fromkeys(_UQ_METRICS, float(survivor[0]["seed"]))
    for size in (5, 11):
        assert by_size[size]["warnings"] == []
        assert by_size[size]["n_bootstrap"] == 2
    assert by_size[25]["n_bootstrap"] == 1, "the full-sample point always runs once"
    model.close()


def test_uq_convergence_curve_keeps_the_axis_when_every_draw_fails(monkeypatch):
    calls = _record_propagation(monkeypatch, fail=lambda size, draw: size == 11)
    model = fit(_samples(), VARIABLES, RESPONSE)
    curve = model.uq_convergence_curve(
        distributions=CURVE_DISTRIBUTIONS,
        min_samples=5,
        max_points=4,
        n_bootstrap=2,
        seed=11,
    )
    assert [point["n_samples"] for point in curve] == [5, 11, 18, 25], (
        "a gap beats a shorter curve"
    )
    dead = curve[1]
    assert all(value is None for value in dead["metrics"].values())
    assert all(value is None for value in dead["metrics_std"].values())
    assert dead["draws_by_metric"] == {token: [] for token in _UQ_METRICS}
    assert dead["n_bootstrap"] == 0
    assert len(dead["warnings"]) == 2
    assert all(call["size"] != 11 for call in calls)
    model.close()


def test_uq_convergence_curve_rejects_caller_errors_before_the_sweep(monkeypatch):
    calls = _record_propagation(monkeypatch)
    model = fit(_samples(), VARIABLES, RESPONSE)
    with pytest.raises(SumoInputError, match="cover variables exactly"):
        model.uq_convergence_curve(
            distributions={"length": CURVE_DISTRIBUTIONS["length"]}
        )
    with pytest.raises(SumoInputError, match="below the 5 samples"):
        model.uq_convergence_curve(distributions=CURVE_DISTRIBUTIONS, min_samples=4)
    with pytest.raises(SumoInputError, match="leave no room"):
        model.uq_convergence_curve(distributions=CURVE_DISTRIBUTIONS, min_samples=30)
    logged = fit(
        _samples(),
        VARIABLES,
        RESPONSE,
        preprocessing=PreprocessingSpec(
            overrides={"length": VariableSpec(scale="log")}
        ),
    )
    with pytest.raises(SumoInputError, match="log-scale"):
        logged.uq_convergence_curve(
            distributions={
                "length": DistributionSpec("uniform", minimum=0.0, maximum=1.0),
                "width": CURVE_DISTRIBUTIONS["width"],
            }
        )
    assert calls == [], "every caller error must surface before the first engine call"
    model.close()
    logged.close()


def test_loaded_handle_rejects_uq_convergence_curve():
    model = fit(_samples(), VARIABLES, RESPONSE)
    sumo_model_id = model.save()
    model.close()
    reopened = load(sumo_model_id)
    with pytest.raises(SumoInputError, match="loaded model"):
        reopened.uq_convergence_curve(distributions=CURVE_DISTRIBUTIONS)
    reopened.close()
