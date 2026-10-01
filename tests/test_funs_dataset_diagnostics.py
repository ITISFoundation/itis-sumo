import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from itis_sumo.data import analyze_dataset
from itis_sumo.data.funs_dataset_diagnostics import detect_raw_outliers

# --- detect_raw_outliers ------------------------------------------------------


def test_flags_single_extreme_value():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    summary = detect_raw_outliers(values)
    assert summary.count == 1
    assert summary.indices == [9]
    assert summary.fence_high < 100.0


def test_constant_column_flags_nothing_with_coincident_fences():
    summary = detect_raw_outliers([2.5] * 20)
    assert summary.count == 0
    assert summary.indices == []
    assert summary.fence_low == summary.fence_high == 2.5


def test_nan_rows_never_flagged_and_excluded_from_quartiles():
    clean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with_nan = [1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0]
    assert detect_raw_outliers(with_nan).indices == detect_raw_outliers(clean).indices
    # a NaN is never itself flagged
    spiky = [1.0, 2.0, 3.0, np.nan, 50.0]
    assert 3 not in detect_raw_outliers(spiky).indices


def test_log_scale_requires_strictly_positive_values():
    with pytest.raises(ValueError, match="strictly positive"):
        detect_raw_outliers([1.0, 0.0, 2.0], scale="log")
    with pytest.raises(ValueError, match="strictly positive"):
        detect_raw_outliers([-1.0, 2.0], scale="log")


def test_unknown_scale_raises():
    with pytest.raises(ValueError, match="Unknown scale"):
        detect_raw_outliers([1.0, 2.0], scale="sqrt")  # ty: ignore[invalid-argument-type]


class TestScaleAwareFences:
    """V46np: the fence is computed in the variable's SELECTED scale. For a
    log-scale (multiplicative) variable that means the geometric fence —
    `exp(k-th Tukey fence of ln values)` — not the arithmetic fence in
    disguise; and the log-basis choice (ln vs log10, an affine change on the
    log variable) must never move a single flag."""

    def test_log_fence_is_the_geometric_analogue(self):
        rng = np.random.default_rng(0)
        values = rng.lognormal(mean=1.0, sigma=1.2, size=400)
        q1_ln, q3_ln = np.percentile(np.log(values), [25, 75])
        iqr_ln = q3_ln - q1_ln
        summary = detect_raw_outliers(values, scale="log")
        assert summary.fence_low == pytest.approx(
            np.exp(q1_ln - 1.5 * iqr_ln), rel=1e-12
        )
        assert summary.fence_high == pytest.approx(
            np.exp(q3_ln + 1.5 * iqr_ln), rel=1e-12
        )
        # geometric (multiplicative) form, to within quantile-interpolation noise:
        q1, q3 = np.percentile(values, [25, 75])
        assert summary.fence_low == pytest.approx(q1 * (q1 / q3) ** 1.5, rel=1e-4)
        assert summary.fence_high == pytest.approx(q3 * (q3 / q1) ** 1.5, rel=1e-4)

    def test_linear_fence_on_lognormal_data_overflags_vs_geometric_fence(self):
        # the flip test: 3 injected outliers among benign multiplicative spread.
        rng = np.random.default_rng(1)
        values = rng.lognormal(mean=1.0, sigma=1.2, size=200)
        values[rng.choice(200, size=3, replace=False)] *= 50.0
        linear = detect_raw_outliers(values, scale="linear")
        log = detect_raw_outliers(values, scale="log")
        assert log.count >= 3  # the injected outliers are found
        assert linear.count > log.count + 3  # linear manufactures extras

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_log_base_choice_moves_no_flags(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.lognormal(mean=1.0, sigma=1.2, size=200)
        values[rng.choice(200, size=3, replace=False)] *= 50.0
        ln_space = detect_raw_outliers(values, scale="log")
        log10_space = detect_raw_outliers(np.log10(values), scale="linear")
        assert ln_space.count == log10_space.count
        assert ln_space.indices == log10_space.indices
        assert ln_space.fence_low == pytest.approx(
            10.0**log10_space.fence_low, rel=1e-9
        )
        assert ln_space.fence_high == pytest.approx(
            10.0**log10_space.fence_high, rel=1e-9
        )

    def test_log_fences_reported_in_original_units(self):
        rng = np.random.default_rng(3)
        values = rng.lognormal(mean=0.0, sigma=0.5, size=300)
        summary = detect_raw_outliers(values, scale="log")
        assert summary.fence_low > 0.0
        assert summary.fence_low < np.median(values) < summary.fence_high
        assert summary.count <= max(
            1, len(values) // 100
        )  # benign spread, tail-sized flags only


# --- analyze_dataset ----------------------------------------------------------


def _synthetic_df(n=200, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x_normal": rng.normal(10.0, 2.0, n),
            "x_lognormal": rng.lognormal(0.0, 0.5, n),
            "x_constant": [3.0] * n,
            "y_out": rng.normal(0.0, 1.0, n),
        }
    )


def test_analyze_dataset_returns_diagnostics_for_every_column():
    df = _synthetic_df()
    diag = analyze_dataset(
        df, input_cols=["x_normal", "x_lognormal", "x_constant"], output_cols=["y_out"]
    )
    assert set(diag.inputs) == {"x_normal", "x_lognormal", "x_constant"}
    assert set(diag.outputs) == {"y_out"}
    assert diag.outputs["y_out"].scale in ("linear", "log")
    assert diag.outputs["y_out"].outliers is not None
    assert diag.inputs["x_constant"].distribution == "constant"
    assert diag.inputs["x_constant"].confident is True


def test_analyze_dataset_selects_log_scale_for_lognormal_column():
    df = _synthetic_df()
    diag = analyze_dataset(df, input_cols=["x_lognormal"], output_cols=["y_out"])
    assert diag.inputs["x_lognormal"].scale == "log"
    assert diag.inputs["x_lognormal"].distribution == "normal"


def test_analyze_dataset_surfaces_injected_raw_outliers():
    df = _synthetic_df()
    df.loc[10, "x_lognormal"] *= 100.0
    diag = analyze_dataset(df, input_cols=["x_lognormal"], output_cols=["y_out"])
    summary = diag.inputs["x_lognormal"].outliers
    assert summary is not None
    assert summary.count >= 1
    assert 10 in summary.indices


def test_analyze_dataset_detail_only_with_flag_and_json_serializable():
    df = _synthetic_df(n=100, seed=8)
    lean = analyze_dataset(df, input_cols=["x_normal"], output_cols=["y_out"])
    assert lean.detail is None

    full = analyze_dataset(
        df, input_cols=["x_normal"], output_cols=["y_out"], include_detail=True
    )
    assert full.detail is not None
    assert "candidates" in full.detail["x_normal"]
    # the whole point of the plain-dataclass schema: crosses the flaskapi boundary
    payload = json.dumps(dataclasses.asdict(full))
    assert "x_normal" in payload
