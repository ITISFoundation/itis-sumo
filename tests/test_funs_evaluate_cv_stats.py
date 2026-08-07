"""Unit tests for the SuMo CV statistical-rigor helpers in `funs_evaluate.py` (§T18).

These cover the pure statistical functions (no Dakota/filesystem dependency) plus the
convergence-series orchestration (Dakota-dependent `evaluate_sumo_manual_crossvalidation`
call is monkeypatched to keep the tests fast and deterministic).
"""

import numpy as np
import pytest

from itis_sumo.evaluate.funs_evaluate import (
    _convergence_subset_sizes,
    compute_cv_accuracy_metrics,
    compute_cv_convergence,
    compute_paired_ttest,
)


class TestComputeCvAccuracyMetrics:
    def test_identical_arrays_yield_zero_error(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        metrics = compute_cv_accuracy_metrics(actual, predicted)
        assert metrics["root_mean_squared"] == pytest.approx(0.0)
        assert metrics["sum_abs"] == pytest.approx(0.0)
        assert metrics["mean_abs"] == pytest.approx(0.0)
        assert metrics["max_abs"] == pytest.approx(0.0)

    def test_known_residuals(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [2.0, 2.0, 3.0, 8.0]
        metrics = compute_cv_accuracy_metrics(actual, predicted)
        # residuals: -1, 0, 0, -4 -> abs: 1, 0, 0, 4
        assert metrics["sum_abs"] == pytest.approx(5.0)
        assert metrics["mean_abs"] == pytest.approx(1.25)
        assert metrics["max_abs"] == pytest.approx(4.0)
        assert metrics["root_mean_squared"] == pytest.approx(np.sqrt((1 + 0 + 0 + 16) / 4))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_cv_accuracy_metrics([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_nan_pair_from_dropped_cv_row_is_excluded(self):
        """B22: a fold row evaluate_sumo_manual_crossvalidation couldn't recover leaves
        NaN in actual/predicted at that position - it must be excluded, not turn every
        metric into NaN."""
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [2.0, 2.0, 3.0, float("nan")]
        metrics = compute_cv_accuracy_metrics(actual, predicted)
        # only the first 3 pairs contribute: residuals -1, 0, 0
        assert metrics["sum_abs"] == pytest.approx(1.0)
        assert metrics["mean_abs"] == pytest.approx(1.0 / 3)
        assert metrics["max_abs"] == pytest.approx(1.0)
        assert not np.isnan(metrics["root_mean_squared"])


class TestComputePairedTtest:
    def test_no_systematic_bias_high_pvalue(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(loc=0.0, scale=1.0, size=200)
        noise = rng.normal(loc=0.0, scale=0.01, size=200)
        predicted = actual + noise  # unbiased surrogate (symmetric noise)
        result = compute_paired_ttest(actual, predicted)
        assert "statistic" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_systematic_bias_detected_low_pvalue(self):
        rng = np.random.default_rng(7)
        actual = rng.normal(loc=0.0, scale=1.0, size=200)
        predicted = actual + 5.0  # constant offset -> strong systematic bias
        result = compute_paired_ttest(actual, predicted)
        assert result["p_value"] < 0.05

    def test_symmetric_residuals_zero_statistic(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [2.0, 1.0, 4.0, 3.0]  # residuals -1,+1,-1,+1 -> mean 0, non-zero variance
        result = compute_paired_ttest(actual, predicted)
        assert result["statistic"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_paired_ttest([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_paired_ttest([1.0], [1.0])

    def test_nan_pair_from_dropped_cv_row_is_excluded(self):
        """B22: same NaN-exclusion behavior as compute_cv_accuracy_metrics."""
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [2.0, 1.0, 4.0, 3.0, float("nan")]  # last pair dropped (NaN)
        result = compute_paired_ttest(actual, predicted)
        # remaining residuals -1,+1,-1,+1 -> mean 0, non-zero variance
        assert result["statistic"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)


class TestConvergenceSubsetSizes:
    def test_below_minimum_returns_empty(self):
        assert _convergence_subset_sizes(n_total=3, min_samples=5, max_points=5) == []

    def test_exactly_minimum_returns_single_point(self):
        assert _convergence_subset_sizes(n_total=5, min_samples=5, max_points=5) == [5]

    def test_evenly_spaced_and_capped(self):
        sizes = _convergence_subset_sizes(n_total=25, min_samples=5, max_points=5)
        assert sizes[0] == 5
        assert sizes[-1] == 25
        assert len(sizes) <= 5
        assert sizes == sorted(sizes)
        assert len(sizes) == len(set(sizes))

    def test_max_points_bounds_dakota_reruns(self):
        sizes = _convergence_subset_sizes(n_total=1000, min_samples=5, max_points=3)
        assert len(sizes) == 3
        assert sizes[0] == 5
        assert sizes[-1] == 1000


class TestComputeCvConvergence:
    def test_series_shape_and_calls_manual_cv_per_subset(self, tmp_path, monkeypatch):
        training_file = tmp_path / "df_processed_jobs.dat"
        n_total = 10
        rng = np.random.default_rng(0)
        x1 = rng.uniform(-1, 1, n_total)
        y = x1 * 2.0
        with open(training_file, "w") as f:
            f.write("x1 y\n")
            f.writelines(f"{xi} {yi}\n" for xi, yi in zip(x1, y))

        call_sizes = []

        def fake_manual_cv(run_dir, subset_file, input_vars, output_response, N_CROSS_VALIDATION=5):
            import pandas as pd

            df = pd.read_csv(subset_file, sep=" ")
            call_sizes.append(len(df))
            actual = df[output_response].astype(float).tolist()
            predicted = [v + 0.1 for v in actual]
            return {
                output_response: actual,
                output_response + "_hat": predicted,
                output_response + "_std_hat": [0.0] * len(actual),
            }

        monkeypatch.setattr(
            "itis_sumo.evaluate.funs_evaluate.evaluate_sumo_manual_crossvalidation",
            fake_manual_cv,
        )

        series = compute_cv_convergence(
            tmp_path, training_file, ["x1"], "y", min_samples=5, max_points=3
        )

        assert len(series) == len(call_sizes)
        assert [point["n_samples"] for point in series] == call_sizes
        assert call_sizes[0] == 5
        assert call_sizes[-1] == n_total
        for point in series:
            assert point["metric"] == pytest.approx(0.1)

    def test_empty_series_when_below_minimum(self, tmp_path):
        training_file = tmp_path / "df_processed_jobs.dat"
        with open(training_file, "w") as f:
            f.write("x1 y\n")
            f.write("0.1 0.2\n")
            f.write("0.2 0.4\n")

        series = compute_cv_convergence(tmp_path, training_file, ["x1"], "y", min_samples=5)
        assert series == []
