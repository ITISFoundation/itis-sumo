"""Unit tests for the SuMo CV statistical-rigor helpers in `funs_evaluate.py` (§T18).

These cover the pure statistical functions (no Dakota/filesystem dependency) plus the
convergence-series orchestration (Dakota-dependent `evaluate_sumo_manual_crossvalidation`
call is monkeypatched to keep the tests fast and deterministic).
"""

import inspect
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from itis_sumo.engine.evaluation import (
    _CV_ACCURACY_METRICS,
    _CV_COVERAGE_METRICS,
    _CV_METRICS,
    _convergence_subset_sizes,
    _uq_quantile_grid,
    compute_coverage,
    compute_cv_accuracy_metrics,
    compute_cv_convergence,
    compute_cv_diagnostics,
    compute_paired_ttest,
    fit_convergence_exponential,
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
        assert metrics["root_mean_squared"] == pytest.approx(
            np.sqrt((1 + 0 + 0 + 16) / 4)
        )

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
        predicted = [
            2.0,
            1.0,
            4.0,
            3.0,
        ]  # residuals -1,+1,-1,+1 -> mean 0, non-zero variance
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


class TestComputeCvDiagnostics:
    def test_returns_rmse_mae_ttest_cohens_d_in_one_call(self):
        """§T19df/V16wq: a single call surfaces every metric the convergence
        study needs -- no second CV rerun required for a different metric."""
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [2.0, 1.0, 4.0, 3.0]  # residuals -1,+1,-1,+1 -> mean 0
        diag = compute_cv_diagnostics(actual, predicted)
        assert diag["root_mean_squared"] == pytest.approx(1.0)
        assert diag["mean_abs"] == pytest.approx(1.0)
        assert diag["ttest_statistic"] == pytest.approx(0.0)
        assert diag["ttest_p_value"] == pytest.approx(1.0)
        assert diag["cohens_d"] == pytest.approx(0.0)
        assert diag["actual"] == pytest.approx(actual)
        assert diag["predicted"] == pytest.approx(predicted)

    def test_tukey_flags_extreme_residual(self):
        """One wildly-off pair among otherwise-tight residuals gets flagged and
        excluded from the filtered MAE, without touching the unfiltered `mean_abs`
        (V17kb: unfiltered MAE stays primary, filtering is a secondary view)."""
        actual = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        predicted = [1.1, 1.9, 3.1, 3.9, 5.1, 0.0]  # last pair: residual +100
        diag = compute_cv_diagnostics(actual, predicted)
        assert diag["n_outliers"] == 1
        assert diag["mean_abs_filtered"] < diag["mean_abs"]

    def test_no_systematic_bias_gives_small_effect_size(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(loc=0.0, scale=1.0, size=200)
        predicted = actual + rng.normal(loc=0.0, scale=0.01, size=200)
        diag = compute_cv_diagnostics(actual, predicted)
        assert abs(diag["cohens_d"]) < 0.2

    def test_constant_bias_gives_large_effect_size_regardless_of_n(self):
        """Cohen's d (unlike the t-test p-value) isn't confounded by sample size --
        the same constant bias yields ~the same effect size at N=20 and N=200."""
        rng = np.random.default_rng(7)
        for n in (20, 200):
            actual = rng.normal(loc=0.0, scale=1.0, size=n)
            predicted = actual + 2.0
            diag = compute_cv_diagnostics(actual, predicted)
            assert abs(diag["cohens_d"]) > 0.8

    def test_fewer_than_two_points_returns_nan_diagnostics_not_raise(self):
        diag = compute_cv_diagnostics([1.0], [2.0])
        assert np.isnan(diag["ttest_p_value"])
        assert np.isnan(diag["cohens_d"])
        assert diag["n_outliers"] == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_cv_diagnostics([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_predicted_std_passed_through_when_provided(self):
        """T24bn/V19cz: predicted_std rides along in the bundle so downstream
        coverage/calibration checks don't need a second CV rerun."""
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [2.0, 1.0, 4.0, 3.0]
        predicted_std = [0.5, 0.6, 0.7, 0.8]
        diag = compute_cv_diagnostics(actual, predicted, predicted_std)
        assert diag["predicted_std"] == pytest.approx(predicted_std)

    def test_predicted_std_defaults_to_nan_when_omitted(self):
        """Back-compat: existing 2-arg callers keep working, predicted_std
        just comes back as NaN instead of a real value."""
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [2.0, 1.0, 4.0, 3.0]
        diag = compute_cv_diagnostics(actual, predicted)
        assert len(diag["predicted_std"]) == len(actual)
        assert all(np.isnan(v) for v in diag["predicted_std"])

    def test_predicted_std_filtered_by_same_nan_mask_as_actual_predicted(self):
        """B22-style: a dropped CV row's NaN in actual/predicted must also drop
        the corresponding predicted_std entry, keeping all three arrays aligned."""
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [2.0, 1.0, 4.0, float("nan")]
        predicted_std = [0.5, 0.6, 0.7, 0.8]
        diag = compute_cv_diagnostics(actual, predicted, predicted_std)
        assert diag["predicted_std"] == pytest.approx([0.5, 0.6, 0.7])


class TestComputeCoverage:
    """One call answers ONE level (V19cz): `level` is a required coverage TARGET
    and the cross-level reading is the caller's list of calls."""

    def test_a_well_calibrated_gaussian_matches_its_own_nominal_level(self):
        """N(0,1) residuals with std 1 reported: each level's own z multiple must
        catch exactly the mass that level promises."""
        rng = np.random.default_rng(0)
        n = 20000
        actual = rng.normal(loc=0.0, scale=1.0, size=n)
        predicted = np.zeros(n)
        predicted_std = np.ones(n)

        for level in (0.68, 0.95, 0.99):
            result = compute_coverage(actual, predicted, predicted_std, level)
            assert result["level"] == level
            assert result["n_points"] == n
            assert result["empirical"] == pytest.approx(level, abs=0.01)

    def test_the_sigma_exact_targets_behave_like_the_sigmas_they_stand_for(self):
        """The conversion a sigma-thinking caller needs: 1sigma = 0.6827,
        2sigma = 0.9545, 3sigma = 0.9973 two-sided mass. Those are what the
        sweep's ROUND tokens (68/95/99) stand for, and `level` takes either
        spelling -- it is always the target mass, never the multiple."""
        rng = np.random.default_rng(0)
        n = 20000
        actual = rng.normal(loc=0.0, scale=1.0, size=n)
        predicted = np.zeros(n)
        predicted_std = np.ones(n)

        for target in (0.6827, 0.9545, 0.9973):
            assert compute_coverage(actual, predicted, predicted_std, target)[
                "empirical"
            ] == pytest.approx(target, abs=0.01)

    def test_the_result_names_the_single_level_it_answers(self):
        """BREAKING shape (T65ux): the vector `levels`/`empirical` pair is gone. A
        result is one level, so `empirical` is a fraction rather than a list to
        index, and it carries its own nominal for a chart to compare itself to."""
        result = compute_coverage([1.0, 2.0], [1.0, 2.0], [1.0, 1.0], 0.95)

        assert set(result) == {"level", "empirical", "n_points"}
        assert result["level"] == 0.95
        assert isinstance(result["empirical"], float)

    def test_the_level_is_required_not_defaulted(self):
        """There is no default level: "at which level" is the caller's question, so
        defaulting it would decide the question for whoever forgets to ask."""
        parameter = inspect.signature(compute_coverage).parameters["level"]

        assert parameter.default is inspect.Parameter.empty

    def test_a_level_must_be_a_probability(self):
        """0 and 1 have no z multiple (erfinv diverges), and a mass outside 0..1 is
        not a question -- fail at the boundary rather than return NaN coverage."""
        for bad in (0.0, 1.0, 1.5, -0.5):
            with pytest.raises(ValueError, match="strictly between 0 and 1"):
                compute_coverage([1.0, 2.0], [1.0, 2.0], [1.0, 1.0], bad)

    def test_overconfident_std_undercounts_coverage(self):
        """A predicted_std much smaller than the actual residual spread should
        show up as empirical coverage well below the nominal level -- the
        exact overconfidence pattern V19cz exists to catch."""
        rng = np.random.default_rng(1)
        n = 2000
        actual = rng.normal(loc=0.0, scale=1.0, size=n)
        predicted = np.zeros(n)
        predicted_std = np.full(n, 0.2)  # much narrower than the true scale=1.0

        result = compute_coverage(actual, predicted, predicted_std, 0.68)

        assert result["empirical"] < 0.68 - 0.05

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_coverage([1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.95)

    def test_nan_and_nonpositive_std_excluded(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        predicted_std = [1.0, float("nan"), 0.0, -1.0]

        result = compute_coverage(actual, predicted, predicted_std, 0.68)

        assert result["n_points"] == 1

    def test_empty_after_filtering_returns_nan_not_raise(self):
        result = compute_coverage([1.0], [1.0], [float("nan")], 0.68)

        assert result["n_points"] == 0
        assert np.isnan(result["empirical"])


class TestFitConvergenceExponentialAsymptotic:
    def test_recovers_known_params_with_nonzero_floor(self):
        """V18wp: raw error metrics have an irreducible-error floor as N->infinity,
        unlike the old zero-asymptote fit -- this is why the 3-parameter model exists."""
        a_true, b_true, c_true = 2.0, 0.1, 0.5
        n = np.linspace(10, 100, 40)
        y = a_true * np.exp(-b_true * n) + c_true

        fit = fit_convergence_exponential(n.tolist(), y.tolist(), asymptotic=True)

        assert fit["a"] == pytest.approx(a_true, rel=1e-2)
        assert fit["b"] == pytest.approx(b_true, rel=1e-2)
        assert fit["c"] == pytest.approx(c_true, rel=1e-2)
        assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_nan_values_are_dropped_before_fitting(self):
        a_true, b_true, c_true = 3.0, 0.1, 0.2
        n = np.linspace(5, 50, 10)
        y = a_true * np.exp(-b_true * n) + c_true
        y_with_nans = y.tolist()
        y_with_nans[2] = float("nan")
        y_with_nans[7] = float("nan")

        fit = fit_convergence_exponential(n.tolist(), y_with_nans, asymptotic=True)

        assert fit["a"] == pytest.approx(a_true, rel=1e-2)
        assert fit["c"] == pytest.approx(c_true, rel=1e-2)

    def test_raises_below_four_valid_points(self):
        with pytest.raises(ValueError):
            fit_convergence_exponential(
                [10.0, 20.0, 30.0], [1.0, 0.5, 0.3], asymptotic=True
            )


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

        def fake_manual_cv(
            run_dir, subset_file, input_vars, output_response, N_CROSS_VALIDATION=5
        ):
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
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation",
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
            # V16wq: diagnostics bundle rides along, one entry per draw, no rerun
            assert len(point["diagnostics"]) == point["n_bootstrap"]
            assert point["diagnostics"][0]["root_mean_squared"] == pytest.approx(0.1)
            # T24bn/V19cz: std_hat rides along through the convergence series too
            assert all(
                v == pytest.approx(0.0)
                for v in point["diagnostics"][0]["predicted_std"]
            )

    def test_empty_series_when_below_minimum(self, tmp_path):
        training_file = tmp_path / "df_processed_jobs.dat"
        with open(training_file, "w") as f:
            f.write("x1 y\n")
            f.write("0.1 0.2\n")
            f.write("0.2 0.4\n")

        series = compute_cv_convergence(
            tmp_path, training_file, ["x1"], "y", min_samples=5
        )
        assert series == []


class TestComputeCvConvergenceBootstrap:
    def test_bootstrap_draws_multiple_distinct_subsets_below_full_size(
        self, tmp_path, monkeypatch
    ):
        training_file = tmp_path / "df_processed_jobs.dat"
        n_total = 10
        rng = np.random.default_rng(1)
        x1 = rng.uniform(-1, 1, n_total)
        y = x1 * 2.0
        with open(training_file, "w") as f:
            f.write("x1 y\n")
            f.writelines(f"{xi} {yi}\n" for xi, yi in zip(x1, y))

        seen_subsets = []

        def fake_manual_cv(
            run_dir, subset_file, input_vars, output_response, N_CROSS_VALIDATION=5
        ):
            import pandas as pd

            df = pd.read_csv(subset_file, sep=" ")
            seen_subsets.append(tuple(sorted(df["x1"].round(6).tolist())))
            actual = df[output_response].astype(float).tolist()
            predicted = [v + 0.1 for v in actual]
            return {
                output_response: actual,
                output_response + "_hat": predicted,
                output_response + "_std_hat": [0.0] * len(actual),
            }

        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation",
            fake_manual_cv,
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=4,
        )

        # sizes: [5, 10] -- 4 bootstrap draws at n=5, exactly 1 run at n=10 (full set)
        assert [point["n_samples"] for point in series] == [5, 10]
        assert len(seen_subsets) == 4 + 1

        small_point, full_point = series
        assert small_point["n_bootstrap"] == 4
        assert full_point["n_bootstrap"] == 1
        assert full_point["metric_std"] == pytest.approx(0.0)

        # the 4 draws at n=5 aren't all the same subset of rows
        small_subsets = set(seen_subsets[:4])
        assert len(small_subsets) > 1

        # raw per-draw values are exposed for pooled downstream analysis
        assert len(small_point["draws"]) == 4
        assert small_point["draws"] == pytest.approx([0.1] * 4)
        assert full_point["draws"] == pytest.approx([0.1])

        # V16wq: diagnostics bundle length matches draws length at each size
        assert len(small_point["diagnostics"]) == 4
        assert len(full_point["diagnostics"]) == 1

    def test_bootstrap_is_deterministic_given_seed(self, tmp_path, monkeypatch):
        training_file = tmp_path / "df_processed_jobs.dat"
        n_total = 10
        rng = np.random.default_rng(2)
        x1 = rng.uniform(-1, 1, n_total)
        y = x1 * 2.0
        with open(training_file, "w") as f:
            f.write("x1 y\n")
            f.writelines(f"{xi} {yi}\n" for xi, yi in zip(x1, y))

        def fake_manual_cv(
            run_dir, subset_file, input_vars, output_response, N_CROSS_VALIDATION=5
        ):
            import pandas as pd

            df = pd.read_csv(subset_file, sep=" ")
            actual = df[output_response].astype(float).tolist()
            predicted = [v + 0.1 for v in actual]
            return {
                output_response: actual,
                output_response + "_hat": predicted,
                output_response + "_std_hat": [0.0] * len(actual),
            }

        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation",
            fake_manual_cv,
        )

        series_a = compute_cv_convergence(
            tmp_path / "a",
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=4,
            seed=7,
        )
        series_b = compute_cv_convergence(
            tmp_path / "b",
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=4,
            seed=7,
        )
        assert _same_structure(series_a, series_b)

    def test_default_n_bootstrap_matches_pre_bootstrap_behavior(
        self, tmp_path, monkeypatch
    ):
        """n_bootstrap defaults to 1: single deterministic first-n-rows subset,
        same as the pre-bootstrap implementation -- back-compat for existing callers."""
        training_file = tmp_path / "df_processed_jobs.dat"
        n_total = 10
        rng = np.random.default_rng(3)
        x1 = rng.uniform(-1, 1, n_total)
        y = x1 * 2.0
        with open(training_file, "w") as f:
            f.write("x1 y\n")
            f.writelines(f"{xi} {yi}\n" for xi, yi in zip(x1, y))

        seen_subsets = []

        def fake_manual_cv(
            run_dir, subset_file, input_vars, output_response, N_CROSS_VALIDATION=5
        ):
            import pandas as pd

            df = pd.read_csv(subset_file, sep=" ")
            seen_subsets.append(df["x1"].round(6).tolist())
            actual = df[output_response].astype(float).tolist()
            predicted = [v + 0.1 for v in actual]
            return {
                output_response: actual,
                output_response + "_hat": predicted,
                output_response + "_std_hat": [0.0] * len(actual),
            }

        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation",
            fake_manual_cv,
        )

        series = compute_cv_convergence(
            tmp_path, training_file, ["x1"], "y", min_samples=5, max_points=2
        )

        assert len(seen_subsets) == 2  # one run per size, no bootstrap fan-out
        assert seen_subsets[0] == pytest.approx(x1[:5].round(6).tolist())
        assert all(
            point["n_bootstrap"] == 1 and point["metric_std"] == 0.0 for point in series
        )


class TestFitConvergenceExponential:
    def test_recovers_known_params_on_noiseless_data(self):
        a_true, b_true = 5.0, 0.05
        n = np.linspace(10, 100, 30)
        y = a_true * np.exp(-b_true * n)

        fit = fit_convergence_exponential(n.tolist(), y.tolist())

        assert fit["a"] == pytest.approx(a_true, rel=1e-3)
        assert fit["b"] == pytest.approx(b_true, rel=1e-3)
        assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_nan_values_are_dropped_before_fitting(self):
        a_true, b_true = 3.0, 0.1
        n = np.linspace(5, 50, 10)
        y = a_true * np.exp(-b_true * n)
        y_with_nans = y.tolist()
        y_with_nans[2] = float("nan")
        y_with_nans[7] = float("nan")

        fit = fit_convergence_exponential(n.tolist(), y_with_nans)

        assert fit["a"] == pytest.approx(a_true, rel=1e-3)
        assert fit["b"] == pytest.approx(b_true, rel=1e-3)

    def test_noisy_data_gives_high_but_imperfect_r_squared(self):
        rng = np.random.default_rng(0)
        n = np.linspace(10, 100, 40)
        y = 5.0 * np.exp(-0.05 * n) + rng.normal(0, 0.05, size=n.size)

        fit = fit_convergence_exponential(n.tolist(), y.tolist())

        assert 0.5 < fit["r_squared"] < 1.0

    def test_raises_below_three_valid_points(self):
        with pytest.raises(ValueError):
            fit_convergence_exponential([10.0, 20.0], [1.0, 0.5])


# ---------------------------------------------------------------------------
# The metric family one convergence sweep returns, and the `uq` block beside it.
#
# Every metric and every predictive statistic is read from data the CV pass
# ALREADY produces (V16wq, V56qx), so every test here also pins the cost claim:
# exactly one CV rerun per (subset, draw), whatever the point reports.
# ---------------------------------------------------------------------------


def _same_structure(left: Any, right: Any) -> bool:
    """Deep equality where NaN matches NaN.

    A convergence series is otherwise unequal to ITSELF under `==`: a surrogate
    that reports no spread yields NaN coverage (V19cz drops the zero-std points
    rather than inventing a level), and NaN != NaN. Determinism is what the
    seeded-bootstrap tests mean to check, so compare structurally.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_structure(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_structure(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    return bool(left == right)


def _write_training_file(path, n_total: int, seed: int):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n_total)
    y = x1 * 2.0
    path.write_text(
        "x1 y\n" + "".join(f"{xi} {yi}\n" for xi, yi in zip(x1, y)), encoding="utf-8"
    )
    return path


def _recording_cv(residual, std):
    """A `evaluate_sumo_manual_crossvalidation` stand-in with a KNOWN output.

    `residual` (`actual - predicted`, the sign convention
    `compute_cv_accuracy_metrics` uses) and `std` are each a scalar/list cycled
    over the held-out rows, or a `callable(subset_df) -> array`. Each call's
    arrays are recorded, so a test can compare the series against
    `compute_cv_accuracy_metrics`/`compute_cv_diagnostics` over exactly the same
    pairs.
    """
    seen: list[dict[str, list[float]]] = []

    def _column(spec, subset, n):
        values = spec(subset) if callable(spec) else spec
        return np.asarray(np.resize(values, n), dtype=float)

    def fake_manual_cv(
        run_dir, subset_file, input_vars, output_response, N_CROSS_VALIDATION=5
    ):
        subset = pd.read_csv(subset_file, sep=" ")
        actual = subset[output_response].astype(float).to_numpy()
        predicted = actual - _column(residual, subset, len(actual))
        predicted_std = _column(std, subset, len(actual))
        seen.append(
            {
                "actual": actual.tolist(),
                "predicted": predicted.tolist(),
                "predicted_std": predicted_std.tolist(),
            }
        )
        return {
            output_response: actual.tolist(),
            output_response + "_hat": predicted.tolist(),
            output_response + "_std_hat": predicted_std.tolist(),
        }

    return fake_manual_cv, seen


# Chosen so rmse / mae / bias are three different numbers (a constant residual
# would make them collapse into one) and bias comes out positive.
_RESIDUAL_CYCLE = [0.5, -0.1, 0.2, -0.3, 0.4]
_FLAT_STD = 0.2


# The vocabulary, spelled out here so widening it in the engine has to be a
# deliberate decision in two places rather than a silent one.
_FAMILY = {"rmse", "mae", "bias", "coverage68", "coverage95", "coverage99"}
_ACCURACY = ("rmse", "mae", "bias")
_ACCURACY_KEYS = {
    "rmse": "root_mean_squared",
    "mae": "mean_abs",
    "bias": "mean_signed_error",
}
# accuracy tokens, and the coverage target each coverage token means
# (the sweep's round 68/95/99 spelling, V19cz's calibration question)
_LEVELS = {"coverage68": 0.68, "coverage95": 0.95, "coverage99": 0.99}


class TestComputeCvConvergenceMetricFamily:
    def test_every_point_carries_the_whole_family(self, tmp_path, monkeypatch):
        """There is no metric argument to ask for one reading: each point reports
        every token, bootstrapped or not, with the same per-metric shape."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=11
        )
        fake, _ = _recording_cv(_RESIDUAL_CYCLE, _FLAT_STD)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=3,
            n_bootstrap=2,
        )

        assert len(series) > 1
        for point in series:
            assert set(point["metrics"]) == _FAMILY
            assert set(point["metrics_std"]) == _FAMILY
            assert set(point["draws_by_metric"]) == _FAMILY
            for name in _FAMILY:
                assert len(point["draws_by_metric"][name]) == point["n_bootstrap"]

    def test_the_family_is_what_the_engine_documents(self):
        """The test vocabulary above and the engine's own tokens agree (it also
        pins how a coverage level spells its token: 0.95 -> coverage95)."""
        assert set(_CV_METRICS) == _FAMILY
        assert _CV_ACCURACY_METRICS == _ACCURACY
        assert _CV_COVERAGE_METRICS == ("coverage68", "coverage95", "coverage99")

    def test_every_metric_equals_the_bundles_own_number_for_the_same_subset(
        self, tmp_path, monkeypatch
    ):
        """V16wq/V55nw: every token is a READING of the diagnostics the subset
        already produced — accuracy straight out of the bundle, coverage from
        `compute_coverage` over the pairs the bundle carries."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=12
        )
        fake, seen = _recording_cv(_RESIDUAL_CYCLE, _FLAT_STD)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
        )

        assert len(series) == len(seen)
        for point, pairs in zip(series, seen):
            metrics = compute_cv_accuracy_metrics(pairs["actual"], pairs["predicted"])
            bundle = compute_cv_diagnostics(
                pairs["actual"], pairs["predicted"], pairs["predicted_std"]
            )
            for name in _ACCURACY:
                key = _ACCURACY_KEYS[name]
                assert point["draws_by_metric"][name] == [pytest.approx(metrics[key])]
                # the bundle riding along agrees with the series number
                assert point["metrics"][name] == pytest.approx(bundle[key])
            for token, level in _LEVELS.items():
                assert point["metrics"][token] == pytest.approx(
                    compute_coverage(
                        pairs["actual"],
                        pairs["predicted"],
                        pairs["predicted_std"],
                        level,
                    )["empirical"]
                )

    def test_the_rmse_named_keys_stay_the_rmse_reading(self, tmp_path, monkeypatch):
        """Additive-only growth: the pre-family keys keep their meaning, so an
        old consumer of this series still gets the RMSE sweep it expects."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=13
        )
        fake, _ = _recording_cv(_RESIDUAL_CYCLE, _FLAT_STD)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )
        legacy_keys = {
            "n_samples",
            "metric",
            "metric_std",
            "n_bootstrap",
            "draws",
            "diagnostics",
        }

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=2,
        )

        for point in series:
            assert legacy_keys <= set(point)
            assert set(point) - legacy_keys == {
                "metrics",
                "metrics_std",
                "draws_by_metric",
                "uq",
            }
            assert point["metric"] == point["metrics"]["rmse"]
            assert point["metric_std"] == point["metrics_std"]["rmse"]
            assert point["draws"] == point["draws_by_metric"]["rmse"]
            if point["n_bootstrap"] == 1:
                # unchanged meaning: the RMSE the bundled diagnostics report
                assert point["metric"] == pytest.approx(
                    point["diagnostics"][0]["root_mean_squared"]
                )
                assert point["metric_std"] == 0.0
                assert all(value == 0.0 for value in point["metrics_std"].values())

    def test_the_accuracy_metrics_stay_distinct_and_bias_is_signed(
        self, tmp_path, monkeypatch
    ):
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=14
        )
        fake, _ = _recording_cv(_RESIDUAL_CYCLE, _FLAT_STD)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        metrics = compute_cv_convergence(
            tmp_path, training_file, ["x1"], "y", min_samples=5, max_points=2
        )[0]["metrics"]

        assert len({round(metrics[name], 9) for name in _ACCURACY}) == 3
        assert metrics["rmse"] > 0.0 and metrics["mae"] > 0.0
        # only the SIGNED metric carries the direction of the residual cycle at all
        assert metrics["bias"] > 0.0

    def test_the_coverage_levels_are_nested_and_ordered(self, tmp_path, monkeypatch):
        """A 68% interval sits inside a 99% interval, so the trio must come out
        ordered — a `coverage99` below its `coverage68` would mean the levels got
        mixed up on the way out, which no single-level test could see."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=15
        )
        fake, _ = _recording_cv(_RESIDUAL_CYCLE, _FLAT_STD)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        metrics = compute_cv_convergence(
            tmp_path, training_file, ["x1"], "y", min_samples=5, max_points=2
        )[0]["metrics"]

        assert metrics["coverage68"] <= metrics["coverage95"] <= metrics["coverage99"]
        # the fixture's residuals straddle the 68%/95%/99% multiples, so
        # the ordering is informative here rather than three copies of one number
        assert metrics["coverage68"] < metrics["coverage99"]
        assert 0.0 <= metrics["coverage68"] and metrics["coverage99"] <= 1.0

    def test_coverage_is_pooled_over_draws_while_draws_stay_per_draw(
        self, tmp_path, monkeypatch
    ):
        """V19cz: one draw's tail coverage is noise, so each coverage number
        covers every draw's held-out points TOGETHER (at every level, since the
        tail levels need the points even more); `draws_by_metric` keeps each
        draw's own value, so the cloud still shows that noise."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=16
        )
        # std tracks |x1|, so |residual|/std straddles the critical z values and
        # the per-draw coverages really differ from each other and from the pool
        fake, seen = _recording_cv(
            0.1, lambda subset: 0.02 + 0.1 * np.abs(subset["x1"])
        )
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=3,
        )
        small, full = series

        draws = seen[:3]
        for token, level in _LEVELS.items():
            expected_metric = compute_coverage(
                [v for pairs in draws for v in pairs["actual"]],
                [v for pairs in draws for v in pairs["predicted"]],
                [v for pairs in draws for v in pairs["predicted_std"]],
                level,
            )["empirical"]
            expected_draws = [
                compute_coverage(
                    pairs["actual"],
                    pairs["predicted"],
                    pairs["predicted_std"],
                    level,
                )["empirical"]
                for pairs in draws
            ]
            assert small["metrics"][token] == pytest.approx(expected_metric)
            assert small["draws_by_metric"][token] == [
                pytest.approx(value) for value in expected_draws
            ]
            # the single (full-sample) point is its own pool: value == its draw
            assert full["metrics"][token] == pytest.approx(
                full["draws_by_metric"][token][0]
            )
        assert 0.0 < small["metrics"]["coverage95"] < 1.0  # a straddle, not 0/1
        # the draws genuinely differ at 95%; at 99 this fixture saturates at
        # 1.0, which is the level-dependence V19cz warns about
        assert len({round(v, 12) for v in small["draws_by_metric"]["coverage95"]}) > 1
        assert full["n_bootstrap"] == 1
        assert all(value == 0.0 for value in full["metrics_std"].values())

    def test_the_family_costs_no_extra_cv_rerun(self, tmp_path, monkeypatch):
        """The cost claim behind returning everything (V16wq, V55nw): exactly one
        CV rerun per (subset size, draw), never one per metric."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=17
        )
        fake, seen = _recording_cv(_RESIDUAL_CYCLE, _FLAT_STD)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=3,
            n_bootstrap=2,
        )

        # sizes [5, 7, 10]: two draws below full size, one at it
        assert len(seen) == sum(point["n_bootstrap"] for point in series) == 2 + 2 + 1


class TestComputeCvConvergenceUq:
    """The per-size predictive-distribution summary (CV replay, Gaussian, V56qx)."""

    def test_percentiles_match_a_known_gaussian_predictive_spread(
        self, tmp_path, monkeypatch
    ):
        """Every held-out point reports N(1.0, 0.2^2), so the pooled predictive
        sample is that same Gaussian whatever the subset: p95/p99 must land on
        1.0 + 0.2 * Phi^-1(0.95 / 0.99) — the percentiles mmux_vite #458 asks
        for, not a boxplot quartile wearing a new name."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=21
        )
        fake, _ = _recording_cv(
            lambda subset: subset["y"].astype(float) - 1.0, _FLAT_STD
        )  # predicted == 1.0 for every held-out row
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=3,
        )
        uq = series[0]["uq"]

        assert uq["basis"] == "cv-replay-gaussian"
        assert uq["mean"] == pytest.approx(1.0)  # symmetric grid: mean == mean(mu)
        assert uq["median"] == pytest.approx(1.0)
        # the deterministic midpoint grid resolves Phi^-1(0.95)/Phi^-1(0.99)
        # itself, so the pooled percentiles land right on the Gaussian tail
        grid = _uq_quantile_grid(uq["quantiles_per_point"])
        assert grid[47] == pytest.approx(1.6448536, abs=1e-5)  # Phi^-1(0.95)
        assert grid[-1] == pytest.approx(2.3263479, abs=1e-5)  # Phi^-1(0.99)
        assert uq["p95"] == pytest.approx(1.0 + 0.2 * 1.6448536, rel=2e-3)
        assert uq["p99"] == pytest.approx(1.0 + 0.2 * 2.3263479, rel=2e-3)
        assert uq["p99"] > uq["p95"] > uq["median"]
        assert uq["std"] == pytest.approx(0.2, rel=0.05)
        assert uq["n_points"] == 5 * 3  # 5 held-out points x 3 draws
        assert uq["n_draws"] == 3

    def test_a_vanishing_spread_collapses_to_the_prediction_percentiles(
        self, tmp_path, monkeypatch
    ):
        """sigma -> 0 is the deterministic limit: the predictive sample IS the
        set of held-out predictions, so the CV-replay percentiles must reproduce
        their plain empirical percentiles."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 41, seed=22
        )
        fake, seen = _recording_cv(_RESIDUAL_CYCLE, 1e-9)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path, training_file, ["x1"], "y", min_samples=41, max_points=1
        )
        assert len(series) == 1 == len(seen)
        predicted = np.asarray(seen[0]["predicted"], dtype=float)
        uq = series[0]["uq"]

        assert uq["n_points"] == 41
        assert uq["mean"] == pytest.approx(predicted.mean())
        for key, level in (("median", 50), ("p95", 95)):
            assert uq[key] == pytest.approx(np.percentile(predicted, level), abs=1e-6)
        # p99 of a 41-point tail interpolates between the two largest held-out
        # predictions differently in a 2050-value pool than in a 41-value one —
        # the honest bound is that interpolation span, not a made-up tolerance
        top_gap = float(np.diff(np.sort(predicted)[-2:])[0])
        assert abs(uq["p99"] - np.percentile(predicted, 99)) <= top_gap + 1e-6

    def test_points_without_a_usable_predicted_std_are_excluded(
        self, tmp_path, monkeypatch
    ):
        """Numerical hygiene matches `compute_coverage`: NaN or non-positive std
        is dropped, never silently counted."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=23
        )
        nan = float("nan")
        fake, seen = _recording_cv(_RESIDUAL_CYCLE, [0.2, nan, 0.0, -1.0])
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=2,
        )
        expected = sum(
            sum(1 for s in pairs["predicted_std"] if np.isfinite(s) and s > 0.0)
            for pairs in seen[:2]
        )
        assert series[0]["uq"]["n_points"] == expected
        assert 0 < expected < 5 * 2  # some points really were dropped
        assert np.isfinite(series[0]["uq"]["p95"])

    def test_zero_spread_column_yields_none_not_a_point_mass(
        self, tmp_path, monkeypatch
    ):
        """A surrogate that reports no spread at all (constant zero std, e.g. a
        fold that never wrote variances.dat) has no predictive distribution to
        summarise: `None` statistics with n_points == 0, no fabricated mass."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=24
        )
        fake, _ = _recording_cv(_RESIDUAL_CYCLE, [0.0])
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        uq = compute_cv_convergence(
            tmp_path, training_file, ["x1"], "y", min_samples=5, max_points=2
        )[0]["uq"]

        assert uq["n_points"] == 0 and uq["n_draws"] == 0
        for key in ("mean", "median", "std", "p95", "p99"):
            assert uq[key] is None
            assert uq["draw_spread"][key] is None

    def test_draw_spread_needs_two_draws(self, tmp_path, monkeypatch):
        """One subset carries no information about how far a statistic would move
        on a redraw: None, not a reassuring 0.0."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=25
        )
        fake, _ = _recording_cv(_RESIDUAL_CYCLE, [0.05, 0.2, 0.4])
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        single = compute_cv_convergence(
            tmp_path / "single", training_file, ["x1"], "y", min_samples=5, max_points=2
        )[0]["uq"]
        assert all(value is None for value in single["draw_spread"].values())
        assert single["n_draws"] == 1

        bootstrapped = compute_cv_convergence(
            tmp_path / "boot",
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=2,
            n_bootstrap=3,
        )[0]["uq"]
        assert bootstrapped["n_draws"] == 3
        assert bootstrapped["draw_spread"]["mean"] > 0.0
        assert bootstrapped["draw_spread"]["p99"] > 0.0

    def test_the_uq_block_costs_no_extra_cv_rerun(self, tmp_path, monkeypatch):
        """V56qx with the V16wq cost claim: adding the distribution view to every
        point must not add ONE more CV pass."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 10, seed=26
        )
        fake, seen = _recording_cv(_RESIDUAL_CYCLE, _FLAT_STD)
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=5,
            max_points=3,
            n_bootstrap=2,
        )

        assert all("uq" in point for point in series)
        assert len(seen) == sum(point["n_bootstrap"] for point in series)


class TestFitConvergenceExponentialIsMetricAgnostic:
    """V18wp on any metric series — for coverage the interesting question is
    whether it settles AT the nominal level or below it, and that is exactly
    what the floor parameter `c` answers."""

    def test_recovers_a_coverage_floor_at_the_nominal_level(self):
        level = 0.95
        n = np.linspace(10, 200, 30)
        values = 0.30 * np.exp(-0.05 * n) + level

        fit = fit_convergence_exponential(n.tolist(), values.tolist(), asymptotic=True)

        assert fit["c"] == pytest.approx(level, rel=1e-3)
        assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_recovers_a_coverage_floor_below_the_nominal_level(self):
        """Miscalibration that does NOT wash out with more data: the floor IS the
        answer, and a zero-asymptote fit would have been nonsense (V19cz)."""
        n = np.linspace(10, 200, 30)
        values = 0.30 * np.exp(-0.05 * n) + 0.78

        fit = fit_convergence_exponential(n.tolist(), values.tolist(), asymptotic=True)

        assert fit["c"] == pytest.approx(0.78, rel=1e-3)
        assert fit["c"] < 0.95

    def test_fits_a_coverage_series_straight_out_of_the_convergence_run(
        self, tmp_path, monkeypatch
    ):
        """No metric-specific branch anywhere: the fit takes whatever column of
        the family you feed it — here the coverage draws, not the RMSE ones."""
        training_file = _write_training_file(
            tmp_path / "df_processed_jobs.dat", 20, seed=31
        )
        # a genuinely improving calibration: the residual shrinks with the subset
        # size while the reported spread varies across rows
        fake, _ = _recording_cv(
            lambda subset: 0.25 * np.exp(-0.15 * len(subset)),
            lambda subset: 0.02 + 0.06 * np.abs(subset["x1"]),
        )
        monkeypatch.setattr(
            "itis_sumo.engine.evaluation.evaluate_sumo_manual_crossvalidation", fake
        )

        series = compute_cv_convergence(
            tmp_path,
            training_file,
            ["x1"],
            "y",
            min_samples=6,
            max_points=4,
            n_bootstrap=2,
        )
        coverages = [point["metrics"]["coverage95"] for point in series]
        assert coverages[-1] > coverages[0]  # the sweep really moves

        coverage_draws = [
            (e["n_samples"], draw)
            for e in series
            for draw in e["draws_by_metric"]["coverage95"]
        ]
        fit = fit_convergence_exponential(
            [n for n, _ in coverage_draws],
            [draw for _, draw in coverage_draws],
            asymptotic=True,
        )

        assert set(fit) == {"a", "b", "c", "r_squared"}
        # curve_fit is unconstrained, so the floor is read as "converges to
        # (about) full coverage" rather than clipped into [0, 1] by the fitter
        assert fit["c"] == pytest.approx(1.0, abs=0.1)
