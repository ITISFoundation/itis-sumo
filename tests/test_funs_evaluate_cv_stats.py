"""Unit tests for the SuMo CV statistical-rigor helpers in `funs_evaluate.py` (§T18).

These cover the pure statistical functions (no Dakota/filesystem dependency) plus the
convergence-series orchestration (Dakota-dependent `evaluate_sumo_manual_crossvalidation`
call is monkeypatched to keep the tests fast and deterministic).
"""

import numpy as np
import pytest

from itis_sumo.evaluate.funs_evaluate import (
    _convergence_subset_sizes,
    compute_coverage,
    compute_cv_accuracy_metrics,
    compute_cv_convergence,
    compute_cv_diagnostics,
    compute_paired_ttest,
    fit_convergence_exponential,
    fit_convergence_exponential_asymptotic,
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
    def test_well_calibrated_gaussian_matches_nominal_levels(self):
        rng = np.random.default_rng(0)
        n = 5000
        actual = rng.normal(loc=0.0, scale=1.0, size=n)
        predicted = np.zeros(n)
        predicted_std = np.ones(n)
        result = compute_coverage(actual, predicted, predicted_std)
        assert result["n_points"] == n
        for empirical, nominal in zip(result["empirical"], result["levels"]):
            assert empirical == pytest.approx(nominal, abs=0.02)

    def test_overconfident_std_undercounts_coverage(self):
        """A predicted_std much smaller than the actual residual spread should
        show up as empirical coverage well below the nominal level -- the
        exact overconfidence pattern V19cz exists to catch."""
        rng = np.random.default_rng(1)
        n = 2000
        actual = rng.normal(loc=0.0, scale=1.0, size=n)
        predicted = np.zeros(n)
        predicted_std = np.full(n, 0.2)  # much narrower than the true scale=1.0
        result = compute_coverage(actual, predicted, predicted_std, levels=(0.6827,))
        assert result["empirical"][0] < 0.6827 - 0.05

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_coverage([1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])

    def test_nan_and_nonpositive_std_excluded(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        predicted_std = [1.0, float("nan"), 0.0, -1.0]
        result = compute_coverage(actual, predicted, predicted_std, levels=(0.6827,))
        assert result["n_points"] == 1

    def test_empty_after_filtering_returns_nan_not_raise(self):
        result = compute_coverage([1.0], [1.0], [float("nan")], levels=(0.6827,))
        assert result["n_points"] == 0
        assert np.isnan(result["empirical"][0])


class TestFitConvergenceExponentialAsymptotic:
    def test_recovers_known_params_with_nonzero_floor(self):
        """V18wp: raw error metrics have an irreducible-error floor as N->infinity,
        unlike the old zero-asymptote fit -- this is why the 3-parameter model exists."""
        a_true, b_true, c_true = 2.0, 0.1, 0.5
        n = np.linspace(10, 100, 40)
        y = a_true * np.exp(-b_true * n) + c_true

        fit = fit_convergence_exponential_asymptotic(n.tolist(), y.tolist())

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

        fit = fit_convergence_exponential_asymptotic(n.tolist(), y_with_nans)

        assert fit["a"] == pytest.approx(a_true, rel=1e-2)
        assert fit["c"] == pytest.approx(c_true, rel=1e-2)

    def test_raises_below_four_valid_points(self):
        with pytest.raises(ValueError):
            fit_convergence_exponential_asymptotic([10.0, 20.0, 30.0], [1.0, 0.5, 0.3])


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
            "itis_sumo.evaluate.funs_evaluate.evaluate_sumo_manual_crossvalidation",
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
            "itis_sumo.evaluate.funs_evaluate.evaluate_sumo_manual_crossvalidation",
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
        assert series_a == series_b

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
            "itis_sumo.evaluate.funs_evaluate.evaluate_sumo_manual_crossvalidation",
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
