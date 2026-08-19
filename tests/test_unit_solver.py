"""Tier 1 unit tests for MMUX solver backend pure/near-pure functions."""

import numpy as np
import pandas as pd
import pytest
from scipy import spatial

from itis_sumo.config.funs_create_dakota_conf import (
    add_continuous_variables,
    add_responses,
    add_surrogate_model,
    create_moga_optimization_conffile,
    create_sumo_crossvalidation_conffile,
    create_sumo_evaluation_conffile,
    create_sumo_manual_crossvalidation_conffile,
    create_uq_propagation_conffile,
    start_dakota_file,
)
from itis_sumo.data.funs_data_processing import (
    _filter_data,
    create_grid_samples,
    create_manual_uq_samples,
    create_samples_along_axes,
    get_bounds_uniform_distribution,
    get_bounds_uniform_distributions,
    get_non_dominated_indices,
    is_dominated,
)
from itis_sumo.evaluate.funs_evaluate import _parse_crossvalidation_outputlogs
from itis_sumo.sampling.lhs import lhs


# ---------------------------------------------------------------------------
# 1. LHS Sampling
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLhs:
    def test_output_shape(self):
        H = lhs(n=3, k=7, seed=42)
        assert H.shape == (7, 3)

    def test_values_in_unit_hypercube(self):
        H = lhs(n=5, k=20, seed=42)
        assert np.all(H >= 0.0)
        assert np.all(H <= 1.0)

    def test_stratification(self):
        n, k = 4, 10
        H = lhs(n=n, k=k, seed=42)
        for col in range(n):
            hist, _ = np.histogram(H[:, col], bins=k, range=(0, 1))
            assert np.all(hist == 1), f"Column {col} failed stratification"

    def test_reproducibility_with_seed(self):
        H1 = lhs(n=3, k=10, seed=123)
        H2 = lhs(n=3, k=10, seed=123)
        np.testing.assert_array_equal(H1, H2)

    def test_different_seeds_differ(self):
        H1 = lhs(n=3, k=10, seed=1)
        H2 = lhs(n=3, k=10, seed=2)
        assert not np.array_equal(H1, H2)

    def test_maximin_min_distance_ge_classic(self):
        rs_classic = np.random.RandomState(42)
        rs_maximin = np.random.RandomState(42)
        H_classic = lhs(n=4, k=15, seed=rs_classic)
        H_maximin = lhs(n=4, k=15, method="maximin", iter=50, seed=rs_maximin)
        d_classic = spatial.distance.pdist(H_classic, "euclidean")
        d_maximin = spatial.distance.pdist(H_maximin, "euclidean")
        assert np.min(d_maximin) >= np.min(d_classic)

    def test_centered_values_at_midpoints(self):
        n, k = 3, 6
        H = lhs(n=n, k=k, method="center", seed=42)
        expected_centers = np.linspace(0, 1, k + 1)
        centers = (expected_centers[:k] + expected_centers[1 : k + 1]) / 2
        for col in range(n):
            col_sorted = np.sort(H[:, col])
            np.testing.assert_allclose(col_sorted, centers, atol=1e-14)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid value"):
            lhs(n=3, k=5, method="invalid", seed=42)

    def test_single_sample(self):
        H = lhs(n=2, k=1, seed=42)
        assert H.shape == (1, 2)
        assert np.all(H >= 0) and np.all(H <= 1)


# ---------------------------------------------------------------------------
# 2. Grid & Sweep Sampling
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCreateGridSamples:
    def test_2d_grid_point_count(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = create_grid_samples(
            run_dir=run_dir,
            grid_vars=["x", "y"],
            input_vars=["x", "y"],
            mins=[0.0, 0.0],
            cut_values=[0.5, 0.5],
            maxs=[1.0, 1.0],
            n_points_per_dimension=[3, 4],
        )
        df = pd.read_csv(result, sep=" ")
        assert len(df) == 12  # 3 * 4

    def test_grid_values_within_bounds(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = create_grid_samples(
            run_dir=run_dir,
            grid_vars=["x", "y"],
            input_vars=["x", "y"],
            mins=[0.0, 0.0],
            cut_values=[0.5, 0.5],
            maxs=[10.0, 20.0],
            n_points_per_dimension=[5, 5],
        )
        df = pd.read_csv(result, sep=" ")
        assert df["x"].min() >= 0.0
        assert df["x"].max() <= 10.0
        assert df["y"].min() >= 0.0
        assert df["y"].max() <= 20.0

    def test_non_grid_vars_fixed_at_cut_values(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = create_grid_samples(
            run_dir=run_dir,
            grid_vars=["x"],
            input_vars=["x", "y"],
            mins=[0.0, 0.0],
            cut_values=[0.5, 7.0],
            maxs=[1.0, 1.0],
            n_points_per_dimension=[3, 1],
        )
        df = pd.read_csv(result, sep=" ")
        assert len(df) == 3
        assert (df["y"] == 7.0).all()


@pytest.mark.unit
class TestCreateSamplesAlongAxes:
    def test_correct_total_point_count(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [10.0, 20.0, 30.0]})
        result = create_samples_along_axes(run_dir, data, ["x", "y"], NSAMPLESPERVAR=5)
        df = pd.read_csv(result, sep=" ")
        assert len(df) == 10  # 2 vars * 5 samples

    def test_sweep_variable_varies_monotonically(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        data = pd.DataFrame({"x": [0.0, 1.0], "y": [10.0, 20.0]})
        result = create_samples_along_axes(run_dir, data, ["x", "y"], NSAMPLESPERVAR=4)
        df = pd.read_csv(result, sep=" ")
        x_values = df["x"].values
        assert x_values[0] < x_values[1] < x_values[2] < x_values[3]

    def test_non_sweep_vars_fixed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [10.0, 20.0, 30.0]})
        cut_values = {"x": 1.0, "y": 20.0}
        result = create_samples_along_axes(
            run_dir, data, ["x", "y"], NSAMPLESPERVAR=3, cut_values=cut_values
        )
        df = pd.read_csv(result, sep=" ")
        first_3_y = df["y"].values[:3]
        assert np.allclose(first_3_y, 20.0)


# ---------------------------------------------------------------------------
# 3. Surrogate Model Building & shared Dakota config primitives
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDakotaConfigGeneration:
    def test_start_dakota_file_contains_environment(self):
        result = start_dakota_file()
        assert "environment" in result
        assert "tabular_data" in result
        assert "results.dat" in result

    def test_start_dakota_file_custom_results(self):
        result = start_dakota_file(results_file_name="my_results.dat")
        assert "my_results.dat" in result

    def test_start_dakota_file_with_method_pointer(self):
        result = start_dakota_file(top_method_pointer="SAMPLING")
        assert "top_method_pointer" in result
        assert "SAMPLING" in result

    def test_add_continuous_variables(self):
        result = add_continuous_variables(variables=["x1", "x2", "x3"])
        assert "continuous_design = 3" in result
        assert "'x1'" in result
        assert "'x2'" in result
        assert "'x3'" in result

    def test_add_continuous_variables_with_bounds(self):
        result = add_continuous_variables(
            variables=["a", "b"],
            lower_bounds=[0.0, -1.0],
            upper_bounds=[10.0, 1.0],
        )
        assert "lower_bounds" in result
        assert "upper_bounds" in result
        assert "0.0" in result
        assert "10.0" in result

    def test_add_responses(self):
        result = add_responses(["stress", "displacement"])
        assert "responses" in result
        assert "objective_functions = 2" in result
        assert "'stress'" in result
        assert "'displacement'" in result
        assert "no_gradients" in result

    def test_add_surrogate_model(self):
        result = add_surrogate_model(
            training_samples_file="/tmp/build.txt", has_eval_id_column=True
        )
        assert "surrogate global" in result
        assert "gaussian_process" in result
        assert "predictions.dat" in result

    def test_add_surrogate_model_with_cv(self):
        result = add_surrogate_model(
            training_samples_file="/tmp/build.txt",
            cross_validation_folds=5,
            has_eval_id_column=True,
        )
        assert "cross_validation folds = 5" in result
        assert "metrics" in result

    def test_create_sumo_evaluation_has_all_blocks(self, tmp_path):
        build = tmp_path / "build.txt"
        samples = tmp_path / "samples.txt"
        result = create_sumo_evaluation_conffile(
            build_file=build,
            samples_file=samples,
            input_variables=["x", "y"],
            output_responses=["response"],
        )
        assert "environment" in result
        assert "model" in result
        assert "method" in result
        assert "variables" in result
        assert "responses" in result
        assert "'x'" in result
        assert "'y'" in result
        assert "'response'" in result


# ---------------------------------------------------------------------------
# 4. Cross-Validation
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestParseCrossvalidationOutputlogs:
    def test_valid_log_extracts_metrics(self):
        log = (
            "Surrogate quality metrics (5-fold CV) for AFpeak:\n"
            "    root_mean_squared 1.23e-04\n"
            "    sum_abs 5.67e-03\n"
            "    mean_abs 2.34e-04\n"
            "    max_abs 8.90e-03\n"
            "build (training) points\n"
        )
        result = _parse_crossvalidation_outputlogs(log, 5)
        assert "AFpeak" in result
        assert result["AFpeak"]["root_mean_squared"] == "1.23e-04"
        assert result["AFpeak"]["sum_abs"] == "5.67e-03"
        assert result["AFpeak"]["mean_abs"] == "2.34e-04"
        assert result["AFpeak"]["max_abs"] == "8.90e-03"

    def test_valid_log_two_variables(self):
        log = (
            "Surrogate quality metrics (5-fold CV) for AFpeak:\n"
            "    root_mean_squared 1.0e-04\n"
            "build (training) points\n"
            "Surrogate quality metrics (5-fold CV) for BField:\n"
            "    root_mean_squared 2.0e-04\n"
            "build (training) points\n"
        )
        result = _parse_crossvalidation_outputlogs(log, 5)
        assert "AFpeak" in result
        assert "BField" in result
        assert result["AFpeak"]["root_mean_squared"] == "1.0e-04"
        assert result["BField"]["root_mean_squared"] == "2.0e-04"

    def test_empty_log_returns_empty_dict(self):
        result = _parse_crossvalidation_outputlogs("", 5)
        assert result == {}

    def test_variable_found_no_metrics(self):
        log = "Surrogate quality metrics (5-fold CV) for AFpeak:\n"
        result = _parse_crossvalidation_outputlogs(log, 5)
        assert result["AFpeak"] == "No surrogate quality metrics found."

    def test_malformed_log_no_crash(self):
        result = _parse_crossvalidation_outputlogs("random garbage text", 5)
        assert result == {}


@pytest.mark.unit
class TestCrossValidationConfigGeneration:
    def _make_build_file(self, tmp_path):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0, 40.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        )
        build_file = tmp_path / "build.txt"
        df.to_csv(build_file, sep=" ", index=False)
        return build_file

    def test_create_sumo_crossvalidation_has_folds_and_metrics(self, tmp_path):
        build_file = self._make_build_file(tmp_path)
        result = create_sumo_crossvalidation_conffile(
            build_file=build_file,
            input_variables=["x", "y"],
            output_responses=["z"],
            N_CROSS_VALIDATION=5,
        )
        assert "cross_validation folds = 5" in result
        assert "root_mean_squared" in result
        assert "'x'" in result
        assert "'y'" in result
        assert "'z'" in result

    def test_create_sumo_crossvalidation_custom_fold_count(self, tmp_path):
        build_file = self._make_build_file(tmp_path)
        result = create_sumo_crossvalidation_conffile(
            build_file=build_file,
            input_variables=["x", "y"],
            output_responses=["z"],
            N_CROSS_VALIDATION=10,
        )
        assert "cross_validation folds = 10" in result

    def test_create_sumo_manual_crossvalidation_has_all_blocks(self, tmp_path):
        build_file = self._make_build_file(tmp_path)
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        result = create_sumo_manual_crossvalidation_conffile(
            fold_run_dir=fold_dir,
            build_file=build_file,
            input_variables=["x", "y"],
            output_response="z",
            validation_indices=[0, 2],
        )
        assert "environment" in result
        assert "model" in result
        assert "method" in result
        assert "variables" in result
        assert "responses" in result
        assert "'x'" in result
        assert "'y'" in result
        assert "'z'" in result

    def test_create_sumo_manual_crossvalidation_splits_training_and_validation(
        self, tmp_path
    ):

        build_file = self._make_build_file(tmp_path)
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        create_sumo_manual_crossvalidation_conffile(
            fold_run_dir=fold_dir,
            build_file=build_file,
            input_variables=["x", "y"],
            output_response="z",
            validation_indices=[1, 3],
        )
        training_files = list(fold_dir.glob("*training*"))
        validation_files = list(fold_dir.glob("*validation*"))
        assert len(training_files) == 1
        assert len(validation_files) == 1
        # 4 total rows minus the 2 held-out validation indices [1, 3] leaves 2 training rows
        training_df = pd.read_csv(training_files[0], sep=" ")
        assert len(training_df) == 2


# ---------------------------------------------------------------------------
# 5. Uncertainty Quantification
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCreateManualUqSamples:
    def test_normal_distribution_mean_and_std(self):
        samples = create_manual_uq_samples(
            input_vars=["x"],
            distributions={"x": {"distribution": "normal", "mean": 5.0, "std": 1.0}},
            num_samples=10_000,
            seed=42,
        )
        assert len(samples["x"]) == 10_000
        assert np.isclose(np.mean(samples["x"]), 5.0, atol=0.05)
        assert np.isclose(np.std(samples["x"]), 1.0, atol=0.05)

    def test_uniform_distribution_range(self):
        samples = create_manual_uq_samples(
            input_vars=["x"],
            distributions={"x": {"distribution": "uniform", "min": 2.0, "max": 8.0}},
            num_samples=10_000,
            seed=42,
        )
        assert np.all(np.array(samples["x"]) >= 2.0)
        assert np.all(np.array(samples["x"]) <= 8.0)

    def test_constant_distribution(self):
        samples = create_manual_uq_samples(
            input_vars=["x"],
            distributions={"x": {"distribution": "constant", "value": 42.0}},
            num_samples=100,
            seed=42,
        )
        assert all(v == 42.0 for v in samples["x"])

    def test_unsupported_distribution_raises(self):
        with pytest.raises(ValueError, match="Unsupported distribution"):
            create_manual_uq_samples(
                input_vars=["x"],
                distributions={"x": {"distribution": "lognormal", "mean": 0, "std": 1}},
                num_samples=10,
                seed=42,
            )

    def test_multiple_variables(self):
        samples = create_manual_uq_samples(
            input_vars=["x", "y"],
            distributions={
                "x": {"distribution": "constant", "value": 1.0},
                "y": {"distribution": "constant", "value": 2.0},
            },
            num_samples=5,
            seed=42,
        )
        assert all(v == 1.0 for v in samples["x"])
        assert all(v == 2.0 for v in samples["y"])


@pytest.mark.unit
class TestGetBoundsUniformDistributions:
    def test_get_bounds_uniform_distribution(self):
        dist = {"distribution": "uniform", "min": -1.0, "max": 5.0}
        lo, hi = get_bounds_uniform_distribution("x", dist)
        assert lo == -1.0
        assert hi == 5.0

    def test_get_bounds_non_uniform_raises(self):
        dist: dict[str, float | str] = {"distribution": "normal", "mean": 0, "std": 1}
        with pytest.raises(ValueError, match="Non-uniform"):
            get_bounds_uniform_distribution("x", dist)

    def test_get_bounds_missing_keys_raises(self):
        dist: dict[str, float | str] = {"distribution": "uniform"}
        with pytest.raises(ValueError, match="not defined"):
            get_bounds_uniform_distribution("x", dist)

    def test_get_bounds_min_ge_max_raises(self):
        dist = {"distribution": "uniform", "min": 5.0, "max": 5.0}
        with pytest.raises(ValueError, match="min >= max"):
            get_bounds_uniform_distribution("x", dist)

    def test_get_bounds_uniform_distributions(self):
        input_vars = ["x", "y"]
        distributions = {
            "x": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "y": {"distribution": "uniform", "min": -2.0, "max": 2.0},
        }
        lower, upper = get_bounds_uniform_distributions(input_vars, distributions)
        assert lower == [0.0, -2.0]
        assert upper == [1.0, 2.0]

    def test_get_bounds_uniform_distributions_missing_var(self):
        input_vars = ["x", "y"]
        distributions = {"x": {"distribution": "uniform", "min": 0.0, "max": 1.0}}
        with pytest.raises(ValueError, match="not defined"):
            get_bounds_uniform_distributions(input_vars, distributions)


@pytest.mark.unit
class TestUqPropagationConfigGeneration:
    def test_create_uq_propagation_has_all_blocks(self, tmp_path):
        build = tmp_path / "build.txt"
        result = create_uq_propagation_conffile(
            build_file=build,
            input_variables=["x", "y"],
            input_means={"x": 0.0, "y": 1.0},
            input_stds={"x": 0.1, "y": 0.2},
            output_responses=["response"],
            n_samples=500,
        )
        assert "environment" in result
        assert "model" in result
        assert "method" in result
        assert "normal_uncertain = 2" in result
        assert "responses" in result
        assert "'x'" in result
        assert "'y'" in result


# ---------------------------------------------------------------------------
# 6. Data Filtering
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestFilterData:
    def test_keep_idxs(self):
        df = pd.DataFrame({"a": [10, 20, 30, 40, 50], "b": [5, 4, 3, 2, 1]})
        result = _filter_data(df, keep_idxs=[1, 3])
        assert list(result["a"]) == [20, 40]

    def test_filter_n_samples(self):
        df = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
        result = _filter_data(df, filter_N_samples=3)
        assert len(result) == 3
        assert list(result["a"]) == [10, 20, 30]

    def test_filter_highest_n_removes_top_rows(self):
        df = pd.DataFrame({"a": [10, 20, 30, 40, 50], "b": [1, 2, 3, 4, 5]})
        result = _filter_data(df, filter_highest_N=2, filter_highest_N_variable="b")
        assert len(result) == 3
        assert list(result["b"]) == [3, 2, 1]

    def test_filter_highest_n_uses_last_column_by_default(self):
        df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 5, 3]})
        result = _filter_data(df, filter_highest_N=1)
        assert len(result) == 2

    def test_mutual_exclusion_assertion(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(AssertionError):
            _filter_data(df, filter_N_samples=1, filter_highest_N=1)


# ---------------------------------------------------------------------------
# DEFERRED: MOGA / Pareto (not this round — see docs/TIER1_TIER2_UNIT_TESTS_PLAN.md)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestIsDominated:
    def test_dominated_returns_true(self):
        point = np.array([3, 4])
        others = np.array([[1, 2]])
        assert is_dominated(point, others) is True

    def test_not_dominated_returns_false(self):
        point = np.array([1, 2])
        others = np.array([[3, 4]])
        assert is_dominated(point, others) is False

    def test_self_dominance_returns_true(self):
        point = np.array([1, 2])
        others = np.array([[1, 2]])
        assert is_dominated(point, others) is True

    def test_multi_objective_dominated(self):
        point = np.array([2, 3])
        others = np.array([[1, 1]])
        assert is_dominated(point, others) is True

    def test_multi_objective_not_dominated(self):
        point = np.array([1, 5])
        others = np.array([[5, 1]])
        assert is_dominated(point, others) is False

    def test_multiple_others_one_dominates(self):
        point = np.array([3, 4])
        others = np.array([[5, 5], [1, 2], [4, 1]])
        assert is_dominated(point, others) is True

    def test_multiple_others_none_dominates(self):
        point = np.array([1, 3])
        others = np.array([[3, 1], [2, 2]])
        assert is_dominated(point, others) is False


@pytest.mark.unit
class TestGetNonDominatedIndices:
    def test_known_2d_front_minimization(self):
        df = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [3, 1, 2, 4, 5]})
        result = get_non_dominated_indices(df, ["f1", "f2"], ["min", "min"])
        assert sorted(result) == [0, 1]

    def test_all_points_on_front(self):
        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [3, 2, 1]})
        result = get_non_dominated_indices(df, ["f1", "f2"], ["min", "min"])
        assert sorted(result) == [0, 1, 2]

    def test_single_point(self):
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        result = get_non_dominated_indices(df, ["f1", "f2"], ["min", "min"])
        assert result == [0]

    def test_maximization_sign_flip(self):
        df = pd.DataFrame({"f1": [5, 4, 3, 2, 1], "f2": [5, 4, 3, 2, 1]})
        result = get_non_dominated_indices(df, ["f1", "f2"], ["max", "max"])
        assert result == [0]

    def test_sort_by_column(self):
        df = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5],
                "f2": [3, 1, 2, 4, 5],
            }
        )
        result = get_non_dominated_indices(
            df, ["f1", "f2"], ["min", "min"], sort_by_column="f1"
        )

        assert list(result) == [0, 1]

    def test_mismatched_modes_raises(self):
        df = pd.DataFrame({"f1": [1], "f2": [2]})
        with pytest.raises(ValueError, match="must match"):
            get_non_dominated_indices(df, ["f1", "f2"], ["min"])

    def test_invalid_mode_raises(self):
        df = pd.DataFrame({"f1": [1], "f2": [2]})
        with pytest.raises(ValueError, match="not recognized"):
            get_non_dominated_indices(df, ["f1", "f2"], ["min", "invalid"])  # ty: ignore[invalid-argument-type]


@pytest.mark.unit
class TestMogaConfigGeneration:
    def test_create_moga_optimization_has_all_blocks(self, tmp_path):
        build = tmp_path / "build.txt"
        result = create_moga_optimization_conffile(
            build_file=build,
            input_variables=["a", "b"],
            output_responses=["obj1", "obj2"],
            moga_kwargs={},
        )
        assert "environment" in result
        assert "model" in result
        assert "moga" in result
        assert "variables" in result
        assert "responses" in result
        assert "'a'" in result
        assert "'b'" in result
        assert "'obj1'" in result
        assert "'obj2'" in result
