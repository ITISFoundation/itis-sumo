"""Tier 2 property-based / invariant tests for the MMUX solver backend."""

import numpy as np
import pandas as pd
import pytest

from itis_sumo.data.funs_data_processing import (
    create_grid_samples,
    create_samples_along_axes,
    get_non_dominated_indices,
    is_dominated,
    load_data,
)
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor
from itis_sumo.sampling.lhs import _lhscentered, _lhsclassic, _lhsmaximin, lhs


# ---------------------------------------------------------------------------
# P1. LHS Invariants
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLHSInvariants:
    def test_bounds(self):
        """All LHS values must be in [0, 1]."""
        H = lhs(5, 20, seed=42)
        assert np.all(H >= 0)
        assert np.all(H <= 1)

    def test_shape(self):
        """Output shape is (k, n) — k samples, n factors."""
        n, k = 4, 15
        H = lhs(n, k, seed=42)
        assert H.shape == (k, n)

    def test_stratification_one_per_bin(self):
        """For each column, floor(value * k) gives a permutation of 0..k-1."""
        n, k = 3, 10
        H = lhs(n, k, seed=42)
        for j in range(n):
            bins = np.floor(H[:, j] * k).astype(int)
            assert sorted(bins) == list(range(k)), (
                f"Column {j} does not produce a full stratification permutation"
            )

    @pytest.mark.parametrize("seed_val", [1, 42, 123, 999])
    def test_bounds_parametrized(self, seed_val):
        """Bounds hold across multiple seeds."""
        H = lhs(6, 30, seed=seed_val)
        assert np.all(H >= 0)
        assert np.all(H <= 1)

    def test_maximin_monotonicity(self):
        """_lhsmaximin min pairwise distance >= _lhsclassic min pairwise distance."""
        from scipy import spatial

        seed = 42
        rs_classic = np.random.RandomState(seed)
        rs_maximin = np.random.RandomState(seed)
        n, k = 5, 20
        H_classic = _lhsclassic(n, k, rs_classic)
        H_maximin = _lhsmaximin(n, k, 10, "maximin", rs_maximin)
        d_classic = spatial.distance.pdist(H_classic, "euclidean")
        d_maximin = spatial.distance.pdist(H_maximin, "euclidean")
        assert np.min(d_maximin) >= np.min(d_classic) - 1e-12

    def test_centered_exact_midpoints(self):
        """_lhscentered values are exact midpoints (i+0.5)/k for each stratum."""
        n, k = 3, 8
        seed = 42
        rs = np.random.RandomState(seed)
        H = _lhscentered(n, k, rs)
        expected_centers = np.array([(i + 0.5) / k for i in range(k)])
        for j in range(n):
            col_sorted = np.sort(H[:, j])
            np.testing.assert_allclose(col_sorted, expected_centers, atol=1e-14)


# ---------------------------------------------------------------------------
# P2. Grid Sample Invariants
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGridSampleInvariants:
    def test_total_points(self, tmp_path):
        """Total rows = product of n_points_per_dimension for grid variables only.

        n_points_per_dimension must have the same length as input_vars.
        Non-grid vars still need an entry (it's ignored by meshgrid).
        """
        grid_vars = ["x1", "x2"]
        input_vars = ["x1", "x2", "x3"]
        mins = [0.0, 0.0, 5.0]
        cut_values = [0.5, 0.5, 10.0]
        maxs = [1.0, 1.0, 15.0]
        n_points = [3, 4, 1]
        path = create_grid_samples(
            tmp_path, grid_vars, input_vars, mins, cut_values, maxs, n_points
        )
        df = load_data(path)
        assert len(df) == 3 * 4

    def test_grid_values_within_bounds(self, tmp_path):
        """Grid variable values are >= min and <= max."""
        grid_vars = ["x1", "x2"]
        input_vars = ["x1", "x2"]
        mins = [0.0, -1.0]
        cut_values = [0.5, -0.5]
        maxs = [1.0, 1.0]
        n_points = [5, 5]
        path = create_grid_samples(
            tmp_path, grid_vars, input_vars, mins, cut_values, maxs, n_points
        )
        df = load_data(path)
        for i, var in enumerate(grid_vars):
            assert np.all(df[var].astype(float) >= mins[i] - 1e-12)
            assert np.all(df[var].astype(float) <= maxs[i] + 1e-12)

    def test_non_grid_variables_fixed(self, tmp_path):
        """Non-grid variables are exactly equal to cut_value."""
        grid_vars = ["x1"]
        input_vars = ["x1", "x2"]
        mins = [0.0, 0.0]
        cut_values = [0.5, 42.0]
        maxs = [1.0, 100.0]
        n_points = [4, 1]
        path = create_grid_samples(
            tmp_path, grid_vars, input_vars, mins, cut_values, maxs, n_points
        )
        df = load_data(path)
        np.testing.assert_allclose(df["x2"].astype(float).values, 42.0, atol=1e-12)

    def test_2d_meshgrid_all_combinations(self, tmp_path):
        """2 grid vars with 3 points each produces 9 rows covering all combos."""
        grid_vars = ["x1", "x2"]
        input_vars = ["x1", "x2"]
        mins = [0.0, 0.0]
        cut_values = [0.5, 0.5]
        maxs = [1.0, 1.0]
        n_points = [3, 3]
        path = create_grid_samples(
            tmp_path, grid_vars, input_vars, mins, cut_values, maxs, n_points
        )
        df = load_data(path)
        assert len(df) == 9
        x1_vals = sorted(df["x1"].astype(float).unique())
        x2_vals = sorted(df["x2"].astype(float).unique())
        assert len(x1_vals) == 3
        assert len(x2_vals) == 3
        np.testing.assert_allclose(x1_vals, np.linspace(0, 1, 3), atol=1e-12)
        np.testing.assert_allclose(x2_vals, np.linspace(0, 1, 3), atol=1e-12)


# ---------------------------------------------------------------------------
# P3. Axis Sweep Invariants
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestAxisSweepInvariants:
    def test_total_points(self, tmp_path):
        """Total rows = NSAMPLESPERVAR * number of input variables."""
        df = pd.DataFrame({"x1": np.linspace(0, 10, 50), "x2": np.linspace(-5, 5, 50)})
        NSP = 20
        path = create_samples_along_axes(tmp_path, df, ["x1", "x2"], NSP)
        result = load_data(path)
        assert len(result) == NSP * 2

    def test_sweep_monotonicity(self, tmp_path):
        """Within each sweep segment, the swept variable increases monotonically."""
        data = pd.DataFrame(
            {"x1": np.linspace(0, 10, 50), "x2": np.linspace(-5, 5, 50)}
        )
        NSP = 15
        path = create_samples_along_axes(tmp_path, data, ["x1", "x2"], NSP)
        result = load_data(path)
        for i, var in enumerate(["x1", "x2"]):
            segment = result[var].astype(float).values[i * NSP : (i + 1) * NSP]
            diffs = np.diff(segment)
            assert np.all(diffs >= -1e-12), (
                f"Sweep segment for {var} is not monotonically increasing"
            )

    def test_non_sweep_constancy(self, tmp_path):
        """Within each sweep segment, non-sweep variables equal the cut value."""
        data = pd.DataFrame(
            {"x1": np.linspace(0, 10, 50), "x2": np.linspace(-5, 5, 50)}
        )
        NSP = 10
        path = create_samples_along_axes(tmp_path, data, ["x1", "x2"], NSP)
        result = load_data(path)
        mean_vals = data.mean().values
        for i, var in enumerate(["x1", "x2"]):
            other_vars = [v for v in ["x1", "x2"] if v != var]
            for ov in other_vars:
                segment = result[ov].astype(float).values[i * NSP : (i + 1) * NSP]
                np.testing.assert_allclose(
                    segment, mean_vals[list(data.columns).index(ov)], atol=1e-12
                )

    def test_uniform_spacing(self, tmp_path):
        """Consecutive differences within each sweep segment are equal."""
        data = pd.DataFrame(
            {"x1": np.linspace(0, 10, 50), "x2": np.linspace(-5, 5, 50)}
        )
        NSP = 12
        path = create_samples_along_axes(tmp_path, data, ["x1", "x2"], NSP)
        result = load_data(path)
        for i, var in enumerate(["x1", "x2"]):
            segment = result[var].astype(float).values[i * NSP : (i + 1) * NSP]
            diffs = np.diff(segment)
            np.testing.assert_allclose(
                diffs,
                diffs[0],
                atol=1e-12,
                err_msg=f"Sweep for {var} is not uniformly spaced",
            )


# ---------------------------------------------------------------------------
# P4. DataPreprocessor Roundtrip
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDataPreprocessorRoundtrip:
    def _make_df(self):
        return pd.DataFrame(
            {
                "length": [1.0, 2.0, 3.0, 4.0, 5.0],
                "width": [10.0, 20.0, 30.0, 40.0, 50.0],
                "stress": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    def test_z_score_roundtrip(self):
        df = self._make_df()
        pp = DataPreprocessor()
        pp.setup_variables(["length", "width"], ["stress"])
        pp.setup_normalization(
            input_normalizations={"length": "z_score", "width": "z_score"},
            output_normalizations={"stress": "z_score"},
        )
        pp.fit(df)
        transformed = pp.transform(df)
        restored = pp.inverse_transform(transformed)
        for col in ["length", "width", "stress"]:
            np.testing.assert_allclose(restored[col], df[col].tolist(), atol=1e-10)

    def test_min_max_roundtrip(self):
        df = self._make_df()
        pp = DataPreprocessor()
        pp.setup_variables(["length", "width"], ["stress"])
        pp.setup_normalization(
            input_normalizations={"length": "min_max", "width": "min_max"},
            output_normalizations={"stress": "min_max"},
        )
        pp.fit(df)
        transformed = pp.transform(df)
        restored = pp.inverse_transform(transformed)
        for col in ["length", "width", "stress"]:
            np.testing.assert_allclose(restored[col], df[col].tolist(), atol=1e-10)

    def test_sign_switch_plus_normalization_roundtrip(self):
        df = self._make_df()
        pp = DataPreprocessor()
        pp.setup_variables(["length", "width"], ["stress"])
        pp.setup_normalization(
            input_normalizations={"length": "z_score"},
            output_normalizations={"stress": "min_max"},
        )
        pp.setup_sign_switching(
            input_sign_switches=["width"], output_sign_switches=["stress"]
        )
        pp.fit(df)
        transformed = pp.transform(df)
        restored = pp.inverse_transform(transformed)
        for col in ["length", "width", "stress"]:
            np.testing.assert_allclose(restored[col], df[col].tolist(), atol=1e-10)

    def test_config_save_load_roundtrip(self, tmp_path):
        df = self._make_df()
        pp = DataPreprocessor()
        pp.setup_variables(["length", "width"], ["stress"])
        pp.setup_normalization(
            input_normalizations={"length": "z_score"},
            output_normalizations={"stress": "min_max"},
        )
        pp.fit(df)
        config_path = tmp_path / "config.json"
        pp.save_config(config_path)

        pp2 = DataPreprocessor().load_config(config_path)
        t1 = pp.transform(df)
        t2 = pp2.transform(df)
        pd.testing.assert_frame_equal(t1, t2)


# ---------------------------------------------------------------------------
# P5. Normalization Invariants (DataPreprocessor)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestNormalizationInvariants:
    def _make_df(self):
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
                "c": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    def test_z_score_output_mean_zero_std_one(self):
        df = self._make_df()
        pp = DataPreprocessor()
        pp.setup_variables(["a", "b"], ["c"])
        pp.setup_normalization(
            input_normalizations={"a": "z_score", "b": "z_score"},
            output_normalizations={"c": "z_score"},
        )
        pp.fit(df)
        transformed = pp.transform(df)
        for col in transformed.columns:
            vals = transformed[col].values
            np.testing.assert_allclose(
                np.mean(vals),
                0.0,
                atol=1e-10,
                err_msg=f"Column {col} mean is not ~0 after z-score",
            )
            np.testing.assert_allclose(
                np.std(vals),
                1.0,
                atol=1e-10,
                err_msg=f"Column {col} std is not ~1 after z-score",
            )

    def test_min_max_output_range_zero_to_one(self):
        df = self._make_df()
        pp = DataPreprocessor()
        pp.setup_variables(["a", "b"], ["c"])
        pp.setup_normalization(
            input_normalizations={"a": "min_max", "b": "min_max"},
            output_normalizations={"c": "min_max"},
        )
        pp.fit(df)
        transformed = pp.transform(df)
        for col in transformed.columns:
            np.testing.assert_allclose(
                transformed[col].min(),
                0.0,
                atol=1e-10,
                err_msg=f"Column {col} min is not ~0 after min-max",
            )
            np.testing.assert_allclose(
                transformed[col].max(),
                1.0,
                atol=1e-10,
                err_msg=f"Column {col} max is not ~1 after min-max",
            )

    def test_sign_flip_double_negation(self):
        """Negating a variable and negating again returns the original value."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        pp = DataPreprocessor()
        pp.setup_variables(["a"], ["b"])
        pp.setup_sign_switching(input_sign_switches=["a"])
        pp.fit(df)
        transformed = pp.transform(df)
        restored = pp.inverse_transform(transformed)
        np.testing.assert_allclose(restored["a"], df["a"].tolist(), atol=1e-10)


# ---------------------------------------------------------------------------
# DEFERRED: Pareto Dominance Invariants (formerly P1) — not this round,
# see docs/TIER1_TIER2_UNIT_TESTS_PLAN.md
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestParetoDominance:
    def test_irreflexivity_self_dominated(self):
        """is_dominated uses `>=`, so a point IS dominated by itself (reflexive).

        This is a known behavior of the implementation: `all(a >= a)` is True for
        any point, meaning the function treats self-comparison as domination.
        We test the actual (reflexive) behavior here.
        """
        a = np.array([1.0, 2.0, 3.0])
        assert is_dominated(a, [a]) is True

    def test_asymmetry(self):
        """If a dominates b in the is_dominated sense (a >= b in all dims),
        then b does NOT dominate a."""
        a = np.array([2.0, 3.0])
        b = np.array([1.0, 2.0])
        # is_dominated(a, [b]) checks a >= b → [True, True] → True
        assert is_dominated(a, [b]) is True
        # is_dominated(b, [a]) checks b >= a → [False, False] → False
        assert is_dominated(b, [a]) is False

    def test_asymmetry_strict(self):
        """For strictly unequal points where one dominates, the reverse is false."""
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([1.0, 1.0, 1.0])
        assert is_dominated(a, [b]) is True
        assert is_dominated(b, [a]) is False

    def test_transitivity_on_dominance(self):
        """If a dominates b and b dominates c, then a dominates c."""
        a = np.array([3.0, 3.0])
        b = np.array([2.0, 2.0])
        c = np.array([1.0, 1.0])
        assert is_dominated(a, [b]) is True
        assert is_dominated(b, [c]) is True
        assert is_dominated(a, [c]) is True

    def test_non_dominated_output_no_dominated_pairs(self):
        """No two points in the non-dominated set should dominate each other."""
        df = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 0.5, 1.5], "y": [3.0, 2.0, 1.0, 0.5, 2.5]}
        )
        indices = get_non_dominated_indices(df, ["x", "y"], ["min", "min"])
        nd_points = df.loc[indices, ["x", "y"]].values
        for i, p in enumerate(nd_points):
            others = [q for j, q in enumerate(nd_points) if j != i]
            if others:
                assert not is_dominated(p, others), (
                    f"Non-dominated point {p} is dominated by another in the set"
                )

    def test_non_dominated_completeness(self):
        """All non-dominated points must appear in the output."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0], "y": [10.0, 3.0, 2.0, 1.0]})
        indices = get_non_dominated_indices(df, ["x", "y"], ["min", "min"])
        nd_set = set(indices)
        data_np = df[["x", "y"]].values
        for i, point in enumerate(data_np):
            others = np.delete(data_np, i, axis=0)
            if not is_dominated(point, others):
                assert i in nd_set, (
                    f"Point {point} at index {i} is truly non-dominated but missing from output"
                )

    def test_non_dominated_with_max_mode(self):
        """Non-dominance works correctly when optimization mode is 'max'."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})
        indices = get_non_dominated_indices(df, ["x", "y"], ["max", "max"])
        nd_points = df.loc[indices, ["x", "y"]].values
        for i, p in enumerate(nd_points):
            others = [q for j, q in enumerate(nd_points) if j != i]
            if others:
                assert not is_dominated(p, others)
