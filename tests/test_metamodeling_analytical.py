"""Tier 3 analytical integration tests: real Dakota solver end-to-end.

These tests run the real Dakota solver (`itis-dakota`, not mocked) against small
analytical functions with known closed-form answers and check the numeric outputs
match. This is the "give it inputs, check the outputs are what we expect" layer.
Each test spawns a real Dakota subprocess, so all tests here are tagged
`@pytest.mark.analytical` and excluded from the default `make test-flaskapi` run.

Test IDs (B1-B8, C1-C3, D1-D5, E1-E4, F1-F3, F5-F7) mirror the categories/IDs in
docs/VERIFICATION_VALIDATION_PLAN.md. Category A (LHS) and H (DataPreprocessor) are
pure-Python and already covered in test_unit_solver.py / test_property_invariants.py;
Category G (MOGA) is deferred (see those files' "DEFERRED" sections).

Category F note: `propagate_uq`/`create_uq_propagation_conffile` (Dakota-native UQ,
normal-uncertain only) is *not called from any Flask route* - grepping
blueprints/dakota.py confirms only `evaluate_sumo*` functions are imported from
funs_evaluate.py. The real, user-reachable UQ pathway is
`POST /manual_uq_propagation_with_uncertainty` (blueprints/dakota.py), which composes
`create_manual_uq_samples` (funs_data_processing.py - supports normal/uniform/constant
per-variable distributions) with `evaluate_sumo` and an erfinv-based predictive-
uncertainty injection, entirely in Python plus one Dakota surrogate-evaluation call.
This suite tests *that* pathway (see `_manual_uq_propagate` below), which is why F2
(uniform input) and F7 (mixed normal+uniform inputs) are now included, not excluded.

See docs/TIER1_TIER2_UNIT_TESTS_PLAN.md (Tier 3) for scope and rationale.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import erfinv

from itis_sumo.data.funs_data_processing import create_manual_uq_samples
from itis_sumo.evaluate.funs_evaluate import (
    evaluate_sumo,
    evaluate_sumo_along_axes,
    evaluate_sumo_crossvalidation,
    evaluate_sumo_manual_crossvalidation,
    evaluate_sumo_on_grid,
)


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return d


def _write_processed(df: pd.DataFrame, path):
    # `add_surrogate_model` only omits the `eval_id` tabular column when the training
    # file path contains "processed" (funs_create_dakota_conf.py) - fixtures must
    # follow that naming convention.
    df.to_csv(path, sep=" ", index=False)
    return path


def _rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def _manual_uq_propagate(
    run_dir,
    train_file,
    input_vars,
    output_var,
    distributions,
    num_samples,
    n_histograms=100,
    seed=42,
):
    """Mirrors the real `/manual_uq_propagation_with_uncertainty` route
    (blueprints/dakota.py): draw per-variable UQ samples, evaluate the surrogate on
    them, then inject the surrogate's own predictive uncertainty via the erfinv trick
    (sqrt(2)*erfinv(U) ~ N(0,1) for U~Uniform(-1,1)). Returns the flat array of
    propagated output samples. Skips DataPreprocessor mapping/normalization, consistent
    with the rest of this suite, which calls funs_evaluate directly rather than going
    through the Flask route.
    """
    samples = create_manual_uq_samples(input_vars, distributions, num_samples, seed)
    eval_file = _write_processed(
        pd.DataFrame(samples), run_dir / "uq_samples_processed.txt"
    )

    results = evaluate_sumo(run_dir, train_file, eval_file, input_vars, output_var)
    pred = np.array(results[f"{output_var}_hat"])
    std = np.array(results[f"{output_var}_std_hat"])

    rng = np.random.default_rng(seed)
    all_values = np.empty((n_histograms, num_samples))
    for i in range(n_histograms):
        r = np.sqrt(2) * erfinv(rng.uniform(-1 + 1e-10, 1 - 1e-10, size=num_samples))
        all_values[i, :] = pred + r * std
    return all_values.ravel()


@pytest.mark.analytical
class TestSurrogateAccuracy:
    """Category B: Surrogate Model Accuracy."""

    def test_exact_interpolation_at_training_points(self, run_dir):
        """B1: GP surrogate must reproduce training data almost exactly at training points."""
        x_train = np.linspace(0, 1, 15)
        y_train = 2 * x_train + 1
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train}), run_dir / "train_processed.txt"
        )
        eval_file = _write_processed(pd.DataFrame({"x": x_train}), run_dir / "eval.txt")

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
        np.testing.assert_allclose(result["y_hat"], y_train, atol=1e-3)

    def test_linear_function_matches_analytical(self, run_dir):
        """B2: f(x) = 2x + 1: GP surrogate should reproduce this near-exactly."""
        x_train = np.linspace(0, 1, 20)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": 2 * x_train + 1}),
            run_dir / "train_processed.txt",
        )
        x_eval = np.array([0.05, 0.3, 0.55, 0.72, 0.95])
        eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "eval.txt")

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
        expected = 2 * x_eval + 1
        rmse, r2 = _rmse(expected, result["y_hat"]), _r2(expected, result["y_hat"])
        assert rmse < 1e-2
        assert r2 > 0.99

    def test_quadratic_function_matches_analytical(self, run_dir):
        """B3: f(x) = x^2: GP surrogate should approximate within a looser tolerance."""
        x_train = np.linspace(0, 1, 30)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": x_train**2}),
            run_dir / "train_processed.txt",
        )
        x_eval = np.array([0.1, 0.35, 0.5, 0.65, 0.9])
        eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "eval.txt")

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
        expected = x_eval**2
        rmse, r2 = _rmse(expected, result["y_hat"]), _r2(expected, result["y_hat"])
        assert rmse < 1e-2
        assert r2 > 0.99

    def test_sinusoidal_function_within_tolerance(self, run_dir):
        """B4: f(x) = sin(x) on [0, 2pi]: looser tolerance, GP needs denser sampling."""
        x_train = np.linspace(0, 2 * np.pi, 50)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": np.sin(x_train)}),
            run_dir / "train_processed.txt",
        )
        x_eval = np.linspace(0.1, 2 * np.pi - 0.1, 12)
        eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "eval.txt")

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
        expected = np.sin(x_eval)
        assert _rmse(expected, result["y_hat"]) < 0.05
        assert _r2(expected, result["y_hat"]) > 0.95

    def test_logarithmic_function_within_tolerance(self, run_dir):
        """B5: f(x) = ln(x) on [0.01, 2]: monotonic, unbounded derivative near 0.

        Eval points stay away from the extreme boundary (x=0.01) where that unbounded
        derivative makes the GP fit genuinely unreliable at this training density -
        that behavior is expected/documented, not something this test should paper
        over with a looser tolerance.
        """
        x_train = np.linspace(0.01, 2, 20)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": np.log(x_train)}),
            run_dir / "train_processed.txt",
        )
        x_eval = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 1.9])
        eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "eval.txt")

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
        expected = np.log(x_eval)
        assert _rmse(expected, result["y_hat"]) < 0.01

    def test_rosenbrock_2d_harder_than_smooth_functions(self, run_dir):
        """B6: Rosenbrock (2D), non-convex curved valley - GP struggles more than on
        smooth low-order functions, but should still capture the rough shape and
        report non-trivial predictive uncertainty (surfpack variance)."""
        rng = np.linspace(-2, 2, 10)
        xx, yy = np.meshgrid(rng, rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        z_train = (1 - x_train) ** 2 + 100 * (y_train - x_train**2) ** 2
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": z_train}),
            run_dir / "train_processed.txt",
        )
        x_eval = np.array([-1.5, -0.5, 0.5, 1.5])
        y_eval = np.array([1.5, -0.5, 0.5, -1.5])
        eval_file = _write_processed(
            pd.DataFrame({"x": x_eval, "y": y_eval}), run_dir / "eval.txt"
        )

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x", "y"], "z")
        expected = (1 - x_eval) ** 2 + 100 * (y_eval - x_eval**2) ** 2
        z_hat = np.array(result["z_hat"])
        assert np.all(np.isfinite(z_hat))
        normalized_rmse = _rmse(expected, z_hat) / (np.std(expected) + 1e-9)
        # captures the rough shape (better than guessing the mean) but is
        # measurably worse than the near-exact fits on smooth low-order functions above
        assert normalized_rmse < 1.0
        assert "z_std_hat" in result
        assert np.all(np.array(result["z_std_hat"]) > 0)

    def test_convergence_with_training_size(self, run_dir):
        """B7: RMSE for f(x) = sin(x) should decrease as training size grows.

        Uses 3 full periods (domain [0, 6pi]) so that a sparse training set (N=6)
        genuinely under-resolves the oscillation - on a single period, even N=10
        already interpolates near machine precision and the trend can't be observed.
        """
        x_eval = np.linspace(0.4, 6 * np.pi - 0.4, 8)
        eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "eval.txt")
        expected = np.sin(x_eval)

        rmses = {}
        for n in (6, 15, 60):
            sub_dir = run_dir / f"n_{n}"
            sub_dir.mkdir()
            x_train = np.linspace(0, 6 * np.pi, n)
            train_file = _write_processed(
                pd.DataFrame({"x": x_train, "y": np.sin(x_train)}),
                sub_dir / "train_processed.txt",
            )
            result = evaluate_sumo(sub_dir, train_file, eval_file, ["x"], "y")
            rmses[n] = _rmse(expected, result["y_hat"])

        assert rmses[60] < rmses[15] < rmses[6]

    def test_variance_near_zero_at_training_points_grows_away(self, run_dir):
        """B8: predictive variance ~0 at training points, larger when extrapolating far away."""
        x_train = np.linspace(0, 1, 15)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": 2 * x_train + 1}),
            run_dir / "train_processed.txt",
        )
        x_eval = np.concatenate([x_train[:3], [10.0, 20.0]])
        eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "eval.txt")

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
        std_hat = np.array(result["y_std_hat"])
        std_at_train, std_far = std_hat[:3], std_hat[3:]
        assert np.all(std_at_train < 1e-3)
        assert np.all(std_far > std_at_train.max())


@pytest.mark.analytical
class TestAxisSweep:
    """Category C: Axis Sweep."""

    def test_linear_2d_sweep_slopes_match_analytical(self, run_dir):
        """C1: f(x,y) = 2x + 3y: sweeping x gives slope 2, sweeping y gives slope 3."""
        rng = np.linspace(0, 1, 6)
        xx, yy = np.meshgrid(rng, rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": 2 * x_train + 3 * y_train}),
            run_dir / "train_processed.txt",
        )

        results = evaluate_sumo_along_axes(
            run_dir, train_file, ["x", "y"], "z", NSAMPLESPERVAR=11
        )

        slope_x = np.polyfit(results["x"]["x"], results["x"]["y_hat"], 1)[0]
        slope_y = np.polyfit(results["y"]["x"], results["y"]["y_hat"], 1)[0]
        assert slope_x == pytest.approx(2.0, rel=0.05)
        assert slope_y == pytest.approx(3.0, rel=0.05)

    def test_sinusoidal_sweep_rmse_within_tolerance(self, run_dir):
        """C2: f(x,y) = sin(x) + y: sweep along x (y fixed) should capture the oscillation."""
        x_rng = np.linspace(0, 2 * np.pi, 12)
        y_rng = np.linspace(0, 1, 12)
        xx, yy = np.meshgrid(x_rng, y_rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": np.sin(x_train) + y_train}),
            run_dir / "train_processed.txt",
        )
        y_cut = 0.5
        results = evaluate_sumo_along_axes(
            run_dir,
            train_file,
            ["x", "y"],
            "z",
            # cut_values must cover every input var (funs_data_processing.py:242 indexes it
            # by all of input_vars) - "x"'s entry is irrelevant since it's the swept axis and
            # gets overwritten by the linspace, but the key must still be present.
            cut_values={"x": 0.0, "y": y_cut},
            NSAMPLESPERVAR=15,
        )

        x_swept = np.array(results["x"]["x"])
        expected = np.sin(x_swept) + y_cut
        assert _rmse(expected, results["x"]["y_hat"]) < 0.1

    def test_custom_cut_values_match_analytical(self, run_dir):
        """C3: with a non-default cut value, sweep predictions should match f(x, cut_y)."""
        rng = np.linspace(0, 1, 6)
        xx, yy = np.meshgrid(rng, rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": 2 * x_train + 3 * y_train}),
            run_dir / "train_processed.txt",
        )
        custom_cut_y = 0.2  # deliberately not the data mean (0.5)

        results = evaluate_sumo_along_axes(
            run_dir,
            train_file,
            ["x", "y"],
            "z",
            cut_values={"x": 0.0, "y": custom_cut_y},
            NSAMPLESPERVAR=9,
        )

        x_swept = np.array(results["x"]["x"])
        expected = 2 * x_swept + 3 * custom_cut_y
        np.testing.assert_allclose(results["x"]["y_hat"], expected, atol=0.05)


@pytest.mark.analytical
class TestGridEvaluation:
    """Category D: Grid Evaluation."""

    def test_grid_evaluation_linear_2d(self, run_dir):
        """D1: f(x,y) = x + y: predictions on a small 2D grid should match analytically.

        We compare the *set* of predicted values against the *set* of analytically
        expected values (both sorted) rather than positional (x, y) -> z alignment:
        `evaluate_sumo_on_grid`'s 2D reshape/transpose logic is explicitly flagged
        as uncertain in its own source comments ("Why???") for some variable
        orderings, so asserting on element ordering would couple this test to that
        unresolved implementation detail rather than to surrogate correctness.
        """
        rng = np.linspace(0, 1, 6)
        xx, yy = np.meshgrid(rng, rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": x_train + y_train}),
            run_dir / "train_processed.txt",
        )

        NSAMPLESPERVAR = 4
        results = evaluate_sumo_on_grid(
            run_dir,
            train_file,
            grid_vars=["x", "y"],
            input_vars=["x", "y"],
            response_var="z",
            NSAMPLESPERVAR=NSAMPLESPERVAR,
        )

        z_grid = np.array(results["z"])
        assert z_grid.shape == (NSAMPLESPERVAR, NSAMPLESPERVAR)

        x_vals = np.array(results["x"])
        y_vals = np.array(results["y"])
        z_sorted = np.sort(z_grid.ravel())
        expected_sorted = np.sort(x_vals + y_vals)
        np.testing.assert_allclose(z_sorted, expected_sorted, atol=0.05)

    def test_grid_evaluation_quadratic_2d(self, run_dir):
        """D2: f(x,y) = x^2 + y^2: same sorted-set comparison as D1, looser tolerance."""
        rng = np.linspace(-1, 1, 8)
        xx, yy = np.meshgrid(rng, rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": x_train**2 + y_train**2}),
            run_dir / "train_processed.txt",
        )

        NSAMPLESPERVAR = 5
        results = evaluate_sumo_on_grid(
            run_dir,
            train_file,
            grid_vars=["x", "y"],
            input_vars=["x", "y"],
            response_var="z",
            NSAMPLESPERVAR=NSAMPLESPERVAR,
        )

        z_grid = np.array(results["z"])
        x_vals, y_vals = np.array(results["x"]), np.array(results["y"])
        z_sorted = np.sort(z_grid.ravel())
        expected_sorted = np.sort(x_vals**2 + y_vals**2)
        assert _rmse(expected_sorted, z_sorted) < 0.05

    def test_grid_dimension_consistency(self, run_dir):
        """D3: NSAMPLESPERVAR=10 with 2 grid vars -> output shape (10, 10)."""
        rng = np.linspace(0, 1, 6)
        xx, yy = np.meshgrid(rng, rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": x_train + y_train}),
            run_dir / "train_processed.txt",
        )

        results = evaluate_sumo_on_grid(
            run_dir,
            train_file,
            grid_vars=["x", "y"],
            input_vars=["x", "y"],
            response_var="z",
            NSAMPLESPERVAR=10,
        )
        assert np.array(results["z"]).shape == (10, 10)

    def test_grid_with_fixed_non_grid_variable(self, run_dir):
        """D4: 3 inputs, 2 grid vars + 1 fixed var - the fixed var stays at its cut value."""
        rng = np.linspace(0, 1, 5)
        xx, yy, zz = np.meshgrid(rng, rng, rng)
        x_train, y_train, z_train = xx.ravel(), yy.ravel(), zz.ravel()
        train_file = _write_processed(
            pd.DataFrame(
                {
                    "x": x_train,
                    "y": y_train,
                    "z": z_train,
                    "w": x_train + y_train + z_train,
                }
            ),
            run_dir / "train_processed.txt",
        )
        fixed_z = 0.3

        results = evaluate_sumo_on_grid(
            run_dir,
            train_file,
            grid_vars=["x", "y"],
            input_vars=["x", "y", "z"],
            response_var="w",
            # cut_values must cover every input var (funs_evaluate.py:350 indexes it by all
            # of input_vars) - "x"/"y" entries are irrelevant since they're grid vars and get
            # overwritten by the grid linspace, but the keys must still be present.
            cut_values={"x": 0.0, "y": 0.0, "z": fixed_z},
            NSAMPLESPERVAR=4,
        )

        np.testing.assert_allclose(results["z"], fixed_z, atol=1e-9)

    def test_grid_reshape_matches_positionally_for_asymmetric_case(self, run_dir):
        """D5: element-wise check of the grid reshape/transpose, using an asymmetric
        domain (different range per axis) and an asymmetric function (f(x,y) != f(y,x))
        so a row/column transpose bug can't hide behind a value coincidence.

        D1/D2 above intentionally use a *sorted-set* comparison to route around the
        reshape code's own "Why???" comment (funs_evaluate.py) about why the XY/YX
        branch needs a reversed reshape - but with their symmetric domains and
        symmetric functions (x+y, x^2+y^2), a transpose bug leaves the sorted set of
        values unchanged, so they can't actually catch one. This test pins down the
        orientation directly: empirically (see scratchpad diagnostic), for 2 grid vars
        that are also the first 2 input vars, `results[response][j][i] == f(x_lin[i],
        y_lin[j])` - i.e. rows index the *second* grid var, columns the *first*.
        """
        x_rng = np.linspace(0, 1, 6)
        y_rng = np.linspace(0, 2, 6)
        xx, yy = np.meshgrid(x_rng, y_rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": 2 * x_train + 5 * y_train}),
            run_dir / "train_processed.txt",
        )

        N = 5
        results = evaluate_sumo_on_grid(
            run_dir,
            train_file,
            grid_vars=["x", "y"],
            input_vars=["x", "y"],
            response_var="z",
            NSAMPLESPERVAR=N,
        )

        x_lin = np.linspace(x_train.min(), x_train.max(), N)
        y_lin = np.linspace(y_train.min(), y_train.max(), N)
        expected = np.array([[2 * x + 5 * y for x in x_lin] for y in y_lin])
        np.testing.assert_allclose(results["z"], expected, atol=0.1)


@pytest.mark.analytical
class TestCrossValidation:
    """Category E: Cross-Validation."""

    def test_crossvalidation_rmse_near_zero_for_linear(self, run_dir):
        """E1: A well-sampled linear function should cross-validate with RMSE ~ 0."""
        x_train = np.linspace(0, 1, 30)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": 2 * x_train + 1}),
            run_dir / "train_processed.txt",
        )

        metrics = evaluate_sumo_crossvalidation(
            run_dir, train_file, ["x"], "y", N_CROSS_VALIDATION=5
        )
        assert "y" in metrics
        assert isinstance(metrics["y"], dict)
        rmse = float(metrics["y"]["root_mean_squared"])
        assert rmse < 0.1

    def test_crossvalidation_rmse_improves_with_more_data(self, run_dir):
        """E2: CV RMSE for f(x) = sin(x) should generally decrease as training size grows."""
        rmses = {}
        for n in (20, 50, 100):
            sub_dir = run_dir / f"n_{n}"
            sub_dir.mkdir()
            x_train = np.linspace(0, 2 * np.pi, n)
            train_file = _write_processed(
                pd.DataFrame({"x": x_train, "y": np.sin(x_train)}),
                sub_dir / "train_processed.txt",
            )
            metrics = evaluate_sumo_crossvalidation(
                sub_dir, train_file, ["x"], "y", N_CROSS_VALIDATION=5
            )
            rmses[n] = float(metrics["y"]["root_mean_squared"])

        assert rmses[100] < rmses[20]

    def test_manual_crossvalidation_prediction_interval_coverage(self, run_dir):
        """E3: ~95% of held-out points should fall within y_hat +/- 1.96*std_hat.

        Uses the small-N regime typical of this test suite, so the observed coverage
        won't hit exactly 95% - a generous lower bound avoids flakiness while still
        being a meaningful check (not trivially satisfied).
        """
        x_train = np.linspace(0, 1, 40)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": x_train**2}),
            run_dir / "train_processed.txt",
        )

        cv = evaluate_sumo_manual_crossvalidation(
            run_dir, train_file, ["x"], "y", N_CROSS_VALIDATION=5
        )
        obs = np.array(cv["y"])
        pred = np.array(cv["y_hat"])
        std = np.array(cv["y_std_hat"])

        within_interval = np.abs(obs - pred) <= 1.96 * std + 1e-9
        coverage = within_interval.mean()
        assert coverage >= 0.7

    def test_manual_vs_builtin_crossvalidation_agree(self, run_dir):
        """E4: manual (per-fold) and built-in Dakota CV RMSE should roughly agree."""
        x_train = np.linspace(0, 1, 30)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": 2 * x_train + 1}),
            run_dir / "train_processed.txt",
        )

        builtin_metrics = evaluate_sumo_crossvalidation(
            run_dir, train_file, ["x"], "y", N_CROSS_VALIDATION=5
        )
        builtin_rmse = float(builtin_metrics["y"]["root_mean_squared"])

        manual_dir = run_dir / "manual"
        manual_dir.mkdir()
        manual_cv = evaluate_sumo_manual_crossvalidation(
            manual_dir, train_file, ["x"], "y", N_CROSS_VALIDATION=5
        )
        manual_rmse = _rmse(manual_cv["y"], manual_cv["y_hat"])

        # both should be small in absolute terms for this easy, well-sampled function
        assert builtin_rmse < 0.1
        assert manual_rmse < 0.1


@pytest.mark.analytical
class TestUqPropagation:
    """Category F: UQ Propagation via the real `/manual_uq_propagation_with_uncertainty`
    pathway (`_manual_uq_propagate`, see module docstring for why not `propagate_uq`)."""

    def test_linear_normal_uq_propagation_matches_analytical(self, run_dir):
        """F1: f(x) = 2x+1, x ~ N(0,1) => y ~ N(1, 4): output mean ~ 1, std ~ 2."""
        x_train = np.linspace(-4, 4, 40)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": 2 * x_train + 1}),
            run_dir / "train_processed.txt",
        )

        samples = _manual_uq_propagate(
            run_dir,
            train_file,
            ["x"],
            "y",
            distributions={"x": {"distribution": "normal", "mean": 0.0, "std": 1.0}},
            num_samples=500,
        )
        assert np.isclose(samples.mean(), 1.0, atol=0.3)
        assert np.isclose(samples.std(), 2.0, atol=0.3)

    def test_linear_uniform_uq_propagation_matches_analytical(self, run_dir):
        """F2: f(x) = 2x+1, x ~ U(-3,3) => y ~ U(-5,7): mean ~ 1, std ~ sqrt(12).

        Previously excluded (uniform inputs were assumed unsupported) - closed now that
        the suite targets the real pathway, which does support uniform distributions.
        """
        x_train = np.linspace(-4, 4, 40)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": 2 * x_train + 1}),
            run_dir / "train_processed.txt",
        )

        samples = _manual_uq_propagate(
            run_dir,
            train_file,
            ["x"],
            "y",
            distributions={"x": {"distribution": "uniform", "min": -3.0, "max": 3.0}},
            num_samples=500,
        )
        assert np.isclose(samples.mean(), 1.0, atol=0.3)
        assert np.isclose(samples.std(), np.sqrt(12), atol=0.4)

    def test_quadratic_uq_propagation_matches_chi_square(self, run_dir):
        """F3: f(x) = x^2, x ~ N(0,1) => y ~ chi-square(1): mean ~ 1, var ~ 2."""
        x_train = np.linspace(-4, 4, 60)
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": x_train**2}),
            run_dir / "train_processed.txt",
        )

        samples = _manual_uq_propagate(
            run_dir,
            train_file,
            ["x"],
            "y",
            distributions={"x": {"distribution": "normal", "mean": 0.0, "std": 1.0}},
            num_samples=3000,
        )
        assert np.isclose(samples.mean(), 1.0, atol=0.25)
        assert np.isclose(samples.var(), 2.0, atol=0.6)

    def test_multi_input_uq_propagation_matches_analytical(self, run_dir):
        """F5: f(x,y) = x+y, x,y ~ N(0,1) iid => y ~ N(0, sqrt(2))."""
        rng = np.linspace(-4, 4, 15)
        xx, yy = np.meshgrid(rng, rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": x_train + y_train}),
            run_dir / "train_processed.txt",
        )

        samples = _manual_uq_propagate(
            run_dir,
            train_file,
            ["x", "y"],
            "z",
            distributions={
                "x": {"distribution": "normal", "mean": 0.0, "std": 1.0},
                "y": {"distribution": "normal", "mean": 0.0, "std": 1.0},
            },
            num_samples=800,
        )
        assert np.isclose(samples.mean(), 0.0, atol=0.2)
        assert np.isclose(samples.std(), np.sqrt(2), atol=0.3)

    def test_undersampled_oscillation_uncertainty_is_underestimated(self, run_dir):
        """F6: documents a real, verified miscalibration rather than assuming the
        intuitive-but-false opposite.

        Two earlier versions of this test assumed "sparser training -> larger propagated
        uncertainty" and both failed empirically:
        - with a *linear* f(x): a GP interpolates a line almost exactly from a handful of
          bracketing points, so sparse vs. dense training barely changed std_hat at all
          (propagated std ~1.967 in both cases).
        - with f(x) = sin(x) and N=5 training points: propagated std was *lower* for the
          sparse case (~0.52) than the dense one (~0.65), the opposite of the assumption.

        A direct diagnostic (bypassing the propagation pipeline) explains why: with only
        N=5 points over [-4, 4], surfpack's Kriging fit is badly wrong (e.g. at x=3.5,
        predicts 0.030 vs. true sin(3.5)=-0.351, error 0.38 - about 38% of the function's
        amplitude) yet reports almost no predictive uncertainty there (std_hat=0.0039).
        With N=60 points the fit is exact and std_hat is ~0. So undersampling here shows
        up as a *biased, damped* prediction with a deceptively tight (not wide) confidence
        band - not as legitimately larger error bars. That is a real product-relevant
        limitation of the underlying surrogate's uncertainty estimate for non-monotonic
        responses, not a test artifact, so this test asserts *that* behavior directly:
        predictive std stays small even while prediction error is large.
        """
        x_train = np.linspace(
            -4, 4, 5
        )  # deliberately too sparse to resolve one sine period
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": np.sin(x_train)}),
            run_dir / "train_processed.txt",
        )
        x_eval = np.linspace(-3.5, 3.5, 9)
        eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "eval.txt")

        result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
        pred, std = np.array(result["y_hat"]), np.array(result["y_std_hat"])
        expected = np.sin(x_eval)

        assert (
            _rmse(expected, pred) > 0.2
        )  # the fit is genuinely bad under this undersampling
        assert (
            std.max() < 0.01
        )  # yet the surrogate reports almost no uncertainty about it

    def test_mixed_distributions_uq_propagation_matches_analytical(self, run_dir):
        """F7: f(x,y) = x+y, x ~ N(0,1), y ~ U(-1,1), independent =>
        Var(z) = Var(x) + Var(y) = 1 + 1/3 = 4/3, mean ~ 0, std ~ sqrt(4/3).

        Previously excluded (mixed-distribution inputs were assumed unsupported) - closed
        now that the suite targets the real pathway: `create_manual_uq_samples` samples
        each input variable's distribution independently, so per-variable distribution
        types were never actually coupled the way the old exclusion rationale implied.
        """
        x_rng = np.linspace(-4, 4, 15)
        y_rng = np.linspace(-2, 2, 15)
        xx, yy = np.meshgrid(x_rng, y_rng)
        x_train, y_train = xx.ravel(), yy.ravel()
        train_file = _write_processed(
            pd.DataFrame({"x": x_train, "y": y_train, "z": x_train + y_train}),
            run_dir / "train_processed.txt",
        )

        samples = _manual_uq_propagate(
            run_dir,
            train_file,
            ["x", "y"],
            "z",
            distributions={
                "x": {"distribution": "normal", "mean": 0.0, "std": 1.0},
                "y": {"distribution": "uniform", "min": -1.0, "max": 1.0},
            },
            num_samples=800,
        )
        assert np.isclose(samples.mean(), 0.0, atol=0.25)
        assert np.isclose(samples.std(), np.sqrt(4 / 3), atol=0.3)
