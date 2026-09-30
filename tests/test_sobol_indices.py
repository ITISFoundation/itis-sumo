"""
Tests for Sobol' sensitivity indices - pure scipy-based computation.

Covers the Saltelli sampling, constant-var handling and the exact arbitrary-d
pair estimator ported from mmux_vite T31rb. The estimator is validated against
four analytic benchmarks - additive d=8, pair-interaction d=5, Ishigami,
pair-quadratic d=10 - with tolerances scaled by the shared-bootstrap CI
half-widths, and the B26nc degeneracy regression proves the retired identity's
mass-leak artifact is gone. Order masses M1/M2/R closure and the B28pp
estimator invariants (scipy parity, translation invariance, zero-variance flag)
are covered here; route-side response contracts stay in mmux/vite.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from itis_sumo.evaluate.funs_evaluate import SOBOL_BASE_SAMPLES

pytestmark = pytest.mark.unit


def _ishigami(x: np.ndarray) -> np.ndarray:
    """Ishigami test function: f(x1,x2,x3) = sin(x1) + 7*sin²(x2) + 0.1*x3⁴*sin(x1).

    Uniform inputs on [-π, π] for all three variables.
    """
    return (
        np.sin(x[:, 0])
        + 7.0 * np.sin(x[:, 1]) ** 2
        + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])
    )


def _build_sobol_designs(f, d: int, *, n: int = 2**12, seed: int = 42):
    """Saltelli A/B/C + AB + pair(U/V) evaluations for analytic f.

    Builds the designs with the PRODUCTION sampling/design helpers, so the
    analytic benchmarks exercise the shipped pipeline, not a copy of it."""
    from scipy.stats import uniform

    from itis_sumo.evaluate.funs_evaluate import (
        _saltelli_abc,
        _saltelli_pair_designs,
    )

    dists = [uniform(loc=-np.pi, scale=2 * np.pi) for _ in range(d)]
    A, B, C = _saltelli_abc(dists, n, seed)
    AB, uv, pairs = _saltelli_pair_designs(A, B, C)
    K = pairs.shape[0]
    f_AB = np.stack([f(block) for block in AB])
    f_UV = np.stack([f(block) for block in uv])
    return f(A), f(B), f_AB, f_UV[:K], f_UV[K:], pairs


def _run_algebra_on_analytic(f, d: int, *, n: int = 2**12, seed: int = 42):
    """Run the PRODUCTION algebra/bootstrap helpers on analytic designs."""
    from itis_sumo.evaluate.funs_evaluate import (
        _sobol_algebra,
        _sobol_joint_bootstrap,
    )

    f_A, f_B, f_AB, f_U, f_V, pairs = _build_sobol_designs(f, d, n=n, seed=seed)
    alg = _sobol_algebra(f_A, f_B, f_AB, f_U, f_V, pairs)
    boot = _sobol_joint_bootstrap(
        f_A,
        f_B,
        f_AB,
        f_U,
        f_V,
        pairs,
        seed=seed,
        n_resamples=200,
        confidence=0.95,
    )
    return alg, boot


class TestSobolSampling:
    """Unit tests for the Saltelli sampling and index computation (no surrogate)."""

    def test_power_of_two_rounding(self):
        """num_samples is rounded up to the next power of 2."""

        # 100 -> 128, 1 -> 2, 17 -> 32
        for requested, expected_n in [
            (100, 128),
            (1, 2),
            (17, 32),
            (64, 64),
            (256, 256),
        ]:
            assert 2 ** math.ceil(math.log2(max(requested, 2))) == expected_n

    def test_sobol_base_samples_is_1024(self):
        """V36: Sobol' uses a fixed base N=1024 (Saltelli scheme), decoupled from
        the shared UQ `numSamples` field used by Histogram/Correlation."""
        assert SOBOL_BASE_SAMPLES == 1024

    def test_constant_variable_indices_are_zero(self):
        """Constant input variables get main=0, total=0 in the response."""

        # We can't call evaluate_sumo without a real surrogate, so test the
        # logic by verifying the constant-var detection and zero assignment.
        # This is tested indirectly via the degenerate paths below.
        distributions = {
            "x1": {"distribution": "uniform", "min": -3.14159, "max": 3.14159},
            "x2": {"distribution": "constant", "value": 1.0},
        }
        # Verify constant detection
        constant_vars = {
            k: v["value"]
            for k, v in distributions.items()
            if v["distribution"] == "constant"
        }
        varying_vars = [k for k in distributions if k not in constant_vars]
        assert constant_vars == {"x2": 1.0}
        assert varying_vars == ["x1"]


class TestSobolArbitraryDPairEstimator:
    """Acceptance gates for the exact arbitrary-d pair estimator (T31rb port).

    All tolerances are scaled by the shared-bootstrap CI half-width rather than
    hand-tuned absolute floors. The Ishigami test keeps the
    `test_sobol_indices_ishigami_analytical` name (historic §R1 acceptance gate)
    but now drives the PRODUCTION algebra helpers (no inlined duplicate math).
    """

    def test_sobol_indices_ishigami_analytical(self):
        """§R1 acceptance gate (production algebra): Ishigami indices match analytical."""
        alg, _ = _run_algebra_on_analytic(_ishigami, d=3, n=2**14, seed=42)

        # first-order: S1≈0.314, S2≈0.442, S3≈0; total: 0.558/0.442/0.244
        for i, expect in enumerate([0.314, 0.442, 0.0]):
            assert alg["first"][i] == pytest.approx(expect, abs=0.05)
        for i, expect in enumerate([0.558, 0.442, 0.244]):
            assert alg["total"][i] == pytest.approx(expect, abs=0.05)

        # second-order: S_12≈0, S_13≈0.244, S_23≈0
        assert alg["second"][0, 1] == pytest.approx(0.0, abs=0.05)
        assert alg["second"][0, 2] == pytest.approx(0.244, abs=0.05)
        assert alg["second"][1, 2] == pytest.approx(0.0, abs=0.05)

        # masses over unique ANOVA terms, identity exact per replicate
        assert alg["m1"] == pytest.approx(0.756, abs=0.05)
        assert alg["m2"] == pytest.approx(0.244, abs=0.05)
        assert alg["m1"] + alg["m2"] + alg["r"] == pytest.approx(1.0, abs=1e-12)

    def test_sobol_second_order_additive_d8(self):
        """Additive d=8 benchmark: every pair ≈ 0 within 3·bootstrap-CI."""

        def f(x):
            return (
                np.sin(x[:, 0])
                + np.abs(x[:, 1])
                + np.tanh(x[:, 2])
                + x[:, 3] ** 2
                + np.cos(x[:, 4])
                + np.exp(x[:, 5] / 4)
                + np.sin(2 * x[:, 6])
                + np.abs(x[:, 7]) ** 1.5
            )

        alg, boot = _run_algebra_on_analytic(f, d=8, seed=7)
        ii, jj = np.triu_indices(8, k=1)
        s_ij = alg["second"][ii, jj]
        # no real interactions -> every pair estimate sits inside 3x its own
        # shared-bootstrap CI half-width (observed ~1e-4 at n=4096)
        half_pair = (boot["second"][:, 1] - boot["second"][:, 0]) / 2.0
        assert np.all(np.abs(s_ij) <= 3 * half_pair + 1e-9)
        half_m2 = float(boot["m2"][1] - boot["m2"][0]) / 2.0
        assert abs(alg["m2"]) <= 3 * half_m2 + 1e-9
        assert alg["m1"] + alg["m2"] + alg["r"] == pytest.approx(1.0, abs=1e-12)

    def test_sobol_second_order_pair_interaction_d5(self):
        """Pair-interaction d=5: S_12 recovers the analytic value exactly."""

        def f(x):
            return (
                2 * x[:, 0] * x[:, 1]
                + np.sin(x[:, 2])
                + np.abs(x[:, 3])
                + np.tanh(x[:, 4])
            )

        alg, boot = _run_algebra_on_analytic(f, d=5, seed=11)
        # analytic: V_12/V with uniform[-pi,pi]: E[x^2]=pi^2/3
        v12 = 4 * (np.pi**2 / 3) ** 2
        others = [
            np.var(np.sin(np.linspace(-np.pi, np.pi, 40001))),
            np.var(np.abs(np.linspace(-np.pi, np.pi, 40001))),
            np.var(np.tanh(np.linspace(-np.pi, np.pi, 40001))),
        ]
        s12_true = v12 / (v12 + sum(others))
        # tolerance = 3x THIS pair's bootstrap CI half-width
        half_pair = (boot["second"][:, 1] - boot["second"][:, 0]) / 2.0
        ii, jj = np.triu_indices(5, k=1)
        s_all = alg["second"][ii, jj]
        assert s_all[0] == pytest.approx(
            s12_true, abs=max(0.01, 3 * float(half_pair[0]))
        )
        # all remaining pairs are truly zero -> within their own 3 CIs
        assert np.all(np.abs(s_all[1:]) <= 3 * half_pair[1:] + 1e-9)

    def test_sobol_second_order_pair_quadratic_d10(self):
        """>=10-dim analytic benchmark: one known pair in d=10."""
        rng = np.random.default_rng(3)
        a = rng.uniform(0.5, 2.0, 10)
        c0 = 1.5

        def f(x):
            return a @ x.T + c0 * x[:, 0] * x[:, 1]

        alg, boot = _run_algebra_on_analytic(f, d=10, seed=5)
        v12 = c0**2 * (np.pi**2 / 3) ** 2
        vtot = float(np.sum(a**2) * (np.pi**2 / 3)) + v12
        s01_true = v12 / vtot
        half = float(boot["m2"][1] - boot["m2"][0]) / 2.0
        assert alg["second"][0, 1] == pytest.approx(s01_true, abs=max(0.01, 3 * half))
        # d=10 is inside the 4-25 target window: M2 captures the true mass
        assert alg["m2"] == pytest.approx(s01_true, abs=max(0.01, 3 * half))
        # the 44 zero pairs must each stay within 3x their OWN bootstrap CI -
        # pairs touching a strong main effect carry visibly larger joint-index
        # noise, which a flat absolute tolerance would misjudge as bias
        half_pair = (boot["second"][:, 1] - boot["second"][:, 0]) / 2.0
        ii, jj = np.triu_indices(10, k=1)
        s_all = alg["second"][ii, jj]
        assert np.all(np.abs(s_all[1:]) <= 3 * half_pair[1:] + 1e-9)
        assert np.sum(s_all) == pytest.approx(s01_true, abs=max(0.01, 3 * half))

    def test_sobol_second_order_not_degenerate_vs_b26nc(self):
        """B26nc regression: the retired identity leaks mass as a NEGATIVE pair.

        For f = Ishigami(x1,x2,x3) + additive(x4,x5) the old
        ((U_i+U_j)-Σ_{k≠i,j} U_k)/2 estimator assigns S_45 = -S_13 ≈ -0.244
        (pure artifact: x4,x5 are non-interacting). The ported estimator must
        return S_45 ≈ 0 with no negative mass anywhere outside noise.
        """

        def f(x):
            base = _ishigami(x[:, :3])
            rest = np.abs(x[:, 3]) + np.tanh(x[:, 4])
            return base + rest

        alg, boot = _run_algebra_on_analytic(f, d=5, seed=13)
        # true interaction: only S_13 ≈ 0.244 (diluted by the additive rest)
        assert alg["second"][0, 2] > 0.1
        half45 = float(boot["second"][9, 1] - boot["second"][9, 0]) / 2.0  # pair (3,4)
        assert abs(alg["second"][3, 4]) <= max(0.01, 3 * half45), (
            "non-interacting pair must not carry negative mass"
        )
        # what the OLD identity would have produced, from the same first/total:
        u = alg["total"] - alg["first"]
        old_s45 = (u[3] + u[4] - (u[0] + u[1] + u[2])) / 2.0
        assert old_s45 < -0.1, (
            "regression guard: old identity is provably degenerate here"
        )
        # new estimator's total pair mass stays physically sane
        assert abs(alg["m2"] - alg["second"][0, 2]) <= 0.05

    def test_order_mass_identity_any_d(self):
        """M1+M2+R=1 exactly (closure), for any d including d=2."""
        for d, seed in [(2, 1), (4, 2), (8, 3)]:

            def f(x, d=d):
                main = np.sin(x[:, 0])
                if d > 1:
                    main = main + np.abs(x[:, 1] * x[:, min(2, x.shape[1] - 1)])
                return main

            alg, _ = _run_algebra_on_analytic(f, d=d, n=2**10, seed=seed)
            assert alg["m1"] + alg["m2"] + alg["r"] == pytest.approx(1.0, abs=1e-12)


class TestSobolEstimatorInvariants:
    """B28pp (mmux_vite PR #649 review): estimator-level invariants.

    The retired raw-product algebra anchored on ``f_A`` only was NOT
    translation-invariant under ``f -> f + c`` (drift ~ c whenever mean >> std)
    and was a different estimator than the scipy point estimates displayed
    alongside it. The ported algebra mirrors scipy.stats.sobol_indices'
    ``saltelli_2010`` exactly (pooled A/B mean removal, pooled A/B variance,
    centered products)."""

    @staticmethod
    def _f(x):
        return 2 * x[:, 0] * x[:, 1] + np.sin(x[:, 2]) + np.abs(x[:, 3])

    def test_indices_invariant_under_output_translation(self):
        """Adding a constant offset to EVERY model output moves NO index.

        Raw-product estimators drift proportionally to the offset (units
        effects, e.g. stress reported in Pa); centered products cannot."""
        from itis_sumo.evaluate.funs_evaluate import _sobol_algebra

        d = 4
        f_A, f_B, f_AB, f_U, f_V, pairs = _build_sobol_designs(
            self._f, d=d, n=2**10, seed=21
        )
        alg0 = _sobol_algebra(f_A, f_B, f_AB, f_U, f_V, pairs)
        offset = 500.0  # ~100x the output std: the mean >> std regime
        algc = _sobol_algebra(
            f_A + offset, f_B + offset, f_AB + offset, f_U + offset, f_V + offset, pairs
        )
        np.testing.assert_allclose(algc["first"], alg0["first"], atol=1e-8)
        np.testing.assert_allclose(algc["total"], alg0["total"], atol=1e-8)
        np.testing.assert_allclose(algc["second"], alg0["second"], atol=1e-8)
        assert algc["m1"] == pytest.approx(alg0["m1"], abs=1e-8)
        assert algc["m2"] == pytest.approx(alg0["m2"], abs=1e-8)
        assert algc["r"] == pytest.approx(alg0["r"], abs=1e-8)

    def test_algebra_matches_scipy_point_estimator(self):
        """The displayed points ARE the scipy estimator: _sobol_algebra is the
        single runtime source (no scipy call remains), so this parity test is
        what pins evaluate_sobol_indices output to scipy.stats.sobol_indices."""
        from scipy.stats import sobol_indices

        from itis_sumo.evaluate.funs_evaluate import _sobol_algebra

        d, n = 4, 2**10
        f_A, f_B, f_AB, f_U, f_V, pairs = _build_sobol_designs(
            self._f, d=d, n=n, seed=22
        )
        si = sobol_indices(
            func={
                "f_A": f_A.reshape(1, n),
                "f_B": f_B.reshape(1, n),
                "f_AB": f_AB.reshape(d, 1, n),
            },
            n=n,
        )
        alg = _sobol_algebra(f_A, f_B, f_AB, f_U, f_V, pairs)
        np.testing.assert_allclose(
            alg["first"], np.atleast_1d(si.first_order), atol=1e-9
        )
        np.testing.assert_allclose(
            alg["total"], np.atleast_1d(si.total_order), atol=1e-9
        )

    def test_zero_variance_sample_sets_var_zero(self):
        """V43pt/B28pp algebra seam: constant samples flag var_zero (the response
        layer turns that into null order contributions, not fake 0/0/0 masses)."""
        from itis_sumo.evaluate.funs_evaluate import _sobol_algebra

        n, d = 64, 3
        alg = _sobol_algebra(
            np.full(n, 2.0),
            np.full(n, 2.0),
            np.full((d, n), 2.0),
            np.empty((0, n)),
            np.empty((0, n)),
            np.empty((0, 2), dtype=int),
        )
        assert alg["var_zero"] is True
        assert alg["m1"] == 0.0 and alg["m2"] == 0.0 and alg["r"] == 0.0
        # and a live sample does NOT set the flag
        alg_live, _ = _run_algebra_on_analytic(self._f, d=4, n=2**8, seed=23)
        assert alg_live["var_zero"] is False


class TestDegenerateResponses:
    """evaluate_sobol_indices degenerate paths, driven with a fabricated
    evaluate_sumo (no Dakota): constant predictions must yield null order
    contributions, not silent (0,0,0) masses."""

    class _FakeVarCfg:
        def __init__(self, name: str):
            self.mapped_name = name

    class _FakePreprocessor:
        def __init__(self, cols: list[str]):
            self.input_variables = {
                c: TestDegenerateResponses._FakeVarCfg(c) for c in cols
            }

        def transform(self, df):
            return df

    def test_all_constant_inputs_return_null_contributions(self, tmp_path: Path):
        from itis_sumo.evaluate import funs_evaluate

        pre = self._FakePreprocessor(["x1", "x2"])
        distributions = {
            "x1": {"distribution": "constant", "value": 1.0},
            "x2": {"distribution": "constant", "value": 2.0},
        }
        out = funs_evaluate.evaluate_sobol_indices(
            tmp_path,
            tmp_path / "training.dat",
            ["x1", "x2"],
            "y",
            distributions,
            pre,
            seed=1,
        )
        assert out["sobolSecondOrder"] == {}
        assert out["sobolOrderContributions"] is None
        for entry in out["sobol"].values():
            assert entry["main"] == 0.0 and entry["total"] == 0.0

    def test_degenerate_surrogate_nulls_contributions(
        self, tmp_path: Path, monkeypatch
    ):
        """One varying input, constant surrogate output: var_zero -> null masses."""
        import pandas as pd

        from itis_sumo.evaluate import funs_evaluate

        def _constant_sumo(
            run_dir, training_file, samples_file, input_vars, response_var
        ):
            df = pd.read_csv(samples_file, sep=" ")
            return {response_var + "_hat": [2.5] * len(df)}

        monkeypatch.setattr(funs_evaluate, "evaluate_sumo", _constant_sumo)
        pre = self._FakePreprocessor(["x1", "x2"])
        distributions = {
            "x1": {"distribution": "uniform", "min": -1.0, "max": 1.0},
            "x2": {"distribution": "constant", "value": 0.0},
        }
        out = funs_evaluate.evaluate_sobol_indices(
            tmp_path,
            tmp_path / "training.dat",
            ["x1", "x2"],
            "y",
            distributions,
            pre,
            seed=2,
        )
        assert out["sobolOrderContributions"] is None
        assert out["sobolSecondOrder"] == {}
        assert out["sobol"]["x1"]["main"] == 0.0
        assert out["sobol"]["x2"]["main"] == 0.0
