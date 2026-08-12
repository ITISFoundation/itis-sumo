# Verification & Validation Report

## Purpose

Demonstrates the correctness of the itis-sumo computational pipeline
(sampling → surrogate building → prediction → UQ propagation → MOGA
optimization) against known analytical solutions. This is not test
coverage in the usual sense — every result below has a closed-form
reference value it's checked against, not just "didn't crash."

Full test plan and category design:
[`VERIFICATION_VALIDATION_PLAN.md`](https://github.com/ITISFoundation/itis-sumo/blob/main/docs/VERIFICATION_VALIDATION_PLAN.md).

## Current status

Live results as of the last run of the ported V&V suite:

| Suite | Tests | Result |
|---|---|---|
| Full standalone suite (`uv run pytest`) | 245 | **all passing** |
| Analytical/integration tier (`-m analytical`, real Dakota subprocess, no mocking) | 26 | **all passing** |
| Sobol' / Ishigami acceptance gate (`test_sobol_indices.py`) | 4 | **all passing** |

The analytical tier spawns a real `itis-dakota` (1.5.9 / Dakota 6.20)
process per test — nothing here is mocked. Re-run locally with:

```sh
uv run pytest -m analytical    # Dakota-backed integration tests
uv run pytest                  # full suite, standalone
```

## Test functions

Escalating complexity, from trivial to genuinely hard for a GP surrogate:

| ID | Function | Domain | Why |
|----|----------|--------|-----|
| F1 | Constant `f(x) = c` | x ∈ [0, 1] | Trivial baseline — surrogate must predict flat |
| F2 | Linear `f(x) = ax + b` | x ∈ [0, 1] | Exact interpolation; analytically tractable UQ |
| F3 | Quadratic `f(x) = ax² + bx + c` | x ∈ [0, 1] | Curvature capture; error grows near boundaries |
| F4 | Sinusoidal `f(x) = A·sin(ωx + φ)` | x ∈ [0, 2π] | Oscillatory — struggles without dense sampling |
| F5 | Logarithmic `f(x) = a·ln(x) + b` | x ∈ [0.01, 2] | Monotonic; unbounded derivative near x=0 |
| F6 | Rosenbrock (2D) | x, y ∈ [-2, 2] | Non-convex, narrow curved valley |
| F7 | Branin (2D) | x ∈ [-5, 10], y ∈ [0, 15] | Multiple global minima — MOGA test |
| Ishigami (3D) | `sin(x1) + 7sin²(x2) + 0.1x3⁴sin(x1)` | xᵢ ∈ [-π, π] | Standard Sobol'-index benchmark; nonlinear + non-monotonic, has a known-zero main effect (x3) and known second-order interaction (x1×x3) |

## Category A: Sampling Quality (LHS)

Pure-Python, no Dakota involved — verifies the LHS implementation itself
(`itis_sumo.sampling.lhs`), covered in `tests/test_unit_solver.py` /
`tests/test_property_invariants.py`.

| Test | Pass criterion | Status |
|---|---|---|
| Stratification | every 1D projection has exactly one sample per `[i/k, (i+1)/k)` | ✅ |
| Value range | all values in `[0, 1]` | ✅ |
| Reproducibility | same seed ⟹ identical output | ✅ |
| Maximin | `_lhsmaximin` min pairwise distance ≥ `_lhsclassic` | ✅ |
| Correlation | `_lhscorrelate` max off-diagonal correlation ≤ `_lhsclassic` | ✅ |
| Marginal uniformity | empirical CDF ≈ uniform (KS test) for large k | ✅ |

## Category B: Surrogate Model Accuracy

GP surrogates built via `evaluate_sumo`, evaluated against held-out
analytical values (`tests/test_metamodeling_analytical.py`, real Dakota).

| Test | Function | Pass criterion | Status |
|---|---|---|---|
| B1 | Exact interpolation at training points | `y_hat ≈ y_train` to ~1e-3 | ✅ |
| B2 | Linear `2x + 1`, N=20 | RMSE < 1e-6, R² ≈ 1.0 | ✅ |
| B3 | Quadratic `x²`, N=30 | RMSE < 1e-4, R² > 0.99 | ✅ |
| B4 | Sinusoidal `sin(x)`, N=50 | RMSE < 0.05, R² > 0.95 | ✅ |
| B5 | Logarithmic `ln(x)`, N=20 | RMSE < 0.01 | ✅ |
| B6 | Rosenbrock (2D), N=100 | error larger than F1-F5; uncertainty visible in valley | ✅ |
| B7 | Convergence: `sin(x)`, N = 10…200 | RMSE monotonically decreasing | ✅ |
| B8 | Variance behavior | `std_hat[train] ≈ 0`; grows away from training data | ✅ |

## Evaluation pathways (Categories C–E)

Axis sweep, grid evaluation, and cross-validation — see
[Evaluate: core evaluation & cross-validation](reference/evaluate.md).

| Category | What | Status |
|---|---|---|
| C — Axis sweep | Slope/interpolation accuracy along 1D sweeps | ✅ |
| D — Grid evaluation | 2D grid predictions, dimension-consistency, fixed non-grid variables | ✅ |
| E — Cross-validation | Manual K-fold CV metrics, convergence with more data, ~95% prediction-interval coverage | ✅ |

**E4** (manual vs. built-in Dakota CV) documents a known discrepancy: the
two pathways agree only to within ~20%, not exactly — this is the observed
signature of the built-in-CV parsing gap in [known limitations](#known-limitations),
not a bug in the manual path (which is the one whose absolute accuracy is
independently verified by B1-B8-style checks).

## Category F: UQ Propagation

Tests the pathway actually reachable from the web UI — manual per-variable
sampling (`create_manual_uq_samples`) + surrogate evaluation + erfinv-based
predictive-uncertainty injection (see
[Sensitivity & UQ propagation](reference/sensitivity-uq.md)) —
against closed-form output distributions.

| Test | Function | Input | Analytical output | Status |
|---|---|---|---|---|
| F1 | `2x + 1` | `x ~ N(0,1)` | `y ~ N(1, 4)` | ✅ |
| F2 | `3x` | `x ~ U(0,1)` | `y ~ U(0,3)` | ✅ |
| F3 | `x²` | `x ~ N(0,1)` | `y ~ χ²(1)` | ✅ |
| F4 | Convergence, `2x+1` | `x ~ N(0,1)` | error ↓ with n_samples | ✅ |
| F5 | Multi-input `x+y` | `x,y ~ N(0,1)` | `y ~ N(0, 2)` | ✅ |
| F6 | Surrogate-uncertainty effect | — | propagated std > analytical std when surrogate uncertain | ✅ |
| F7 | Mixed dists `x+y` | `x~U(0,1)`, `y~N(1,0.5)` | mean≈1.5, std≈0.6–0.7 | ✅ |

## Category G: MOGA Optimization

See [MOGA optimization](reference/moga.md) for the known
`max_function_evaluations` limitation.

| Test | What | Status |
|---|---|---|
| G1 | Single-objective `(x-0.5)²` finds x ≈ 0.5 | ✅ |
| G2 | Bi-objective front spans (0,1)→(1,0), all points non-dominated | ✅ |
| G3 | Front improves (dominates) with more iterations | ✅ |
| G4 | All Pareto points respect variable bounds | ✅ |

## Category H: Data Preprocessor

See [Data preprocessing](reference/preprocess.md).

| Test | What | Status |
|---|---|---|
| H1-H3 | Z-score / min-max / sign-switch round-trip to 1e-10 | ✅ |
| H4 | Round-trip on 1000×20 dataset | ✅ |
| H5 | Normalization improves accuracy on badly-scaled `f(x) = 1000x+1` | ✅ |

## Ishigami analytical acceptance gate

`SPEC.md` §R1 — the acceptance test for the entire Sobol' sensitivity
pipeline (`evaluate_sobol_indices`). Bypasses the GP surrogate entirely and
evaluates the Ishigami function analytically on Saltelli/QMC samples
(n=2¹⁴), so it isolates "is the sampling → scipy → closed-form second-order
math correct" from surrogate accuracy.

| Index | Reference value | Tolerance | Status |
|---|---|---|---|
| S1 (first-order, x1) | 0.314 | ±0.05 | ✅ |
| S2 (first-order, x2) | 0.442 | ±0.05 | ✅ |
| S3 (first-order, x3) | 0.0 | ±0.05 | ✅ |
| S_T1 (total-order, x1) | 0.558 | ±0.05 | ✅ |
| S_T2 (total-order, x2) | 0.442 | ±0.05 | ✅ |
| S_T3 (total-order, x3) | 0.244 | ±0.05 | ✅ |
| S_12 (second-order) | 0.0 | ±0.05 | ✅ |
| S_13 (second-order) | 0.244 | ±0.05 | ✅ |
| S_23 (second-order) | 0.0 | ±0.05 | ✅ |

x3's zero first-order-but-nonzero total-order index is the point of the
benchmark: it has no *main* effect on its own, but a real interaction
effect through the `0.1·x3⁴·sin(x1)` term — a surrogate/sensitivity
pipeline that got the interaction term wrong would still pass a
first-order-only check and fail this one.

## Known limitations

Carried over from the original test plan, still true of the pinned engine
(`itis-dakota==1.5.9`, Dakota 6.20):

- **Built-in Dakota CV parsing is unreliable** — `log_output` comes back
  hardcoded empty on some study configurations, which is why
  `evaluate_sumo_manual_crossvalidation` (Python-side K-fold) is the
  correctness-verified cross-validation pathway, not
  `evaluate_sumo_crossvalidation`.
- **MOGA `max_function_evaluations` is not enforced** — deprecated
  parameter on the Dakota side; use iteration/generation controls instead.
- **GP interpolation vs. approximation tradeoffs** — surrogate accuracy
  degrades predictably on non-smooth functions (Rosenbrock, B6) and
  under-sampled oscillatory functions (sinusoidal, B4); this is expected
  GP behavior, not a bug, but worth remembering when interpreting
  predictions on functions unlike the ones tested here.
- **Grid reshaping complexity for >2 dimensions** — `evaluate_sumo_on_grid`
  is verified for 2D grids; higher-dimensional grid reshaping is less
  exercised.
- **`propagate_uq` (Dakota-native UQ)** is implemented and tested but not
  the pathway actually wired to the web UI — see
  [Evaluate § Uncertainty propagation](reference/evaluate.md#uncertainty-propagation).
  Don't assume it's the one production traffic exercises.

## Summary

Every category in the original V&V plan (A through H, plus the Ishigami
acceptance gate) currently passes against its analytical reference, with
real (unmocked) Dakota execution for every category that requires the
engine. The known limitations above are pre-existing engine/pipeline
characteristics documented so consumers of this package don't rediscover
them the hard way — not test failures.
