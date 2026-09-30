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
| Full standalone suite (`uv run pytest`) | 387 | **all passing** |
| Analytical/integration tier (`-m analytical`, real Dakota subprocess, no mocking) | 26 | **all passing** |
| Sobol' / Ishigami acceptance gate (`test_sobol_indices.py`) | 14 | **all passing** |

The analytical tier spawns a real `itis-dakota` (6.24.7)
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
| H6 | `log_transform` round-trip (`np.log`/`np.exp`), non-positive values refused, delta-method std inverse `std_orig ≈ |y_hat|·std_log` | ✅ |

## Category I: Scale semantics (log)

`scale` is a first-class, un-forgettable axis (SPEC V44ls/V45ls): every public
value-producing entry point accepts it (defaulting to linear, so existing calls
are unchanged) and every value the api yields is computed under it. The one
`unit→value` map behind all of it is `scale_distribution` (`uniform` vs
`scipy.stats.loguniform`), and it requires the scale argument — unwired code
breaks with `TypeError`, it can never silently default.

| Test | What | Status |
|---|---|---|
| I1 | LHS + grid samplers: log domain ⇒ log-uniform/geometric fill; linear default bit-identical; non-positive log domain ⇒ `SumoInputError` | ✅ |
| I2 | Correlations: Pearson moves under log, Spearman provably unchanged (monotone-invariant), untouched columns bit-identical; correlator's scale args are required (`TypeError` tripwire) | ✅ |
| I3 | Surrogate / CV / along-axes / grid eval: log response exp-restored to original units; non-positive training outputs rejected pre-Dakota | ✅ |
| I4 | UQ propagation: log inputs drawn log-uniform (response skews low vs linear, directionally asserted); log⊗normal drawn lognormal — ln-space μ/σ, mean shifts high vs linear by Jensen, directionally asserted (V46rn); log+min≤0 / log+constant rejected; log response ⇒ multiplicative (not additive) spread | ✅ |
| I5 | CV accuracy metrics: inherit log through `cross_validate` (metrics differ from linear); reject non-positive log responses | ✅ |
| I6 | MOGA: log variable explored in ln-space (domain mapped, positivity-guarded); log objective exp-restored for **both** minimize and maximize (sign-after-log inverse order verified) | ✅ |
| I7 | Sobol: log input shifts the variance decomposition in the expected direction (log-uniform box draws compress toward the low end, so the variable explains less); a DECLARED narrow box concentrates sensitivity onto the other variable — the decomposition follows the sampled domain, not the training spread (V26dd) | ✅ |
| I8 | Flip matrix: all 11 public value-producing entry points' outputs move when a column turns log, ∀ 2 `distributions`-taking entry points also move when a NORMAL column turns log (V46rn; Sobol's box is domain-only after V26dd, so its flip rides on DOMAIN boxes and it has no NORMAL matrix row) — the V45ls machine guard against any silent scale-ignore, shipped or future | ✅ |
| I9 | MC-through-surrogate correlation (`evaluate_correlations`, #470 workflow): dominant variable recovered over the shared sample set, seed-reproducible, log-scale coefficients move (uniform ∧ normal, V46rn), log+constant / log+non-positive / non-covering distributions rejected, engine producer requires its scales (`TypeError` tripwire) | ✅ |
| I10 | Boundary input guards (V47st): `DomainSpec`/`DistributionSpec` reject inverted/degenerate bounds at construction; unused `preprocessing` overrides rejected by the table-mode + sampler entry points (not just the session path); log-uniform upper bound required ∧ ordered at the api boundary; a variable missing from the correlator's `input_scales` raises instead of defaulting to linear | ✅ |
| I11 | Domain⊥distribution split (V26dd): `evaluate_sobol` samples the exploration DOMAIN (explicit `DomainSpec` boxes echoed verbatim ∧ auto-inferred from observed bounds when omitted), ⊥ modeller distributions ∧ ⊥ a box re-derived from `mean±3σ`; a column constant in the samples is pinned (reports in `fixed`, indices zero, ⊥ a fabricated box); unknown domain names rejected; log-scale domain non-positive rejected at the boundary | ✅ |

Tests: `tests/test_api_workflows.py` (`TestLogScale*`, `TestScaleGapCoverage`,
`TestScaleAwareSamplers`, `TestScaleFlipMatrix`),
`tests/test_correlation_indices.py`, `tests/test_dakota_funs_data_processing.py`,
`tests/test_data_preprocessor.py`.

## Ishigami analytical acceptance gate

`SPEC.md` §R1 — the acceptance tests for the Sobol' sensitivity pipeline
(`evaluate_sobol_indices`). They bypass the GP surrogate entirely and evaluate
the Ishigami function analytically on designs built with the PRODUCTION
sampling, pair-design and algebra helpers (n=2¹⁴), so they isolate "is the
sampling → splitting → pair-design → algebra math correct" from surrogate
accuracy.

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
| M1 (order mass, ΣᵢSᵢ) | 0.756 | ±0.05 | ✅ |
| M2 (order mass, Σ_{i<j}S_ij) | 0.244 | ±0.05 | ✅ |

x3's zero first-order-but-nonzero total-order index is the point of the
benchmark: it has no *main* effect on its own, but a real interaction
effect through the `0.1·x3⁴·sin(x1)` term — a surrogate/sensitivity
pipeline that got the interaction term wrong would still pass a
first-order-only check and fail this one.

### Estimator tier (T31rb port, V48tr)

Same production helpers, more analytic benchmarks — all tolerances scaled by
the shared-bootstrap CI half-width, never hand-tuned floors:

- **Additive d=8** — every pair estimate inside 3× its own CI (no false
  interactions); **pair-interaction d=5** and **pair-quadratic d=10** — the
  one true pair recovered at its analytic value, the other pairs silent.
- **B26nc degeneracy regression** — the retired first/total-gap identity
  assigns a NON-interacting pair S_45 ≈ −0.244 (mass leak) on a fixture where
  the exact joint-pair estimator returns ≈0; the test pins both directions.
- **scipy parity** — `_sobol_algebra` first/total equal
  `scipy.stats.sobol_indices` (`saltelli_2010`) to 1e-9: the algebra is the
  single runtime source, the parity test is what pins it.
- **Translation invariance** — adding a constant offset (~100× the output
  std, the "stress in Pa" regime) moves no index, mass or CI bound.
- **Zero-variance flag** — constant samples set `var_zero` (the api turns
  that into null order contributions, never fake `(0, 0, 0)` masses).

## Known limitations

Carried over from the original test plan, still true of the pinned engine
(`itis-dakota==6.24.7`):

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

Every category in the original V&V plan (A through I, plus the Ishigami
acceptance gate) currently passes against its analytical reference, with
real (unmocked) Dakota execution for every category that requires the
engine. The known limitations above are pre-existing engine/pipeline
characteristics documented so consumers of this package don't rediscover
them the hard way — not test failures.
