# Sensitivity (Sobol) & UQ propagation

See [Worked examples § Sensitivity analysis on a surrogate](../tutorials/examples.md#3-sensitivity-analysis-on-a-surrogate-does-it-recover-the-right-physics)
for a runnable demonstration of this pipeline against the Ishigami function,
GP surrogate included.

## Sobol' sensitivity indices

`evaluate_sobol_indices` (`evaluate/funs_evaluate.py`) computes first-,
total-, and exact second-order Sobol' sensitivity indices plus ANOVA order
masses for an already-built surrogate — **scipy-design, not Dakota-native**
(`V9gh`): Dakota's own Sobol study type is not used; instead the module draws
Saltelli A/B/C sample matrices, evaluates the fitted surrogate on them via
`evaluate_sumo`, and applies `_sobol_algebra` — the exact
`scipy.stats.sobol_indices` `saltelli_2010` algebra (pinned equal by an
estimator-parity test), the single runtime source for every index.

- **Sampling box = exploration domain** (`V26dd`): each varying input is
  drawn uniformly across its box (log-uniform under a `log_scale` entry);
  modeller distributions are NOT an input — they belong to UQ propagation.
  The public `itis_sumo.api.evaluate_sobol` accepts optional `DomainSpec`
  boxes; omitted boxes are auto-inferred from the observed sample bounds, and
  a column constant in the samples is held fixed.
- **Base sample size**: fixed at `SOBOL_BASE_SAMPLES = 1024`, independent
  of the general UQ `numSamples` field used by histogram/correlation
  computations (`V36`) — rounded up to the next power of two if a caller
  requests fewer (`ceil(log2(max(n, 2)))`).
- **Second-order indices** come from EXACT joint-pair designs
  (`U^ij`/`V^ij` mixed evaluations against an independent third stream `C`):
  `S_ij = Var(E[Y|X_i,X_j])/V − S_i − S_j`, exact for ANY input count —
  `V48tr`. The retired Jansen/Saltelli closed-form relation was provably
  underdetermined for `d ≥ 4` (B20qt). Surrogate cost is
  `n·(2 + d + d·(d−1))` — the documented `O(n·d²)` price of exact pairs.
- **Order masses** `M1`/`M2`/`R = 1 − M1 − M2` close to 1 by construction and
  are returned with bootstrap CIs; zero sample variance reports them as
  `None` (undefined fractions), never `(0, 0, 0)`.
- **Confidence intervals** (all orders + masses) come from ONE shared-row
  bootstrap over the already-computed evaluations — no extra surrogate calls,
  estimator covariance and the closure identity preserved per replicate.
- **Fixed input variables** (a `{"value": ...}` entry, or a column constant
  in the samples) are short-circuited to `main=0, total=0` rather than passed
  through the sampler.

### Validation: Ishigami analytical acceptance gate

This pipeline's correctness is pinned to acceptance tests — `SPEC.md` §R1 —
that bypass the surrogate entirely and evaluate the
[Ishigami function](https://www.sfu.ca/~ssurjano/ishigami.html) analytically
on designs built with the PRODUCTION sampling/design/algebra helpers, to
isolate "is the sampling → splitting → pair-design → algebra math correct"
from "is the GP surrogate accurate" — alongside additive d=8,
pair-interaction d=5, and pair-quadratic d=10 analytic benchmarks with
bootstrap-CI-scaled tolerances. See
[Verification & Validation](../verification-validation.md#ishigami-analytical-acceptance-gate)
for the reference values and current pass status.

## UQ propagation

Two distinct pathways exist — see
[Evaluate § Uncertainty propagation](evaluate.md#uncertainty-propagation)
for which one is actually reachable from the web UI and which this site's
verification suite exercises:

- `propagate_uq` — Dakota-native, normal-uncertain variables only.
- Manual pathway — `create_manual_uq_samples` (normal / uniform / constant
  per variable, seeded) + `evaluate_sumo` + an erfinv-based injection of
  the surrogate's own predictive uncertainty into the propagated samples.

Both are validated against closed-form output distributions for simple
transforms (e.g. `y = 2x + 1, x ~ N(0,1) ⟹ y ~ N(1, 4)`;
`y = x², x ~ N(0,1) ⟹ y ~ χ²(1)`) — see
[Category F](../verification-validation.md#category-f-uq-propagation).
