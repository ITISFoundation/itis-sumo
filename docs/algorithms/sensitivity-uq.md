# Sensitivity (Sobol) & UQ propagation

See [Worked examples § Sensitivity analysis on a surrogate](../theory/examples.md#3-sensitivity-analysis-on-a-surrogate-does-it-recover-the-right-physics)
for a runnable demonstration of this pipeline against the Ishigami function,
GP surrogate included.

## Sobol' sensitivity indices

`evaluate_sobol_indices` (`evaluate/funs_evaluate.py`) computes first-,
total-, and (closed-form) second-order Sobol' sensitivity indices for an
already-built surrogate — **scipy-based, not Dakota-native** (`V9gh`):
Dakota's own Sobol study type is not used; instead the module draws
Saltelli-style A/B/AB sample matrices, evaluates the fitted surrogate on
them via `evaluate_sumo`, and calls `scipy.stats.sobol_indices` directly.

- **Base sample size**: fixed at `SOBOL_BASE_SAMPLES = 1024`, independent
  of the general UQ `numSamples` field used by histogram/correlation
  computations (`V36`) — rounded up to the next power of two if a caller
  requests fewer (`ceil(log2(max(n, 2)))`).
- **Second-order indices** are not directly returned by
  `scipy.stats.sobol_indices`; they're derived via the Jansen/Saltelli
  (2010) closed-form relation from the total-order indices:
  `S_ij = (S_Ti + S_Tj - sum(higher-order terms)) / 2`.
- **Constant input variables** are detected and short-circuited to
  `main=0, total=0` rather than passed through the sampler.

### Validation: Ishigami analytical acceptance gate

This pipeline's correctness is pinned to a single acceptance test —
`SPEC.md` §R1 — that bypasses the surrogate entirely and evaluates the
[Ishigami function](https://www.sfu.ca/~ssurjano/ishigami.html) analytically
on the Saltelli samples, to isolate "is the sampling → splitting → scipy
call → closed-form second-order math correct" from "is the GP surrogate
accurate." See
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
