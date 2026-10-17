# Surrogate evaluation & cross-validation

`itis_sumo.evaluate.sumo_evaluation` is the highest-level module: it drives a
Dakota study through `DakotaObject`, then parses the resulting tabular
output back into pandas DataFrames and derived metrics.

## Core evaluation

- **`evaluate_sumo(run_dir, train_file, eval_file, input_vars, output_var)`**
  — fits a Gaussian-process surrogate on `train_file`, evaluates it at the
  points in `eval_file`, and returns predictions (`{output}_hat`) plus
  predictive standard deviation (`{output}_std_hat`, `V8df` — always sourced
  from the surrogate's own posterior, never raw job outputs). At training
  points the surrogate reproduces the training values almost exactly
  (verified: Verification & Validation Category B, test B1).
- **`evaluate_sumo_along_axes`** — 1D sweeps (pairs with
  `create_samples_along_axes`/`extract_predictions_along_axes`).
- **`evaluate_sumo_on_grid`** — full-factorial grid evaluation, reshaping
  Dakota's flat tabular output back into an N-D array matching the grid
  shape.

## Cross-validation

One pathway: Dakota's built-in CV chain (conf composer + stdout-log parser)
was removed as dead — the product never called it, and its log-parsing
was a known gap (see
[known limitations](../verification-validation.md#known-limitations)).

- **`evaluate_sumo_manual_crossvalidation`** — K-fold CV done in Python
  (validation indices from a seeded `np.random.RandomState` permutation
  split via `np.array_split`, a layout frozen to be fold-for-fold identical
  to `sklearn`'s `KFold(shuffle=True)` before that dependency was dropped)
  driving repeated `evaluate_sumo` calls, one fold at a time. This is the
  pathway that's actually correctness-verified end to end (Verification &
  Validation Category E).

Metrics on top of the fold predictions:

- `compute_cv_accuracy_metrics` — RMSE / R² from predictions vs. held-out
  truth.
- `compute_paired_ttest` — paired t-test for systematic bias between two
  sets of predictions (`scipy.stats.ttest_rel`).
- `compute_cv_convergence` — re-runs CV at increasing subset sizes
  (`_convergence_subset_sizes`) to produce a convergence curve; verified
  monotonically decreasing RMSE with more training data (Category B7).

## Uncertainty propagation

The **user-reachable** UQ pathway (the one wired to the
`/manual_uq_propagation_with_uncertainty` route in mmux/vite) composes
`create_manual_uq_samples` (supports normal/uniform/constant per
variable) with `evaluate_sumo`, then injects the surrogate's own
predictive uncertainty via an erfinv-based transform
(`sqrt(2)·erfinv(U)  ~  N(0,1)` for `U ~ Uniform(-1, 1)`) — this is the
pathway [Verification & Validation Category F](../verification-validation.md#category-f-uq-propagation)
exercises. (Dakota's native `propagate_uq` chain — normal-uncertain inputs
only — was dead on every real path and has been removed.)

## Optimization

**`perform_moga_optimization`** — drives
`create_moga_optimization_conffile`, returns the Pareto-optimal set. See
[MOGA optimization](moga.md) for detail and known caveats.

## Sensitivity

**`evaluate_sobol_indices`** — see
[Sensitivity (Sobol) & UQ propagation](sensitivity-uq.md).
