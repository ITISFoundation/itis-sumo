# Surrogate evaluation & cross-validation

`itis_sumo.evaluate.funs_evaluate` is the highest-level module: it drives a
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

Two independent pathways, because Dakota's built-in one has a known
parsing gap:

- **`evaluate_sumo_crossvalidation`** — Dakota-native CV
  (`create_sumo_crossvalidation_conffile`). Parses Dakota's CV log output
  via `_parse_crossvalidation_outputlogs`.
- **`evaluate_sumo_manual_crossvalidation`** — K-fold CV done in Python
  (`sklearn.model_selection.KFold`) driving repeated `evaluate_sumo` calls,
  one fold at a time. This is the pathway that's actually
  correctness-verified end to end (Verification & Validation Category E);
  the built-in pathway's `log_output` is currently hardcoded empty on the
  Dakota side, which the manual path exists to work around — see
  [known limitations](../verification-validation.md#known-limitations).

Metrics on top of either pathway:

- `compute_cv_accuracy_metrics` — RMSE / R² from predictions vs. held-out
  truth.
- `compute_paired_ttest` — paired t-test for systematic bias between two
  sets of predictions (`scipy.stats.ttest_rel`).
- `compute_cv_convergence` — re-runs CV at increasing subset sizes
  (`_convergence_subset_sizes`) to produce a convergence curve; verified
  monotonically decreasing RMSE with more training data (Category B7/E2).

## Uncertainty propagation

- **`propagate_uq`** — Dakota-native forward UQ propagation
  (`create_uq_propagation_conffile`); normal-uncertain inputs only.
- The **actually user-reachable** UQ pathway (the one wired to the
  `/manual_uq_propagation_with_uncertainty` route in mmux/vite) composes
  `create_manual_uq_samples` (supports normal/uniform/constant per
  variable) with `evaluate_sumo`, then injects the surrogate's own
  predictive uncertainty via an erfinv-based transform
  (`sqrt(2)·erfinv(U)  ~  N(0,1)` for `U ~ Uniform(-1, 1)`) — this is the
  pathway [Verification & Validation Category F](../verification-validation.md#category-f-uq-propagation)
  actually exercises, not `propagate_uq`.

## Optimization

**`perform_moga_optimization`** — drives
`create_moga_optimization_conffile`, returns the Pareto-optimal set. See
[MOGA optimization](moga.md) for detail and known caveats.

## Sensitivity

**`evaluate_sobol_indices`** — see
[Sensitivity (Sobol) & UQ propagation](sensitivity-uq.md).
