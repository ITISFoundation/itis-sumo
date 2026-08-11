# MOGA optimization

`perform_moga_optimization` (`evaluate/funs_evaluate.py`) drives Dakota's
Multi-Objective Genetic Algorithm (MOGA) method
(`create_moga_optimization_conffile` in `config/`) over one or more
objective functions defined by a fitted surrogate, and returns the
Pareto-optimal set.

## Pareto-front support (`data/funs_data_processing.py`)

- `is_dominated` / `get_non_dominated_indices` — standard Pareto dominance
  check/filter, used both to validate MOGA's own output and (independently)
  as a reusable utility.
- `get_bounds_uniform_distribution(s)` — resolves variable bounds from a
  uniform-distribution spec, feeding the `variables` block MOGA searches
  over.

## Verified behavior

(Verification & Validation Category G)

- Single-objective minimization finds the known optimum within tolerance.
- A bi-objective front spans the expected trade-off curve and every
  returned point is genuinely non-dominated (cross-checked against
  `get_non_dominated_indices`).
- The front improves (dominates the lower-iteration front) as the
  iteration budget increases.
- Every returned point respects the input variable bounds.

## Known limitation

`max_function_evaluations` is accepted by the composer but not enforced by
the underlying Dakota MOGA method in the pinned engine version (deprecated
parameter on the Dakota side) — don't rely on it as a hard evaluation-count
cap; use the iteration/generation controls instead. See
[Known limitations](../verification-validation.md#known-limitations).
