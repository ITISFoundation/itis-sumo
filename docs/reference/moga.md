# MOGA optimization

`perform_moga_optimization` (`evaluate/sumo_evaluation.py`) drives Dakota's
Multi-Objective Genetic Algorithm (MOGA) method
(`create_moga_optimization_conffile` in `config/`) over one or more
objective functions defined by a fitted surrogate, and returns the
Pareto-optimal set.

## Pareto-front support (`data/tabular.py`)

- `get_bounds_uniform_distribution(s)` — resolves variable bounds from a
  uniform-distribution spec, feeding the `variables` block MOGA searches
  over.

(Standalone Python dominance helpers — `is_dominated` /
`get_non_dominated_indices` — existed as a reusable cross-check but had no
product callers; removed, nondominance is MOGA's own job on the Dakota side.)

## Verified behavior

(Verification & Validation Category G)

- Single-objective minimization finds the known optimum within tolerance.
- A bi-objective front spans the expected trade-off curve.
- The front improves (dominates the lower-iteration front) as the
  iteration budget increases.
- Every returned point respects the input variable bounds.

## Known limitation

`max_function_evaluations` is accepted by the composer but not enforced by
the underlying Dakota MOGA method in the pinned engine version (deprecated
parameter on the Dakota side) — don't rely on it as a hard evaluation-count
cap; use the iteration/generation controls instead. See
[Known limitations](../verification-validation.md#known-limitations).
