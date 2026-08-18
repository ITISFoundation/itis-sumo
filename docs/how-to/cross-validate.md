# How to cross-validate a surrogate

**Goal:** know how much to trust a fitted surrogate on points it wasn't
trained on, before using it for anything downstream.

Assumes you already have a training file in the shape
[Getting started](../tutorials/getting-started.md) builds — standardized
`x1, x2, …` / `y1` columns, written space-separated.

## Run K-fold cross-validation

```python
from itis_sumo.evaluate.funs_evaluate import (
    evaluate_sumo_manual_crossvalidation,
    compute_cv_accuracy_metrics,
)

cv = evaluate_sumo_manual_crossvalidation(
    run_dir, training_file, ["x1", "x2"], "y1", N_CROSS_VALIDATION=5,
)
# cv["y1"] is the held-out actual values, cv["y1_hat"] the CV predictions
# (both length-N, NaN at any fold Dakota couldn't complete)
metrics = compute_cv_accuracy_metrics(cv["y1"], cv["y1_hat"])
print(metrics["root_mean_squared"], metrics["mean_abs"], metrics["max_abs"])
```

Use `evaluate_sumo_manual_crossvalidation`, not `evaluate_sumo_crossvalidation`
(the Dakota-native pathway) — the manual K-fold pathway is the one whose
numbers are independently verified end to end; Dakota's own CV log parsing
has a known gap. Full detail: [Reference → Evaluate § Cross-validation](../reference/evaluate.md#cross-validation).

## Check for systematic bias

`compute_paired_ttest` runs a paired t-test on the same actual/predicted CV
pair, testing whether the surrogate is systematically biased (not just
noisy) relative to the held-out truth:

```python
from itis_sumo.evaluate.funs_evaluate import compute_paired_ttest

compute_paired_ttest(cv["y1"], cv["y1_hat"])  # -> {"statistic": ..., "p_value": ...}
```

A low p-value (e.g. < 0.05) means the surrogate's errors aren't just noise —
they're skewed in a consistent direction.

## Check that more data actually helps

Before trusting a surrogate's accuracy at your current training-set size,
confirm the error is actually shrinking as you add points — a flat or
increasing RMSE curve as `N` grows is a sign of something wrong (wrong
kernel assumptions, noisy/mislabeled data, insufficient sampling coverage)
rather than "just needs more points":

```python
from itis_sumo.evaluate.funs_evaluate import compute_cv_convergence

curve = compute_cv_convergence(run_dir, training_file, ["x1", "x2"], "y1")
# list of {"n_samples": ..., "metric": <RMSE at that training-set size>}
```

## Reading the numbers

- **RMSE / MAE / max-abs error** (`root_mean_squared`, `mean_abs`, `max_abs`
  from `compute_cv_accuracy_metrics`) — same units as your output variable;
  compare against what "close enough" means for your decision, there's no
  universal threshold. Expect worse numbers on oscillatory or
  high-dimensional targets (see
  [Why surrogate modeling § What makes a function a good surrogate target](../explanation/surrogate-modeling.md#what-makes-a-function-a-good-or-bad-surrogate-target)
  for why).
- If accuracy isn't good enough: add training points (ideally via
  [Latin Hypercube sampling](../reference/sampling.md), not more of whatever
  design you started with), or reconsider whether the target function is a
  good surrogate candidate at all.
