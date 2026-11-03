# How to cross-validate a surrogate

**Goal:** know how much to trust a fitted surrogate on points it wasn't
trained on, before using it for anything downstream.

Assumes you already have a training table in the shape
[Getting started](../tutorials/getting-started.md) builds — your own column
names, your own units.

## Run K-fold cross-validation

```python
from itis_sumo import cross_validation_metrics, fit

model = fit(samples, ["width", "height"], "stress")
cv = model.cross_validate(folds=5, seed=42)
# cv.observed / cv.predicted / cv.predicted_std — aligned to your rows,
# NaN exactly where a fold Dakota couldn't complete, reason in cv.warnings

metrics = cross_validation_metrics(samples, ["width", "height"], "stress", seed=42)
metrics.root_mean_squared, metrics.mean_abs, metrics.max_abs
```

!!! danger "The manual K-fold path is the only CV path"
    The Dakota-native CV chain was removed as dead: its log-parsing had a
    known gap (see
    [known limitations](../verification-validation.md#known-limitations))
    and it had no callers. These manual K-fold numbers are independently
    verified end to end.

## Check for systematic bias, and for calibration

The one-pass diagnostics bundle runs the paired t-test, Cohen's *d* effect
size, and the outlier-fenced MAE on the CV pair:

```python
from itis_sumo import compute_cv_diagnostics

bundle = compute_cv_diagnostics(cv.observed, cv.predicted, cv.predicted_std)
bundle["ttest_p_value"]  # low p-value: errors are skewed, not just noise
bundle["cohens_d"]
```

A low p-value (e.g. < 0.05) means the surrogate's errors aren't just noise —
they're skewed in a consistent direction.

Accuracy closeness is half the question; the other half is whether the
surrogate's **own reported spread** earns trust. Coverage compares, at the
1σ/2σ/3σ nominal masses, how many observations actually fall within
`predicted ± z·predicted_std`:

```python
from itis_sumo import compute_coverage

coverage = compute_coverage(cv.observed, cv.predicted, cv.predicted_std)
coverage["levels"], coverage["empirical"]  # compare entry-wise
```

A persistent (non-shrinking) gap between empirical and nominal is a
miscalibrated variance model, not a data shortage (V19cz). Both readings have
renderings: `itis_sumo.report.render_cv_diagnostics`,
`render_coverage`.

## Check that more data actually helps

Before trusting accuracy at your current training-set size, confirm the error
is actually shrinking as you add points — a flat or increasing RMSE curve as
`N` grows signals something wrong (kernel assumptions, noisy/mislabeled data,
poor design coverage) rather than "just needs more points":

```python
from itis_sumo import compute_cv_convergence, fit_convergence_exponential

curve = compute_cv_convergence(run_dir, training_file, ["width", "height"], "stress")
# series of {n_samples, metric, metric_std, n_bootstrap, draws, diagnostics}

fit = fit_convergence_exponential(
    [e["n_samples"] for e in curve for _ in e["draws"]],
    [d for e in curve for d in e["draws"]],
    asymptotic=True,  # a raw error metric floors above zero (V18wp)
)
fit["c"], fit["r_squared"]  # irreducible floor + fit quality
```

!!! note "The one explicitly-staged function"
    `compute_cv_convergence` re-runs K-fold CV per subset size, so it is the
    only manifest function that asks for an explicit run directory and
    space-separated training file (e.g. staged via
    `fit(..., workspace=...)`) instead of a `SuMo` handle. Everything else on
    this page runs through the handle. `render_convergence` /
    `render_convergence_fit` present the curve and the fit.

## Reading the numbers

- **RMSE / MAE / max-abs error** (`root_mean_squared`, `mean_abs`, `max_abs`)
  — same units as your output variable; compare against what "close enough"
  means for your decision, there's no universal threshold. Expect worse
  numbers on oscillatory or high-dimensional targets (see
  [Why surrogate modeling § What makes a function a good surrogate target](../explanation/surrogate-modeling.md#what-makes-a-function-a-good-or-bad-surrogate-target)
  for why).
- If accuracy isn't good enough: add training points (ideally via
  [Latin Hypercube sampling](../reference/functions.md#sampling), not more of
  whatever design you started with), or reconsider whether the target
  function is a good surrogate candidate at all.
