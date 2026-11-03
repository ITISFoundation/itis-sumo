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
# one point per subset size: {n_samples, metrics, metrics_std, draws_by_metric,
#                             n_bootstrap, metric, metric_std, draws, diagnostics}

fit = fit_convergence_exponential(
    [e["n_samples"] for e in curve for _ in e["draws"]],
    [d for e in curve for d in e["draws"]],
    asymptotic=True,  # a raw error metric floors above zero (V18wp)
)
fit["c"], fit["r_squared"]  # irreducible floor + fit quality
```

### Which metric to look at

The sweep takes **no metric argument**: every point already carries all of them,
because the CV pass produces all of them — asking for MAE or the tails instead of
RMSE costs **no extra cross-validation** (V16wq, V55nw). You choose at
presentation time instead of paying for another sweep:

```python
from itis_sumo.report import render_convergence

render_convergence(curve, metric="coverage95")  # axis and reference line follow the metric
coverage95 = [point["metrics"]["coverage95"] for point in curve]  # or take the numbers
```

| token | what it is | units |
| --- | --- | --- |
| `rmse` | root mean squared error — the default, and what a point's `metric` / `metric_std` / `draws` keys still mean | your QoI's units |
| `mae` | mean absolute error | your QoI's units |
| `bias` | mean **signed** error, `mean(actual − predicted)` — the systematic over/under-prediction RMSE and MAE can net away | your QoI's units, signed |
| `coverage68.27` | fraction of held-out observations inside `predicted ± 1σ·predicted_std` | fraction, 0…1 |
| `coverage95` | … inside `predicted ± 1.96σ·predicted_std` (V19cz) | fraction, 0…1 |
| `coverage99.73` | … inside `predicted ± 3σ·predicted_std` — the tail, and the level that needs the most points before it says anything | fraction, 0…1 |

Read the coverage trio together: short of 0.68 already at `coverage68.27` means
the variance model is off everywhere; fine at 68.27 but short at 99.73 means the
centre is right and the tails are too narrow; `metrics_std` says whether the
shortfall is larger than the redraw noise.

Aggregation differs by kind on purpose. An accuracy metric is the **mean over
draws**; a coverage metric is computed over all draws' held-out points **pooled**
(one draw's tail coverage is noise below ~50 points, V19cz) — so a coverage number
is not the mean of its own `draws_by_metric`, which keep each draw's value. Each
point also keeps the whole `compute_cv_diagnostics` bundle per draw in
`diagnostics`, so a number that is not in the table is one line of arithmetic
away. `fit_convergence_exponential` is metric-agnostic — feed it any token's
column; for coverage the floor is the interesting number: `fit["c"] ≈ 0.95` means
coverage settles **at** nominal, `fit["c"] < 0.95` means the variance model stays
miscalibrated no matter how much data you add.

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
