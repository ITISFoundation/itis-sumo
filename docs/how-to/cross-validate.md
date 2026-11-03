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
surrogate's **own reported spread** earns trust. Coverage asks, at a level you
name, how many observations actually fall within `predicted ± z·predicted_std`
(z is that level's two-sided Gaussian multiple — derived, never an input):

```python
from itis_sumo import compute_coverage

# one call answers ONE level: which level is the question you are asking
coverage = [
    compute_coverage(cv.observed, cv.predicted, cv.predicted_std, level)
    for level in (0.68, 0.95, 0.99)
]
[(c["level"], c["empirical"], c["n_points"]) for c in coverage]
```

`0.95` means "did the interval that should catch 95% catch 95%" (z = 1.96);
`0.68` is the one-sigma question (z = 0.99), `0.99` the tail (z = 2.58). A result
is `{level, empirical, n_points}` — `n_points` matters, because a fraction
measured over 4 points is not a calibration reading. If you think in sigmas,
`level` is the mass and not the multiple: ±1σ is `0.6827`, ±2σ `0.9545`, ±3σ
`0.9973`. Levels are separate calls because that is cheap (arithmetic on points
already in memory) and it keeps the choice of levels with you rather than in a
default.

A persistent (non-shrinking) gap between empirical and nominal is a
miscalibrated variance model, not a data shortage (V19cz). Both readings have
renderings: `itis_sumo.report.render_cv_diagnostics`, and `render_coverage` for
the list above (one bar per level, each compared against its own nominal).

## Check that more data actually helps

Before trusting accuracy at your current training-set size, confirm the error
is actually shrinking as you add points — a flat or increasing RMSE curve as
`N` grows signals something wrong (kernel assumptions, noisy/mislabeled data,
poor design coverage) rather than "just needs more points":

```python
from itis_sumo import compute_cv_convergence, fit_convergence_exponential

curve = compute_cv_convergence(run_dir, training_file, ["width", "height"], "stress")
# one point per subset size: {n_samples, metrics, metrics_std, draws_by_metric,
#                             n_bootstrap, metric, metric_std, draws, diagnostics, uq}

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
| `coverage68` | fraction of held-out observations inside `predicted ± 0.99σ·predicted_std` — the one-sigma interval at a round 68% target | fraction, 0…1 |
| `coverage95` | … inside `predicted ± 1.96σ·predicted_std` (V19cz) | fraction, 0…1 |
| `coverage99` | … inside `predicted ± 2.58σ·predicted_std` — the tail, and the level that needs the most points before it says anything | fraction, 0…1 |

The levels are the round 68 / 95 / 99 because a metric token ends up as an axis
label, and `coverage99` reads as what was asked where `coverage99.73` has to be
decoded. Nothing is approximated by that: each level uses its own exact two-sided
Gaussian multiple, and `compute_coverage` takes any target you prefer (`0.6827`,
`0.9545`, `0.9973` if you want the ±1σ/±2σ/±3σ intervals themselves).

Read the coverage trio together: short of 0.68 already at `coverage68` means
the variance model is off everywhere; fine at 68 but short at 99 means the
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

### Watching the distribution settle, not just the error

An accuracy curve can plateau while the **predictive distribution** is still
moving — its centre drifting, its upper tail still fat. Each series point
therefore carries a `uq` block beside the metric family: the surrogate's own
predictive distribution at that training-sample count, from the same single sweep.

```python
from itis_sumo import compute_cv_convergence
from itis_sumo.report import render_convergence, render_convergence_uq

curve = compute_cv_convergence(run_dir, training_file, ["width", "height"], "stress")
for point in curve:
    uq = point["uq"]
    print(
        point["n_samples"],
        point["metrics"]["mae"],
        uq["mean"], uq["median"], uq["p95"], uq["p99"], uq["n_points"],
    )

render_convergence(curve, metric="mae")  # axis labelled MAE, draws as a cloud
render_convergence_uq(curve)             # mean / median / p95 / p99 vs sample count
```

`uq` holds `mean`, `median`, `std`, `p95`, `p99` (in your QoI's units),
`n_points`/`n_draws`, `quantiles_per_point`, and `draw_spread` — how much each
statistic still moves when the whole subset is redrawn (a single-draw point has
no redraw to compare with, so those entries are `None`, not a flattering `0.0`).

!!! warning "This is a CV replay, not a fresh UQ propagation"
    The `uq` numbers are rebuilt from the `predicted`/`predicted_std` arrays the
    CV run already returned — Gaussian per held-out point, evaluated on a fixed
    50-point quantile grid, pooled over points and draws. That is why they cost
    nothing, and why they describe the **surrogate's** predictive distribution
    over whatever region your CV design happens to cover. For a propagation of
    *stated* input uncertainty, use `sample_uncertainty` +
    `summarize_uncertainty` (which now reports `p95`/`p99` too); points whose
    `predicted_std` is missing or non-positive are excluded, exactly as
    `compute_coverage` excludes them, so a zero-spread column yields `None`
    statistics rather than invented ones. Propagating per subset size instead of
    replaying it is itis-sumo#81.

!!! note "The one explicitly-staged function"
    `compute_cv_convergence` re-runs K-fold CV per subset size, so it is the
    only manifest function that asks for an explicit run directory and
    space-separated training file (e.g. staged via
    `fit(..., workspace=...)`) instead of a `SuMo` handle. Everything else on
    this page runs through the handle. `render_convergence` /
    `render_convergence_fit` present the curve and the fit,
    `render_convergence_uq` the distribution statistics.

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
- **Convergence `uq` percentiles** (`p95`, `p99`) — same units as your output
  variable, and read them against the physics: a `p99` still sitting outside the
  physically possible range means the surrogate's tails are unresolved even when
  MAE already looks flat. Flat accuracy + moving `uq` = the model has stopped
  improving where you measure it but has not become trustworthy yet.
