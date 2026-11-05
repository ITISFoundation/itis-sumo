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

!!! note "The one explicitly-staged function"
    `compute_cv_convergence` re-runs K-fold CV per subset size, so it is the
    only manifest function that asks for an explicit run directory and
    space-separated training file (e.g. staged via
    `fit(..., workspace=...)`) instead of a `SuMo` handle. Everything else on
    this page runs through the handle. `render_convergence` /
    `render_convergence_fit` present the curve and the fit.

### The other convergence question: propagated-distribution convergence

The CV sweep answers *"are the surrogate's own predictions and spread
trustworthy at N?"*. If what you want is *"what do the **distributions I
declared** on the inputs do to the QoI at N?"* — how the propagated tails,
centre and shape move as the training set grows — that is
`SuMo.uq_convergence_curve`: the surrogate is **refit at each subset size**
and the same stated distributions are propagated through each refit.

```python
from itis_sumo import DistributionSpec, fit
from itis_sumo.report import render_convergence

model = fit(samples, ["width", "height"], "stress", workspace=run_dir)
curve = model.uq_convergence_curve(
    distributions={
        "width": DistributionSpec("normal", mean=0.0, std=0.02),
        "height": DistributionSpec("normal", mean=0.0, std=0.01),
    },
    convergence_steps=[10, 25, len(samples)],  # the Ns this curve will use
    n_bootstrap=3,
)
render_convergence(curve, metric="uq_p99")  # takes the typed result directly
```

The result is a typed, `dataclasses.asdict`-serializable object (V22rs) whose
points carry the CV curve's shape, so nothing new is learned to plot them; the
tokens are `uq_mean`, `uq_std`, `uq_p1`, `uq_p5`, `uq_median`, `uq_p95`,
`uq_p99`, `uq_skew`, `uq_kurt` — in your QoI's units except the last two,
which are dimensionless (`uq_skew`/`uq_kurt` follow scipy's bias-corrected and
Fisher conventions, and `uq_kurt`'s sampling error is ≈ √(24/M) over the M
propagated values behind it — read it as shape direction, not shape value).
The tail is **two-sided** on purpose: a QoI going negative is a lower-tail
failure that an upper-tail curve cannot see.

Which sizes the sweep runs is explicit: `convergence_steps` names them
outright (nothing is added silently), or `min_samples` +
`convergence_samples_step` walks up to the full count, which is always the
last point — that last point is the number the UQ panel shows for the whole
training set. With neither given you get the same evenly-spaced, up-to-5-point
sweep `compute_cv_convergence` draws, so the two curves share an x-axis out of
the box. Wild tails at small N are the expected shape of this instrument —
the declared distributions reaching further than the subset ever covered —
not something the sweep clips. `seed` must be a strictly positive integer
(Dakota's own constraint, stated here so you get a clear error instead of an
engine failure); every point echoes the `draw_seeds` its draws actually ran
with, so a warning is re-runnable from the result alone. Pass `n_histograms=1`
to propagate the inputs alone, without the surrogate's own spread at that N.

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
