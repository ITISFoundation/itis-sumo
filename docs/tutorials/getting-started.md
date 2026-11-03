# Getting started

Install itis-sumo, then build, cross-validate, and interrogate one surrogate
end to end. Every step below is real, runnable code — the same pipeline
[`examples/headless_smoke.py`](https://github.com/ITISFoundation/itis-sumo/blob/develop/examples/headless_smoke.py)
runs as a publish-verify smoke test.

## 1. Install

```sh
uv pip install "itis-sumo[report]"     # core + report renderings (plotly)
itis-sumo validate                     # engine probe (expect: Version 6.24.13)
```

!!! warning "Engine gate"
    If `validate` fails, the Dakota engine wheel didn't resolve — an install
    problem, not an itis-sumo problem. Nothing below works until it passes.

Skip the `[report]` extra if you only want the compute core; everything below
that mentions `itis_sumo.report` needs it.

## 2. The marimo way

Report-style work belongs in a [marimo](https://marimo.io) notebook: data
cells call the API and own typed results; rendering cells hand each result —
untouched — to one rendering from `itis_sumo.report`. A ready-made template
with a version stamp ships with the repo:

```sh
marimo run examples/report_notebook_template.py    # or: marimo edit  <same file>
```

The template covers the full arc — `fit`, cross-validation, Sobol', sweeps,
grids, uncertainty draws, predictions, a Pareto front, and the catalog tail
(`report.sections()` says which renderings exist and what each one eats).
The rest of this page is the same flow as plain Python.

## 3. Some training data

itis-sumo trains on whatever `x -> y` observations you already have — from a
simulator, an experiment, a spreadsheet. Here we fake 30 noisy observations
of a made-up `stress = f(length, width)` relationship so the tutorial doesn't
depend on external data:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
length = rng.uniform(0.0, 1.0, size=30)
width = rng.uniform(0.0, 1.0, size=30)
stress = (
    3.0 * length
    + 2.0 * width**2
    + 0.5 * np.sin(10.0 * length)
    + rng.normal(0, 0.05, 30)
)
train = pd.DataFrame({"length": length, "width": width, "stress": stress})
```

## 4. Fit — you get a handle

```python
from itis_sumo import fit

model = fit(train, ["length", "width"], "stress")
```

`fit` returns a `SuMo` handle. Its methods are the **catalog**: everything you
can ask the trained surrogate. Dakota names, normalization, and run
directories are handled inside; your names and units are the only ones you
ever see. The run directory lives exactly as long as the handle: leave the
`with` block (or call `close()`) and the working files are discarded on
success — kept on failure so you can diagnose (V24af).

## 5. Predict, and how good is it?

```python
predictions = model.predict(train[["length", "width"]])
predictions.values       # one prediction per row, in stress's units
predictions.std          # the surrogate's own predictive spread

cv = model.cross_validate(folds=5, seed=42)
cv.observed              # aligned to your rows — NaN exactly where a fold
cv.predicted             # Dakota abandoned, with the reason in cv.warnings
```

5-fold CV, held-out predictions per sample. Pair it with the metric row:

```python
from itis_sumo import cross_validation_metrics

metrics = cross_validation_metrics(train, ["length", "width"], "stress", seed=42)
metrics.root_mean_squared
```

This is the number that tells you whether to trust the surrogate on points it
didn't train on — see
[How-to: cross-validate a surrogate](../how-to/cross-validate.md) for
calibration (does the reported spread earn trust?) and convergence checks.
## 6. Which inputs actually matter? Sobol' sensitivity

```python
sobol = model.sobol(seed=42)
for var, indices in sobol.indices.items():
    print(var, indices["main"], indices["total"])
```

First- and total-order Sobol' indices per variable, with bootstrap CIs, over
the exploration domain — see
[How-to: run sensitivity analysis](../how-to/sensitivity-analysis.md) for
pinning factors and for propagating explicit input uncertainty
(`sample_uncertainty`) on the same handle.

## 7. Make the picture — renderings

```python
from itis_sumo.report import render_cross_validation, render_sobol

render_cross_validation(cv)     # observed vs predicted, hover = row index
render_sobol(sobol)             # indices with their CI whiskers
```

Renderings consume exactly one result and are deletable suggestions: kwargs
pass through to plotly, and `fig.write_html(...)` is always there when the
suggestion is wrong (V54hz).

```python
model.close()    # run directory discarded (V24af)
```

## Run it for real

```sh
uv run python examples/headless_smoke.py   # steps 3-6 above, in one script
uv run pytest                              # itis-sumo's own test suite
```

## Where next

- Building something real? Start with the [How-to guides](../how-to/cross-validate.md) —
  task recipes for cross-validation, sensitivity/UQ, and MOGA optimization.
- Want to see it against known closed-form answers? [Worked examples](examples.md).
- Curious *why* surrogate modeling and GPs work the way they do?
  [Why surrogate modeling](../explanation/surrogate-modeling.md).
- Need exact signatures? [Reference](../reference/index.md).
