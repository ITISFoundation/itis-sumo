# Getting started

Install itis-sumo, then build, cross-validate, and interrogate one surrogate
end to end. Every step below is real, runnable code — the same pipeline
[`examples/headless_smoke.py`](https://github.com/ITISFoundation/itis-sumo/blob/main/examples/headless_smoke.py)
runs as a CI smoke test.

## 1. Install

```sh
uv sync                          # Python 3.10-3.12 currently resolve itis-dakota==1.5.9
uv run itis-sumo validate        # engine probe (expect: Version 1.5.9)
```

If `validate` fails, the Dakota engine wheel didn't resolve correctly — that's
an install problem, not a itis-sumo problem; nothing below will work until
`validate` passes.

## 2. Some training data

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
stress = 3.0 * length + 2.0 * width**2 + 0.5 * np.sin(10.0 * length) + rng.normal(0, 0.05, 30)
train_raw = pd.DataFrame({"length": length, "width": width, "stress": stress})
```

## 3. Preprocess: raw names → Dakota-safe names

itis-sumo's Dakota layer expects standardized variable names (`x1, x2, …` /
`y1, y2, …`), not your original column names — `DataPreprocessor` handles
that mapping (plus optional normalization) both ways:

```python
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
preprocessor.setup_variables(["length", "width"], ["stress"])
preprocessor.fit(train_raw)
train_processed = preprocessor.transform(train_raw)

training_file = run_dir / "df_processed_jobs.dat"
train_processed.to_csv(training_file, sep=" ", index=False)
```

`run_dir` is any writable directory — `itis_sumo.utils.helpers.create_run_dir`
makes one for you with a timestamp-scoped name.

## 4. Fit a surrogate and evaluate it

```python
from itis_sumo.evaluate.funs_evaluate import evaluate_sumo

preds = evaluate_sumo(run_dir, training_file, eval_samples_file, ["x1", "x2"], "y1")
y_hat = preds["y1_hat"]  # posterior mean prediction at each eval point
```

`evaluate_sumo` fits a Gaussian Process on `training_file` and predicts at
every point in `eval_samples_file` (built the same way as the training file —
see the [full script](https://github.com/ITISFoundation/itis-sumo/blob/main/examples/headless_smoke.py)
for the grid-construction step omitted here for brevity). What a GP is and
why itis-sumo uses one: [How Gaussian Processes work](../explanation/gaussian-processes.md).

## 5. How good is it? Cross-validate

```python
from itis_sumo.evaluate.funs_evaluate import evaluate_sumo_crossvalidation

cv_metrics = evaluate_sumo_crossvalidation(run_dir, training_file, ["x1", "x2"], "y1", N_CROSS_VALIDATION=5)
print(cv_metrics["y1"]["root_mean_squared"])
```

5-fold CV, held-out RMSE. This is the number that tells you whether to trust
the surrogate on points you didn't train on — see
[How-to: cross-validate a surrogate](../how-to/cross-validate.md) for the
full recipe including R² and convergence checks.

## 6. Which inputs actually matter? Sobol' sensitivity

```python
from itis_sumo.evaluate.funs_evaluate import evaluate_sobol_indices

distributions = {
    "length": {"distribution": "uniform", "min": 0.0, "max": 1.0},
    "width": {"distribution": "uniform", "min": 0.0, "max": 1.0},
}
sobol = evaluate_sobol_indices(
    run_dir, training_file, ["length", "width"], "y1", distributions, preprocessor, seed=42
)["sobol"]
for var, indices in sobol.items():
    print(var, indices["main"], indices["total"])
```

First- and total-order Sobol' indices per variable — see
[How-to: run sensitivity analysis](../how-to/sensitivity-analysis.md).

## Run it for real

```sh
uv run python examples/headless_smoke.py   # the whole thing above, in one script
uv run pytest                              # itis-sumo's own test suite
```

## Where next

- Building something real? Start with the [How-to guides](../how-to/cross-validate.md) —
  task recipes for cross-validation, sensitivity/UQ, MOGA optimization, and preprocessing.
- Want to see it against real data end to end? [Worked examples](examples.md).
- Curious *why* surrogate modeling and GPs work the way they do?
  [Why surrogate modeling](../explanation/surrogate-modeling.md).
- Need exact function signatures? [Reference](../reference/index.md).
