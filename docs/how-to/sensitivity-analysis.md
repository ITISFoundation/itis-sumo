# How to run a sensitivity analysis

**Goal:** find out which inputs actually drive an output's variance, so you
know where to spend sampling budget and which variables are safe to fix.

!!! info "Key concept"
    Sobol' indices decompose output variance into per-input contributions —
    so you know where to spend sampling budget and which variables are safe to fix.

Assumes a fitted preprocessor and a training file in the shape
[Getting started](../tutorials/getting-started.md) builds.

## Compute Sobol' indices

```python
from itis_sumo.evaluate.funs_evaluate import evaluate_sobol_indices

distributions = {
    "length": {"distribution": "uniform", "min": 0.0, "max": 1.0},
    "width": {"distribution": "uniform", "min": 0.0, "max": 1.0},
}
result = evaluate_sobol_indices(
    run_dir,
    training_file,
    ["length", "width"],
    "y1",
    distributions,
    preprocessor,
    seed=42,
)
sobol = result["sobol"]  # {var: {"main", "total", "main_ci_low", ...}}
second_order = result["sobolSecondOrder"]  # {varA: {varB: float}}

for var, indices in sobol.items():
    print(var, indices["main"], indices["total"])
```

`distributions` needs one entry per name in `input_vars`, each shaped as one
of:

- `{"distribution": "uniform", "min": ..., "max": ...}`
- `{"distribution": "normal", "mean": ..., "std": ...}`
- `{"distribution": "constant", "value": ...}` — held fixed; contributes zero
  variance and is excluded from the sampling budget

Costs `SOBOL_BASE_SAMPLES * (d_varying + 2)` surrogate evaluations
(`SOBOL_BASE_SAMPLES = 1024`), where `d_varying` is the number of
non-constant inputs — set variables you don't care about to `"constant"`
rather than leaving them `"uniform"` with a narrow range, since that cost
scales with count, not width.

## Reading the indices

- **`main`** (first-order) — variance explained by that variable alone. A
  variable with `main ≈ 0` has no effect *on its own*.
- **`total`** — variance explained by that variable including all its
  interactions with others. `total > main` means interactions matter;
  `total ≈ main` means the variable acts independently.
- **`total - main` gap** — the size of the gap tells you how much of that
  variable's influence is only visible through interaction with another
  input. A variable can have `main ≈ 0` but `total` well above zero — it has
  no effect alone but a real interaction effect (see the Ishigami `x3` case
  in the [V&V report](../verification-validation.md#ishigami-analytical-acceptance-gate)).
- **`main_ci_low` / `main_ci_high` / `total_ci_low` / `total_ci_high`** —
  bootstrap 95% confidence bounds, computed by resampling the already-run
  evaluations (no extra surrogate cost). Treat a `main` index as
  indistinguishable from zero if its CI straddles zero.
- **`sobolSecondOrder`** — pairwise interaction variance between two
  variables, `{varA: {varB: value}}`. Useful once `total - main` flags a
  variable as interaction-driven and you want to know *with which other
  variable*.

## Propagate input uncertainty to output uncertainty

Sobol' indices tell you *which* inputs matter; if instead you want the
output distribution itself (mean/std of `y` given input distributions),
that's `propagate_uq` — a separate, Dakota-native pathway not currently
wired to the web UI. See
[Reference → Evaluate § Uncertainty propagation](../reference/evaluate.md#uncertainty-propagation)
for its signature and known caveats before using it.
