# How to tell itis-sumo how a column behaves

**Goal:** tell the engine what your columns *are* (a log-scale length, a
minimize-direction response) without ever naming a transform — and read back
what was actually applied.

You never preprocess by hand for itis-sumo. Column names, units, and scales
stay yours on the way in and on the way out; normalization, internal renaming,
and inverse transforms are engine business (V21pf). What you can control is
one level of intent:

## The default is usually right

```python
from itis_sumo import fit

model = fit(samples, ["width", "height"], "stress")
model.effective_config  # what was selected per column, and why
```

`analyze_dataset(samples)` shows the same reading before you fit: per-column
selected `scale` and `distribution`, a `confident` flag, and scale-aware
Tukey outlier fences (V46np) — presented as a table by
`itis_sumo.report.render_diagnostics`.

## Override in domain terms only

```python
from itis_sumo import PreprocessingSpec, VariableSpec, fit

log_height = PreprocessingSpec({"height": VariableSpec(scale="log")})
model = fit(samples, ["width", "height"], "stress", preprocessing=log_height)
```

`scale` describes the column; it does not name a transform. A `log` column is
fitted in ln space, sampled log-uniformly, drawn log-uniformly in Monte-Carlo
UQ, and reported exp-restored — everywhere, consistently (V45ls). Which
transform each override implies never appears in a signature; you can see it
in `model.effective_config`, but you cannot set transforms directly.

!!! warning "Say it once, mean it everywhere"
    The same spec travels with the handle into every query — cross-validation,
    Sobol' (log-uniform boxes), `sample_uncertainty` (log-uniform draws),
    `optimize` (log exploration, exp-restored front). A log reading that only
    half-arrived is exactly the failure the flip-matrix tests exist to catch.

!!! note "A log column must be positive"
    A `log`-scale column whose values or bounds include zero or less is a
    `SumoInputError`, everywhere the scale is consulted — no silent clipping.

## The engine layer, for the curious

Inside, a `DataPreprocessor` (internal, `itis_sumo.tabular`) maps your names
to Dakota's (`x1, x2, …` / `y1, y2, …`), applies the implied transforms, and
maps everything back — including through the fitted-state round-trip persisted
in a saved model (`model.save()` / `itis_sumo.load()`; the preprocessing
config travels with the model, T26eq). Consumer code never constructs it
(V50kv); the `PreprocessingSpec` is the door.
