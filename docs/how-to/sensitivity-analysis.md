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

sampling = {
    "length": {"minimum": 0.0, "maximum": 1.0},
    "width": {"minimum": 0.0, "maximum": 1.0},
}
result = evaluate_sobol_indices(
    run_dir,
    training_file,
    ["length", "width"],
    "y1",
    sampling,
    preprocessor,
    seed=42,
)
sobol = result["sobol"]  # {var: {"main", "total", "main_ci_low", ...}}
second_order = result["sobolSecondOrder"]  # {varA: {varB: float}}
masses = result["sobolOrderContributions"]  # M1/M2/R order masses or None

for var, indices in sobol.items():
    print(var, indices["main"], indices["total"])
```

`sampling` needs one entry per name in `input_vars`, each shaped as one
of (domain vocabulary only — V26dd: the sensitivity box is a DOMAIN, modeller
distributions belong to `evaluate_uncertainty`, never here):

- `{"minimum": ..., "maximum": ...}` — sampled uniformly across the box
- `{"minimum": ..., "maximum": ..., "log_scale": True}` — log-uniform draw
  (box must be strictly positive)
- `{"value": ...}` — held fixed; contributes zero variance and is excluded
  from the sampling budget

Costs `SOBOL_BASE_SAMPLES * (2 + d + d·(d−1))` surrogate evaluations
(`SOBOL_BASE_SAMPLES = 1024`), where `d` is the number of varying inputs —
the `d·(d−1)` term is the exact joint-pair second-order design, the
documented price of interactions that are EXACT for any input count. Set
variables you don't care about to `{"value": ...}` rather than leaving them
as boxes with a narrow range, since cost scales with count, not width.

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
  variables, `{varA: {varB: value}}`, from the EXACT joint-pair estimator
  (valid for any input count). Useful once `total - main` flags a
  variable as interaction-driven and you want to know *with which other
  variable*.
- **`sobolOrderContributions`** — the unique ANOVA order masses
  `first_order` (M1), `second_order` (M2) and the closure residual
  `third_and_higher` (R = 1 − M1 − M2), each with a shared-bootstrap CI and a
  rough `heuristic_noise_floor`; `None` when the output variance is zero
  (there is no partition to report).

## Propagate input uncertainty to output uncertainty

Sobol' indices tell you *which* inputs matter; if instead you want the
output distribution itself (mean/std of `y` given input distributions),
that's `propagate_uq` — a separate, Dakota-native pathway not currently
wired to the web UI. See
[Reference → Evaluate § Uncertainty propagation](../reference/evaluate.md#uncertainty-propagation)
for its signature and known caveats before using it.
