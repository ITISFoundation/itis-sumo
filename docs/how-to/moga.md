# How to find a Pareto front with MOGA

**Goal:** given a fitted surrogate and one or more objectives to minimize,
find the set of non-dominated trade-off points (single objective → a single
optimum; multiple objectives → a Pareto front).

Assumes a training file in the shape
[Getting started](../tutorials/getting-started.md) builds. MOGA searches
over **uniform-distribution bounds only** — see below.

## Run the optimization

```python
from itis_sumo.evaluate.funs_evaluate import perform_moga_optimization

distributions = {
    "length": {"distribution": "uniform", "min": 0.0, "max": 1.0},
    "width": {"distribution": "uniform", "min": 0.0, "max": 1.0},
}
moga_kwargs = {
    "populationSize": 32,
    "maxIterations": 100,
    "seed": 42,
}
pareto = perform_moga_optimization(
    run_dir,
    training_file,
    ["length", "width"],
    distributions,
    ["y1"],
    moga_kwargs,
)
# pareto["y1"] — objective values on the front
# pareto["length"], pareto["width"] — the input points that produced them
```

!!! warning "Uniform bounds only"
    MOGA raises `ValueError` on any non-uniform distribution. Use
    `{"distribution": "uniform", ...}` for every input; pass several outputs
    for a true multi-objective Pareto front.

## `moga_kwargs` options

| Key | Default | Notes |
|---|---|---|
| `populationSize` | 32 | JEGA initial population size |
| `maxIterations` | 100 | Generation budget — the actual evaluation-count control |
| `fitnessType` | `"layer_rank"` | or `"domination_count"` |
| `replacementType` | `"elitist"` | or `"unique_roulette_wheel"`, `"below_limit"` |
| `seed` | 12345 | reproducibility |
| `max_function_evaluations` | — | **accepted but silently ignored** — deprecated on the Dakota side; use `maxIterations` to bound cost, not this |

## Reading the front

- **Single objective**: `perform_moga_optimization` still returns the same
  shape; treat the returned point(s) as converging toward the single
  optimum as `maxIterations` grows.
- **Multiple objectives**: every point in the returned front is
  non-dominated by construction (verified in V&V Category G) — no point is
  simultaneously worse on every objective than another returned point.
  There's no single "best" point; which one to pick is a decision about
  your objectives' relative priority, not something MOGA can answer.
- **More iterations → a better (dominating) front**, not just a bigger one —
  if you're unsure the front has converged, rerun with a higher
  `maxIterations` and check whether the new front dominates the old one.
- All returned points respect the `distributions` bounds by construction.

Details on the underlying Pareto-dominance utilities and the
`max_function_evaluations` limitation:
[Reference → MOGA optimization](../reference/moga.md).
