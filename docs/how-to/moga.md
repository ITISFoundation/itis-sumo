# How to find a Pareto front with MOGA

**Goal:** given training data and one or more objectives, find the set of
non-dominated trade-off points (single objective → a single optimum;
multiple objectives → a Pareto front).

Assumes a training table in the shape
[Getting started](../tutorials/getting-started.md) builds.

## Run the optimization

```python
from itis_sumo import DomainSpec, optimize

front = optimize(
    samples,
    ["length", "width"],
    {"stress": "minimize", "mass": "minimize"},
    domains={
        "length": DomainSpec(minimum=0.0, maximum=1.0),
        "width": DomainSpec(minimum=0.0, maximum=1.0),
    },
    max_evaluations=1000,
)
front.data        # objective + variable columns, original units
front.objectives  # {"stress": "minimize", "mass": "minimize"}
```

One surrogate is fitted per objective over the domains — MOGA explores a
**uniform domain**, never a real-world distribution (T27fr): it walks a box
looking for the best point, and a box is all it can use. A `scale="log"`
override on a variable explores it in log space; on an objective it fits in
ln space and reports the front exp-restored, in your units (V45ls).
`itis_sumo.report.render_pareto` draws the front with every point
individually readable.

!!! warning "Domains, not distributions"
    Anything non-uniform is a caller error: `optimize` takes `DomainSpec`
    boxes precisely so a "distribution" cannot sneak into a search it cannot
    honor.

## Tuning

| Lever | Notes |
|---|---|
| `max_evaluations` | Surrogate evaluation budget — the cost control |
| `domains` | Tighten to the region you would actually build in; cost is per-evaluation, width is free |
| `seed` | Reproducibility (V3er) |

## Reading the front

- **Single objective**: the front degenerates toward the single optimum as
  the evaluation budget grows.
- **Multiple objectives**: every point on the returned front is
  non-dominated by construction (verified in V&V Category G) — no point is
  simultaneously worse on every objective than another returned point.
  There's no single "best" point; which one to pick is a decision about your
  objectives' relative priority, not something MOGA can answer.
- **More evaluations → a better (dominating) front**, not just a bigger
  one — if you're unsure the front has converged, rerun with a higher
  `max_evaluations` and check whether the new front dominates the old one.
- All returned points respect the `domains` bounds by construction.
