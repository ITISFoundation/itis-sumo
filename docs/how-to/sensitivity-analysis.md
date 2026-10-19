# How to run a sensitivity analysis

**Goal:** find out which inputs actually drive an output's variance, so you
know where to spend sampling budget and which variables are safe to fix.

!!! info "Key concept"
    Sobol' indices decompose output variance into per-input contributions —
    so you know where to spend sampling budget and which variables are safe to fix.

Assumes the training table and `SuMo` handle from
[Getting started](../tutorials/getting-started.md).

## Compute Sobol' indices

```python
from itis_sumo import DomainSpec, evaluate_sobol

result = evaluate_sobol(
    samples,
    ["length", "width"],
    "stress",
    domains={
        "length": DomainSpec(minimum=0.0, maximum=1.0),
        "width": DomainSpec(minimum=0.0, maximum=1.0),
    },
    seed=42,
)
for var, indices in result.indices.items():
    print(var, indices["main"], indices["total"])  # + main_ci_* / total_ci_*
result.second_order          # {varA: {varB: value}} exact joint pairs
result.order_contributions   # M1 / M2 / R masses, or None
```

On a fitted handle the same study is `model.sobol(domains=..., fixed=..., seed=...)`.

Domain vocabulary only — V26dd: the sensitivity box is a DOMAIN, modeller
distributions belong to uncertainty propagation, never here. Omit `domains`
and the observed bounds are used. A log-scaled variable (via the `preprocessing`
spec) is sampled log-uniformly across its box (V45ls); the box must be strictly
positive. Variables you don't care about go into `fixed={"var": value}` — they
contribute zero variance by construction and cost nothing — rather than into
boxes with a narrow range, since cost scales with count, not width.

Costs `SOBOL_BASE_SAMPLES * (2 + d + d·(d−1))` surrogate evaluations
(`SOBOL_BASE_SAMPLES = 1024`), where `d` is the number of varying inputs —
the `d·(d−1)` term is the exact joint-pair second-order design, the
documented price of interactions that are EXACT for any input count.

## Reading the indices

- **`main`** (first-order) — variance explained by that variable alone. A
  variable with `main ≈ 0` has no effect *on its own*.
- **`total`** — variance explained by that variable including all its
  interactions with others. `total > main` means interactions matter;
  `total ≈ main` means the variable acts independently.
- **`total - main` gap** — how much of that variable's influence is only
  visible through interaction with another input. A variable can have
  `main ≈ 0` but `total` well above zero — no effect alone, a real
  interaction effect (see the Ishigami `x3` case in the
  [V&V report](../verification-validation.md#ishigami-analytical-acceptance-gate)).
- **`main_ci_low` / `main_ci_high` / `total_ci_low` / `total_ci_high`** —
  bootstrap 95% confidence bounds, computed by resampling the already-run
  evaluations (no extra surrogate cost). Treat a `main` index as
  indistinguishable from zero if its CI straddles zero.
- **`second_order`** — pairwise interaction variance between two variables,
  from the EXACT joint-pair estimator (valid for any input count). Useful
  once `total - main` flags a variable as interaction-driven and you want to
  know *with which other variable*.
- **`order_contributions`** — the unique ANOVA order masses `first_order`
  (M1), `second_order` (M2) and the closure residual R = 1 − M1 − M2, each
  with a shared-bootstrap CI; `None` when the output variance is zero (no
  partition to report).

`itis_sumo.report.render_sobol` draws this whole result — indices, CI
whiskers — as one figure.

## Propagate input uncertainty to output uncertainty

Sobol' indices tell you *which* inputs matter over a domain; if instead you
want the output distribution itself given per-variable distributions you
believe, that's `sample_uncertainty` (or `model.sample_uncertainty`): it
draws from your declared normal/uniform/constant distributions, evaluates the
surrogate, injects its own predictive spread — and hands back the raw
material (V53md):

```python
from itis_sumo import DistributionSpec, sample_uncertainty, summarize_uncertainty

draws = sample_uncertainty(
    samples,
    ["length", "width"],
    "stress",
    distributions={
        "length": DistributionSpec("normal", mean=0.5, std=0.1),
        "width": DistributionSpec("uniform", minimum=0.0, maximum=1.0),
    },
    seed=42,
)
summary = summarize_uncertainty(draws)  # pure re-derivation, no second run
summary.median, summary.std
```

## …and correlate through the surrogate

The Monte-Carlo-through-surrogate correlation (the #470 workflow) is open
composition (V52rc), not a fused call: take the draws the propagation already
made, add the predictions, and correlate the table.

```python
import pandas as pd

from itis_sumo import compute_correlations

table = pd.DataFrame({**draws.draws, "stress": draws.predictions})
correlated = compute_correlations(table, ["length", "width"], "stress")
correlated.coefficients  # per variable: pearson + spearman
```

Pearson moves under a log reparametrization, Spearman cannot (V45ls) —
reading both is the scale-honesty check. `render_correlations` presents it.
