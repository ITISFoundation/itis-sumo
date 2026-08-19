# Glossary

itis-sumo uses one word per concept, everywhere: in the public API, in these docs, in
docstrings, in tests. If you see a synonym, it is drift and it is a bug.

## Data shape

| Term | Means | Not |
|---|---|---|
| **sample** | One row of tabular data — a single observation. | "job", "point", "record" |
| **variable**, **parameter** | One *input* column. | "feature", "factor" |
| **response**, **quantity of interest (QoI)** | One *output* column. | "target", "label" |

`variable` and `parameter` are interchangeable, as are `response` and `quantity of
interest`. Inputs and outputs stay distinguished throughout, because they receive
different downstream treatment — normalization, sign handling, sampling roles, and
their position in a Sobol' decomposition all depend on which side a column is on.

itis-sumo takes **plain tabular data**: a `pandas.DataFrame`, or an itis-sumo dataclass
that converts trivially to one. It has no notion of an oSPARC `FunctionJob`. Converting
job records into a table — and filtering out the ones that did not complete — belongs to
the consumer. Deciding that the resulting table has too few samples to build a surrogate
belongs to itis-sumo, which raises so the consumer can surface the problem.

## Two things that are never the same thing

| Term | Means | Can it be inferred from your samples? |
|---|---|---|
| **domain** | Where a variable may be drawn from or explored — its bounds and scale. A property of the *design space*. | **Yes.** Observed bounds plus a detected scale are a defensible default. |
| **distribution** | The real-world shape of a variable. A claim about *the world*. | **No.** A training set sampled uniformly over a box tells you nothing about whether the real input is normal. |

Conflating these is the single easiest way to produce a confidently wrong uncertainty
band. Sobol' analysis and MOGA optimization want a **domain** — a region to explore.
Uncertainty propagation and correlation analysis want a **distribution** — something to
draw from. They are configured separately.

## Modeling

| Term | Means |
|---|---|
| **surrogate**, **SuMo** | The trained metamodel that stands in for an expensive simulation. |
| **surrogate model ID** (`sumo_model_id`) | The server-minted UUID identifying a persisted surrogate in the model store. |

## What you pass, and what stays hidden

You pass **samples** and **configuration**. Everything else is itis-sumo's business:
training-file layout, Dakota configuration, normalization, sign handling, internal
variable renaming, run directories, and mapping predictions back to your original units.

Sensible defaults are generated for you, so the common path requires no configuration at
all. Advanced users may override them — but only in terms of the *domain*, saying things
like `scale="log"` or `direction="maximize"`. The transforms those choices imply are an
implementation detail and never appear in a signature. You can always read back the
configuration that was actually used; you cannot set it in transform terms.
