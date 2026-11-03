# Reference overview

The public surface is small on purpose (V50kv): the root manifest
`itis_sumo.__all__`, the methods on the `SuMo` handle, and the rendering
namespace `itis_sumo.report`. The pages on the left are generated from the
function docstrings themselves — the docstring is the visible product surface,
so these pages cannot drift from the code.

## The three doors

| Door | What it is | Where |
|---|---|---|
| `fit` / `load` → `SuMo` | the fitted-model handle; its **methods are the catalog** of everything you can ask a trained surrogate | [SuMo handle](su-mo.md) |
| one-shot functions | fit-and-forget compositions per module (≤2 public functions each) | [Function modules](functions.md) |
| `itis_sumo.report` | renderings: one function result in, one plotly Figure or pandas DataFrame out | [Report renderings](report.md) |

```python
from itis_sumo import fit

model = fit(samples, ["width", "height"], "stress")   # -> SuMo handle
cv = model.cross_validate(folds=5, seed=42)           # a typed result
from itis_sumo.report import render_cross_validation
render_cross_validation(cv)                           # a deletable figure
```

Everything the result carries is in your units under your column names
(V22rs); everything that happened inside — Dakota names, normalization, run
directories — is engine business ([Engine internals](engine.md)).

## Vocabulary

One word per concept, everywhere. The [Glossary](glossary.md) is the
authority — and the spec's VOCAB block is kept in sync with it: a **sample**
is a row, a **variable** is an input column, a **response** an output column,
a **catalog** is what a fitted `SuMo` can answer, and a **rendering** is the
presentation of exactly one result.
