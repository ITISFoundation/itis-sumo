# Consumer API (`itis_sumo.api`)

`itis_sumo.api` is the whole of what an application embedding itis-sumo is expected to
import — a web service, a notebook, a script. Pass a table of samples and some
configuration; get back a typed result in your own units.

Everything in between belongs to itis-sumo and is not reachable from here: training-file
layout, Dakota configuration, normalization, internal variable renaming, run directories,
and inverse transforms.

```python
from itis_sumo.api import cross_validate

result = cross_validate(samples, variables=["width", "height"], response="stress")
result.predicted  # one value per sample, in the units of `stress`
result.warnings  # anything that went wrong but did not stop the run
```

The vocabulary is the one defined in the [Glossary](glossary.md): a **sample** is a row,
a **variable** (parameter) is an input column, a **response** (quantity of interest) is
an output column.

## Workflows

### `cross_validate`

Predicts every sample from a surrogate that never saw it. The samples are split into
folds; each fold is predicted by a surrogate trained on the others, so the result is an
honest picture of how the surrogate behaves on data it has not memorised.

Returns a `CrossValidationResult`: the observed values, the predicted values and their
standard deviations aligned positionally with your rows, plus any warnings.

If Dakota abandons a fold — which it occasionally does when the training points are
nearly degenerate — the samples in that fold keep a `NaN` prediction and the reason
appears in `warnings`, rather than the whole run failing.

### `evaluate_along_axes`

Sweeps each variable in turn across its observed range while every other variable is held
still. This is the shape behind a one-dimensional profile plot.

Returns an `AlongAxesResult` containing one `AxisSweep` per variable, keyed by the
variable's own name. Use `at=` to pin particular variables; anything you leave out is held
at its mean across the samples.

## Configuration

Sensible defaults are derived from your samples, so the common path needs no
configuration at all. Advanced users may override behaviour per column with
`PreprocessingSpec`:

```python
from itis_sumo.api import PreprocessingSpec, VariableSpec

spec = PreprocessingSpec(overrides={"width": VariableSpec(scale="linear")})
```

Overrides are expressed in terms of the *domain* — what a column is like — never in terms
of a transform. Which normalization or sign convention a choice implies is an
implementation detail. You can always read back what was actually used, via
`result.effective_config`; you cannot set it in transform terms.

Stochastic workflows take a `seed` that already has a value, so results are reproducible
by default without you having to think about it. The seed that was used comes back in the
result.

## Errors

Every error that escapes `itis_sumo.api` is a `SumoError`. Catch the subclass rather than
matching on message text.

| Error | Meaning | Typical HTTP status |
|---|---|---|
| `SumoInputError` | The samples or configuration are unusable. Fixable by sending something different. | 400 |
| `SumoResultError` | The run finished but produced none of the requested values. Retrying unchanged will not help. | 422 |
| `SumoEngineError` | Dakota itself failed. | 500 |

Dakota reports its failures opaquely. Because of that, working files are discarded when a
run succeeds but **kept when one fails** — and `SumoEngineError` carries the surviving run
directory along with the tail of Dakota's own stderr, so there is something to look at.

```python
try:
    result = cross_validate(samples, variables, response)
except SumoInputError as error:
    return bad_request(str(error))
except SumoEngineError as error:
    logger.error("surrogate build failed; evidence at %s", error.run_dir)
    return server_error("Could not build a surrogate from these samples")
```

Pass `workspace=` to keep working files from a successful run too, when you want to
inspect what Dakota was given.
