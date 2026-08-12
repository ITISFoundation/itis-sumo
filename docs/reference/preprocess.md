# Data preprocessing

`itis_sumo.preprocess` — an optional layer that sits *before* training and
*after* prediction. It is the one part of the package that carries the
pydantic job models originating in the web layer (`FunctionJob`,
`JobVariableSelection`), moved in from `mmux_flaskapi` and de-webbed
(`V4ty`: no flask/oSPARC imports even here — just the pydantic models
themselves).

## `DataPreprocessor` (`data_preprocessor.py`)

Handles, in one reusable object:

- **Variable mapping** — standardized names (`x1, x2, …` for inputs;
  `y1, y2, …` for outputs), so downstream Dakota confs never see raw,
  possibly-unsafe user variable names.
- **Normalization / denormalization** — min-max or z-score scaling, applied
  before training and inverted after prediction.
- **Sign switching** — restorable sign flips (useful for
  maximize-vs-minimize framing without re-deriving a surrogate).
- **Configuration persistence** — the applied transform is serialized to
  JSON alongside the run, so predictions can be inverse-transformed later
  by a different process/session without re-deriving the mapping.

## Integration helpers (`data_preprocessor_integration.py`)

- `create_training_file_with_preprocessor` — builds a training file
  through the preprocessor rather than raw data, so the surrogate is
  trained in normalized space.
- `setup_preprocessor_from_config` / `get_preprocessing_summary` — round-trip
  a preprocessor from its persisted JSON config.
- `load_and_inverse_transform_results` — the inverse leg: read raw
  surrogate predictions and map them back to original variable
  names/scale.
- `create_filtered_preprocessor`, `get_variable_statistics`,
  `filter_variables_by_statistics` — support restricting preprocessing to
  a subset of variables, driven by their statistics (e.g. drop
  near-constant columns).

## Verified behavior

(Verification & Validation Category H) — transform/inverse-transform
round-trips to `1e-10` for z-score, min-max, and sign-switch, including on
a 1000-row × 20-variable dataset; and normalization measurably improves
surrogate accuracy on badly-scaled inputs (e.g. `f(x) = 1000x + 1`) where
an unnormalized fit degrades.
