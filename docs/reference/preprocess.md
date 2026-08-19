# Data preprocessing

`itis_sumo.preprocess` -- an optional layer that sits *before* training and
*after* prediction.

## `DataPreprocessor` (`data_preprocessor.py`)

Handles, in one reusable object:

- **Variable mapping** -- standardized names (`x1, x2, ...` for inputs;
  `y1, y2, ...` for outputs), so downstream Dakota confs never see raw,
  possibly-unsafe user variable names.
- **Normalization / denormalization** -- min-max or z-score scaling, applied
  before training and inverted after prediction.
- **Sign switching** -- restorable sign flips (useful for
  maximize-vs-minimize framing without re-deriving a surrogate).
- **Configuration persistence** -- the applied transform is serialized to
  JSON alongside the run, so predictions can be inverse-transformed later
  by a different process/session without re-deriving the mapping.

## `required_completed_jobs` (`models.py`)

Minimum completed-sample-count check for building a Dakota (surfpack) GP
surrogate -- `max(floor, len(input_vars) + 1)`, since Dakota aborts surrogate
construction when given `<= len(input_vars)` training points.

## Verified behavior

(Verification & Validation Category H) -- transform/inverse-transform
round-trips to `1e-10` for z-score, min-max, and sign-switch, including on
a 1000-row x 20-variable dataset; and normalization measurably improves
surrogate accuracy on badly-scaled inputs (e.g. `f(x) = 1000x + 1`) where
an unnormalized fit degrades.
