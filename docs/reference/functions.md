# Function modules

The one-shot door: each module carries at most two public functions (V50kv) —
either a fit-and-forget composition over the `SuMo` handle or pure maths over
caller-owned data. Generated from the docstrings; `itis_sumo.__all__` is the
manifest the tests pin.

## validation

::: itis_sumo.validation

## sensitivity

::: itis_sumo.sensitivity

## uncertainty

::: itis_sumo.uncertainty

## exploration

::: itis_sumo.exploration

## sampling

::: itis_sumo.sampling

## optimization

::: itis_sumo.optimization

## diagnostics

::: itis_sumo.diagnostics

## calibration

::: itis_sumo.calibration

## convergence

::: itis_sumo.convergence

!!! note "Convergence plumbing"
    `compute_cv_convergence` re-runs cross-validation per subset size, so it
    operates on an explicit run directory and training file rather than a
    `SuMo` handle. `fit(..., workspace=...)` is the supported way to keep
    those files.
