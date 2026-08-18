"""The itis-sumo consumer API.

This is the whole of what an application embedding itis-sumo -- a web service, a
notebook, a script -- is expected to import. Pass a table of samples and some
configuration; get back a typed result in your own units. Everything else is
itis-sumo's business (SPEC V16qf, §G).

    >>> from itis_sumo.api import cross_validate
    >>> result = cross_validate(samples, variables=["width", "height"], response="stress")
    >>> result.predicted[:3]

Vocabulary follows SPEC VOCAB throughout: a *sample* is a row, a *variable*
(parameter) is an input column, a *response* (quantity of interest) is an output
column. See the Glossary in the documentation.
"""

from itis_sumo.api.errors import (
    SumoEngineError,
    SumoError,
    SumoInputError,
    SumoResultError,
)
from itis_sumo.api.types import (
    DEFAULT_SEED,
    AlongAxesResult,
    AxisSweep,
    CorrelationResult,
    CrossValidationResult,
    CVAccuracyMetrics,
    DistributionSpec,
    GridResult,
    PreprocessingSpec,
    Scale,
    SobolResult,
    VariableSpec,
)
from itis_sumo.api.workflows import (
    compute_correlations,
    cross_validate,
    evaluate_along_axes,
    evaluate_cv_metrics,
    evaluate_grid,
    evaluate_sobol,
)

__all__ = [
    "DEFAULT_SEED",
    "AlongAxesResult",
    "AxisSweep",
    "CVAccuracyMetrics",
    "CorrelationResult",
    "CrossValidationResult",
    "DistributionSpec",
    "GridResult",
    "PreprocessingSpec",
    "Scale",
    "SobolResult",
    "SumoEngineError",
    "SumoError",
    "SumoInputError",
    "SumoResultError",
    "VariableSpec",
    "compute_correlations",
    "cross_validate",
    "evaluate_along_axes",
    "evaluate_cv_metrics",
    "evaluate_grid",
    "evaluate_sobol",
]
