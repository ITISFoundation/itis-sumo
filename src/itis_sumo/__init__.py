"""itis-sumo: headless core of the MetaModeling tools.

The root package IS the consumer manifest (SPEC V50kv, §I target): import
functions, spec/result types and the error taxonomy from here, the methods from
a fitted SuMo handle, and report renderings from `itis_sumo.report`.
Everything else is itis-sumo's business.

    >>> from itis_sumo import cross_validate
    >>> result = cross_validate(samples, variables=["width", "height"], response="stress")
    >>> result.predicted[:3]

Vocabulary follows SPEC VOCAB throughout: a *sample* is a row, a *variable*
(parameter) is an input column, a *response* (quantity of interest) is an output
column. See the Glossary in the documentation.
"""

from importlib.metadata import version as package_version

__version__ = package_version("itis-sumo")

from itis_sumo.calibration import compute_coverage, compute_cv_diagnostics
from itis_sumo.convergence import (
    compute_cv_convergence,
    fit_convergence_exponential,
    fit_convergence_exponential_asymptotic,
)
from itis_sumo.diagnostics import analyze_dataset, detect_raw_outliers
from itis_sumo.errors import (
    SumoEngineError,
    SumoError,
    SumoInputError,
    SumoResultError,
)
from itis_sumo.exploration import evaluate_along_axes, evaluate_grid
from itis_sumo.optimization import optimize
from itis_sumo.sampling import generate_grid_samples, generate_lhs_samples
from itis_sumo.sensitivity import compute_correlations, evaluate_sobol
from itis_sumo.types import (
    DEFAULT_SEED,
    AlongAxesResult,
    AxisSweep,
    CorrelationResult,
    CrossValidationResult,
    CVAccuracyMetrics,
    DatasetDiagnostics,
    Direction,
    DistributionSpec,
    DomainSpec,
    GridResult,
    OrderMasses,
    OutlierSummary,
    ParetoFrontResult,
    PreprocessingSpec,
    Scale,
    SobolResult,
    UncertaintyResult,
    VariableDiagnostics,
    VariableSpec,
)
from itis_sumo.uncertainty import evaluate_correlations, evaluate_uncertainty
from itis_sumo.validation import cross_validate, evaluate_cv_metrics

__all__ = [
    "DEFAULT_SEED",
    "AlongAxesResult",
    "AxisSweep",
    "CVAccuracyMetrics",
    "CorrelationResult",
    "CrossValidationResult",
    "DatasetDiagnostics",
    "Direction",
    "DistributionSpec",
    "DomainSpec",
    "GridResult",
    "OrderMasses",
    "OutlierSummary",
    "ParetoFrontResult",
    "PreprocessingSpec",
    "Scale",
    "SobolResult",
    "SumoEngineError",
    "SumoError",
    "SumoInputError",
    "SumoResultError",
    "UncertaintyResult",
    "VariableDiagnostics",
    "VariableSpec",
    "__version__",
    "analyze_dataset",
    "compute_correlations",
    "compute_coverage",
    "compute_cv_convergence",
    "compute_cv_diagnostics",
    "cross_validate",
    "detect_raw_outliers",
    "evaluate_along_axes",
    "evaluate_correlations",
    "evaluate_cv_metrics",
    "evaluate_grid",
    "evaluate_sobol",
    "evaluate_uncertainty",
    "fit_convergence_exponential",
    "fit_convergence_exponential_asymptotic",
    "generate_grid_samples",
    "generate_lhs_samples",
    "optimize",
]
