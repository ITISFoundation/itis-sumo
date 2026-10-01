"""Report-authoring surface (V49rp): the curated entrypoints automated report
notebooks (marimo, T51np pattern; tissue-conductivity LHS instance, T52rv)
build on — the report-facing counterpart of `itis_sumo.api` (which serves the
flaskapi/mmux_vite request-response consumer, V16qf).

One flat import (`from itis_sumo.report import ...`) for the whole authoring
arc: dataset diagnostics -> CV validation & calibration (coverage, Cohen's dz,
bootstrap convergence sweep, asymptotic error fits) -> sensitivity & correlation
-> manual UQ propagation. Re-exports only: no logic lives here, so engine
refactors stay invisible to report code as long as this surface holds.
"""

from itis_sumo.api import (
    DEFAULT_SEED,
    DistributionSpec,
    compute_correlations,
    cross_validate,
    evaluate_along_axes,
    evaluate_correlations,
    evaluate_sobol,
)
from itis_sumo.data import analyze_dataset
from itis_sumo.evaluate import (
    compute_coverage,
    compute_cv_accuracy_metrics,
    compute_cv_convergence,
    compute_cv_diagnostics,
    compute_paired_ttest,
    fit_convergence_exponential,
    fit_convergence_exponential_asymptotic,
)
from itis_sumo.evaluate.funs_evaluate import propagate_manual_uq_with_uncertainty

__all__ = [
    "DEFAULT_SEED",
    "DistributionSpec",
    "analyze_dataset",
    "compute_correlations",
    "compute_coverage",
    "compute_cv_accuracy_metrics",
    "compute_cv_convergence",
    "compute_cv_diagnostics",
    "compute_paired_ttest",
    "cross_validate",
    "evaluate_along_axes",
    "evaluate_correlations",
    "evaluate_sobol",
    "fit_convergence_exponential",
    "fit_convergence_exponential_asymptotic",
    "propagate_manual_uq_with_uncertainty",
]
