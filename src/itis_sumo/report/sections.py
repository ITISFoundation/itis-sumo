"""`sections()` — the report catalog (V50kv: queryable from the namespace).

One entry per report section: the rendering, the result type it consumes, and
a sentence of intent. Automated notebooks iterate the catalog (or import
renderings directly); the catalog is the authority on which renderings exist
and what each one eats.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from itis_sumo.report.figures import (
    render_along_axes,
    render_convergence,
    render_correlations,
    render_coverage,
    render_cross_validation,
    render_grid,
    render_pareto,
    render_predictions,
    render_sobol,
    render_uncertainty,
)
from itis_sumo.report.text import (
    render_convergence_fit,
    render_cv_accuracy,
    render_cv_diagnostics,
    render_diagnostics,
    render_uncertainty_bins,
    render_uncertainty_summary,
)

__all__ = ["Section", "sections"]


@dataclass(frozen=True)
class Section:
    """One report section: a rendering, the result it consumes, a sentence."""

    name: str
    rendering: Callable[..., object]
    consumes: str
    description: str


_SECTIONS: tuple[Section, ...] = (
    Section(
        "diagnostics",
        render_diagnostics,
        "DatasetDiagnostics",
        "what the dataset healer selected per column (scale, distribution, outliers)",
    ),
    Section(
        "cross_validation",
        render_cross_validation,
        "CrossValidationResult",
        "held-out predictions against observations, per sample",
    ),
    Section(
        "cv_accuracy",
        render_cv_accuracy,
        "CVAccuracyMetrics",
        "the accuracy row for the record",
    ),
    Section(
        "coverage",
        render_coverage,
        "compute_coverage() result",
        "does the surrogate's own spread earn trust (V19cz)",
    ),
    Section(
        "cv_diagnostics",
        render_cv_diagnostics,
        "compute_cv_diagnostics() result",
        "one-pass statistical bundle: effect size, t-test, outlier count",
    ),
    Section(
        "convergence",
        render_convergence,
        "compute_cv_convergence() result",
        "does more data still help, with its bootstrap spread visible",
    ),
    Section(
        "convergence_fit",
        render_convergence_fit,
        "fit_convergence_exponential() result",
        "the fitted decay — floor included when the fit is asymptotic (V18wp)",
    ),
    Section(
        "sobol",
        render_sobol,
        "SobolResult",
        "variance decomposition over the sampled domain, CIs attached",
    ),
    Section(
        "correlations",
        render_correlations,
        "CorrelationResult",
        "scale-honest sensitivity read: Pearson and Spearman side by side",
    ),
    Section(
        "along_axes",
        render_along_axes,
        "AlongAxesResult",
        "response profile per variable, all others held central",
    ),
    Section(
        "grid",
        render_grid,
        "GridResult",
        "every grid point kept, no smoothed surfaces",
    ),
    Section(
        "uncertainty",
        render_uncertainty,
        "UncertaintyDraws",
        "the propagated cloud — draws, not a pre-chewed summary",
    ),
    Section(
        "uncertainty_summary",
        render_uncertainty_summary,
        "UncertaintySummary",
        "percentile row of the propagated spread",
    ),
    Section(
        "uncertainty_bins",
        render_uncertainty_bins,
        "UncertaintySummary",
        "per-bin table for the methods appendix",
    ),
    Section(
        "predictions",
        render_predictions,
        "PredictionsResult",
        "prediction rows with their spread, from a live or loaded SuMo",
    ),
    Section(
        "pareto",
        render_pareto,
        "ParetoFrontResult",
        "the front itself, each point individually readable",
    ),
)


def sections() -> dict[str, Section]:
    """The ordered catalog of report sections, keyed by name."""
    return {section.name: section for section in _SECTIONS}
