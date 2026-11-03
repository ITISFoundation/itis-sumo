"""Tabular narrative renderings (V54hz): one function result in, one DataFrame out.

The narrative half of a V&V report is prose around NUMBERS the producing
function already computed. Each `render_*` here lifts exactly one result's
fields into a pandas DataFrame a notebook can print or write verbatim; no
statistics are re-derived, no engine call happens, and units/names are the
caller's own. Escape hatch: the returned frame (`df.to_csv(...)`,
`df.to_markdown()`, `df.to_html(...)`).
"""

from __future__ import annotations

import pandas as pd

from itis_sumo.types import (
    CVAccuracyMetrics,
    DatasetDiagnostics,
    UncertaintySummary,
)

__all__ = [
    "render_convergence_fit",
    "render_cv_accuracy",
    "render_cv_diagnostics",
    "render_diagnostics",
    "render_uncertainty_bins",
    "render_uncertainty_summary",
]

_SCALAR_TYPES = (int, float, bool)


def render_cv_accuracy(result: CVAccuracyMetrics, **kwargs) -> pd.DataFrame:
    """The accuracy-metric row: one line, the result's fields verbatim.

    kwargs pass through to the `DataFrame` constructor; escape hatch:
    `df.to_markdown()` for a report cell, `df.to_csv()` for the archive.
    """
    row = {
        "response": result.response,
        "seed": result.seed,
        "rmse": result.root_mean_squared,
        "sum_abs": result.sum_abs,
        "mean_abs": result.mean_abs,
        "max_abs": result.max_abs,
        "mean_signed_error": result.mean_signed_error,
    }
    return pd.DataFrame([row], **kwargs)


def render_cv_diagnostics(diagnostics: dict, **kwargs) -> pd.DataFrame:
    """Scalar bundle of one CV subset (the `compute_cv_diagnostics` result).

    Only scalar fields are tabulated (accuracy, Cohen's dz, t-test, outlier
    count); the filtered `actual`/`predicted` pairs stay in the dict — they
    belong to `render_cross_validation`'s data, not the narrative row.

    kwargs pass through to the `DataFrame` constructor; escape hatch: the
    frame (`df.to_csv(...)`, `df.to_markdown()`, `df.to_html(...)`).
    """
    scalars = {
        key: value
        for key, value in diagnostics.items()
        if isinstance(value, _SCALAR_TYPES)
    }
    return pd.DataFrame([scalars], **kwargs)


def render_convergence_fit(fit: dict, **kwargs) -> pd.DataFrame:
    """The exponential-fit row (`fit_convergence_exponential` result).

    Carries `a`, `b`, `r_squared` — and `c` for an `asymptotic=True` fit,
    which is the irreducible-error floor (V18wp).

    kwargs pass through to the `DataFrame` constructor; escape hatch: the
    frame (`df.to_csv(...)`, `df.to_markdown()`, `df.to_html(...)`).
    """
    return pd.DataFrame([fit], **kwargs)


def render_diagnostics(result: DatasetDiagnostics, **kwargs) -> pd.DataFrame:
    """Per-variable diagnostic rows: selected scale, distribution, outliers.

    One row per column (inputs then outputs); outlier fences are the
    scale-aware Tukey fences the diagnostics function selected (V46np),
    reported as computed.

    kwargs pass through to the `DataFrame` constructor; escape hatch: the
    frame (`df.to_csv(...)`, `df.to_markdown()`, `df.to_html(...)`).
    """
    rows = []
    for kind, group in (("input", result.inputs), ("output", result.outputs)):
        for name, variable in group.items():
            rows.append(
                {
                    "column": name,
                    "kind": kind,
                    "scale": variable.scale,
                    "distribution": variable.distribution,
                    "confident": variable.confident,
                    "outlier_count": (
                        variable.outliers.count if variable.outliers else 0
                    ),
                    "fence_low": (
                        variable.outliers.fence_low if variable.outliers else None
                    ),
                    "fence_high": (
                        variable.outliers.fence_high if variable.outliers else None
                    ),
                }
            )
    return pd.DataFrame(rows, **kwargs)


def render_uncertainty_summary(result: UncertaintySummary, **kwargs) -> pd.DataFrame:
    """The boxplot/percentile row of a propagated-uncertainty summary.

    Every number is `summarize_uncertainty`'s own; the frame is a verbatim
    narrative lift, not a recomputation.

    kwargs pass through to the `DataFrame` constructor; escape hatch: the
    frame (`df.to_csv(...)`, `df.to_markdown()`, `df.to_html(...)`).
    """
    row = {
        "response": result.response,
        "seed": result.seed,
        "q1": result.q1,
        "median": result.median,
        "q3": result.q3,
        "whisker_min": result.whisker_min,
        "whisker_max": result.whisker_max,
        "n_outliers": len(result.outliers),
        "mean": result.mean,
        "std": result.std,
        "min": result.minimum,
        "max": result.maximum,
    }
    return pd.DataFrame([row], **kwargs)


def render_uncertainty_bins(result: UncertaintySummary, **kwargs) -> pd.DataFrame:
    """Per-bin means/stds of the same summary, one row per histogram bin.

    Bin positions are the summary's own `bins_start`/`bins_end` endpoints and
    the bin count — layout of a table, not a statistic.

    kwargs pass through to the `DataFrame` constructor; escape hatch: the
    frame (`df.to_csv(...)`, `df.to_markdown()`, `df.to_html(...)`).
    """
    n_bins = len(result.bin_means)
    if n_bins == 0:
        return pd.DataFrame(
            {"bin": [], "bin_start": [], "bin_end": [], "mean": [], "std": []}
        )
    edges = [
        result.bins_start + i * (result.bins_end - result.bins_start) / n_bins
        for i in range(n_bins + 1)
    ]
    frame = pd.DataFrame(
        {
            "bin": list(range(n_bins)),
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "mean": result.bin_means,
            "std": result.bin_stds,
        },
        **kwargs,
    )
    return frame
