"""Report renderings: one function result in, a figure or table out (V54hz).

The report namespace is where a V&V story gets told. Each RENDERING consumes
exactly ONE function result — a typed result from the root manifest or one
documented producer dict — never runs the engine, and never recomputes a
statistic: it presents what the producing function already decided, in the
caller's original units, with hover carrying per-sample traceability.
Everything here is a deletable "serving suggestion", never the contract:
kwargs pass through to the plotly layout (figures) or the DataFrame
constructor (tables), and the escape hatch is always the returned object
(`fig.write_html` / `df.to_csv`).

`sections()` is the catalog: every report section, the rendering it names,
and the result it consumes. Automated notebooks pair data cells (root
manifest functions, via `fit`/`load`) with rendering cells (this namespace) —
see `examples/report_notebook_template.py` for the pattern with a version
stamp.

plotly (and marimo, for the template) are optional dependencies:
`pip install itis-sumo[report]` (V54hz: presentation deps never sit in core,
and core never imports this package).
"""

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
from itis_sumo.report.sections import Section, sections
from itis_sumo.report.text import (
    render_convergence_fit,
    render_cv_accuracy,
    render_cv_diagnostics,
    render_diagnostics,
    render_uncertainty_bins,
    render_uncertainty_summary,
)

__all__ = [
    "Section",
    "render_along_axes",
    "render_convergence",
    "render_convergence_fit",
    "render_correlations",
    "render_coverage",
    "render_cross_validation",
    "render_cv_accuracy",
    "render_cv_diagnostics",
    "render_diagnostics",
    "render_grid",
    "render_pareto",
    "render_predictions",
    "render_sobol",
    "render_uncertainty",
    "render_uncertainty_bins",
    "render_uncertainty_summary",
    "sections",
]
