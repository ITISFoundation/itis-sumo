"""Plotly figure renderings (V54hz): one function result in, one Figure out.

Every `render_*` here consumes exactly ONE result object (a typed dataclass
from the root manifest or one documented producer dict), never runs the
engine, and never recomputes a statistic -- it plots what the producing
function already computed, in the original units the caller passed in, and
hover carries per-sample traceability (the sample row index) wherever a point
maps to a row. kwargs pass through to the plotly LAYOUT
(`fig.update_layout(**kwargs)`); when the suggestion is wrong the escape
hatch is the Figure itself: `fig.data`, `fig.write_html(...)`,
`fig.to_image(...)`.

plotly is an optional dependency: `pip install itis-sumo[report]` (V54hz:
presentation deps never sit in core, and core never imports this package).
"""

from __future__ import annotations

import math

import plotly.graph_objects as go

from itis_sumo.types import (
    AlongAxesResult,
    CorrelationResult,
    CrossValidationResult,
    GridResult,
    ParetoFrontResult,
    PredictionsResult,
    SobolResult,
    UncertaintyDraws,
)

__all__ = [
    "render_along_axes",
    "render_convergence",
    "render_correlations",
    "render_coverage",
    "render_cross_validation",
    "render_grid",
    "render_pareto",
    "render_predictions",
    "render_sobol",
    "render_uncertainty",
]

# Presentation for the metric family `compute_cv_convergence` returns, keyed by
# its metric tokens: the wording of the axis the metric sits on, and for a
# coverage token the nominal level its curve is read against (a definition, like
# render_cross_validation's parity line — layout decoration, not a statistic, and
# the numbers themselves stay the engine's: V54hz). A token this table does not
# know still plots, labelled by its own name, so an engine-side metric added
# later never renders as a mystery axis.
_METRIC_PRESENTATION: dict[str, tuple[str, float | None]] = {
    "rmse": ("RMSE (original units)", None),
    "mae": ("MAE (original units)", None),
    "bias": ("signed bias (original units)", None),
    "coverage68": ("68% interval coverage (fraction)", 0.68),
    "coverage95": ("95% interval coverage (fraction)", 0.95),
    "coverage99": ("99% interval coverage (fraction)", 0.99),
}


def _convergence_point_metric(entry: dict, metric: str) -> tuple[float, float, list]:
    """`(value, spread, per-draw cloud)` for one convergence point's `metric`.

    A point from `compute_cv_convergence` carries the whole metric family; a
    hand-written point, or one from a build that predates the family maps, carries
    only the single historical RMSE reading — which is what `metric`/`metric_std`/
    `draws` always meant, so those still render (as RMSE) rather than silently
    becoming something else.
    """
    if "metrics" in entry:
        if metric not in entry["metrics"]:
            raise ValueError(
                f"this convergence series has no {metric!r} metric; it has "
                f"{', '.join(sorted(entry['metrics']))}"
            )
        return (
            entry["metrics"][metric],
            entry["metrics_std"][metric],
            entry["draws_by_metric"][metric],
        )
    if metric != "rmse":
        raise ValueError(
            f"a convergence series without a `metrics` map carries only its "
            f"historical RMSE reading, so {metric!r} cannot be plotted from it"
        )
    return entry["metric"], entry["metric_std"], entry["draws"]


def _parity_line(values: list[float]) -> tuple[float, float]:
    usable = [v for v in values if not math.isnan(v)]
    if not usable:
        return (0.0, 1.0)
    low, high = min(usable), max(usable)
    span = (high - low) or abs(high) or 1.0
    return (low - 0.05 * span, high + 0.05 * span)


def render_cross_validation(result: CrossValidationResult, **kwargs) -> go.Figure:
    """Observed vs held-out-predicted, one point per SAMPLE row.

    The parity line is layout decoration; the only numbers plotted are the
    result's own `observed`/`predicted`(/`predicted_std` as error bars). A
    sample whose fold Dakota abandoned keeps NaN and simply does not appear
    (its explanation travels in `result.warnings`).

    kwargs pass through to the layout; escape hatch: the returned Figure.
    """
    lo, hi = _parity_line(result.observed)
    error = (
        {"type": "data", "array": result.predicted_std}
        if result.predicted_std is not None
        else {}
    )
    fig = go.Figure(
        [
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="parity",
                line={"dash": "dot", "color": "grey"},
                hoverinfo="skip",
            ),
            go.Scatter(
                x=result.observed,
                y=result.predicted,
                mode="markers",
                name="samples",
                error_y=error,
                customdata=list(range(len(result.observed))),
                hovertemplate=(
                    "sample %{customdata}<br>observed %{x:.4g}"
                    "<br>predicted %{y:.4g}<extra></extra>"
                ),
            ),
        ]
    )
    fig.update_layout(
        title=f"Cross-validation — {result.response} (seed {result.seed})",
        xaxis_title="observed (original units)",
        yaxis_title="predicted (original units)",
    )
    fig.update_layout(**kwargs)
    return fig


def render_sobol(result: SobolResult, **kwargs) -> go.Figure:
    """First- and total-order indices with their bootstrap CIs, per variable.

    The CI whiskers are the result's `main_ci_*`/`total_ci_*` fields lifted
    from the shared-row resample (no recomputation); fixed factors show at
    zero by construction.

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    variables = list(result.indices)
    main = [result.indices[v]["main"] for v in variables]
    total = [result.indices[v]["total"] for v in variables]
    fig = go.Figure(
        [
            go.Bar(
                x=variables,
                y=main,
                name="first-order",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        result.indices[v]["main_ci_high"] - result.indices[v]["main"]
                        for v in variables
                    ],
                    "arrayminus": [
                        result.indices[v]["main"] - result.indices[v]["main_ci_low"]
                        for v in variables
                    ],
                },
            ),
            go.Bar(
                x=variables,
                y=total,
                name="total",
                opacity=0.55,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        result.indices[v]["total_ci_high"] - result.indices[v]["total"]
                        for v in variables
                    ],
                    "arrayminus": [
                        result.indices[v]["total"] - result.indices[v]["total_ci_low"]
                        for v in variables
                    ],
                },
            ),
        ]
    )
    fig.update_layout(
        title=f"Sensitivity (Sobol) — {result.response} (seed {result.seed})",
        yaxis_title="index over the sampled domain",
        barmode="group",
    )
    fig.update_layout(**kwargs)
    return fig


def render_correlations(result: CorrelationResult, **kwargs) -> go.Figure:
    """Pearson and Spearman of each variable against the response.

    Two bars per variable: Pearson moves under a log reparametrization,
    Spearman cannot (V45ls) — reading both is the scale honesty check.

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    variables = list(result.coefficients)
    fig = go.Figure(
        [
            go.Bar(
                x=variables,
                y=[result.coefficients[v]["pearson"] for v in variables],
                name="pearson",
            ),
            go.Bar(
                x=variables,
                y=[result.coefficients[v]["spearman"] for v in variables],
                name="spearman",
                opacity=0.55,
            ),
        ]
    )
    title = f"Correlations — {result.response}"
    if result.seed is not None:
        title += f" (seed {result.seed})"
    fig.update_layout(title=title, yaxis_title="coefficient", barmode="group")
    fig.update_layout(**kwargs)
    return fig


def render_along_axes(result: AlongAxesResult, **kwargs) -> go.Figure:
    """One sweep line per variable (every other factor held at its centre).

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    fig = go.Figure()
    for variable, sweep in result.sweeps.items():
        fig.add_trace(
            go.Scatter(
                x=sweep.x,
                y=sweep.predicted,
                mode="lines+markers",
                name=variable,
                customdata=list(range(len(sweep.x))),
                hovertemplate=(
                    f"{variable} = %{{x:.4g}}<br>{result.response} = %{{y:.4g}}"
                    "<br>point %{customdata}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Axis sweeps — {result.response}",
        yaxis_title=f"{result.response} (original units)",
    )
    fig.update_layout(**kwargs)
    return fig


def render_grid(result: GridResult, **kwargs) -> go.Figure:
    """Every grid point as a marker — never a smoothed surface (V54hz).

    Two grid variables become position + colour; three-plus become position +
    colour of the first two with the rest readable from `result.data`.

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    grid_vars = list(result.grid_variables)
    y = result.data[result.response]
    x = result.data[grid_vars[0]]
    marker: dict = {"colorscale": "Viridis"}
    if len(grid_vars) >= 2:
        marker["color"] = result.data[grid_vars[1]]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=marker,
            customdata=list(range(len(y))),
            hovertemplate=(
                f"{grid_vars[0]} = %{{x:.4g}}<br>{result.response} = %{{y:.4g}}"
                "<br>grid point %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Grid — {result.response} over {', '.join(grid_vars)}",
        xaxis_title=f"{grid_vars[0]} (original units)",
        yaxis_title=f"{result.response} (original units)",
    )
    fig.update_layout(**kwargs)
    return fig


def render_uncertainty(draws: UncertaintyDraws, **kwargs) -> go.Figure:
    """Histogram of the propagated response across all realisations.

    The binning is plotly's rendering mechanism; the data are the result's
    `propagated` grid flattened — nothing is summarised on the way in.
    Per-draw predictions stay available in `draws.predictions` for a
    render-side overlay if an author wants one.

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    flat = [value for row in draws.propagated for value in row]
    fig = go.Figure(
        go.Histogram(
            x=flat,
            name="propagated",
            hovertemplate="response bin %{x}<br>realisations %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"Uncertainty propagation — {draws.response} "
            f"({draws.num_samples} draws × {draws.n_histograms} realisations, "
            f"seed {draws.seed})"
        ),
        xaxis_title=f"{draws.response} (original units)",
        bargap=0.0,
    )
    fig.update_layout(**kwargs)
    return fig


def render_predictions(result: PredictionsResult, **kwargs) -> go.Figure:
    """Predicted response per sample row, ± predictive std when supplied.

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    error = {"type": "data", "array": result.std} if result.std is not None else {}
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(result.values))),
            y=result.values,
            mode="lines+markers",
            error_y=error,
            customdata=result.values,
            hovertemplate=("sample %{x}<br>predicted %{y:.4g}<extra></extra>"),
        )
    )
    title = f"Predictions — {result.response}"
    if result.sumo_model_id is not None:
        title += f" (loaded model {result.sumo_model_id[:8]})"
    fig.update_layout(title=title, xaxis_title="sample row", yaxis_title="response")
    fig.update_layout(**kwargs)
    return fig


def render_pareto(result: ParetoFrontResult, **kwargs) -> go.Figure:
    """Scatter-plot matrix of the front: objectives on their own, variables
    as colour — every Pareto point stays individually readable (hover).

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    dimensions = [
        {"label": column, "values": values} for column, values in result.data.items()
    ]
    fig = go.Figure(
        go.Splom(
            dimensions=dimensions,
            text=[str(i) for i in range(len(next(iter(result.data.values()))))],
            hoverinfo="text",
        )
    )
    objectives = ", ".join(f"{k} ({v})" for k, v in result.objectives.items())
    fig.update_layout(title=f"Pareto front — {objectives}")
    fig.update_layout(**kwargs)
    return fig


def render_coverage(coverage: list, **kwargs) -> go.Figure:
    """Empirical vs nominal prediction-interval coverage (V19cz).

    `compute_coverage` answers ONE level, so this consumes the LIST of its
    results — one entry per level you asked about, in the order you want them
    read. Bars are each result's `empirical` fraction, markers are each result's
    own `level` (nominal), and the point count each fraction was measured over
    rides on the hover: a 95% bar computed over 4 points looks identical to one
    computed over 400 until you ask. A bar sitting off its marker is
    miscalibration evidence, not a data shortage — the reading is the plot.

    kwargs pass through to the plotly layout; escape hatch: the returned
    Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take `fig.data` apart).
    """
    if isinstance(coverage, dict):
        raise TypeError(
            "render_coverage takes the LIST of compute_coverage results, one per "
            f"level it should plot; got a single {coverage.get('level')!r} result "
            "— pass [result] for one level"
        )
    levels = [float(entry["level"]) for entry in coverage]
    empirical = [float(entry["empirical"]) for entry in coverage]
    labels = [f"{level:.1%}" for level in levels]
    fig = go.Figure(
        [
            go.Bar(
                x=labels,
                y=empirical,
                name="empirical",
                customdata=[int(entry["n_points"]) for entry in coverage],
                hovertemplate=(
                    "%{x}: %{y:.1%} empirical<br>measured over %{customdata} "
                    "points<extra></extra>"
                ),
            ),
            go.Scatter(
                x=labels,
                y=levels,
                mode="markers",
                name="nominal",
                marker={"symbol": "line-ew", "size": 24, "color": "black"},
                hovertemplate="nominal %{y:.1%}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        title="Prediction-interval coverage",
        xaxis_title="nominal interval mass",
        yaxis_title="empirical fraction",
        yaxis_range=[0.0, 1.0],
    )
    fig.update_layout(**kwargs)
    return fig


def render_convergence(series: list, **kwargs) -> go.Figure:
    """One CV metric vs training-sample count, with bootstrap draws as a cloud.

    `compute_cv_convergence` returns the whole metric family at every size, so
    WHICH curve you look at is this rendering's one option, `metric` (default
    `"rmse"`): a token of that family — `"mae"`, `"bias"`, `"coverage68"`,
    `"coverage95"`, `"coverage99"`. The line and its error bars are the
    series' own value and `metrics_std` for that token, the faint markers are its
    `draws_by_metric` — spread is shown, never re-derived, and no metric is
    recomputed here (V54hz). An unknown token fails loudly, listing the tokens the
    series actually carries.

    A coverage metric additionally gets a fixed 0–1 axis and its nominal level as
    a reference line, because "did coverage settle AT the nominal level or below
    it" is the whole question there (layout decoration, like
    render_cross_validation's parity line).

    kwargs (other than `metric`) pass through to the plotly layout; escape hatch:
    the returned Figure (`fig.write_html(...)`, `fig.to_image(...)`, or take
    `fig.data` apart).
    """
    metric = kwargs.pop("metric", "rmse")
    if not isinstance(metric, str):
        raise TypeError(
            f"metric must be one of the convergence metric tokens, got "
            f"{type(metric).__name__}"
        )
    metric = metric.strip().lower()
    selected = [_convergence_point_metric(entry, metric) for entry in series]
    label, nominal = _METRIC_PRESENTATION.get(metric, (metric, None))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[entry["n_samples"] for entry in series],
            y=[value for value, _, _ in selected],
            mode="lines+markers",
            name=metric,
            error_y={
                "type": "data",
                "array": [spread for _, spread, _ in selected],
            },
            customdata=[entry["n_bootstrap"] for entry in series],
            hovertemplate=(
                f"n=%{{x}}<br>{metric} %{{y:.4g}}<br>%{{customdata}} bootstrap "
                "draws<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[
                entry["n_samples"]
                for entry, (_, _, draws) in zip(series, selected)
                for _ in draws
            ],
            y=[draw for _, _, draws in selected for draw in draws],
            mode="markers",
            name="bootstrap draws",
            opacity=0.3,
            marker={"size": 4, "color": "grey"},
            hoverinfo="skip",
        )
    )
    layout: dict = {
        "title": f"CV convergence — {metric}",
        "xaxis_title": "training samples",
        "yaxis_title": label,
    }
    if nominal is not None:
        # a coverage curve is a fraction: the 0-1 axis is what makes a level
        # drifting below its nominal line visible instead of merely low
        layout["yaxis_range"] = [0.0, 1.0]
    fig.update_layout(**layout)
    if nominal is not None:
        fig.add_hline(
            y=nominal,
            line={"dash": "dot", "color": "black"},
            annotation_text=f"nominal {nominal * 100:g}%",
        )
    fig.update_layout(**kwargs)
    return fig
