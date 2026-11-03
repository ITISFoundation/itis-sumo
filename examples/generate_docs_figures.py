"""Generate the worked-example figures embedded in the docs site.

T61pn: PLOTLY, not matplotlib — one visual language across docs, the report
renderings, and the mmux front-end (V54hz). Figures are interactive HTML,
embedded from the docs pages via `<iframe>`. Regenerate with:

    uv run --extra report python examples/generate_docs_figures.py

Every figure is produced by calling the root manifest (real Dakota GP fits,
real LHS sampler, real Sobol' pipeline) against small analytical problems with
known closed-form answers — nothing is hand-drawn or faked. The Sobol' figure
starts FROM the report rendering (`itis_sumo.report.render_sobol`) and adds
the analytical reference bars: docs figures are built from renderings, not
beside them. Outputs land in docs/assets/examples/, referenced from
docs/tutorials/*.md and docs/index.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from itis_sumo import DomainSpec, evaluate_sobol, fit, generate_lhs_samples
from itis_sumo.report import render_sobol

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "examples"
SEED = 42


def _embed(fig: go.Figure, name: str) -> None:
    fig.update_layout(width=760, height=430)
    fig.write_html(
        OUT_DIR / name, include_plotlyjs="cdn", include_mathjax=False, full_html=True
    )
    print(name)


def fig_gp_fit_uncertainty() -> None:
    """GP surrogate mean + 95% prediction interval on a sparsely-sampled sine."""
    rng = np.random.default_rng(SEED)
    x_train = np.sort(rng.uniform(0.0, 2 * np.pi, size=9))
    train = pd.DataFrame({"x": x_train, "y": np.sin(x_train)})

    x_eval = np.linspace(0.0, 2 * np.pi, 200)
    model = fit(train, ["x"], "y")
    predictions = model.predict(pd.DataFrame({"x": x_eval}))
    y_hat = np.asarray(predictions.values)
    y_std = np.asarray(predictions.std)
    model.close()

    fig = go.Figure()
    fig.add_scatter(
        x=x_eval, y=np.sin(x_eval), name="true f(x) = sin(x)", line={"dash": "dash"}
    )
    fig.add_scatter(
        x=x_eval,
        y=y_hat + 1.96 * y_std,
        name="95% interval",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=x_eval,
        y=y_hat - 1.96 * y_std,
        fill="tonexty",
        fillcolor="rgba(63, 81, 181, 0.2)",
        line={"width": 0},
        name="95% prediction interval (±1.96 σ̂)",
        hoverinfo="skip",
    )
    fig.add_scatter(x=x_eval, y=y_hat, name="GP mean prediction", line={"width": 2})
    fig.add_scatter(
        x=x_train,
        y=np.sin(x_train),
        mode="markers",
        name="training points (n=9)",
        marker={"size": 8},
    )
    fig.update_layout(
        title="GP surrogate: mean prediction and uncertainty band",
        xaxis_title="x",
        yaxis_title="y",
        legend={"x": 0.01, "y": 0.01},
    )
    _embed(fig, "gp_fit_uncertainty.html")


def fig_lhs_vs_random() -> None:
    """LHS vs. plain uniform random: LHS covers each 1D marginal stratum exactly once."""
    n = 25
    lhs_pts = generate_lhs_samples(
        {
            "x1": DomainSpec(minimum=0.0, maximum=1.0),
            "x2": DomainSpec(minimum=0.0, maximum=1.0),
        },
        n,
        seed=SEED,
    )
    rng = np.random.default_rng(SEED)
    rand_pts = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Plain uniform random (n=25)",
            "Latin Hypercube, maximin (n=25)",
        ),
    )
    for col, pts, name, show in (
        (1, rand_pts, "uniform random", True),
        (2, lhs_pts, "LHS (maximin)", False),
    ):
        for edge in np.linspace(0, 1, n + 1):
            fig.add_shape(
                {"type": "line", "x0": edge, "x1": edge, "y0": 0, "y1": 1},
                row=1,
                col=col,
                line={"color": "lightgrey", "width": 0.5},
                layer="below",
            )
            fig.add_shape(
                {"type": "line", "x0": 0, "x1": 1, "y0": edge, "y1": edge},
                row=1,
                col=col,
                line={"color": "lightgrey", "width": 0.5},
                layer="below",
            )
        fig.add_trace(
            go.Scatter(
                x=pts["x1"],
                y=pts["x2"],
                mode="markers",
                name=name if show else None,
                showlegend=show,
                hovertemplate="x1 %{x:.3f}<br>x2 %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="x1", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="x1", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="x2", range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    fig.update_layout(
        title="Design-of-experiments: LHS stratifies every marginal, random sampling doesn't"
    )
    _embed(fig, "lhs_vs_random.html")


def fig_sobol_ishigami() -> None:
    """Sobol' indices for Ishigami, GP-surrogate-based, via the report rendering."""

    def ishigami(x: np.ndarray) -> np.ndarray:
        return (
            np.sin(x[:, 0])
            + 7.0 * np.sin(x[:, 1]) ** 2
            + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])
        )

    rng = np.random.default_rng(SEED)
    x_train = rng.uniform(-np.pi, np.pi, size=(300, 3))
    train = pd.DataFrame(x_train, columns=pd.Index(["x1", "x2", "x3"]))
    train["y"] = ishigami(x_train)

    result = evaluate_sobol(
        train,
        ["x1", "x2", "x3"],
        "y",
        domains={
            "x1": DomainSpec(minimum=-np.pi, maximum=np.pi),
            "x2": DomainSpec(minimum=-np.pi, maximum=np.pi),
            "x3": DomainSpec(minimum=-np.pi, maximum=np.pi),
        },
        seed=SEED,
    )

    fig = render_sobol(result)
    variables = ["x1", "x2", "x3"]
    ref_main = [0.314, 0.442, 0.0]
    ref_total = [0.558, 0.442, 0.244]
    fig.add_scatter(
        x=variables,
        y=ref_main,
        name="first-order (analytical)",
        mode="markers",
        marker={"symbol": "diamond", "size": 10},
    )
    fig.add_scatter(
        x=variables,
        y=ref_total,
        name="total-order (analytical)",
        mode="markers",
        marker={"symbol": "diamond-open", "size": 10},
    )
    fig.update_layout(
        title="Ishigami function: Sobol' indices, GP surrogate vs. analytical reference"
    )
    _embed(fig, "sobol_ishigami.html")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_gp_fit_uncertainty()
    fig_lhs_vs_random()
    fig_sobol_ishigami()
    print(f"\nAll docs figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
