"""`itis_sumo.report` surface contract (V54hz, T59hz): renderings, not façades.

The report namespace is a catalog of RENDERINGS — each consumes exactly ONE
function result, never runs the engine, never recomputes statistics, and
returns a plotly Figure or a pandas DataFrame (the deletable "serving
suggestion"; the escape hatch is the returned object). This replaces the old
T54rp re-export-façade completeness test — the façade itself died with T59hz.

All renderings are exercised against SYNTHETIC result objects: if a rendering
needed the engine to be shown, these calls would touch Dakota and the
engine-tripwire test would fail.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import itis_sumo
from itis_sumo.engine.evaluation import _CV_METRICS
from itis_sumo.types import (
    AlongAxesResult,
    AxisSweep,
    CorrelationResult,
    CrossValidationResult,
    CVAccuracyMetrics,
    DatasetDiagnostics,
    GridResult,
    OutlierSummary,
    ParetoFrontResult,
    PredictionsResult,
    SobolResult,
    UncertaintyDraws,
    UncertaintySummary,
    VariableDiagnostics,
)

try:
    import plotly.graph_objects as go

    from itis_sumo import report
    from itis_sumo.report import Section, sections

    print("report-depencencies imported correctly")

except ImportError:
    print("report import failed. Make sure report-extra-dependencies are installed.")


REPO = Path(__file__).resolve().parents[2]

# this module imports plotly which is a "report" extra dependency
# thus cannot run against core itis-sumo package
pytestmark = pytest.mark.integration


def test_report_all_importable_and_sorted() -> None:
    names = report.__all__
    assert names == sorted(names), "__all__ must stay sorted (ruff RUF022 precedent)"
    for name in names:
        assert hasattr(report, name), f"itis_sumo.report missing {name}"


# --------------------------------------------------------------- the catalog


def test_sections_is_the_catalog_of_the_namespace() -> None:
    catalog = sections()
    render_names = {n for n in report.__all__ if n.startswith("render_")}
    # every section is named for its rendering: render_<name>
    assert {f"render_{name}" for name in catalog} == render_names
    for name, section in catalog.items():
        assert isinstance(section, Section)
        assert section.name == name
        assert getattr(report, f"render_{name}") is section.rendering
        assert section.consumes, f"{name}: a rendering must name its result"
        assert section.description.strip(), f"{name}: a section must say its intent"


def test_catalog_covers_the_result_types() -> None:
    """Every manifest result type either has a section consuming it directly
    or is rendered as part of one (fields of a rendered parent)."""
    consumed = {section.consumes for section in sections().values()}
    results = {
        name
        for name in itis_sumo.__all__
        if name.endswith(("Result", "Summary", "Draws", "Diagnostics", "Metrics"))
        and dataclasses.is_dataclass(getattr(itis_sumo, name, None))
    }
    nested = {
        "OutlierSummary",  # row fields of render_diagnostics
        "VariableDiagnostics",  # rendered row-wise inside DatasetDiagnostics
    }
    assert results - consumed == nested


def test_every_rendering_takes_one_result_plus_kwargs() -> None:
    for name in report.__all__:
        if not name.startswith("render_"):
            continue
        signature = inspect.signature(getattr(report, name))
        parameters = list(signature.parameters.values())
        assert len(parameters) == 2, f"{name}: must take exactly one result + **kwargs"
        assert parameters[1].kind is inspect.Parameter.VAR_KEYWORD, (
            f"{name}: kwargs must pass through to the backend (V54hz)"
        )
        doc = getattr(report, name).__doc__ or ""
        assert "escape hatch" in doc.lower(), f"{name}: docstring names the hatch"


# --------------------------------------------------------- synthetic results


def _cross_validation_result() -> CrossValidationResult:
    return CrossValidationResult(
        response="stress",
        observed=[3.0, 4.0, 5.0, float("nan")],
        predicted=[3.1, 3.9, 5.2, 4.0],
        predicted_std=[0.2, 0.2, 0.3, 0.2],
        warnings=["fold 3 abandoned"],
        seed=7,
        effective_config={},
    )


def _sobol_result() -> SobolResult:
    return SobolResult(
        response="stress",
        indices={
            "width": {
                "main": 0.8,
                "total": 0.85,
                "main_ci_low": 0.7,
                "main_ci_high": 0.9,
                "total_ci_low": 0.75,
                "total_ci_high": 0.95,
            },
            "height": {
                "main": 0.1,
                "total": 0.15,
                "main_ci_low": 0.05,
                "main_ci_high": 0.2,
                "total_ci_low": 0.05,
                "total_ci_high": 0.25,
            },
        },
        second_order={},
        order_contributions=None,
        seed=7,
        domains={},
        fixed={},
        effective_config={},
    )


_CONVERGENCE_SPREAD = 0.1


def _convergence_series() -> list[dict[str, Any]]:
    """A two-size convergence series in the shape `compute_cv_convergence` returns.

    Every metric family token at each size (the accuracy trio plus the
    sweep's coverage levels), the three legacy RMSE keys that mean its `rmse` entry, and
    two bootstrap draws per metric at a fixed spread — enough for a rendering to
    show a line, an error bar and a cloud per metric without any of them being
    equal by accident.
    """
    per_size = [
        (
            10,
            {
                "rmse": 0.5,
                "mae": 0.42,
                "bias": 0.1,
                "coverage68": 0.62,
                "coverage95": 0.90,
                "coverage99": 0.97,
            },
        ),
        (
            20,
            {
                "rmse": 0.3,
                "mae": 0.25,
                "bias": -0.02,
                "coverage68": 0.68,
                "coverage95": 0.94,
                "coverage99": 0.99,
            },
        ),
    ]
    series = []
    for n_samples, metrics in per_size:
        draws_by_metric = {
            name: [value - _CONVERGENCE_SPREAD, value + _CONVERGENCE_SPREAD]
            for name, value in metrics.items()
        }
        series.append(
            {
                "n_samples": n_samples,
                "metrics": metrics,
                "metrics_std": dict.fromkeys(metrics, _CONVERGENCE_SPREAD),
                "draws_by_metric": draws_by_metric,
                "n_bootstrap": 2,
                "metric": metrics["rmse"],
                "metric_std": _CONVERGENCE_SPREAD,
                "draws": draws_by_metric["rmse"],
                "diagnostics": [],
            }
        )
    return series


def _synthetics() -> dict[str, Any]:
    draws = UncertaintyDraws(
        response="stress",
        distributions={},
        seed=7,
        num_samples=4,
        n_histograms=3,
        draws={"width": [1.0, 2.0, 3.0, 4.0], "height": [100.0] * 4},
        predictions=[3.0, 4.0, 5.0, 6.0],
        prediction_stds=[0.2, 0.2, 0.3, 0.2],
        propagated=[[3.0, 4.0, 5.0, 6.0]] * 3,
    )
    summary = UncertaintySummary(
        response="stress",
        seed=7,
        bins_start=2.0,
        bins_end=7.0,
        bin_means=[3.0, 4.0, 5.0],
        bin_stds=[0.1, 0.1, 0.1],
        q1=3.5,
        median=4.5,
        q3=5.5,
        whisker_min=2.5,
        whisker_max=6.5,
        outliers=[7.5],
        mean=4.5,
        std=1.0,
        minimum=2.5,
        maximum=6.5,
    )
    diagnostics = DatasetDiagnostics(
        inputs={
            "width": VariableDiagnostics(
                name="width",
                scale="linear",
                distribution="uniform",
                confident=True,
                outliers=OutlierSummary(1, [2], 0.5, 6.0),
            )
        },
        outputs={
            "stress": VariableDiagnostics(
                name="stress",
                scale="log",
                distribution="normal",
                confident=False,
            )
        },
    )
    return {
        "cross_validation": _cross_validation_result(),
        "cv_accuracy": CVAccuracyMetrics(
            response="stress",
            root_mean_squared=0.2,
            sum_abs=0.8,
            mean_abs=0.2,
            max_abs=0.4,
            mean_signed_error=0.05,
            seed=7,
        ),
        "coverage": [
            {"level": 0.68, "empirical": 0.7, "n_points": 100},
            {"level": 0.95, "empirical": 0.94, "n_points": 100},
            {"level": 0.99, "empirical": 0.99, "n_points": 100},
        ],
        "cv_diagnostics": {
            "root_mean_squared": 0.2,
            "mean_abs": 0.15,
            "mean_abs_filtered": 0.12,
            "n_outliers": 2,
            "ttest_statistic": 1.1,
            "ttest_p_value": 0.3,
            "cohens_d": 0.2,
            "actual": [1.0, 2.0],  # non-scalar: must be left out of the table
        },
        "convergence": _convergence_series(),
        "convergence_fit": {"a": 1.0, "b": 0.1, "r_squared": 0.98},
        "sobol": _sobol_result(),
        "correlations": CorrelationResult(
            response="stress",
            coefficients={
                "width": {"pearson": 0.9, "spearman": 0.9},
                "height": {"pearson": 0.1, "spearman": 0.1},
            },
            seed=7,
        ),
        "along_axes": AlongAxesResult(
            response="stress",
            sweeps={
                "width": AxisSweep(variable="width", x=[1.0, 3.0], predicted=[3.0, 9.0])
            },
            effective_config={},
        ),
        "grid": GridResult(
            response="stress",
            grid_variables=("width", "height"),
            data={
                "width": [1.0, 1.0, 5.0, 5.0],
                "height": [100.0, 500.0, 100.0, 500.0],
                "stress": [3.0, 8.0, 15.0, 20.0],
            },
            effective_config={},
        ),
        "uncertainty": draws,
        "uncertainty_summary": summary,
        "uncertainty_bins": summary,
        "predictions": PredictionsResult(
            response="stress",
            values=[3.0, 4.0, 5.0],
            std=[0.2, 0.2, 0.2],
            sumo_model_id=None,
        ),
        "pareto": ParetoFrontResult(
            objectives={"stress": "minimize"},
            variables=("width",),
            data={"stress": [3.0, 4.0], "width": [1.0, 2.0]},
        ),
        "diagnostics": diagnostics,
    }


@pytest.mark.parametrize("name", sorted(sections()))
def test_each_rendering_consumes_its_result_without_the_engine(
    name: str, monkeypatch
) -> None:
    """A rendering shown from a synthetic result must never reach Dakota:
    the engine door is welded shut here."""
    from itis_sumo.engine import dakota_object

    def _welded(*args, **kwargs):
        raise AssertionError(f"{name} touched the engine")

    monkeypatch.setattr(dakota_object.DakotaObject, "run", _welded)

    section = sections()[name]
    out = section.rendering(_synthetics()[name])
    if isinstance(out, go.Figure):
        assert out.data, f"{name}: empty figure is not a rendering"
    else:
        assert isinstance(out, pd.DataFrame), f"{name}: {type(out)} is neither"


@pytest.mark.parametrize("name", sorted(sections()))
def test_kwargs_pass_through_to_the_backend(name: str) -> None:
    section = sections()[name]
    module = getattr(section.rendering, "__module__", "")
    is_figure = module.endswith("figures")
    if is_figure:
        out = section.rendering(_synthetics()[name], title="PASSTHROUGH")
        assert cast(go.Figure, out).layout.title.text == "PASSTHROUGH"
    else:
        # tables pass kwargs to the DataFrame constructor; `index` must be
        # as long as the table, so the two multi-row renderings get their own
        rows = {"diagnostics": 2, "uncertainty_bins": 3}.get(name, 1)
        out = section.rendering(
            _synthetics()[name], index=[f"row-{i}" for i in range(rows)]
        )
        assert list(cast(pd.DataFrame, out).index) == [f"row-{i}" for i in range(rows)]


def test_uncertainty_summary_renders_scalars_only_and_bins_separately() -> None:
    summary = _synthetics()["uncertainty_summary"]
    row = report.render_uncertainty_summary(summary)
    assert list(row.columns) != ["response"] and "median" in row.columns
    assert "outliers" not in row.columns  # the LIST became a count
    assert row["n_outliers"].iloc[0] == 1
    bins = report.render_uncertainty_bins(summary)
    assert len(bins) == len(summary.bin_means)
    assert bins["bin_end"].iloc[0] == pytest.approx(bins["bin_start"].iloc[1])


def test_render_coverage_compares_the_levels_the_caller_asked_for() -> None:
    """`compute_coverage` answers one level, so the level comparison is a LIST of
    results: one bar per entry, each marker at its own result's nominal level, and
    the point count each fraction was measured over kept visible."""
    results = _synthetics()["coverage"]

    figure = report.render_coverage(results)

    assert [bar.name for bar in figure.data] == ["empirical", "nominal"]
    assert list(figure.data[0].x) == ["68.0%", "95.0%", "99.0%"]
    assert list(figure.data[0].y) == pytest.approx([0.7, 0.94, 0.99])
    # the nominal markers are the results' own levels, not a fixed trio
    assert list(figure.data[1].y) == pytest.approx([0.68, 0.95, 0.99])
    assert list(figure.data[0].customdata) == [100, 100, 100]

    # one level asked is one level plotted
    assert len(report.render_coverage(results[:1]).data[0].y) == 1

    # a single result passed by mistake says what to pass instead of failing oddly
    with pytest.raises(TypeError, match="LIST of compute_coverage results"):
        report.render_coverage(results[0])


def test_render_convergence_plots_the_metric_you_ask_for() -> None:
    """One sweep returns every metric, so WHICH curve you see is decided here:
    line, legend name and axis all follow `metric`, and nothing is recomputed."""
    series = _synthetics()["convergence"]

    default = report.render_convergence(series)
    assert list(default.data[0].y) == pytest.approx([0.5, 0.3])
    assert default.data[0].name == "rmse"
    assert default.layout.yaxis.title.text == "RMSE (original units)"

    mae = report.render_convergence(series, metric="mae")
    assert list(mae.data[0].y) == pytest.approx([0.42, 0.25])
    assert mae.layout.yaxis.title.text == "MAE (original units)"
    assert "mae" in mae.layout.title.text
    assert not mae.layout.shapes, "an accuracy curve has no nominal level"

    # the tokens are matched the way a human types them
    bias = report.render_convergence(series, metric=" BIAS ")
    assert list(bias.data[0].y) == pytest.approx([0.1, -0.02])
    assert bias.layout.yaxis.title.text == "signed bias (original units)"


def test_render_convergence_frames_coverage_against_its_own_nominal_level() -> None:
    """A coverage curve is a fraction read against a level: the 0-1 axis and that
    level's line come from the token in play, so 68 and 99 do not render as
    the same question."""
    series = _synthetics()["convergence"]

    tail = report.render_convergence(series, metric="coverage99")
    assert list(tail.data[0].y) == pytest.approx([0.97, 0.99])
    assert tail.layout.yaxis.title.text == "99% interval coverage (fraction)"
    assert list(tail.layout.yaxis.range) == [0.0, 1.0]
    assert tail.layout.shapes, "a coverage plot needs its nominal reference line"
    assert tail.layout.shapes[0].y0 == pytest.approx(0.99)

    one_sigma = report.render_convergence(series, metric="coverage68")
    assert one_sigma.layout.yaxis.title.text == "68% interval coverage (fraction)"
    assert one_sigma.layout.shapes[0].y0 == pytest.approx(0.68)


def test_render_convergence_reads_spread_from_the_series() -> None:
    """V54hz: error bars are the point's own `metrics_std` and the cloud is its own
    `draws_by_metric` for that metric — the rendering derives no statistic."""
    series = _synthetics()["convergence"]

    figure = report.render_convergence(series, metric="coverage95")
    assert list(figure.data[0].error_y.array) == pytest.approx([0.1, 0.1])
    assert list(figure.data[1].y) == pytest.approx([0.8, 1.0, 0.84, 1.04])
    assert list(figure.data[1].x) == [10, 10, 20, 20]


def test_render_convergence_fails_loud_on_a_metric_it_cannot_plot() -> None:
    series = _synthetics()["convergence"]

    with pytest.raises(ValueError, match="no 'r2' metric"):
        report.render_convergence(series, metric="r2")
    wrong: Any = 0.95  # what a caller who never read the docstring sends
    with pytest.raises(TypeError, match="metric must be"):
        report.render_convergence(series, metric=wrong)


def test_render_convergence_still_renders_a_series_without_a_metric_family() -> None:
    """Back-compat: a hand-built or pre-family series carries only the
    historical RMSE keys and renders as the RMSE sweep it always was; asking it
    for another metric says so instead of plotting RMSE under a MAE label."""
    legacy = [
        {
            "n_samples": point["n_samples"],
            "metric": point["metric"],
            "metric_std": point["metric_std"],
            "n_bootstrap": point["n_bootstrap"],
            "draws": point["draws"],
            "diagnostics": [],
        }
        for point in _synthetics()["convergence"]
    ]

    figure = report.render_convergence(legacy)
    assert list(figure.data[0].y) == pytest.approx([0.5, 0.3])
    assert figure.layout.yaxis.title.text == "RMSE (original units)"

    with pytest.raises(ValueError, match="historical RMSE reading"):
        report.render_convergence(legacy, metric="mae")


def test_every_engine_metric_wears_a_label() -> None:
    """Contract between the two sides of V54hz: the engine's metric family is what
    the presentation table knows, so a level added to the engine's coverage
    defaults cannot arrive in a report as an unlabelled axis."""
    presentation = report.figures._METRIC_PRESENTATION

    assert set(_CV_METRICS) <= set(presentation)
    assert all(label for label, _ in presentation.values())


def test_cv_diagnostics_table_drops_the_pair_arrays() -> None:
    frame = report.render_cv_diagnostics(_synthetics()["cv_diagnostics"])
    assert "cohens_d" in frame.columns
    assert "actual" not in frame.columns


# ----------------------------------------------------- V54hz structural rules


def test_core_never_imports_report() -> None:
    """V54hz: the report namespace sits ON TOP of core; core must stay clean."""
    src = REPO / "src" / "itis_sumo"
    for path in src.rglob("*.py"):
        if path.relative_to(src).parts[0] == "report":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for module in names:
                assert not module.startswith("itis_sumo.report"), (
                    f"{path}: core imports the report namespace (V54hz)"
                )


def test_report_never_imports_the_engine() -> None:
    """Renderings consume RESULTS; importing engine plumbing would be the
    first step back towards façade behaviour."""
    for path in (REPO / "src" / "itis_sumo" / "report").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert not module.startswith("itis_sumo.engine"), (
                    f"{path}: report imports engine internals (V54hz)"
                )


def test_marimo_template_pairs_data_and_rendering_cells() -> None:
    template = REPO / "examples" / "report_notebook_template.py"
    source = template.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(template))
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for module in modules:
            assert module.startswith("itis_sumo") is False or module in {
                "itis_sumo",
                "itis_sumo.report",
            }, f"template imports non-public module {module!r} (V50kv)"
    assert "__version__" in source, "the template must carry the version stamp"
    assert "sections()" in source, "the template must show the catalog"
    assert "# %%" in source, "the template must mark its cells"
