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
import plotly.graph_objects as go
import pytest

import itis_sumo
from itis_sumo import report
from itis_sumo.report import Section, sections
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

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


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
        "coverage": {
            "levels": [0.6827, 0.95, 0.9973],
            "empirical": [0.7, 0.94, 0.99],
            "n_points": 100,
        },
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
        "convergence": [
            {
                "n_samples": 10,
                "metric": 0.5,
                "metric_std": 0.1,
                "n_bootstrap": 2,
                "draws": [0.4, 0.6],
                "diagnostics": [],
            },
            {
                "n_samples": 20,
                "metric": 0.3,
                "metric_std": 0.05,
                "n_bootstrap": 2,
                "draws": [0.25, 0.35],
                "diagnostics": [],
            },
        ],
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
