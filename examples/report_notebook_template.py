#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "itis-sumo[report]"]
# ///
"""SuMo V&V report template (T59hz) — `marimo edit examples/report_notebook_template.py`.

The pattern (V54hz): DATA cells call root-manifest functions (`fit`/`load`
and the one-shots) and own typed results; RENDERING cells hand each result —
untouched — to exactly one `itis_sumo.report` rendering. No cell calls Dakota
directly and no cell recomputes a statistic; renderings are deletable serving
suggestions, and `sections()` is the catalog of which renderings exist.

The version stamp travels in the title cell: every generated report records
the itis-sumo build that produced its numbers. The `convergence` /
`convergence_fit` sections have renderings in the catalog but need a run
directory by hand (`compute_cv_convergence` is engine-level); they are left
out here on purpose — this template only shows what the manifest offers.
"""

# %% ── version stamp
import marimo as mo

import itis_sumo

mo.md(
    f"""# SuMo validation report

Generated with **itis-sumo {itis_sumo.__version__}**. Every number below is
a result object produced by the function named in its cell; every picture is
one rendering from `itis_sumo.report` consuming exactly one of those objects.
"""
)

# %% ── data cell: the dataset, and the one fit everything else reuses
import numpy as np
import pandas as pd

from itis_sumo import (
    DEFAULT_SEED,
    DistributionSpec,
    DomainSpec,
    fit,
)

rng = np.random.default_rng(DEFAULT_SEED)
samples = pd.DataFrame(
    {
        "width": rng.uniform(1.0, 5.0, 20),
        "height": rng.uniform(100.0, 500.0, 20),
    }
)
samples["stress"] = 3.0 * samples["width"] + 0.01 * samples["height"]

VARIABLES = ["width", "height"]
RESPONSE = "stress"

model = fit(samples, VARIABLES, RESPONSE)

# %% ── data cells: one result object per report section
cv = model.cross_validate(folds=5, seed=DEFAULT_SEED)
sobol = model.sobol(seed=DEFAULT_SEED)
axes = model.along_axes(at=None, points_per_variable=11)
grid = model.grid(points_per_variable=5)
draws = model.sample_uncertainty(
    distributions={
        "width": DistributionSpec("uniform", minimum=1.0, maximum=5.0),
        "height": DistributionSpec("uniform", minimum=100.0, maximum=500.0),
    },
    num_samples=400,
    n_histograms=10,
    seed=DEFAULT_SEED,
)
predictions = model.predict(samples)

# open composition (V52rc): draws -> caller-owned table -> correlations
correlated = pd.DataFrame({**draws.draws, RESPONSE: draws.predictions})

from itis_sumo import (
    analyze_dataset,
    compute_correlations,
    compute_coverage,
    compute_cv_diagnostics,
    cross_validation_metrics,
    optimize,
    summarize_uncertainty,
)

diagnostics = analyze_dataset(samples)
accuracy = cross_validation_metrics(samples, VARIABLES, RESPONSE, seed=DEFAULT_SEED)
coverage = compute_coverage(cv.observed, cv.predicted, cv.predicted_std or [])
diagnostics_bundle = compute_cv_diagnostics(
    cv.observed, cv.predicted, cv.predicted_std or []
)
correlations = compute_correlations(correlated, VARIABLES, RESPONSE)
summary = summarize_uncertainty(draws)
front = optimize(
    samples,
    VARIABLES,
    {RESPONSE: "minimize"},
    domains={
        "width": DomainSpec(minimum=1.0, maximum=5.0),
        "height": DomainSpec(minimum=100.0, maximum=500.0),
    },
    max_evaluations=200,
)

# %% ── rendering cells: one result in, one figure/table out
from itis_sumo.report import (
    render_along_axes,
    render_correlations,
    render_coverage,
    render_cross_validation,
    render_cv_accuracy,
    render_cv_diagnostics,
    render_diagnostics,
    render_grid,
    render_pareto,
    render_predictions,
    render_sobol,
    render_uncertainty,
    render_uncertainty_summary,
    sections,
)

mo.md("## Dataset diagnostics")
render_diagnostics(diagnostics)

mo.md("## Cross-validation")
render_cross_validation(cv)
render_cv_accuracy(accuracy)

mo.md("## Calibration")
render_coverage(coverage)
render_cv_diagnostics(diagnostics_bundle)

mo.md("## Sensitivity")
render_sobol(sobol)
render_correlations(correlations)

mo.md("## Exploration")
render_along_axes(axes)
render_grid(grid)

mo.md("## Uncertainty")
render_uncertainty(draws)
render_uncertainty_summary(summary)

mo.md("## Predictions & optimization")
render_predictions(predictions)
render_pareto(front)

# %% ── catalog tail: which renderings exist, and what each one eats
mo.md(
    "## Section catalog\n\n"
    + "\n".join(
        f"- **{name}** — {section.description} (consumes `{section.consumes}`)"
        for name, section in sections().items()
    )
)

# %% ── end the handle's lifetime (run dir discarded on success, V24af)
model.close()
