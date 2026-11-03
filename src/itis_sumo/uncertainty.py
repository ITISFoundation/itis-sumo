"""Uncertainty propagation through a fitted SuMo (V52rc grammar).

Two verbs, one seam: `sample_uncertainty` produces the Monte-Carlo data (input
draws, per-draw predictions, propagated realisation grid) as
`UncertaintyDraws`, and `summarize_uncertainty` recomputes the histogram +
boxplot statistics from those draws -- pure maths, no engine call. Correlation
through the surrogate is deliberately NOT fused here: assemble the draws and
predictions into a table and hand it to
`itis_sumo.sensitivity.compute_correlations` (V52rc open composition).
"""

from __future__ import annotations

__all__ = [
    "sample_uncertainty",
    "summarize_uncertainty",
]

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from itis_sumo.engine.evaluation import summarize_uncertainty_samples
from itis_sumo.errors import SumoInputError
from itis_sumo.surrogate._session import SumoSession
from itis_sumo.types import (
    DEFAULT_SEED,
    DistributionSpec,
    PreprocessingSpec,
    UncertaintyDraws,
    UncertaintySummary,
)


def sample_uncertainty(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    distributions: Mapping[str, DistributionSpec],
    num_samples: int = 1000,
    n_histograms: int = 100,
    seed: int = DEFAULT_SEED,
    preprocessing: PreprocessingSpec | None = None,
    workspace: Path | None = None,
) -> UncertaintyDraws:
    """Propagate explicit per-variable uncertainty through the surrogate.

    Draws ``num_samples`` from ``distributions``, evaluates the surrogate once,
    then repeats ``n_histograms`` times injecting the surrogate's own predictive
    uncertainty -- and hands back the whole raw material in original units:
    per-variable input draws, per-draw predictions and spreads, and the
    propagated grid. Summarise it with ``summarize_uncertainty``, correlate it
    with ``compute_correlations``, or keep it; the numbers are yours (V53md).

    Args:
        samples: Training samples (the surrogate is fitted on these).
        variables: Input variable columns.
        response: Response column, named as in the table.
        distributions: Per-variable Monte Carlo distributions; must cover the
            variables exactly.
        num_samples: Monte Carlo sample size.
        n_histograms: Propagated realisations per draw (the predictive-spread
            injection count).
        seed: Controls the draw and the surrogate.
        preprocessing: Per-column scale overrides (V45ls).
        workspace: If given, working files are written here and kept. If
            omitted, they are discarded on success and kept on failure.

    Raises:
        SumoInputError: Distributions do not cover the variables exactly, or a
            log-scale variable's distribution cannot support log sampling.
        SumoEngineError: Dakota failed; the error carries the surviving run
            directory and the tail of Dakota's stderr.
    """
    with SumoSession(
        samples, variables, response, preprocessing=preprocessing, workspace=workspace
    ) as session:
        return session.fit().sample_uncertainty(
            distributions=distributions,
            num_samples=num_samples,
            n_histograms=n_histograms,
            seed=seed,
        )


def summarize_uncertainty(
    draws: UncertaintyDraws,
    *,
    num_bins: int | None = None,
) -> UncertaintySummary:
    """Recompute the histogram + boxplot statistics from propagated draws.

    Pure function (V52rc): no engine run. Given the ``UncertaintyDraws`` from
    ``sample_uncertainty`` it reproduces exactly the bin statistics the old
    fused call returned, so the summary is always re-derivable from its data.

    Args:
        draws: The propagation result to summarise.
        num_bins: Optional explicit histogram bin count.

    Raises:
        SumoInputError: The draws carry no values to summarise.
    """
    values = np.asarray(draws.propagated, dtype=float)
    if values.size == 0:
        raise SumoInputError("The draws carry no propagated values to summarise.")
    summary = summarize_uncertainty_samples(values, num_bins=num_bins)
    return UncertaintySummary(
        response=draws.response,
        seed=draws.seed,
        bins_start=summary["bins_start"],
        bins_end=summary["bins_end"],
        bin_means=summary["bin_means"],
        bin_stds=summary["bin_stds"],
        q1=summary["q1"],
        median=summary["median"],
        q3=summary["q3"],
        whisker_min=summary["whisker_min"],
        whisker_max=summary["whisker_max"],
        outliers=summary["outliers"],
        mean=summary["mean"],
        std=summary["std"],
        minimum=summary["min"],
        maximum=summary["max"],
    )
