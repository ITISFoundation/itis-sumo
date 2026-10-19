"""Uncertainty propagation through a fitted SuMo (T58md retires these fuses for open composition)."""

from __future__ import annotations

__all__ = [
    "evaluate_correlations",
    "evaluate_uncertainty",
]
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from itis_sumo.surrogate._session import (
    SumoSession,
)
from itis_sumo.types import (
    DEFAULT_SEED,
    CorrelationResult,
    DistributionSpec,
    PreprocessingSpec,
    UncertaintyResult,
)


def evaluate_uncertainty(
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
) -> UncertaintyResult:
    """Propagate explicit per-variable uncertainty through the surrogate.

    Draws ``num_samples`` from ``distributions``, evaluates the surrogate once,
    then repeats ``n_histograms`` times injecting the surrogate's own predictive
    uncertainty, returning a histogram + boxplot summary in the response's
    original units.
    """
    with SumoSession(
        samples, variables, response, preprocessing=preprocessing, workspace=workspace
    ) as session:
        return session.fit().uncertainty(
            distributions=distributions,
            num_samples=num_samples,
            n_histograms=n_histograms,
            seed=seed,
        )


def evaluate_correlations(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    distributions: Mapping[str, DistributionSpec],
    num_samples: int = 1000,
    seed: int = DEFAULT_SEED,
    preprocessing: PreprocessingSpec | None = None,
    workspace: Path | None = None,
) -> CorrelationResult:
    """Correlate a response against every variable over the SAME Monte Carlo
    sample set the surrogate predicts.

    Draws ``num_samples`` from ``distributions``, evaluates the fitted surrogate
    once, and reports Pearson + Spearman of each variable's samples against the
    prediction (original units, original names). This is the sampling-through-
    the-surrogate correlation the web UI needs -- distinct from
    ``compute_correlations``, which correlates columns of a table the caller
    already owns.

    Args:
        samples: Training samples (the surrogate is fitted on these).
        variables: Input variable columns.
        response: Response column, named as in the table.
        distributions: Per-variable Monte Carlo distributions; must cover the
            variables exactly.
        num_samples: Monte Carlo sample size.
        seed: Controls the draw and the surrogate.
        preprocessing: Per-column scale overrides (V45ls): each column's samples
            and the prediction are correlated on their declared scale, so a
            log-scale variable's Pearson moves while its Spearman is provably
            unchanged.
        workspace: If given, working files are written here and kept. If
            omitted, they are discarded on success and kept on failure.

    Raises:
        SumoInputError: Distributions do not cover the variables exactly, or a
            log-scale variable's distribution cannot support log sampling.
    """
    with SumoSession(
        samples, variables, response, preprocessing=preprocessing, workspace=workspace
    ) as session:
        return session.fit().correlations(
            distributions=distributions, num_samples=num_samples, seed=seed
        )
