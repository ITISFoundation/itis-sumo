"""Exploration of a fitted SuMo: axis sweeps and grid evaluation."""

from __future__ import annotations

__all__ = [
    "evaluate_along_axes",
    "evaluate_grid",
]
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from itis_sumo.surrogate._session import (
    SumoSession,
)
from itis_sumo.types import (
    AlongAxesResult,
    GridResult,
    PreprocessingSpec,
)


def evaluate_along_axes(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    at: Mapping[str, float] | None = None,
    points_per_variable: int = 21,
    preprocessing: PreprocessingSpec | None = None,
    workspace: Path | None = None,
) -> AlongAxesResult:
    """Sweep each variable in turn, holding the others still.

    This is the shape behind a one-dimensional profile plot: for each variable,
    the response is predicted across that variable's observed range while every
    other variable stays at a fixed value.

    Args:
        samples: One row per sample, one column per variable or response.
        variables: Columns to treat as inputs.
        response: Column to predict.
        at: Values at which to hold the variables that are not being swept.
            Any variable left out is held at its mean across the samples,
            which is also what happens when ``at`` is omitted entirely.
        points_per_variable: How many points to evaluate along each sweep.
        preprocessing: Optional per-column overrides. Omit for sensible defaults.
        workspace: If given, working files are written here and kept. If omitted,
            they are discarded on success and kept on failure.

    Raises:
        SumoInputError: The samples or configuration cannot support this run.
        SumoResultError: The run finished without producing any sweep.
        SumoEngineError: Dakota failed; the error carries the surviving run
            directory and the tail of Dakota's stderr.
    """
    with SumoSession(
        samples,
        variables,
        response,
        preprocessing=preprocessing,
        workspace=workspace,
    ) as session:
        return session.fit().along_axes(at=at, points_per_variable=points_per_variable)


def evaluate_grid(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    grid_variables: Sequence[str],
    at: Mapping[str, float] | None = None,
    points_per_variable: int = 21,
    preprocessing: PreprocessingSpec | None = None,
    workspace: Path | None = None,
) -> GridResult:
    """Evaluate a surrogate across a grid of selected variables."""
    with SumoSession(
        samples, variables, response, preprocessing=preprocessing, workspace=workspace
    ) as session:
        return session.fit().grid(
            grid_variables=grid_variables,
            at=at,
            points_per_variable=points_per_variable,
        )
