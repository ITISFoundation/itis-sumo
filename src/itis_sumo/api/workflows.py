"""The workflows itis-sumo offers its consumers.

Each function takes a table of samples plus configuration, and returns a typed
result in the caller's own units and column names. Everything in between --
training-file layout, Dakota configuration, normalization, internal variable
renaming, run directories, inverse transforms -- belongs to itis-sumo and is not
reachable from here (SPEC §G).

They are one-shot by design: each call fits a surrogate and queries it once.
Internally they are already split into a fit step and a query step, so that
holding on to a fitted surrogate across several queries can be offered later
without changing what these functions do (SPEC V27fq).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from itis_sumo.api._session import SumoSession
from itis_sumo.api.errors import SumoInputError
from itis_sumo.api.types import (
    DEFAULT_SEED,
    AlongAxesResult,
    CrossValidationResult,
    PreprocessingSpec,
)


def cross_validate(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    preprocessing: PreprocessingSpec | None = None,
    folds: int = 5,
    seed: int = DEFAULT_SEED,
    workspace: Path | None = None,
) -> CrossValidationResult:
    """Predict every sample from a surrogate that never saw it.

    The samples are split into ``folds`` groups; each group is predicted by a
    surrogate trained on the others. The result is therefore an honest picture of
    how the surrogate behaves on data it has not memorised.

    Args:
        samples: One row per sample, one column per variable or response.
        variables: Columns to treat as inputs.
        response: Column to predict.
        preprocessing: Optional per-column overrides. Omit for sensible defaults.
        folds: Number of cross-validation groups.
        seed: Controls how samples are assigned to folds.
        workspace: If given, working files are written here and kept. If omitted,
            they are discarded on success and kept on failure.

    Raises:
        SumoInputError: The samples or configuration cannot support this run.
        SumoResultError: The run finished without producing predictions.
        SumoEngineError: Dakota failed; the error carries the surviving run
            directory and the tail of Dakota's stderr.
    """
    if folds < 2:
        raise SumoInputError(f"Cross-validation needs at least 2 folds, got {folds}")

    with SumoSession(
        samples,
        variables,
        response,
        preprocessing=preprocessing,
        workspace=workspace,
    ) as session:
        return session.fit().cross_validate(folds=folds, seed=seed)


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
