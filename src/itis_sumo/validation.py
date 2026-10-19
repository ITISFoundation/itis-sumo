"""Model validation: k-fold cross-validation of a trained SuMo (V8df)."""

from __future__ import annotations

__all__ = [
    "cross_validate",
    "evaluate_cv_metrics",
]
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from itis_sumo.engine.evaluation import compute_cv_accuracy_metrics
from itis_sumo.errors import SumoInputError
from itis_sumo.surrogate._session import (
    SumoSession,
)
from itis_sumo.types import (
    DEFAULT_SEED,
    CrossValidationResult,
    CVAccuracyMetrics,
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


def evaluate_cv_metrics(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    preprocessing: PreprocessingSpec | None = None,
    folds: int = 5,
    seed: int = DEFAULT_SEED,
    workspace: Path | None = None,
) -> CVAccuracyMetrics:
    """Fit cross-validation predictions and return stable accuracy metrics."""
    result = cross_validate(
        samples,
        variables,
        response,
        preprocessing=preprocessing,
        folds=folds,
        seed=seed,
        workspace=workspace,
    )
    metrics = compute_cv_accuracy_metrics(result.observed, result.predicted)
    return CVAccuracyMetrics(response=response, seed=result.seed, **metrics)
