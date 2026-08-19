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

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from itis_sumo.api._session import SumoSession, optimize_pareto_front
from itis_sumo.api.errors import SumoInputError
from itis_sumo.api.types import (
    DEFAULT_SEED,
    AlongAxesResult,
    CorrelationResult,
    CrossValidationResult,
    CVAccuracyMetrics,
    Direction,
    DistributionSpec,
    DomainSpec,
    GridResult,
    ParetoFrontResult,
    PreprocessingSpec,
    SobolResult,
    UncertaintyResult,
)
from itis_sumo.data.funs_data_processing import (
    compute_correlation_indices,
    create_grid_samples,
    load_data,
)
from itis_sumo.evaluate.funs_evaluate import compute_cv_accuracy_metrics
from itis_sumo.sampling.lhs import lhs as _lhs
from itis_sumo.utils.helpers import create_run_dir


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


def evaluate_sobol(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    distributions: Mapping[str, DistributionSpec],
    preprocessing: PreprocessingSpec | None = None,
    seed: int = DEFAULT_SEED,
    workspace: Path | None = None,
) -> SobolResult:
    """Compute Sobol sensitivity indices from explicit distributions."""
    with SumoSession(
        samples, variables, response, preprocessing=preprocessing, workspace=workspace
    ) as session:
        return session.fit().sobol(distributions=distributions, seed=seed)


def compute_correlations(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
) -> CorrelationResult:
    """Compute response correlations from a caller-owned sample table."""
    missing = sorted((set(variables) | {response}) - set(samples.columns))
    if missing:
        raise SumoInputError(f"Samples do not contain columns: {missing}")
    try:
        coefficients = compute_correlation_indices(
            samples, samples[response].tolist(), list(variables)
        )
    except ValueError as exc:
        raise SumoInputError(str(exc)) from exc
    return CorrelationResult(response=response, coefficients=coefficients)


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


def optimize(
    samples: pd.DataFrame,
    variables: Sequence[str],
    objectives: Mapping[str, Direction],
    *,
    domains: Mapping[str, DomainSpec],
    max_evaluations: int = 1000,
    workspace: Path | None = None,
) -> ParetoFrontResult:
    """Find the Pareto-optimal trade-off front across one or more objectives.

    Unlike the other workflows, this fits one surrogate per objective over a
    domain (where exploration is allowed), not a real-world uncertainty
    distribution -- MOGA cannot use anything but a uniform domain (SPEC T27fr).
    """
    return optimize_pareto_front(
        samples,
        variables,
        objectives,
        domains=domains,
        max_evaluations=max_evaluations,
        workspace=workspace,
    )


def generate_lhs_samples(
    domains: Mapping[str, DomainSpec],
    n_samples: int,
    *,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Draw a Latin-hypercube design over the given variable domains.

    Args:
        domains: Variable name -> allowed range to draw from.
        n_samples: Number of sample rows to generate.
        seed: Controls the draw.

    Raises:
        SumoInputError: No domains given.
    """
    if not domains:
        raise SumoInputError("At least one variable domain is required.")
    names = list(domains)
    design = _lhs(len(names), n_samples, seed=seed)
    return pd.DataFrame(
        {
            name: design[:, i] * (domains[name].maximum - domains[name].minimum)
            + domains[name].minimum
            for i, name in enumerate(names)
        }
    )


def generate_grid_samples(
    domains: Mapping[str, DomainSpec],
    points_per_variable: Mapping[str, int],
    *,
    workspace: Path | None = None,
) -> pd.DataFrame:
    """Generate a full-factorial grid of samples over the given variable domains.

    Args:
        domains: Variable name -> allowed range to draw from.
        points_per_variable: Variable name -> number of grid points along that axis.
        workspace: If given, working files are written here and kept. If omitted,
            they are discarded on success and kept on failure.

    Raises:
        SumoInputError: No domains given, or a variable is missing its point count.
    """
    if not domains:
        raise SumoInputError("At least one variable domain is required.")
    names = list(domains)
    missing = [name for name in names if name not in points_per_variable]
    if missing:
        raise SumoInputError(f"Missing points_per_variable for: {', '.join(missing)}")

    run_dir = (
        create_run_dir(Path(workspace), "grid-sample")
        if workspace is not None
        else Path(tempfile.mkdtemp(prefix="itis-sumo-"))
    )
    try:
        grid_file = create_grid_samples(
            run_dir=run_dir,
            grid_vars=names,
            input_vars=names,
            mins=[domains[name].minimum for name in names],
            cut_values=[
                (domains[name].minimum + domains[name].maximum) / 2 for name in names
            ],
            maxs=[domains[name].maximum for name in names],
            n_points_per_dimension=[points_per_variable[name] for name in names],
        )
        return load_data(grid_file)[names].astype(float)
    finally:
        if workspace is None:
            shutil.rmtree(run_dir, ignore_errors=True)
