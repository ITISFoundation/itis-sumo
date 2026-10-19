"""Global sensitivity: Sobol' indices and correlation structure of a fitted SuMo."""

from __future__ import annotations

__all__ = [
    "compute_correlations",
    "evaluate_sobol",
]
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from itis_sumo.errors import SumoInputError
from itis_sumo.surrogate._session import (
    SumoSession,
    column_scale,
)
from itis_sumo.tabular._guards import _reject_unused_overrides
from itis_sumo.tabular.processing import (
    compute_correlation_indices,
)
from itis_sumo.types import (
    DEFAULT_SEED,
    CorrelationResult,
    DomainSpec,
    PreprocessingSpec,
    SobolResult,
)


def evaluate_sobol(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    domains: Mapping[str, DomainSpec] | None = None,
    fixed: Mapping[str, float] | None = None,
    preprocessing: PreprocessingSpec | None = None,
    seed: int = DEFAULT_SEED,
    workspace: Path | None = None,
) -> SobolResult:
    """Compute Sobol sensitivity indices over the exploration domain.

    The sampling box is DOMAIN vocabulary (V26dd): pass per-variable
    :class:`DomainSpec` boxes, or omit them and the box is auto-inferred from
    the observed sample bounds; a variable constant in the samples stays
    fixed. ``fixed`` pins caller-stated factors at given values (the
    domain-vocabulary freeze; a variable cannot be both boxed and fixed).
    Draws are uniform across the box -- log-uniform for a ``scale="log"``
    override (V44ls). Modeller distributions are NOT an input here; they drive
    :func:`evaluate_uncertainty`.

    Second-order pairs come from the exact joint-pair estimator (valid for any
    input count), and the result carries the M1/M2/R order-mass partition --
    ``None`` when the sample variance is zero and the fractions undefined.
    """
    with SumoSession(
        samples, variables, response, preprocessing=preprocessing, workspace=workspace
    ) as session:
        return session.fit().sobol(domains=domains, fixed=fixed, seed=seed)


def compute_correlations(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    preprocessing: PreprocessingSpec | None = None,
) -> CorrelationResult:
    """Compute response correlations from a caller-owned sample table.

    Each column is correlated on its own scale (V45ls): Pearson is not
    invariant to log, so a log-scaled variable's coefficient shifts; Spearman
    is monotone-invariant and therefore unchanged.
    """
    missing = sorted((set(variables) | {response}) - set(samples.columns))
    if missing:
        raise SumoInputError(f"Samples do not contain columns: {missing}")
    _reject_unused_overrides(preprocessing, [*variables, response])
    spec = preprocessing or PreprocessingSpec()
    try:
        coefficients = compute_correlation_indices(
            samples,
            samples[response].tolist(),
            list(variables),
            input_scales={v: column_scale(spec, v) for v in variables},
            output_scale=column_scale(spec, response),
        )
    except ValueError as exc:
        raise SumoInputError(str(exc)) from exc
    return CorrelationResult(response=response, coefficients=coefficients)
