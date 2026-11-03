"""Multi-objective optimization over a fitted SuMo (MOGA)."""

from __future__ import annotations

__all__ = [
    "optimize",
]
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from itis_sumo.surrogate._session import (
    optimize_pareto_front,
)
from itis_sumo.types import (
    Direction,
    DomainSpec,
    ParetoFrontResult,
    PreprocessingSpec,
)


def optimize(
    samples: pd.DataFrame,
    variables: Sequence[str],
    objectives: Mapping[str, Direction],
    *,
    domains: Mapping[str, DomainSpec],
    max_evaluations: int = 1000,
    preprocessing: PreprocessingSpec | None = None,
    workspace: Path | None = None,
) -> ParetoFrontResult:
    """Find the Pareto-optimal trade-off front across one or more objectives.

    Unlike the other workflows, this fits one surrogate per objective over a
    domain (where exploration is allowed), not a real-world uncertainty
    distribution -- MOGA cannot use anything but a uniform domain (SPEC T27fr).
    A ``scale="log"`` override on a variable explores it in log space; on an
    objective it fits in ln space and reports the front exp-restored, in the
    original units the public result contract promises.
    """
    return optimize_pareto_front(
        samples,
        variables,
        objectives,
        domains=domains,
        max_evaluations=max_evaluations,
        preprocessing=preprocessing,
        workspace=workspace,
    )
