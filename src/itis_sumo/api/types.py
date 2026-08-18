"""Configuration and result types for the itis-sumo consumer API.

The vocabulary here is fixed by SPEC VOCAB and is the same vocabulary used in the
documentation: a **sample** is a row, a **variable** (equivalently *parameter*) is
an input column, and a **response** (equivalently *quantity of interest*) is an
output column.

Every result type is a plain frozen dataclass carrying values in the caller's
original units under the caller's original column names, and is JSON-serializable
via :func:`dataclasses.asdict` (SPEC V22rs).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

#: Seed used by every stochastic entrypoint unless the caller overrides it.
#: Fixed rather than required, so that results are reproducible by default
#: without the caller having to think about it (SPEC V25sd).
DEFAULT_SEED = 42

Scale = Literal["linear", "log"]
DistributionKind = Literal["constant", "uniform", "normal"]


@dataclass(frozen=True)
class DistributionSpec:
    """A real-world uncertainty distribution for one variable.

    This is deliberately not a domain object. A domain says where exploration is
    allowed; this says what shape real-world uncertainty has. The two are split
    fully in the fitted-model transformation (SPEC T27fr).
    """

    distribution: DistributionKind
    value: float | None = None
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def as_engine_dict(self) -> dict[str, float | str]:
        """Translate stable public names to the current Dakota adapter shape."""
        values: dict[str, float | str] = {"distribution": self.distribution}
        if self.value is not None:
            values["value"] = self.value
        if self.mean is not None:
            values["mean"] = self.mean
        if self.std is not None:
            values["std"] = self.std
        if self.minimum is not None:
            values["min"] = self.minimum
        if self.maximum is not None:
            values["max"] = self.maximum
        return values


@dataclass(frozen=True)
class SobolResult:
    """First-, total-, and second-order sensitivity indices."""

    response: str
    indices: dict[str, dict[str, float]]
    second_order: dict[str, dict[str, float]]
    seed: int
    distributions: dict[str, DistributionSpec]


@dataclass(frozen=True)
class VariableSpec:
    """An optional override for how one column behaves.

    Expressed in domain terms only: ``scale`` describes the column, it does not
    name a transform. Which transform that implies is itis-sumo's business and
    never appears in a public signature (SPEC V21pf).
    """

    scale: Scale = "linear"


@dataclass(frozen=True)
class PreprocessingSpec:
    """Per-column overrides, keyed by the column's own name.

    Omit this entirely -- the common case -- and suitable defaults are derived
    from the samples themselves.
    """

    overrides: Mapping[str, VariableSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossValidationResult:
    """Held-out predictions for every sample, in the response's original units.

    ``predicted`` and ``predicted_std`` are aligned positionally with the rows of
    the samples that were passed in. A sample whose fold was abandoned by Dakota
    keeps a ``NaN`` prediction and is explained in :attr:`warnings` rather than
    failing the whole run.
    """

    response: str
    observed: list[float]
    predicted: list[float]
    predicted_std: list[float] | None
    warnings: list[str]
    seed: int
    effective_config: dict[str, VariableSpec]


@dataclass(frozen=True)
class AxisSweep:
    """One variable swept across its observed range.

    Every other variable is held fixed for the duration of the sweep.
    """

    variable: str
    x: list[float]
    predicted: list[float]
    predicted_std: list[float] | None = None


@dataclass(frozen=True)
class GridResult:
    """Predictions on a grid, keyed by the original variable names."""

    response: str
    grid_variables: tuple[str, ...]
    data: dict[str, list[float] | list[list[float]]]
    effective_config: dict[str, VariableSpec]


@dataclass(frozen=True)
class AlongAxesResult:
    """One :class:`AxisSweep` per variable, keyed by the variable's name."""

    response: str
    sweeps: dict[str, AxisSweep]
    effective_config: dict[str, VariableSpec]
