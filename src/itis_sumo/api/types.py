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

from itis_sumo.api.errors import SumoInputError

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

    def __post_init__(self) -> None:
        # V47st: a declared interval must be increasing; catching it at
        # construction beats an inverted axis surfacing mid-engine later.
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.maximum <= self.minimum
        ):
            raise SumoInputError(
                f"DistributionSpec bounds must increase: maximum "
                f"({self.maximum}) <= minimum ({self.minimum})"
            )


Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class DomainSpec:
    """Where a variable is allowed to be explored (not what its uncertainty is).

    Optimization walks a domain looking for the best point; it does not need,
    and MOGA specifically cannot use, a real-world uncertainty shape (SPEC
    T27fr keeps this split explicit rather than overloading DistributionSpec).
    """

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        # V47st: an inverted (or degenerate) box is always a caller mistake --
        # a silently descending grid axis or a mid-engine MOGA failure.
        if self.maximum <= self.minimum:
            raise SumoInputError(
                f"DomainSpec bounds must increase: maximum "
                f"({self.maximum}) <= minimum ({self.minimum})"
            )

    def as_engine_dict(self) -> dict[str, float | str]:
        return {"distribution": "uniform", "min": self.minimum, "max": self.maximum}


@dataclass(frozen=True)
class ParetoFrontResult:
    """Pareto-optimal points from a multi-objective optimization."""

    objectives: dict[str, Direction]
    variables: tuple[str, ...]
    data: dict[str, list[float]]


@dataclass(frozen=True)
class UncertaintyResult:
    """Histogram and summary statistics from propagating explicit uncertainty
    through a surrogate's own predictive uncertainty."""

    response: str
    distributions: dict[str, DistributionSpec]
    seed: int
    bins_start: float
    bins_end: float
    bin_means: list[float]
    bin_stds: list[float]
    q1: float
    median: float
    q3: float
    whisker_min: float
    whisker_max: float
    outliers: list[float]
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class OrderMasses:
    """Unique ANOVA order masses of one response's variance partition.

    ``first_order`` is M1 = sum of the first-order indices, ``second_order`` is
    M2 = sum of the unordered second-order pairs (each pair once), and
    ``third_and_higher`` is the closure residual R = 1 - M1 - M2, so
    ``M1 + M2 + R = 1`` holds by construction. Bootstrap CIs come from the same
    shared-row resample as the indices, so they preserve estimator covariance
    and the closure per replicate. R's CI covering zero means it is unresolved
    from sampling noise; ``heuristic_noise_floor`` (median CI half-width of the
    first/total indices) is an explicitly rough comparator, not a verdict.
    """

    first_order: float
    second_order: float
    third_and_higher: float
    first_order_ci_low: float
    first_order_ci_high: float
    second_order_ci_low: float
    second_order_ci_high: float
    third_and_higher_ci_low: float
    third_and_higher_ci_high: float
    heuristic_noise_floor: float


@dataclass(frozen=True)
class SobolResult:
    """First-, total-, and second-order sensitivity indices.

    The indices describe sensitivity over the exploration DOMAIN, not over
    modeller distributions (V26dd): ``domains`` are the boxes actually sampled
    (explicit or auto-inferred from the observed bounds), ``fixed`` holds the
    variables that were constant in the samples and therefore pinned (zero
    variance contribution by construction), and ``effective_config`` shows the
    scale each column was sampled on.

    ``second_order`` holds the exact joint-pair estimator
    ``S_ij = Var(E[Y|X_i,X_j])/V - S_i - S_j``, valid for any input count.
    ``order_contributions`` is ``None`` exactly when the sample output variance
    is zero -- there is no variance to partition, so the masses are undefined
    rather than silently ``(0, 0, 0)``.
    """

    response: str
    indices: dict[str, dict[str, float]]
    second_order: dict[str, dict[str, float]]
    order_contributions: OrderMasses | None
    seed: int
    domains: dict[str, DomainSpec]
    fixed: dict[str, float]
    effective_config: dict[str, VariableSpec]


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
class CorrelationResult:
    """Pearson and Spearman sensitivity of a response to each input variable."""

    response: str
    coefficients: dict[str, dict[str, float]]
    seed: int | None = None


@dataclass(frozen=True)
class CVAccuracyMetrics:
    """Error metrics computed from cross-validation predictions."""

    response: str
    root_mean_squared: float
    sum_abs: float
    mean_abs: float
    max_abs: float
    seed: int


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
