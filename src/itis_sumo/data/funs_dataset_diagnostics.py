"""Dataset diagnostics: the narrow, versioned entrypoint flaskapi/mmux_vite
consumes for scale/distribution/outlier information (SPEC.md V16qf, T18ry).

Detection is stitched from pieces promoted off the confidential incubator
branch: `select_variable_scale`/`auto_select_distributions` (T48np) for the
scale/distribution half, and `detect_raw_outliers` (this module, V46np) for
the outlier-surfacing half — the raw-value Tukey detector BRANCH_CONSOLIDATION.md
§3 recorded as "new work, not a port". Flag-only, like `_tukey_outlier_mask`
does for CV residuals (V17kb): nothing here drops data, callers decide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from itis_sumo.data.funs_data_processing import auto_select_distributions

Scale = Literal["linear", "log"]
Distribution = Literal["constant", "uniform", "normal"]


@dataclass
class OutlierSummary:
    count: int
    indices: list[int]
    fence_low: float
    fence_high: float


@dataclass
class VariableDiagnostics:
    name: str
    scale: Scale
    distribution: Distribution
    confident: bool
    outliers: OutlierSummary | None = None


@dataclass
class DatasetDiagnostics:
    inputs: dict[str, VariableDiagnostics]
    outputs: dict[str, VariableDiagnostics]
    detail: dict[str, Any] | None = None


def detect_raw_outliers(
    values: Sequence[float] | np.ndarray,
    *,
    scale: Scale = "linear",
    k: float = 1.5,
) -> OutlierSummary:
    """Flag raw column values outside the Tukey IQR fence (Q1 - k*IQR,
    Q3 + k*IQR) — the raw-column counterpart of `evaluate`'s private
    `_tukey_outlier_mask` (which operates on CV residuals only, V17kb).
    Flags, never drops: the caller decides what a flag means.

    `scale` is the space the fence is *computed* in (V46np): `"log"` requires
    strictly-positive values, computes the fence in ln-space and returns
    `fence_low`/`fence_high` mapped back (`exp`) into the values' own units.
    That makes the log fence the GEOMETRIC analogue `(q1·(q1/q3)^k,
    q3·(q3/q1)^k)` — deliberately NOT the arithmetic fence re-expressed,
    which is the point: linear-space fences over-flag the benign right tail
    of multiplicative/lognormal quantities (the flip test flags ~4x the
    injected count), so imposing them on a log-selected variable manufactures
    phantom outliers. Quantiles DO commute with the affine log-basis change
    (ln = ln10·log10), so the log base can never change which rows are
    flagged — asserted, not assumed.

    NaN entries are excluded from the quartile estimate (nanpercentile) and
    are never flagged (comparison against a finite fence is False for NaN).
    A zero-IQR column yields coincident fences; only values strictly outside
    are flagged (so a constant column flags nothing).
    """
    arr = np.asarray(values, dtype=float)
    if scale not in ("linear", "log"):
        raise ValueError(f"Unknown scale: {scale!r}")
    finite = np.isfinite(arr)
    if scale == "log":
        if not np.all(arr[finite] > 0):
            raise ValueError(
                "detect_raw_outliers(scale='log') requires strictly positive "
                "values; non-positive rows are a scale-selection error"
            )
        with np.errstate(divide="ignore"):
            x = np.where(finite, np.log(np.where(finite, arr, 1.0)), np.nan)
    else:
        x = arr

    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    flagged = finite & ((x < lower) | (x > upper))

    if scale == "log":
        fence_low, fence_high = float(np.exp(lower)), float(np.exp(upper))
    else:
        fence_low, fence_high = float(lower), float(upper)
    return OutlierSummary(
        count=int(np.sum(flagged)),
        indices=[int(i) for i in np.nonzero(flagged)[0]],
        fence_low=fence_low,
        fence_high=fence_high,
    )


def analyze_dataset(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    alpha: float = 0.05,
    include_detail: bool = False,
) -> DatasetDiagnostics:
    """Single stitched entrypoint for scale/distribution/outlier diagnostics
    on a training dataset — the only itis-sumo call flaskapi should need for
    this feature (V16qf).

    Scale/distribution come from `auto_select_distributions` (the same
    decision the incubator's NIH/Merck studies used); outliers from
    `detect_raw_outliers` computed in each variable's own selected scale
    (V46np). Detection is per column of `df` as given — if the caller keeps
    a hand-built `log_`-prefixed column, that column's raw (log-space)
    values are what get inspected.

    Args:
        df: training data, one column per variable.
        input_cols: names of `df` columns to treat as inputs.
        output_cols: names of `df` columns to treat as outputs.
        alpha: significance level for the scale/distribution fit tests.
        include_detail: if True, populate `DatasetDiagnostics.detail` with the
            raw per-candidate fit statistics; omitted by default so the
            stable summary shape doesn't grow with internal diagnostic detail.

    Returns:
        `DatasetDiagnostics`, JSON-serializable via `dataclasses.asdict()`.
    """
    all_cols = list(input_cols) + list(output_cols)
    _, decisions = auto_select_distributions(df, all_cols, alpha=alpha)

    def per_column(col: str) -> VariableDiagnostics:
        decision = decisions[col]
        scale: Scale = (
            decision["scale"] if decision["distribution"] != "constant" else "linear"
        )
        values = df[col].to_numpy(dtype=float)
        return VariableDiagnostics(
            name=col,
            scale=scale,
            distribution=decision["distribution"],
            confident=bool(decision["confident"]),
            outliers=detect_raw_outliers(values, scale=scale),
        )

    return DatasetDiagnostics(
        inputs={col: per_column(col) for col in input_cols},
        outputs={col: per_column(col) for col in output_cols},
        detail={col: dict(decisions[col]) for col in all_cols}
        if include_detail
        else None,
    )
