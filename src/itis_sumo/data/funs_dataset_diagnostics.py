"""Dataset diagnostics schema: the narrow, versioned entrypoint flaskapi/mmux_vite
consumes for scale/distribution/outlier information (SPEC.md V16qf, T18ry).

Schema only for now. The underlying detection logic (`select_variable_scale`,
`auto_select_distributions`, a raw-value outlier detector) is not yet promoted
from the confidential incubator branch — see BRANCH_CONSOLIDATION.md §3 for
per-piece readiness. `analyze_dataset` raises `NotImplementedError` until that
logic is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

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
    raise NotImplementedError(
        "analyze_dataset is scaffolded (schema only, T18ry) — scale/distribution/"
        "outlier detection logic is not yet promoted from feat/nih-in-silico-example; "
        "see BRANCH_CONSOLIDATION.md §3"
    )
