"""Design generation: Latin-hypercube and grid sample tables."""

from __future__ import annotations

__all__ = [
    "generate_grid_samples",
    "generate_lhs_samples",
]
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from itis_sumo.engine.helpers import create_run_dir
from itis_sumo.engine.lhs import lhs as _lhs
from itis_sumo.errors import SumoInputError
from itis_sumo.surrogate._session import (
    column_scale,
)
from itis_sumo.tabular._guards import _reject_unused_overrides
from itis_sumo.tabular.processing import (
    create_grid_samples,
    load_data,
    scale_distribution,
)
from itis_sumo.types import (
    DEFAULT_SEED,
    DomainSpec,
    PreprocessingSpec,
)


def generate_lhs_samples(
    domains: Mapping[str, DomainSpec],
    n_samples: int,
    *,
    preprocessing: PreprocessingSpec | None = None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Draw a Latin-hypercube design over the given variable domains.

    A ``scale="log"`` domain (via ``preprocessing``) is filled log-uniformly --
    the unit design maps through the same ``scale_distribution`` every value
    producer uses (V45ls); linear domains are unchanged.

    Args:
        domains: Variable name -> allowed range to draw from.
        n_samples: Number of sample rows to generate.
        preprocessing: Per-column scale overrides (default: all linear).
        seed: Controls the draw.

    Raises:
        SumoInputError: No domains given, or a log-scale domain is not strictly
            positive.
    """
    if not domains:
        raise SumoInputError("At least one variable domain is required.")
    names = list(domains)
    _reject_unused_overrides(preprocessing, names)
    design = _lhs(len(names), n_samples, seed=seed)
    columns = {}
    for i, name in enumerate(names):
        dom = domains[name]
        try:
            dist = scale_distribution(
                dom.minimum, dom.maximum, scale=column_scale(preprocessing, name)
            )
        except ValueError as exc:
            raise SumoInputError(f"'{name}': {exc}") from exc
        columns[name] = dist.ppf(design[:, i])
    return pd.DataFrame(columns)


def generate_grid_samples(
    domains: Mapping[str, DomainSpec],
    points_per_variable: Mapping[str, int],
    *,
    preprocessing: PreprocessingSpec | None = None,
    workspace: Path | None = None,
) -> pd.DataFrame:
    """Generate a full-factorial grid of samples over the given variable domains.

    A ``scale="log"`` domain (via ``preprocessing``) is gridded log-uniformly:
    the axis is built in log space (``ln`` bounds to the linspace, then exp back),
    so its points are geometrically spaced -- matching the LHS/UQ log spacing
    (V45ls). Linear domains are unchanged.

    Args:
        domains: Variable name -> allowed range to draw from.
        points_per_variable: Variable name -> number of grid points along that axis.
        preprocessing: Per-column scale overrides (default: all linear).
        workspace: If given, working files are written here and kept. If omitted,
            they are discarded on success and kept on failure.

    Raises:
        SumoInputError: No domains given, a variable is missing its point count,
            or a log-scale domain is not strictly positive.
    """
    if not domains:
        raise SumoInputError("At least one variable domain is required.")
    names = list(domains)
    _reject_unused_overrides(preprocessing, names)
    missing = [name for name in names if name not in points_per_variable]
    if missing:
        raise SumoInputError(f"Missing points_per_variable for: {', '.join(missing)}")

    log_names = set()
    build_bounds: dict[str, tuple[float, float]] = {}
    for name in names:
        dom = domains[name]
        if column_scale(preprocessing, name) == "log":
            if dom.minimum <= 0:
                raise SumoInputError(
                    f"'{name}': log-scale domain must be strictly positive, got "
                    f"[{dom.minimum}, {dom.maximum}]"
                )
            log_names.add(name)
            build_bounds[name] = (
                float(np.log(dom.minimum)),
                float(np.log(dom.maximum)),
            )
        else:
            build_bounds[name] = (dom.minimum, dom.maximum)

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
            mins=[build_bounds[name][0] for name in names],
            cut_values=[
                (build_bounds[name][0] + build_bounds[name][1]) / 2 for name in names
            ],
            maxs=[build_bounds[name][1] for name in names],
            n_points_per_dimension=[points_per_variable[name] for name in names],
        )
        grid = load_data(grid_file)[names].astype(float)
        for name in log_names:
            grid[name] = np.exp(grid[name])
        return grid
    finally:
        if workspace is None:
            shutil.rmtree(run_dir, ignore_errors=True)
