"""Headless smoke test: fit -> predict -> cross-validate -> Sobol on a synthetic problem.

Runs the whole MetaModeling pipeline through the root manifest (V50kv) with
real Dakota runs (itis-dakota wheel, no Flask, no subprocess). Publish-verify
tooling: if this passes against an installed wheel, the consumer surface works.
Migrated off engine internals by T60gb — the smoke test now exercises exactly
what consumers import.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itis_sumo import fit

N_TRAINING_SAMPLES = 30
SEED = 42


def make_training_data(n: int = N_TRAINING_SAMPLES) -> pd.DataFrame:
    """Synthetic data: stress = f(length, width) + noise."""
    rng = np.random.default_rng(SEED)
    length = rng.uniform(0.0, 1.0, size=n)
    width = rng.uniform(0.0, 1.0, size=n)
    stress = (
        3.0 * length
        + 2.0 * width**2
        + 0.5 * np.sin(10.0 * length)
        + rng.normal(0, 0.05, n)
    )
    return pd.DataFrame({"length": length, "width": width, "stress": stress})


def main() -> int:
    train = make_training_data()
    grid = pd.DataFrame(
        {
            "length": np.repeat(np.linspace(0.0, 1.0, 5), 5),
            "width": np.tile(np.linspace(0.0, 1.0, 5), 5),
        }
    )

    # Keep the working files in ./smoke (V24af: a workspace survives the run).
    workspace = Path.cwd() / "smoke"
    model = fit(train, ["length", "width"], "stress", workspace=workspace)

    # --- 1. Surrogate prediction on a grid ---
    predictions = model.predict(grid)
    values = np.asarray(predictions.values)
    assert len(values) == len(grid), "grid prediction count mismatch"
    assert np.isfinite(values).all(), "prediction produced non-finite values"
    print(
        f"[1/3] surrogate grid prediction OK ({len(values)} points, "
        f"range [{values.min():.3f}, {values.max():.3f}])"
    )

    # --- 2. Cross-validation ---
    cv = model.cross_validate(folds=5, seed=SEED)
    observed = np.asarray(cv.observed, dtype=float)
    predicted = np.asarray(cv.predicted, dtype=float)
    usable = ~np.isnan(observed) & ~np.isnan(predicted)
    assert usable.any(), "CV produced no usable fold predictions"
    rmse = float(np.sqrt(np.mean((observed[usable] - predicted[usable]) ** 2)))
    print(
        f"[2/3] CV OK: {int(usable.sum())}/{len(train)} usable folds, "
        f"rmse={rmse:.4f} (5-fold)"
    )

    # --- 3. Sobol' sensitivity indices ---
    sobol = model.sobol(seed=SEED)
    assert set(sobol.indices) == {"length", "width"}, "sobol keys mismatch"
    for var, indices in sobol.indices.items():
        print(
            f"[3/3] Sobol {var}: main={indices['main']:.3f} "
            f"total={indices['total']:.3f}"
        )

    model.close()
    print(f"\nAll headless smoke checks passed. Workspace: {workspace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
