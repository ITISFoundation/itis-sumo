"""Headless smoke test: surrogate -> CV -> Sobol on a synthetic problem.

Runs the whole MetaModeling pipeline through the public API with a real
Dakota run (itis-dakota wheel, no Flask, no subprocess). Verifies T11rt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itis_sumo.evaluate.funs_evaluate import (
    evaluate_sobol_indices,
    evaluate_sumo,
    evaluate_sumo_crossvalidation,
)
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor
from itis_sumo.utils.helpers import create_run_dir

N_TRAINING_SAMPLES = 30
SEED = 42


def make_training_data(n: int = N_TRAINING_SAMPLES) -> pd.DataFrame:
    """Synthetic data: stress = f(length, width) + noise."""
    rng = np.random.default_rng(SEED)
    length = rng.uniform(0.0, 1.0, size=n)
    width = rng.uniform(0.0, 1.0, size=n)
    stress = 3.0 * length + 2.0 * width**2 + 0.5 * np.sin(10.0 * length) + rng.normal(0, 0.05, n)
    return pd.DataFrame({"length": length, "width": width, "stress": stress})


def main() -> int:
    run_dir: Path = create_run_dir(Path.cwd(), dir_name="smoke")

    train_raw = make_training_data()

    # --- Preprocess (raw names -> Dakota-mapped x1/x2/y1) ---
    preprocessor = DataPreprocessor()
    preprocessor.setup_variables(["length", "width"], ["stress"])
    preprocessor.fit(train_raw)
    train_processed = preprocessor.transform(train_raw)
    training_file = run_dir / "df_processed_jobs.dat"
    train_processed.to_csv(training_file, sep=" ", index=False)

    # --- 1. Surrogate evaluation on a grid ---
    grid = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    grid_raw = pd.DataFrame(
        {"length": grid[0].ravel(), "width": grid[1].ravel()}
    )
    grid_processed = preprocessor.transform(grid_raw)
    eval_samples_file = run_dir / "eval_samples.dat"
    grid_processed.to_csv(eval_samples_file, sep=" ", index=False)

    preds = evaluate_sumo(
        run_dir,
        training_file,
        eval_samples_file,
        ["x1", "x2"],
        "y1",
    )
    y_hat = np.asarray(preds["y1_hat"])
    assert len(y_hat) == len(grid_raw), "grid evaluation count mismatch"
    print(f"[1/3] surrogate grid evaluation OK ({len(y_hat)} points, "
          f"pred range [{y_hat.min():.3f}, {y_hat.max():.3f}])")

    # --- 2. Cross-validation quality metrics ---
    cv_metrics = evaluate_sumo_crossvalidation(
        run_dir, training_file, ["x1", "x2"], "y1", N_CROSS_VALIDATION=5
    )
    assert cv_metrics, "CV produced no metrics"
    root_meansq = float(cv_metrics["y1"]["root_mean_squared"])
    print(f"[2/3] CV OK: root_mean_squared={root_meansq:.4f} (5-fold)")

    # --- 3. Sobol' sensitivity indices ---
    distributions = {
        "length": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "width": {"distribution": "uniform", "min": 0.0, "max": 1.0},
    }
    sobol = evaluate_sobol_indices(
        run_dir,
        training_file,
        ["length", "width"],
        "y1",
        distributions,
        preprocessor,
        seed=SEED,
    )
    sobol_vars = sobol["sobol"]
    assert set(sobol_vars.keys()) == {"length", "width"}, "sobol keys mismatch"
    for var, indices in sobol_vars.items():
        print(
            f"[3/3] Sobol {var}: main={indices['main']:.3f} "
            f"total={indices['total']:.3f}"
        )

    print(f"\nAll headless smoke checks passed. Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
