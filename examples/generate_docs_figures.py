"""Generate the worked-example figures embedded in the docs site.

Every figure here is produced by calling itis-sumo's real public API (real Dakota
GP fits, real LHS sampler, real Sobol' pipeline) against small analytical problems
with known closed-form answers — nothing is hand-drawn or faked. Regenerate with:

    uv run --group docs python examples/generate_docs_figures.py

Outputs land in docs/assets/examples/, referenced from docs/theory/*.md.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itis_sumo.evaluate.funs_evaluate import evaluate_sobol_indices, evaluate_sumo
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor
from itis_sumo.sampling.lhs import lhs
from itis_sumo.utils.helpers import create_run_dir

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "examples"
SEED = 42


def _write_processed(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, sep=" ", index=False)
    return path


def fig_gp_fit_uncertainty(run_dir: Path) -> None:
    """GP surrogate mean + 95% prediction interval on a sparsely-sampled sine."""
    rng = np.random.default_rng(SEED)
    x_train = np.sort(rng.uniform(0.0, 2 * np.pi, size=9))
    y_train = np.sin(x_train)
    train_file = _write_processed(
        pd.DataFrame({"x": x_train, "y": y_train}), run_dir / "gp_train_processed.txt"
    )

    x_eval = np.linspace(0.0, 2 * np.pi, 200)
    eval_file = _write_processed(pd.DataFrame({"x": x_eval}), run_dir / "gp_eval.txt")

    result = evaluate_sumo(run_dir, train_file, eval_file, ["x"], "y")
    y_hat = np.asarray(result["y_hat"])
    y_std = np.asarray(result["y_std_hat"])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x_eval, np.sin(x_eval), "k--", lw=1.2, label="true f(x) = sin(x)")
    ax.plot(x_eval, y_hat, color="#3f51b5", lw=1.8, label="GP mean prediction")
    ax.fill_between(
        x_eval,
        y_hat - 1.96 * y_std,
        y_hat + 1.96 * y_std,
        color="#3f51b5",
        alpha=0.2,
        label="95% prediction interval (±1.96 σ̂)",
    )
    ax.scatter(x_train, y_train, color="#e91e63", zorder=5, s=40, label="training points (n=9)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("GP surrogate: mean prediction and uncertainty band")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gp_fit_uncertainty.png", dpi=150)
    plt.close(fig)
    print("[1/3] gp_fit_uncertainty.png")


def fig_lhs_vs_random() -> None:
    """LHS vs. plain uniform random: LHS covers each 1D marginal stratum exactly once."""
    n = 25
    rng = np.random.default_rng(SEED)
    lhs_pts = lhs(2, n, method="maximin", seed=SEED)
    rand_pts = rng.uniform(0.0, 1.0, size=(n, 2))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharex=True, sharey=True)
    for ax, pts, title in zip(
        axes, [rand_pts, lhs_pts], ["Plain uniform random (n=25)", "Latin Hypercube, maximin (n=25)"]
    ):
        ax.scatter(pts[:, 0], pts[:, 1], color="#3f51b5", s=35)
        for edge in np.linspace(0, 1, n + 1):
            ax.axvline(edge, color="0.9", lw=0.5, zorder=0)
            ax.axhline(edge, color="0.9", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.suptitle("Design-of-experiments: LHS stratifies every marginal, random sampling doesn't", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lhs_vs_random.png", dpi=150)
    plt.close(fig)
    print("[2/3] lhs_vs_random.png")


def fig_sobol_ishigami(run_dir: Path) -> None:
    """Sobol' first-order and total-order indices for the Ishigami function, GP-surrogate-based."""

    def ishigami(x: np.ndarray) -> np.ndarray:
        return np.sin(x[:, 0]) + 7.0 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])

    rng = np.random.default_rng(SEED)
    x_train = rng.uniform(-np.pi, np.pi, size=(300, 3))
    train_raw = pd.DataFrame(x_train, columns=["x1", "x2", "x3"])
    train_raw["y"] = ishigami(x_train)

    preprocessor = DataPreprocessor()
    preprocessor.setup_variables(["x1", "x2", "x3"], ["y"])
    preprocessor.fit(train_raw)
    train_file = _write_processed(preprocessor.transform(train_raw), run_dir / "sobol_train_processed.txt")

    distributions = {
        "x1": {"distribution": "uniform", "min": -np.pi, "max": np.pi},
        "x2": {"distribution": "uniform", "min": -np.pi, "max": np.pi},
        "x3": {"distribution": "uniform", "min": -np.pi, "max": np.pi},
    }
    sobol = evaluate_sobol_indices(
        run_dir, train_file, ["x1", "x2", "x3"], "y1", distributions, preprocessor, seed=SEED
    )["sobol"]

    variables = ["x1", "x2", "x3"]
    main = [sobol[v]["main"] for v in variables]
    total = [sobol[v]["total"] for v in variables]
    ref_main = [0.314, 0.442, 0.0]
    ref_total = [0.558, 0.442, 0.244]

    x_pos = np.arange(len(variables))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x_pos - 1.5 * width, main, width, label="first-order (GP surrogate)", color="#3f51b5")
    ax.bar(x_pos - 0.5 * width, ref_main, width, label="first-order (analytical)", color="#9fa8da")
    ax.bar(x_pos + 0.5 * width, total, width, label="total-order (GP surrogate)", color="#e91e63")
    ax.bar(x_pos + 1.5 * width, ref_total, width, label="total-order (analytical)", color="#f48fb1")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variables)
    ax.set_ylabel("Sobol' index")
    ax.set_title("Ishigami function: Sobol' indices, GP surrogate vs. analytical reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sobol_ishigami.png", dpi=150)
    plt.close(fig)
    print("[3/3] sobol_ishigami.png")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir(Path.cwd(), dir_name="docs_figures")

    fig_gp_fit_uncertainty(run_dir)
    fig_lhs_vs_random()
    fig_sobol_ishigami(run_dir)

    print(f"\nAll docs figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
