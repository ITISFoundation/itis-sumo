"""Tests for mmux_flaskapi.dakota.funs_evaluate pure/parsing helpers."""

from pathlib import Path

import numpy as np
import pytest

from itis_sumo.engine.evaluation import (
    summarize_uncertainty_samples,
)
from itis_sumo.engine.helpers import create_run_dir


def test_create_run_dir_creates_unique_directory(tmp_path):
    dir1 = create_run_dir(tmp_path, dir_name="sampling")
    dir2 = create_run_dir(tmp_path, dir_name="sampling")

    assert Path(dir1).is_dir()
    assert Path(dir2).is_dir()
    assert dir1 != dir2
    assert str(dir1).startswith(str(tmp_path / "runs"))


class TestSummarizeUncertaintySamples:
    def test_reports_mean_std_and_range_from_flattened_pool(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=10.0, scale=2.0, size=(50, 500))

        summary = summarize_uncertainty_samples(values)

        assert summary["mean"] == pytest.approx(10.0, abs=0.2)
        assert summary["std"] == pytest.approx(2.0, abs=0.2)
        assert summary["bins_start"] < summary["q1"] < summary["median"] < summary["q3"]
        assert len(summary["bin_means"]) == len(summary["bin_stds"])

    def test_flags_far_outliers_beyond_the_whiskers(self):
        values = np.full((10, 100), 5.0)
        values[0, 0] = 500.0

        summary = summarize_uncertainty_samples(values)

        assert 500.0 in summary["outliers"] or summary["max"] == pytest.approx(500.0)

    def test_reports_p95_and_p99_of_the_flattened_pool(self):
        """The upper tail a UQ reader asks for (mmux_vite #458): percentiles of
        the whole pool, not the clipped histogram edges."""
        values = np.arange(1, 101, dtype=float).reshape(1, 100)

        summary = summarize_uncertainty_samples(values)

        assert summary["p95"] == pytest.approx(95.05)
        assert summary["p99"] == pytest.approx(99.01)
        assert summary["q3"] < summary["p95"] < summary["p99"] <= summary["max"]

    def test_a_gaussian_pool_lands_on_the_normal_quantiles(self):
        rng = np.random.default_rng(3)
        values = rng.normal(loc=0.0, scale=1.0, size=(40, 1000))

        summary = summarize_uncertainty_samples(values)

        assert summary["p95"] == pytest.approx(1.6449, abs=0.05)
        assert summary["p99"] == pytest.approx(2.3263, abs=0.08)

    def test_p99_stays_the_percentile_when_the_histogram_range_is_widened(self):
        """A degenerate (constant) pool widens bins_start/bins_end so the
        histogram has any width at all; the reported percentiles must not move
        with that presentation decision."""
        values = np.full((5, 50), 5.0)

        summary = summarize_uncertainty_samples(values)

        assert summary["bins_end"] > 5.0
        assert summary["p95"] == pytest.approx(5.0)
        assert summary["p99"] == pytest.approx(5.0)
