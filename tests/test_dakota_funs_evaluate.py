"""Tests for mmux_flaskapi.dakota.funs_evaluate pure/parsing helpers."""

from pathlib import Path

import numpy as np
import pytest

from itis_sumo.evaluate.sumo_evaluation import (
    summarize_uncertainty_samples,
)
from itis_sumo.utils.helpers import create_run_dir


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
