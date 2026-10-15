"""Tests for mmux_flaskapi.dakota.funs_evaluate pure/parsing helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itis_sumo.core.dakota_object import DakotaObject
from itis_sumo.evaluate.funs_evaluate import (
    _parse_crossvalidation_outputlogs,
    evaluate_sumo_crossvalidation,
    retrieve_csv_result,
    summarize_uncertainty_samples,
)
from itis_sumo.utils.helpers import create_run_dir

SAMPLE_CV_LOG = """
Some preamble text.
Surrogate quality metrics (5-fold CV) for response_1:
      root_mean_squared         0.1234
      sum_abs                   1.2345
      mean_abs                  0.2345
      max_abs                   0.9999
some intervening text about build (training) points here
Surrogate quality metrics (5-fold CV) for response_2:
      root_mean_squared         0.5678
      sum_abs                   nan
      mean_abs                  0.1111
      max_abs                   0.2222
"""


def test_parse_crossvalidation_outputlogs_extracts_all_variables_and_metrics():
    result = _parse_crossvalidation_outputlogs(SAMPLE_CV_LOG, N_CROSS_VALIDATION=5)

    assert set(result.keys()) == {"response_1", "response_2"}
    assert result["response_1"]["root_mean_squared"] == "0.1234"
    assert result["response_1"]["max_abs"] == "0.9999"
    assert result["response_2"]["sum_abs"] == "nan"


def test_parse_crossvalidation_outputlogs_no_matches_returns_message():
    result = _parse_crossvalidation_outputlogs("no metrics here", N_CROSS_VALIDATION=5)
    assert result == {}


def test_parse_crossvalidation_outputlogs_empty_string():
    result = _parse_crossvalidation_outputlogs("", N_CROSS_VALIDATION=5)
    assert result == {}


def test_evaluate_sumo_crossvalidation_parses_captured_dakota_stdout(
    tmp_path, monkeypatch
):
    """Regression test for B22 (V37): evaluate_sumo_crossvalidation must return the
    metrics parsed from the stdout `DakotaObject.run` actually captures to
    `dakota_stdout.txt`, not the metrics from a hardcoded empty string.

    Only `DakotaObject.run` (the actual Dakota engine invocation) is mocked; the
    surrounding conf-file generation and stdout-parsing code runs for real, so a
    regression to `log_output = ""` makes this test fail with `result == {}`
    instead of silently passing (unlike the endpoint-level tests, which accept
    the "No surrogate quality metrics found." fallback as an equally valid
    outcome and therefore don't catch this regression -- see B22 postmortem)."""
    training_file = tmp_path / "df_processed_jobs.dat"
    training_file.write_text("x y\n0.1 1.0\n0.2 2.0\n0.3 3.0\n0.4 4.0\n0.5 5.0\n")

    fake_stdout = (
        "Surrogate quality metrics (5-fold CV) for y:\n"
        "      root_mean_squared         0.1234\n"
        "      sum_abs                   1.2345\n"
        "      mean_abs                  0.2345\n"
        "      max_abs                   0.9999\n"
    )

    def fake_run(self, dakota_conf, run_dir):
        Path(run_dir, "dakota_stdout.txt").write_text(fake_stdout)

    monkeypatch.setattr(DakotaObject, "run", fake_run)

    result = evaluate_sumo_crossvalidation(
        tmp_path, training_file, ["x"], "y", N_CROSS_VALIDATION=5
    )

    assert result == {
        "y": {
            "root_mean_squared": "0.1234",
            "sum_abs": "1.2345",
            "mean_abs": "0.2345",
            "max_abs": "0.9999",
        }
    }


def test_evaluate_sumo_crossvalidation_no_stdout_file_returns_empty(
    tmp_path, monkeypatch
):
    """If DakotaObject.run doesn't produce a stdout file, parsing degrades to {}
    rather than raising (mirrors the `stdout_file.is_file()` guard)."""
    training_file = tmp_path / "df_processed_jobs.dat"
    training_file.write_text("x y\n0.1 1.0\n0.2 2.0\n0.3 3.0\n0.4 4.0\n0.5 5.0\n")

    monkeypatch.setattr(DakotaObject, "run", lambda self, dakota_conf, run_dir: None)

    result = evaluate_sumo_crossvalidation(
        tmp_path, training_file, ["x"], "y", N_CROSS_VALIDATION=5
    )

    assert result == {}


def test_retrieve_csv_result_single_match(tmp_path):
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30], "out": [100, 200, 300]})
    df.to_csv(csv_file, index=False)

    result = retrieve_csv_result(
        str(csv_file), inputs={"x": 2, "y": 20}, outputs=["out"]
    )
    assert result == {"out": 200}


def test_retrieve_csv_result_missing_input_column_raises(tmp_path):
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"x": [1]}).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="not in the csv file"):
        retrieve_csv_result(str(csv_file), inputs={"missing_col": 1})


def test_retrieve_csv_result_no_match_raises(tmp_path):
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="No result found"):
        retrieve_csv_result(str(csv_file), inputs={"x": 999})


def test_retrieve_csv_result_multiple_matches_raises(tmp_path):
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 1], "out": [1, 2]}).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Multiple results found"):
        retrieve_csv_result(str(csv_file), inputs={"x": 1})


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
