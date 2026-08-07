"""
Tests for correlation/sensitivity indices (#470).

Covers the pure `compute_correlation_indices` function (synthetic correlated /
uncorrelated data). The `/flask/dakota/compute_correlation_indices` endpoint
tests stay in mmux/vite (not ported).
"""

import numpy as np
import pandas as pd
import pytest

from itis_sumo.data.funs_data_processing import compute_correlation_indices


class TestComputeCorrelationIndices:
    """Unit tests for the pure correlation-computation function."""

    def test_perfectly_correlated_variable(self):
        """A linear input->output relationship yields pearson/spearman ≈ 1."""
        rng = np.random.default_rng(42)
        x1 = rng.uniform(-1, 1, size=200)
        output = 3.0 * x1 + 2.0  # perfectly linear, positive correlation

        correlations = compute_correlation_indices({"x1": x1.tolist()}, output.tolist(), ["x1"])

        assert correlations["x1"]["pearson"] == pytest.approx(1.0, abs=1e-6)
        assert correlations["x1"]["spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_perfectly_anticorrelated_variable(self):
        """An inverse linear relationship yields pearson/spearman ≈ -1."""
        rng = np.random.default_rng(7)
        x1 = rng.uniform(-1, 1, size=200)
        output = -5.0 * x1 + 1.0

        correlations = compute_correlation_indices({"x1": x1.tolist()}, output.tolist(), ["x1"])

        assert correlations["x1"]["pearson"] == pytest.approx(-1.0, abs=1e-6)
        assert correlations["x1"]["spearman"] == pytest.approx(-1.0, abs=1e-6)

    def test_uncorrelated_variable_near_zero(self):
        """An input independent of the output yields correlation close to 0."""
        rng = np.random.default_rng(123)
        n = 5000
        x1 = rng.uniform(-1, 1, size=n)
        output = rng.uniform(-1, 1, size=n)  # independent of x1

        correlations = compute_correlation_indices({"x1": x1.tolist()}, output.tolist(), ["x1"])

        assert abs(correlations["x1"]["pearson"]) < 0.05
        assert abs(correlations["x1"]["spearman"]) < 0.05

    def test_multiple_input_vars_one_response_per_var(self):
        """One entry per requested input var, sensitive var stands out from noise vars."""
        rng = np.random.default_rng(1)
        n = 1000
        x_sensitive = rng.uniform(-1, 1, size=n)
        x_noise = rng.uniform(-1, 1, size=n)
        output = 10.0 * x_sensitive

        correlations = compute_correlation_indices(
            {"x_sensitive": x_sensitive.tolist(), "x_noise": x_noise.tolist()},
            output.tolist(),
            ["x_sensitive", "x_noise"],
        )

        assert set(correlations.keys()) == {"x_sensitive", "x_noise"}
        assert abs(correlations["x_sensitive"]["pearson"]) > 0.9
        assert abs(correlations["x_noise"]["pearson"]) < 0.2

    def test_accepts_dataframe_input(self):
        """DataFrame input is equivalent to a dict-of-lists input."""
        rng = np.random.default_rng(99)
        x1 = rng.uniform(-1, 1, size=100)
        output = 2.0 * x1

        df = pd.DataFrame({"x1": x1})
        correlations_df = compute_correlation_indices(df, output.tolist(), ["x1"])
        correlations_dict = compute_correlation_indices(
            {"x1": x1.tolist()}, output.tolist(), ["x1"]
        )

        assert correlations_df == correlations_dict

    def test_empty_input_vars_raises(self):
        """Empty input_vars list is rejected."""
        with pytest.raises(ValueError, match="input_vars cannot be empty"):
            compute_correlation_indices({"x1": [1.0, 2.0]}, [1.0, 2.0], [])

    def test_missing_variable_raises(self):
        """Requesting a variable absent from input_samples raises ValueError."""
        with pytest.raises(ValueError, match="not found in input samples"):
            compute_correlation_indices({"x1": [1.0, 2.0, 3.0]}, [1.0, 2.0, 3.0], ["x2"])

    def test_mismatched_lengths_raises(self):
        """Input/output sample length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Sample length mismatch"):
            compute_correlation_indices({"x1": [1.0, 2.0, 3.0]}, [1.0, 2.0], ["x1"])
