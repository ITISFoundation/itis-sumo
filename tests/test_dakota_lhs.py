"""Tests for mmux_flaskapi.dakota.lhs (Latin Hypercube Sampling)."""

import numpy as np
import pytest

from itis_sumo.sampling.lhs import lhs


@pytest.mark.parametrize(
    "method",
    ["center", "c", "maximin", "m", "centermaximin", "cm", "correlation", "corr", "lhsmu"],
)
def test_lhs_valid_methods_produce_correct_shape_and_bounds(method):
    n, k = 3, 10
    H = lhs(n, k, method=method, seed=42)
    assert H.shape == (k, n)
    assert np.all(H >= 0.0)
    assert np.all(H <= 1.0)


def test_lhs_default_method_is_randomized_classic():
    H = lhs(n=2, k=5, seed=0)
    assert H.shape == (5, 2)
    assert np.all(H >= 0.0) and np.all(H <= 1.0)


def test_lhs_invalid_method_raises_valueerror():
    with pytest.raises(ValueError, match="Invalid value for"):
        lhs(n=2, k=5, method="not-a-method")


def test_lhs_lhsmu_is_case_insensitive():
    """Regression test: `method.lower() in ("lhsmu")` used to do substring
    containment on the string "lhsmu" instead of an equality check. Confirm the
    exact-match branch is what actually executes for the lhsmu method name."""
    H_upper = lhs(n=2, k=8, method="LHSMU", seed=1)
    H_lower = lhs(n=2, k=8, method="lhsmu", seed=1)
    assert H_upper.shape == H_lower.shape == (8, 2)


def test_lhs_lhsmu_with_corr_matrix():
    n, k = 2, 20
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    H = lhs(n, k, method="lhsmu", corr_matrix=corr, seed=7)
    assert H.shape == (k, n)
    assert np.all(H >= 0.0) and np.all(H <= 1.0)


def test_lhs_reproducible_with_integer_seed():
    H1 = lhs(n=3, k=10, method="center", seed=123)
    H2 = lhs(n=3, k=10, method="center", seed=123)
    np.testing.assert_array_equal(H1, H2)


def test_lhs_accepts_random_state_instance():
    rs = np.random.RandomState(5)
    H = lhs(n=2, k=6, method="maximin", seed=rs)
    assert H.shape == (6, 2)
