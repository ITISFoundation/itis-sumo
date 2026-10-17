"""Tests for itis_sumo.sampling.lhs (Latin Hypercube Sampling)."""

import numpy as np

from itis_sumo.sampling.lhs import lhs


def test_lhs_default_method_is_randomized_classic():
    H = lhs(n=2, k=5, seed=0)
    assert H.shape == (5, 2)
    assert np.all(H >= 0.0) and np.all(H <= 1.0)


def test_lhs_reproducible_with_integer_seed():
    H1 = lhs(n=3, k=10, seed=123)
    H2 = lhs(n=3, k=10, seed=123)
    np.testing.assert_array_equal(H1, H2)


def test_lhs_accepts_random_state_instance():
    rs = np.random.RandomState(5)
    H = lhs(n=2, k=6, seed=rs)
    assert H.shape == (6, 2)
