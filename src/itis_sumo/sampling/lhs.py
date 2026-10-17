"""
LHS implementation. This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
The present code is a modification of the pyDOE package implementation with some extras, including
an optional seed argument for experiments reproduction.

Only the "classic" method is kept: every product path (api.generate_lhs_samples
-> _lhs) draws classic designs, and the maximimin/centered/correlation/lhsmu
variants pyDOE carries had no caller outside their own tests (ponytail A3).
"""

import numpy as np

__all__ = ["lhs"]


def lhs(
    n: int,
    k: int,
    seed: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Generate a classic latin-hypercube design
    Parameters
    ----------
    n : int
        The number of factors to generate samples for
    k : int
        The number of samples to generate for each factor
    Optional
    --------
    seed : np.random.RandomState, int
         The seed and random draws
    Returns
    -------
    H : 2d-array
        An n-by-k design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    """
    if seed is None:
        rng = np.random.RandomState()
    elif not isinstance(seed, np.random.RandomState):
        rng = np.random.RandomState(seed)
    else:
        rng = seed

    return _lhsclassic(n, k, rng)


def _lhsclassic(n, k, randomstate):
    # generate the intervals
    cut = np.linspace(0, 1, k + 1)

    # fill points uniformly in each interval
    u = randomstate.rand(k, n)
    a = cut[:k]
    b = cut[1 : k + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j] * (b - a) + a

    # make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = randomstate.permutation(range(k))
        H[:, j] = rdpoints[order, j]

    return H
