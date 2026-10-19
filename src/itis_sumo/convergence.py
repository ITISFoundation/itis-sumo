"""CV convergence: how prediction error shrinks with sample size (V17kb).

Two public functions per the §I cap: the sweep, and the exponential fit whose
`asymptotic` flag carries the zero-floor and floored variants (V18wp; the
T56jw cap exception was folded into this kwarg by T58md).
"""

from itis_sumo.engine.evaluation import (
    compute_cv_convergence,
    fit_convergence_exponential,
)

__all__ = ["compute_cv_convergence", "fit_convergence_exponential"]
