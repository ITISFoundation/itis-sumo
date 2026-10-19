"""CV convergence: how prediction error shrinks with sample size (V17kb).

§I cap note (T56jw decision): three public names until T58md folds
`fit_convergence_exponential_asymptotic` into
`fit_convergence_exponential(..., asymptotic=True)`, restoring the <=2 cap.
"""

from itis_sumo.engine.evaluation import (
    compute_cv_convergence,
    fit_convergence_exponential,
    fit_convergence_exponential_asymptotic,
)

__all__ = [
    "compute_cv_convergence",
    "fit_convergence_exponential",
    "fit_convergence_exponential_asymptotic",
]
