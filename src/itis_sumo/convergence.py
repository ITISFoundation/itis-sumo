"""CV convergence: how every CV metric moves with sample size (V17kb).

Two public functions per the §I cap: the sweep, and the exponential fit whose
`asymptotic` flag carries the zero-floor and floored variants (V18wp; the
T56jw cap exception was folded into this kwarg by T58md). The sweep takes no
metric argument — each point carries the accuracy trio and the coverage family,
read off the bundle the CV pass already produces so none of it costs an extra
rerun (V16wq, V55nw) — which makes "which metric do I look at" a presentation
decision (`render_convergence(curve, metric=...)`) rather than a second sweep.
"""

from itis_sumo.engine.evaluation import (
    compute_cv_convergence,
    fit_convergence_exponential,
)

__all__ = ["compute_cv_convergence", "fit_convergence_exponential"]
