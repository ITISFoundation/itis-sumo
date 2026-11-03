"""CV convergence: how every CV metric moves with sample size (V17kb).

Two public functions per the §I cap: the sweep, and the exponential fit whose
`asymptotic` flag carries the zero-floor and floored variants (V18wp; the
T56jw cap exception was folded into this kwarg by T58md). The sweep takes no
metric argument — each point carries the accuracy trio and the coverage family,
read off the bundle the CV pass already produces so none of it costs an extra
rerun (V16wq, V55nw) — which makes "which metric do I look at" a presentation
decision (`render_convergence(curve, metric=...)`) rather than a second sweep.

Each point also carries a `uq` block: the predictive DISTRIBUTION at that size
(centre, spread, upper tail, and how far each still moves across redraws),
likewise replayed from the same bundle so it costs no engine call (V56qx). It is
a Gaussian replay of the CV run's own numbers, not a propagation over stated
input uncertainty — that is itis-sumo#81.
"""

from itis_sumo.engine.evaluation import (
    compute_cv_convergence,
    fit_convergence_exponential,
)

__all__ = ["compute_cv_convergence", "fit_convergence_exponential"]
