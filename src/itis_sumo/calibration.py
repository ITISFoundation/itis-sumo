"""Calibration of a SuMo's predictive spread (V19cz).

Two public functions per the §I cap; the paired-t/Cohen-d machinery lives
behind `compute_cv_diagnostics` (V16wq: one diagnostics call per CV subset).
"""

from itis_sumo.engine.evaluation import compute_coverage, compute_cv_diagnostics

__all__ = ["compute_coverage", "compute_cv_diagnostics"]
