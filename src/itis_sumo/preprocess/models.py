"""Sample-count requirements for surrogate construction."""


def required_completed_jobs(input_vars: list[str], floor: int = 5) -> int:
    """Minimum completed jobs needed to build a Dakota (surfpack) GP surrogate.

    Dakota aborts surrogate construction (opaque "Dakota aborted: Unknown error 250",
    internal MODEL_ERROR) when given <= len(input_vars) training points -- confirmed
    empirically: len(input_vars)+1 points build successfully, len(input_vars) points
    abort. `floor` preserves the historical flat minimum for low-dimensional problems.
    """
    return max(floor, len(input_vars) + 1)
