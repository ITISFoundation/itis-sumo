"""Collection-time rules that hold for the whole suite.

`publish.yml`'s release-gate job installs the BARE core wheel — no plotly, no marimo, because V54hz
keeps presentation deps in the `[report]` extra. Marker deselection cannot save that job: pytest
IMPORTS every test module during collection before `-m` filters anything, so a heavy module that is
merely deselected still raises `ModuleNotFoundError` and interrupts the run. That is how B25vx
blocked every alpha publish from `0.2.0a8` to `0.2.0a12` — `verify` died, `publish` was skipped, and
`auto-tag` kept reporting success because its job really is only to tag.

So a heavy module must leave *collection* when its dependency is absent. Where heavy modules RUN is
the markers' job (`integration` / `analytical`), and what may never enter the gate at all is guarded
by `tests/repo/test_test_taxonomy.py` (V57jk), which recomputes the mapping below from the test
tree's own imports.
"""

import importlib.util

# Module path (relative to this file) -> the packages it imports at module level and the core wheel
# therefore does not provide.
#
# Per module, deliberately, and NOT one shared "is any presentation dep missing" switch: the dev/CI
# environment carries plotly (it is in the `dev` group as well as in `[report]`) but NOT marimo
# (`[report]` only), so a shared switch hid the entire report surface from every CI run — ~40 tests
# that never ran, reported as green. Absence is asked of the dependency each module actually needs.
PRESENTATION_MODULES: dict[str, tuple[str, ...]] = {
    # `from itis_sumo import report` pulls plotly in (V54hz); marimo is never imported, the notebook
    # template is only parsed as text.
    "report/test_report_surface.py": ("plotly",),
}

collect_ignore = [
    module
    for module, deps in PRESENTATION_MODULES.items()
    if any(importlib.util.find_spec(dep) is None for dep in deps)
]
