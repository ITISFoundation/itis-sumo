"""The test taxonomy IS the release gate's contract (B25vx, V57jk).

`publish.yml`'s verify job installs the BARE core wheel — no plotly, no marimo, because V54hz keeps
presentation deps in the `[report]` extra — and runs `RELEASE_GATE_SELECTOR`. A test module that
imports a presentation dep while staying gate-eligible does not fail loudly: it kills `verify`, which
skips `publish`, and the alpha channel stops shipping while every other CI signal stays green,
because `auto-tag`'s job really is only to tag. That is how `0.2.0a8` … `0.2.0a12` were tagged and
never reached TestPyPI.

Marker deselection cannot save that job, either: pytest IMPORTS every test module during collection
BEFORE `-m` filters anything, so a heavy module that is merely deselected still raises
`ModuleNotFoundError` and interrupts the run. So a heavy module must leave *collection* when its
dependency is absent, by SKIPPING itself at module level (`pytest.skip(..., allow_module_level=True)`
around the imports it cannot satisfy — per module, keyed on what that module actually imports). And
these tests are the cheap rehearsal that keeps the rule honest: an AST pass over the test tree plus
collect-only runs, one of them with the presentation extra made genuinely unimportable. No engine, no
wheel build. A misclassification then dies on the PR instead of two weeks later at a release — because
a rule that lives only in prose is how `0.2.0a15` was tagged and never published (B27qx).
"""

import ast
import contextlib
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).parents[2]

# Single source of truth for what may run in the release gate: UNMARKED means "runs against the bare
# core wheel". Workflows spell this literal out (a workflow cannot import Python), so
# `test_gate_selector_has_not_drifted_out_of_the_workflows` keeps the copies honest.
RELEASE_GATE_SELECTOR = "not integration and not analytical"

HEAVY_LEG_SELECTOR = "integration or analytical"

PUBLISH_WORKFLOW = ".github/workflows/publish.yml"
CI_WORKFLOW = ".github/workflows/ci.yml"

GATE_SELECTOR_WORKFLOWS = (PUBLISH_WORKFLOW, CI_WORKFLOW)

# Only CI runs the heavy leg. `publish.yml` deliberately does NOT: it is the sanity gate, and the
# whole point is that releasing never depends on the heavy half (V57jk).
HEAVY_LEG_WORKFLOWS = (CI_WORKFLOW,)

# Module-level imports the bare core wheel CANNOT satisfy (V54hz), and the installed package each one
# actually needs: `itis_sumo.report` is the presentation layer, so importing it requires plotly even
# though it is one of our own modules. Marimo is never imported by tests — the notebook template is
# parsed as text — but a future notebook-executing test would land here.
IMPORT_REQUIRES: dict[str, tuple[str, ...]] = {
    "plotly": ("plotly",),
    "marimo": ("marimo",),
    "itis_sumo.report": ("plotly",),
}

# A pytest plugin that makes named top-level packages unimportable. `-p` plus PYTHONPATH rehearses
# the BARE-WHEEL gate inside a dev interpreter that does have the extras — the only cheap way to
# catch B25vx's failure mode on a PR rather than on a tag that silently never publishes.
IMPORT_BLOCKER_MODULE = "sumo_import_blocker"

_IMPORT_BLOCKER_SOURCE = '''
"""Make the packages named in SUMO_BLOCK_IMPORTS unimportable for one pytest run."""

import os
import sys

BLOCKED = {name for name in os.environ.get("SUMO_BLOCK_IMPORTS", "").split(",") if name}


class _Blocker:
    """RAISE rather than return None — returning None only means "someone else may answer"."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"{fullname!r} is blocked for this run", name=fullname)
        return None


def pytest_configure(config):
    if BLOCKED:
        sys.meta_path.insert(0, _Blocker())
'''


def _test_modules() -> list[Path]:
    return sorted(
        path
        for path in (REPO / "tests").rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    )


def _dotted(node: ast.AST) -> str | None:
    """`pytest.mark.integration` / `itis_sumo.report.figures` -> the dotted string, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _import_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """The import statements an IMPORT of this module executes.

    That includes the body of a module-level `try:` — `try: import plotly / except ImportError:
    pytest.skip(...)` is still an import-time import, and skipping the scan past it would hide the
    very modules the release gate cannot survive (the guard this file exists for).
    """
    statements: list[ast.stmt] = []
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            statements.append(node)
        elif isinstance(node, ast.Try):
            statements.extend(_import_statements(node.body))
    return statements


def _module_level_imports(source: str, path: Path) -> list[str]:
    """Dotted names a module imports at top level — exactly what a collection pass executes."""
    found: list[str] = []
    tree = ast.parse(source, filename=str(path))
    for node in _import_statements(tree.body):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.extend(
                f"{node.module}.{alias.name}" if alias.name != "*" else node.module
                for alias in node.names
            )
    return found


def _markers_used(source: str, path: Path) -> set[str]:
    """Every `pytest.mark.<name>` appearing in the module (pytestmark assignment or decorator)."""
    tree = ast.parse(source, filename=str(path))
    marks: set[str] = set()
    for node in ast.walk(tree):
        dotted = _dotted(node)
        if dotted and dotted.startswith("pytest.mark."):
            marks.add(dotted.removeprefix("pytest.mark."))
    return marks


def _required_packages(source: str, path: Path) -> set[str]:
    """Installed packages this module needs at IMPORT time, beyond the bare core wheel."""
    required: set[str] = set()
    for name in _module_level_imports(source, path):
        for root, packages in IMPORT_REQUIRES.items():
            if name == root or name.startswith(f"{root}."):
                required |= set(packages)
                break
    return required


def _heavy_modules() -> dict[str, set[str]]:
    """Every test module that cannot be imported against the bare core wheel, with what it needs."""
    heavy: dict[str, set[str]] = {}
    for path in _test_modules():
        required = _required_packages(path.read_text(encoding="utf-8"), path)
        if required:
            heavy[path.relative_to(REPO / "tests").as_posix()] = required
    return heavy


def _declared_markers() -> list[str]:
    with open(REPO / "pyproject.toml", "rb") as file:
        declared = tomllib.load(file)["tool"]["pytest"]["ini_options"]["markers"]
    return [entry.split(":", 1)[0].strip() for entry in declared]


def _collect(selector: str, block_imports: tuple[str, ...] = ()) -> tuple[int, str]:
    """Collect one selection in a fresh interpreter: the same filter the gate applies, minus running.

    `block_imports` makes those packages unimportable in the subprocess, so a dev checkout can
    collect the gate the way the BARE wheel does. Without it the interpreter is this one, extras
    included — which is exactly how a gate-breaking import slipped past this file once (B25vx).
    """
    command = [
        sys.executable,
        "-m",
        "pytest",
        *(["-p", IMPORT_BLOCKER_MODULE] if block_imports else []),
        "-q",
        "--collect-only",
        "-p",
        "no:cacheprovider",
    ]
    environment: dict[str, str] | None = None
    with contextlib.ExitStack() as stack:
        if block_imports:
            plugin_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            (plugin_dir / f"{IMPORT_BLOCKER_MODULE}.py").write_text(
                _IMPORT_BLOCKER_SOURCE, encoding="utf-8"
            )
            environment = {
                **os.environ,
                "PYTHONPATH": str(plugin_dir),
                "SUMO_BLOCK_IMPORTS": ",".join(block_imports),
            }
        command += ["-m", selector, "tests"]
        result = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    return result.returncode, result.stdout + result.stderr


def _skips_on_missing_extra(source: str, path: Path) -> bool:
    """Whether this module leaves COLLECTION when a presentation import fails (V57jk).

    The shape that survives is a module-level `try: import heavy / except ImportError:
    pytest.skip(..., allow_module_level=True)`. Two near-misses both shipped an unpublished tag: a
    bare `except ImportError:` that only printed (collection ran on into a NameError, B27qx), and an
    UNGUARDED module-level skip would silence the module in every environment — a quieter way to lose
    tests than to lose the release.
    """
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not _handler_catches_import_error(handler):
                continue
            if any(_skips_module_level(call) for call in ast.walk(handler)):
                return True
    return False


def _skips_module_level(node: ast.AST) -> bool:
    """A `pytest.skip(..., allow_module_level=True)` call."""
    return (
        isinstance(node, ast.Call)
        and _dotted(node.func) == "pytest.skip"
        and any(
            keyword.arg == "allow_module_level"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    )


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    """Does this `except` clause catch a missing package (the only skip that can save collection)?"""
    names: list[ast.expr | None] = (
        list(handler.type.elts)
        if isinstance(handler.type, ast.Tuple)
        else [handler.type]
    )
    return any(
        isinstance(name, ast.Name) and name.id in {"ImportError", "ModuleNotFoundError"}
        for name in names
    )


# pytest phrases the count differently depending on whether anything was deselected, so accept the
# shapes it actually prints rather than guessing one of them.
_COUNT_PATTERNS = (
    r"(\d+) selected",  # "collected 432 items / 111 deselected / 321 selected"
    r"(\d+)/\d+ tests collected",  # summary line when a selector deselected something
    r"(\d+) tests? collected",  # summary line with nothing deselected
)


def _selected_count(output: str) -> int | None:
    for pattern in _COUNT_PATTERNS:
        match = re.search(pattern, output)
        if match:
            return int(match.group(1))
    return None


def _job_text(workflow: str, job: str) -> str:
    """The YAML body of one job, so guards can talk about a single job instead of a whole file."""
    lines = (REPO / workflow).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.rstrip() == f"  {job}:")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if re.match(r"^  \S", line):  # the next job, at the same indentation
            break
        body.append(line)
    return "\n".join(body)


def test_gate_selection_collects_without_error() -> None:
    """The gate's selection must COLLECT in this environment — that alone is B25vx's failure mode."""
    code, output = _collect(RELEASE_GATE_SELECTOR)
    assert code == 0, (
        f"`pytest -m {RELEASE_GATE_SELECTOR!r}` does not collect cleanly; this is what"
        f" publish.yml's verify job runs against the built wheel:\n{output[-2000:]}"
    )


def test_gate_selection_is_not_degenerate() -> None:
    """A gate that selects nothing passes on nothing (V31vp: never trust 0-collected)."""
    code, output = _collect(RELEASE_GATE_SELECTOR)
    assert code == 0, output[-2000:]

    selected = _selected_count(output)
    assert selected is not None, (
        "could not read a collected count from the gate selection; it may have selected nothing:"
        f"\n{output[-2000:]}"
    )
    assert selected > 100, (
        f"the release-gate selection is implausibly small ({selected} tests) — an accidentally"
        " narrow selection fails green"
    )


def test_gate_selection_collects_with_the_presentation_extra_absent() -> None:
    """The gate the release ACTUALLY runs: bare wheel, so plotly is not importable (B25vx, V57jk).

    `_collect` without `block_imports` runs in this interpreter, where the dev group carries plotly —
    which is why `verify` passed its own rehearsal while the tag it guarded died twice. Blocking the
    import is the cheap version of building the wheel and installing it into a clean venv.
    """
    blocked = sorted(
        {package for packages in _heavy_modules().values() for package in packages}
    )
    assert blocked, (
        "no test module imports a presentation dep; this guard has nothing to pin"
    )

    code, output = _collect(RELEASE_GATE_SELECTOR, block_imports=tuple(blocked))
    assert code == 0, (
        f"`pytest -m {RELEASE_GATE_SELECTOR!r}` dies when {'/'.join(blocked)} is unimportable —"
        " that IS publish.yml's verify job. Deselection cannot help: pytest imports the module while"
        " collecting, so it must skip itself at module level"
        " (`pytest.skip(..., allow_module_level=True)` around the imports it cannot satisfy), never"
        f" swallow the ImportError and carry on.\n{output[-2000:]}"
    )


def test_gate_survival_does_not_lose_gate_eligible_tests() -> None:
    """A heavy module leaving the gate must cost it nothing it could have run (V57jk).

    A module that skips at module level unconditionally, or that is gate-eligible and only survives
    because its imports failed, shrinks the sanity gate every time the extra is missing — silently,
    because the gate stays green. Heavy modules are integration-marked precisely so that this
    comparison is exact: the same count with and without the extras.
    """
    blocked = sorted(
        {package for packages in _heavy_modules().values() for package in packages}
    )
    baseline = _selected_count(_collect(RELEASE_GATE_SELECTOR)[1])
    code, output = _collect(RELEASE_GATE_SELECTOR, block_imports=tuple(blocked))
    assert code == 0, output[-2000:]

    selected = _selected_count(output)
    assert selected == baseline, (
        f"the gate selects {baseline} tests with the extras present but {selected} without them:"
        " a module that skipped its way out of collection was contributing gate-eligible tests."
        " Mark it `integration` instead of leaning on the skip"
    )


def test_every_heavy_module_skips_instead_of_breaking_collection() -> None:
    """Name the module that would take `verify` down, rather than only failing the rehearsal.

    The functional guard above (`test_gate_selection_collects_with_the_presentation_extra_absent`)
    proves the mechanism works; this one says WHICH module broke it, and rejects the two shapes that
    look right: a bare `except ImportError:` that carries on, and an unconditional module-level skip
    that also silences the module in CI's heavy leg.
    """
    unguarded = []
    for module in sorted(_heavy_modules()):
        path = REPO / "tests" / module
        if not _skips_on_missing_extra(path.read_text(encoding="utf-8"), path):
            unguarded.append(module)
    assert not unguarded, (
        f"these modules import a presentation dep at module level without skipping on"
        f" ImportError, so they break the bare-wheel release gate: {unguarded} (V57jk, B25vx)"
    )


def test_gate_selector_has_not_drifted_out_of_the_workflows() -> None:
    """The workflow copies still mean what this module says the gate means."""
    for workflow in GATE_SELECTOR_WORKFLOWS:
        text = (REPO / workflow).read_text(encoding="utf-8")
        assert RELEASE_GATE_SELECTOR in text, (
            f"{workflow} no longer runs {RELEASE_GATE_SELECTOR!r}; if the gate definition changed,"
            " change RELEASE_GATE_SELECTOR here in the same commit"
        )

    for workflow in HEAVY_LEG_WORKFLOWS:
        text = (REPO / workflow).read_text(encoding="utf-8")
        assert HEAVY_LEG_SELECTOR in text, (
            f"{workflow} no longer runs the heavy leg {HEAVY_LEG_SELECTOR!r}; the presentation"
            " surface the gate skips has to be covered somewhere (V57jk)"
        )


def test_publish_verify_job_stays_core_only() -> None:
    """The release gate itself, not just its selection: no extras, no heavy install (V57jk).

    `verify` gates `publish`, so every dependency it installs is a dependency of shipping. Installing
    the presentation extra here is what an earlier revision of this job did — the release then
    inherited the heavy half's failure modes for zero extra information, because CI already covers
    that surface on four platforms from the source tree.
    """
    gate = _job_text(PUBLISH_WORKFLOW, "verify")

    assert RELEASE_GATE_SELECTOR in gate, (
        "publish.yml's verify job no longer runs the gate selection; it is the sanity gate"
    )

    # Scoped to install commands, deliberately: the job also ASSERTS the wheel advertises the extra,
    # which mentions `[report]` without installing it — that step is the point, not a violation.
    installs = [
        line.strip()
        for line in gate.splitlines()
        if not line.strip().startswith("#")
        and any(shape in line for shape in ("pip install", "uv sync", "uv add"))
    ]
    assert installs, (
        "publish.yml's verify job installs nothing; it cannot verify a wheel"
    )
    for line in installs:
        assert "[report]" not in line and "--all-extras" not in line, (
            f"the release gate installs presentation deps ({line!r}) — V54hz parks them in the"
            " `[report]` extra precisely so shipping never depends on them (V57jk)"
        )


def test_every_declared_marker_is_used() -> None:
    """Declared-but-unused markers are how B25vx hid: a `unit` label nothing selected on.

    `--strict-markers` stops un-declared marks; this stops the inverse decay.
    """
    used: set[str] = set()
    for path in _test_modules():
        used |= _markers_used(path.read_text(encoding="utf-8"), path)

    for name in _declared_markers():
        assert name in used, (
            f"marker {name!r} is declared in pyproject.toml but used nowhere — delete it or use it"
            " (a decorative taxonomy is what let a plotly-importing module claim gate eligibility)"
        )
