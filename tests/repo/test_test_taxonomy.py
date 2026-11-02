"""The test taxonomy IS the release gate's contract (B25vx, V57jk).

`publish.yml`'s verify job installs the BARE core wheel — no plotly, no marimo, because V54hz keeps
presentation deps in the `[report]` extra — and runs `RELEASE_GATE_SELECTOR`. A test module that
imports a presentation dep while staying gate-eligible does not fail loudly: it kills `verify`, which
skips `publish`, and the alpha channel stops shipping while every other CI signal stays green,
because `auto-tag`'s job really is only to tag. That is how `0.2.0a8` … `0.2.0a12` were tagged and
never reached TestPyPI.

Marker deselection cannot save that job, either: pytest IMPORTS every test module during collection
BEFORE `-m` filters anything, so a heavy module that is merely deselected still raises
`ModuleNotFoundError` and interrupts the run. So heavy modules must leave *collection* when their
dependency is absent — `tests/conftest.py` — and these tests are the cheap rehearsal that keeps both
halves honest: an AST pass over the test tree plus a couple of collect-only runs, no engine, no wheel
build. A misclassification then dies on the PR instead of two weeks later at a release.
"""

import ast
import re
import subprocess
import sys
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

# Read from the conftest rather than copy-pasted, so the guard and the collection exclusion it checks
# cannot drift apart.
_CONFTEST = REPO / "tests" / "conftest.py"


def _conftest_presentation_modules() -> dict[str, tuple[str, ...]]:
    """Ask tests/conftest.py which modules it excludes from collection, and on which packages."""
    tree = ast.parse(_CONFTEST.read_text(encoding="utf-8"), filename=str(_CONFTEST))
    for node in tree.body:
        # Annotated or plain: the conftest carries the type, which makes this an AnnAssign.
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "PRESENTATION_MODULES"
            for target in targets
        ):
            continue
        assert node.value is not None, "PRESENTATION_MODULES has no value to read"
        return dict(ast.literal_eval(node.value))
    raise AssertionError(
        f"{_CONFTEST} no longer defines PRESENTATION_MODULES; the gate's collection exclusion and"
        " this guard have to change in the same commit"
    )


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


def _module_level_imports(source: str, path: Path) -> list[str]:
    """Dotted names a module imports at top level — exactly what a collection pass executes."""
    found: list[str] = []
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
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


def _collect(selector: str) -> tuple[int, str]:
    """Collect one selection in a fresh interpreter: the same filter the gate applies, minus running."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            "-m",
            selector,
            "tests",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


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


def test_no_gate_eligible_module_imports_a_presentation_dep() -> None:
    """A module that imports plotly/marimo/report is NOT gate-eligible — both ways, always.

    Two independent escape hatches close here: it must carry `integration` (so CI's unit leg and the
    gate selection skip it) and it must be declared in `tests/conftest.py` (so it leaves *collection*
    when the dep is absent, which deselection alone cannot do).
    """
    declared = _conftest_presentation_modules()
    found = _heavy_modules()

    for name, required in sorted(found.items()):
        path = REPO / "tests" / name
        assert name in declared, (
            f"tests/{name} imports a presentation dep, so it cannot run against the bare core wheel"
            " the release gate installs (V54hz, B25vx). Add it to PRESENTATION_MODULES in"
            " tests/conftest.py so it leaves collection when that dep is missing"
        )
        assert set(declared[name]) == required, (
            f"tests/{name} needs {sorted(required)} at import time but tests/conftest.py excludes it"
            f" on {sorted(declared[name])}. Keep them equal: an over-broad exclusion drops passing"
            " tests silently whenever the extra dep is absent — the dev group has plotly but not"
            " marimo, so asking for marimo would hide this module from every CI run"
        )
        assert "integration" in _markers_used(path.read_text(encoding="utf-8"), path), (
            f"tests/{name} imports a presentation dep but is not marked `integration` — it would run"
            " in CI's unit leg and inside the release-gate selection (V57jk)"
        )

    for name in declared:
        assert name in found, (
            f"tests/conftest.py excludes {name!r} as presentation-heavy, but nothing there imports a"
            " presentation dep any more — drop the stale entry"
        )


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
