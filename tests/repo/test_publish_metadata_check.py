"""The release path's own assertions ! have been rehearsed against a real wheel (B29jm, V58dq).

`publish.yml`'s `verify` job gates `publish`, so an assertion in that job is a claim about shipping —
and this one was wrong. It asserted the built wheel advertises the `[report]` extra by grepping for
`extra == "report"`, but hatchling emits `extra == 'report'` with SINGLE quotes, so the pattern could
never match a hatchling-built wheel. `verify` therefore failed on a CORRECT wheel, `publish`
(`needs: verify`) stayed skipped, and `v0.2.0a14` … `v0.2.0a16` were tagged and never reached
TestPyPI — while `auto-tag` reported success, because its job really is only to tag. `a14`/`a15` never
got this far (B27qx's collection error came first), which is how a third defect in the same chain
stayed hidden behind two others.

Same root cause as B27qx, one level up: a rule enforced by prose instead of by something that can
fail. So the guard here is not a second copy of the pattern — it EXECUTES the workflow's own block,
extracted from `publish.yml` between its sentinels, over METADATA as the build actually emits it. The
snippet therefore cannot drift from what CI runs, and it is the CI-side mirror of
`tests/repo/test_bump_alpha_version.py` importing the bump script instead of restating its rules.
"""

import os
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
PUBLISH_WORKFLOW = REPO / ".github/workflows" / "publish.yml"

STEP_NAME = "Report-deps are available"
START = "# check:report-extra:start"
END = "# check:report-extra:end"

# Header block captured from dist/itis_sumo-0.2.0a16-py3-none-any.whl's METADATA (the artifact whose
# publication this step blocked), Classifiers elided as irrelevant here. What matters is kept exactly
# as hatchling wrote it: SINGLE-quoted markers, and a folded continuation header (`License:`), which
# is why the block parses headers instead of scanning lines.
HATCHLING_METADATA = """\
Metadata-Version: 2.3
Name: itis-sumo
Version: 0.2.0a16
Summary: Surrogate Modeling functionality for IT'IS Foundation / ZMT Modeling Intelligence suite
License: Copyright (C) IT'IS Foundation - All Rights Reserved
         Unless required by applicable law or agreed to in writing,
         Licensor provides the Work (and each Contributor provides
         its Contributions) on an "AS IS" BASIS
Requires-Dist: itis-dakota==6.24.13
Requires-Dist: numpy>=2.0,<3.0
Requires-Dist: pandas>=2.2,<3.0
Requires-Dist: pydantic>=2.0,<3.0
Requires-Dist: scipy>=1.15,<2.0
Requires-Dist: plotly>=5.24 ; extra == 'report'
Requires-Dist: marimo>=0.9 ; extra == 'report'
Requires-Python: >=3.12, <3.15
Provides-Extra: report
"""

# The form the retired grep demanded. Kept as a test, not as the implementation: whichever quoting a
# backend picks is its business, so the assertion ! depend on it (B29jm was exactly that dependency).
PIP_STYLE_METADATA = HATCHLING_METADATA.replace(
    "extra == 'report'", 'extra == "report"'
).replace("0.2.0a16", "0.0.0")

CORE_ONLY_METADATA = (
    HATCHLING_METADATA.replace("Requires-Dist: plotly>=5.24 ; extra == 'report'\n", "")
    .replace("Requires-Dist: marimo>=0.9 ; extra == 'report'\n", "")
    .replace("Provides-Extra: report\n", "")
)

EXTRA_WITHOUT_DEPS_METADATA = HATCHLING_METADATA.replace(
    "Requires-Dist: plotly>=5.24 ; extra == 'report'\n", ""
).replace("Requires-Dist: marimo>=0.9 ; extra == 'report'\n", "")


def _run_block(workflow_text: str) -> str:
    """The workflow's check block, verbatim, so CI's assertion ∧ this rehearsal cannot diverge."""
    lines = workflow_text.splitlines()
    edges = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith((START, END))
    ]
    assert len(edges) >= 2, (
        f"{PUBLISH_WORKFLOW.name} lost the {START!r}/{END!r} sentinels around the [report]-extra"
        f" assertion. They are how {Path(__file__).name} runs THAT block instead of a copy of it;"
        " deleting them deletes the only rehearsal the release-path assertion has (V58dq)."
    )
    # The block lives inside a YAML literal scalar; only its own relative indentation survives to
    # the shell, so dedent to the same top-level script CI hands to `python -`.
    body = textwrap.dedent("\n".join(lines[edges[0] + 1 : edges[-1]]))
    assert "import" in body, f"nothing executable between {START!r} and {END!r}"
    return body


def _step_body(workflow_text: str) -> str:
    """The whole `run:` body of one step, dedented as the shell receives it."""
    lines = workflow_text.splitlines()
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == f"- name: {STEP_NAME}"
        )
    except StopIteration as exc:  # pragma: no cover - the guard IS the failure
        raise AssertionError(
            f"{PUBLISH_WORKFLOW.name} no longer has a {STEP_NAME!r} step; the [report] extra is"
            f" what makes `itis-sumo[report]` installable, so its assertion stays (V54hz)"
        ) from exc
    run_at = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].strip() == "run: |"
    )
    indent = len(lines[run_at]) - len(lines[run_at].lstrip())
    body: list[str] = []
    for line in lines[run_at + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return textwrap.dedent("\n".join(body))


def _wheel(tmp_path: Path, metadata: str) -> Path:
    """A minimal `*.whl` carrying METADATA — a wheel is a zip, so the fixture is one too."""
    wheel = tmp_path / "dist" / "itis_sumo-0.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("itis_sumo-0.0.0.dist-info/METADATA", metadata)
    return wheel


def _run_check(tmp_path: Path, metadata: str) -> subprocess.CompletedProcess[str]:
    block = _run_block(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))
    wheel = _wheel(tmp_path, metadata)
    return subprocess.run(
        [sys.executable, "-c", block, str(wheel)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_step_passes_on_metadata_as_hatchling_emits_it(tmp_path: Path) -> None:
    """The bug this guard exists for: a CORRECT wheel failed the step, so a passing rehearsal here
    is the difference between an alpha and eight of them."""
    result = _run_check(tmp_path, HATCHLING_METADATA)
    assert result.returncode == 0, (
        f"the release-path assertion rejects metadata hatchling really produced:\n"
        f"{result.stderr}"
    )


def test_the_step_does_not_depend_on_the_backends_quoting(tmp_path: Path) -> None:
    """Single- OR double-quoted markers both pass — B29jm was a quoting-style dependency, ⊥ a defect."""
    result = _run_check(tmp_path, PIP_STYLE_METADATA)
    assert result.returncode == 0, result.stderr


def test_the_step_fails_when_the_extra_advertises_nothing(tmp_path: Path) -> None:
    """It ! bite: an extra that names no presentation deps is the regression V54hz guards."""
    result = _run_check(tmp_path, EXTRA_WITHOUT_DEPS_METADATA)
    assert result.returncode != 0, (
        "the assertion passes on a `[report]` extra that lists neither plotly nor marimo; it is"
        " decoration, not a guard"
    )
    assert {"marimo", "plotly"} <= set(re.findall(r"\w+", result.stderr)), result.stderr


def test_the_step_fails_when_the_extra_is_absent(tmp_path: Path) -> None:
    result = _run_check(tmp_path, CORE_ONLY_METADATA)
    assert result.returncode != 0, result.stderr
    assert "report" in result.stderr


@pytest.mark.skipif(
    os.name != "posix",
    reason="the step is a bash heredoc and the release runs on ubuntu runners",
)
def test_the_whole_step_survives_a_fixture_wheel(tmp_path: Path) -> None:
    """Not just the Python block: the step AS WRITTEN, shell included.

    Extracted from the workflow and run with bash, so a broken heredoc, a stray `${...}` or a lost
    `set -euo pipefail` fails here instead of at a release — the failure this file is about arrived
    from the shell half, not from the Python half.
    """
    body = _step_body(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))
    _wheel(tmp_path, HATCHLING_METADATA)
    script = tmp_path / "step.sh"
    script.write_text(body, encoding="utf-8")
    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"the step as written fails on a valid wheel:\n{result.stderr}"
    )
