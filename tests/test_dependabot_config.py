"""Regression tests for SPEC.md V19nd/V20qx: Dependabot policy is explicit.

Each ecosystem must produce one grouped weekly update on Monday at 03:00 in
the repository's explicit timezone. The known itis-dakota incompatibility
must remain ignored and tracked until the modernization task resolves it.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".github" / "dependabot.yml"


def _update_blocks() -> list[str]:
    content = CONFIG.read_text()
    return re.findall(
        r"(?ms)^  - package-ecosystem:.*?(?=^  - package-ecosystem:|\Z)", content
    )


def test_v19nd_dependabot_updates_are_grouped_and_scheduled():
    blocks = _update_blocks()

    assert len(blocks) == 2
    assert {
        re.search(r"package-ecosystem: (\S+)", block).group(1) for block in blocks
    } == {
        "github-actions",
        "uv",
    }

    for block in blocks:
        assert "target-branch: develop" in block
        assert "interval: weekly" in block
        assert "day: monday" in block
        assert 'time: "03:00"' in block
        assert "timezone: Europe/Zurich" in block
        assert re.search(
            r"(?m)^    groups:\n      [\w-]+:\n        patterns:\n          - \"\*\"",
            block,
        )


def test_v20qx_itis_dakota_ignore_is_documented_and_tracked():
    content = CONFIG.read_text()

    assert 'dependency-name: "itis-dakota"' in content
    assert "T16mo" in content
    assert "interface-cache regression" in content
