"""V50kv import-surface test: consumers touch the manifest, nothing else.

The public surface is root `itis_sumo.__all__` ∪ `SuMo` methods ∪
`report.__all__` (T56jw). Consumer-shaped code — examples and documentation
snippets — must import ONLY through those names; internal modules
(`itis_sumo.engine`, `itis_sumo.tabular`, `itis_sumo.surrogate`, ...) are
in-package business. This test is the CI enforcement of that rule; the
in-repo mirror tests (tests/test_dakota_*, test_data_preprocessor, ...) are
unit tests of the internals and exempt by construction.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The public root manifest itself resolves to `itis_sumo` exactly; report is a
# sanctioned namespace; the remaining public modules are §I's per-module
# surfaces, reachable as `itis_sumo.<module>` (V50kv: surface = manifest ∪
# SuMo methods ∪ report.__all__; per-module import is the manifest's layout).
PUBLIC_PREFIXES = (
    "itis_sumo",
    "itis_sumo.report",
)

_PUBLIC_MODULES = {
    "itis_sumo",
    "itis_sumo.report",
}


def _imported_modules(source: str, path: Path) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno or 0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno or 0, node.module))
    return found


@pytest.mark.parametrize(
    "path",
    sorted((REPO / "examples").rglob("*.py")) + sorted((REPO / "docs").rglob("*.md")),
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_consumer_surfaces_import_only_the_public_namespace(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        # markdown code fences only; prose mentions are documentation talk.
        # EVERY line of every fence counts (B11sc: the first line after the
        # fence opener used to be dropped, hiding the import lines themselves)
        source = "\n".join(
            "\n".join(block.splitlines())
            for block in re.findall(r"```[^\n]*\n(.*?)```", source, re.DOTALL)
        )
        if not source.strip():
            return
        haystack = "\n".join(
            line for line in source.splitlines() if re.match(r"\s*(from|import) ", line)
        )
        modules = re.findall(r"(?:from|import) (itis_sumo[\w.]*)", haystack)
        for module in modules:
            assert module in _PUBLIC_MODULES, (
                f"{path}: consumer code imports non-public module {module!r} "
                "(V50kv: import from the root manifest or itis_sumo.report only)"
            )
        return

    for lineno, module in _imported_modules(source, path):
        if not module.startswith("itis_sumo"):
            continue
        assert module in _PUBLIC_MODULES, (
            f"{path}:{lineno}: consumer example imports non-public module "
            f"{module!r} (V50kv)"
        )


def test_dissolved_names_stay_dissolved() -> None:
    """T56jw killed the old namespaces; nothing may resurrect them."""
    import importlib

    import itis_sumo

    for dead in ("api", "core", "config", "evaluate", "preprocess", "data", "utils"):
        assert not hasattr(itis_sumo, dead), f"itis_sumo.{dead} must not exist (V50kv)"
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"itis_sumo.{dead}")


def test_internal_packages_exist_but_are_not_exported() -> None:
    """engine/tabular/surrogate are real modules in-repo, not manifest members."""
    import importlib

    import itis_sumo

    assert set(itis_sumo.__all__) & {"engine", "tabular", "surrogate"} == set()
    for internal in ("engine", "tabular", "surrogate"):
        importlib.import_module(f"itis_sumo.{internal}")  # in-repo importable
