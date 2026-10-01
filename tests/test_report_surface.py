"""`itis_sumo.report` surface contract (V49rp, T54rp): the flat re-export set
automated report notebooks import against stays complete and identical to the
underlying api/data/evaluate objects (the facade must not fork behavior)."""

import importlib

import pytest

from itis_sumo import report

#: modules the facade is allowed to source from, in lookup order
_SOURCE_MODULES = (
    "itis_sumo.api",
    "itis_sumo.data",
    "itis_sumo.evaluate",
    "itis_sumo.evaluate.funs_evaluate",
)


def test_report_all_importable_and_sorted() -> None:
    names = report.__all__
    assert names == sorted(names), "__all__ must stay sorted (ruff RUF022 precedent)"
    for name in names:
        assert hasattr(report, name), f"itis_sumo.report missing {name}"


@pytest.mark.parametrize("name", report.__all__)
def test_reexports_are_identical_objects(name: str) -> None:
    """Facade members must BE the underlying functions, not copies/wrappers —
    so a fix in the engine lands in reports without a second promotion step."""
    facade_obj = getattr(report, name)
    origins = [
        module
        for module in _SOURCE_MODULES
        if getattr(importlib.import_module(module), name, None) is facade_obj
    ]
    assert origins, f"{name} is not the same object as in any source module"
