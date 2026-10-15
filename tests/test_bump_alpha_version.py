from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from packaging.version import Version

_spec = spec_from_file_location(
    "bump_alpha_version", Path(__file__).parents[1] / "scripts/bump_alpha_version.py"
)
assert _spec is not None
bump_alpha_version = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(bump_alpha_version)


def test_non_alpha_current_version_is_left_to_a_human():
    assert bump_alpha_version.next_version("0.1.0", [Version("0.1.0a1")]) is None


def test_already_ahead_of_every_tag_needs_no_bump():
    assert bump_alpha_version.next_version("0.1.0a2", [Version("0.1.0a1")]) is None


def test_colliding_alpha_bumps_past_the_highest_existing_tag():
    assert (
        bump_alpha_version.next_version(
            "0.1.0a1", [Version("0.1.0a1"), Version("0.1.0a2")]
        )
        == "0.1.0a3"
    )


def test_different_base_tag_still_blocking_is_left_to_a_human():
    # bumping the alpha counter alone can't clear a stable v0.2.0 tag from
    # the same major.minor.patch's later series; needs a human base bump
    assert bump_alpha_version.next_version("0.1.0a1", [Version("0.2.0")]) is None
