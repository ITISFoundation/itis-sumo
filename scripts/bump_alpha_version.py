#!/usr/bin/env python3
"""Compute and optionally write the next develop-channel (`aN`) alpha version.

Base-version bumps (major/minor/patch) stay a human decision: if the
current version isn't already an alpha prerelease, or if incrementing the
alpha counter alone still can't clear an existing tag (e.g. a differently
based tag), this prints the unchanged current version and leaves
`version-check.yml`'s PR-time gate to fail and prompt a manual bump.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from packaging.version import InvalidVersion, Version

PROJECT_FILE = Path("pyproject.toml")
VERSION_LINE = re.compile(r'^(version = ")([^"]+)(")$', re.MULTILINE)


def current_version() -> str:
    with PROJECT_FILE.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def existing_tags() -> list[Version]:
    tags = subprocess.check_output(["git", "tag", "--list", "v*"], text=True)
    versions: list[Version] = []
    for tag in tags.splitlines():
        try:
            versions.append(Version(tag.removeprefix("v")))
        except InvalidVersion:
            continue
    return versions


def next_version(current: str, tags: list[Version]) -> str | None:
    parsed = Version(current)
    if parsed.pre is None or parsed.pre[0] != "a":
        return None
    if all(tag < parsed for tag in tags):
        return None  # already a valid, newer-than-every-tag alpha

    base = f"{parsed.major}.{parsed.minor}.{parsed.micro}"
    same_base_alpha_numbers: list[int] = []
    for tag in tags:
        if tag.release == parsed.release and tag.pre is not None and tag.pre[0] == "a":
            same_base_alpha_numbers.append(tag.pre[1])
    highest_number = max(same_base_alpha_numbers, default=0)
    candidate_number = max(highest_number, parsed.pre[1]) + 1
    candidate = Version(f"{base}a{candidate_number}")
    if any(tag >= candidate for tag in tags):
        return None  # some other tag (different base/channel) still blocks; needs a human bump

    return str(candidate)


def write_version(version: str) -> None:
    content = PROJECT_FILE.read_text()
    updated, count = VERSION_LINE.subn(rf"\g<1>{version}\g<3>", content, count=1)
    if count != 1:
        raise RuntimeError("Could not find project version in pyproject.toml")
    PROJECT_FILE.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the bumped version to pyproject.toml, if any",
    )
    args = parser.parse_args()
    version = next_version(current_version(), existing_tags())
    if version is None:
        print(current_version())
        return
    if args.write:
        write_version(version)
    print(version)


if __name__ == "__main__":
    main()
