#!/usr/bin/env python3
"""Compute and optionally write the next feature-branch development version."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

PROJECT_FILE = Path("pyproject.toml")
VERSION_LINE = re.compile(r'^(version = ")([^"]+)(")$', re.MULTILINE)
DEV_VERSION = re.compile(r"^(?P<base>.+)\.dev(?P<number>[0-9]+)$")


def current_version() -> str:
    with PROJECT_FILE.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def existing_dev_versions(base: str) -> list[Version]:
    tags = subprocess.check_output(["git", "tag", "--list", f"v{base}.dev*"], text=True)
    versions: list[Version] = []
    for tag in tags.splitlines():
        try:
            version = Version(tag.removeprefix("v"))
        except InvalidVersion:
            continue
        if DEV_VERSION.fullmatch(str(version)):
            versions.append(version)
    return versions


def next_version() -> str:
    current = current_version()
    match = DEV_VERSION.fullmatch(current)
    base = match.group("base") if match else current
    candidate = Version(f"{base}.dev1")
    versions = existing_dev_versions(base)
    if versions:
        highest_number = max(int(DEV_VERSION.fullmatch(str(version)).group("number")) for version in versions)
        candidate = Version(f"{base}.dev{highest_number + 1}")
    return str(candidate)


def write_version(version: str) -> None:
    content = PROJECT_FILE.read_text()
    updated, count = VERSION_LINE.subn(rf"\g<1>{version}\g<3>", content, count=1)
    if count != 1:
        raise RuntimeError("Could not find project version in pyproject.toml")
    PROJECT_FILE.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the next version to pyproject.toml")
    args = parser.parse_args()
    version = next_version()
    if args.write:
        write_version(version)
    print(version)


if __name__ == "__main__":
    main()
