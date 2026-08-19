#!/usr/bin/env python3
"""Compute and optionally write the next feature-branch development version."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from packaging.version import InvalidVersion, Version

PROJECT_FILE = Path("pyproject.toml")
TESTPYPI_JSON_URL = "https://test.pypi.org/pypi/itis-sumo/json"
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


def published_dev_versions(base: str) -> list[Version]:
    try:
        with urlopen(TESTPYPI_JSON_URL, timeout=10) as response:
            releases = json.load(response).get("releases", {})
    except HTTPError as error:
        if error.code == 404:
            return []
        raise RuntimeError("Could not query TestPyPI release versions") from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("Could not query TestPyPI release versions") from error

    versions: list[Version] = []
    for release in releases:
        try:
            version = Version(release)
        except InvalidVersion:
            continue
        match = DEV_VERSION.fullmatch(str(version))
        if match and match.group("base") == base:
            versions.append(version)
    return versions


def next_version() -> str:
    current = current_version()
    match = DEV_VERSION.fullmatch(current)
    base = match.group("base") if match else current
    versions = existing_dev_versions(base) + published_dev_versions(base)

    def _dev_number(version: Version) -> int:
        number_match = DEV_VERSION.fullmatch(str(version))
        assert number_match is not None
        return int(number_match.group("number"))

    highest_number = max((_dev_number(version) for version in versions), default=0)
    return str(Version(f"{base}.dev{highest_number + 1}"))


def write_version(version: str) -> None:
    content = PROJECT_FILE.read_text()
    updated, count = VERSION_LINE.subn(rf"\g<1>{version}\g<3>", content, count=1)
    if count != 1:
        raise RuntimeError("Could not find project version in pyproject.toml")
    PROJECT_FILE.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="write the next version to pyproject.toml"
    )
    args = parser.parse_args()
    version = next_version()
    if args.write:
        write_version(version)
    print(version)


if __name__ == "__main__":
    main()
