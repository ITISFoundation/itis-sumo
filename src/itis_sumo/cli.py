#!/usr/bin/env python3
"""Command-line interface for itis-sumo (recycled from itis_dakota_projects)."""

import argparse
import sys

from itis_sumo.utils.config_guard import validate_nidr_config
from itis_sumo.utils.helpers import validate_dakota_installation


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="itis-sumo - MetaModeling (Surrogate Model) core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  itis-sumo validate                 # Validate Dakota installation
  itis-sumo validate --config conf.in  # Also sanity-check a NIDR config file
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    validate_parser = subparsers.add_parser("validate", help="Validate Dakota installation")
    validate_parser.add_argument(
        "--config",
        metavar="PATH",
        help="NIDR (Dakota input) configuration file to sanity-check",
    )
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "validate":
            return _handle_validate(config_path=args.config)
        else:
            parser.print_help()
            return 1

    except Exception as e:  # noqa: BLE001 - CLI boundary: always fail cleanly
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_validate(config_path: str | None = None) -> int:
    """Handle the validate command."""
    print("Validating Dakota installation...")
    result = validate_dakota_installation()

    if result["dakota_installed"]:
        print("OK - ITIS Dakota is installed and accessible")
        print(f"  Path: {result['dakota_path']}")
        if result["dakota_version"]:
            print(f"  Version: {result['dakota_version']}")
    else:
        print(f"FAIL - Dakota is not installed in {result['python_executable']}")

    if result["errors"]:
        print("\nErrors:")
        for error in result["errors"]:
            print(f"  [error] {error}")

    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  [warn] {warning}")

    rc = 0 if result["dakota_installed"] else 1

    if config_path:
        print(f"\nSanity-checking NIDR config: {config_path}")
        from pathlib import Path

        conf = Path(config_path).read_text()
        problems = validate_nidr_config(conf)
        if problems:
            print("FAIL - config problems found:")
            for problem in problems:
                print(f"  [error] {problem}")
            rc = 1
        else:
            print("OK - config looks sane")

    return rc


if __name__ == "__main__":
    sys.exit(main())
