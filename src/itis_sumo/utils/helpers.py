"""Helper utility functions (recycled from itis_dakota_projects / mmux_flaskapi)."""

import datetime
import os
import sys
import uuid
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as lib_version
from importlib.util import find_spec
from pathlib import Path
from typing import Any


def validate_dakota_installation() -> dict[str, Any]:
    """Validate that ITIS Dakota Python package is properly installed and accessible.

    Returns:
        Dictionary containing validation results
    """
    result = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "dakota_installed": False,
        "dakota_path": None,
        "dakota_version": None,
        "package_name": "itis-dakota",
    }
    errors = []
    warnings = []

    # Check if itis-dakota package is installed
    try:
        import itis_dakota

        result["dakota_installed"] = True

        # Get the package path
        spec = find_spec("itis_dakota")
        if spec and spec.origin:
            result["dakota_path"] = str(Path(spec.origin).parent)

        # Try to get version from the package
        try:
            if hasattr(itis_dakota, "__version__"):
                result["dakota_version"] = itis_dakota.__version__
            else:
                # Try to get version using importlib.metadata
                try:
                    version = lib_version("itis-dakota")
                    result["dakota_version"] = version
                except PackageNotFoundError:
                    warnings.append("Could not determine itis-dakota version")
        except Exception as e:
            warnings.append(f"Error getting itis-dakota version: {e}")

    except ImportError:
        errors.append(
            "itis-dakota Python package not found. "
            "Install with: pip install itis-dakota"
        )

    result["errors"] = errors
    result["warnings"] = warnings

    return result


def get_dakota_version() -> str | None:
    """Get the ITIS Dakota version string.

    Returns:
        ITIS Dakota version string or None if not available
    """
    try:
        import itis_dakota

        if hasattr(itis_dakota, "__version__"):
            return itis_dakota.__version__  # type: ignore
        else:
            # Try to get version using importlib.metadata
            try:
                import importlib.metadata

                return importlib.metadata.version("itis-dakota")
            except importlib.metadata.PackageNotFoundError:
                return None
    except ImportError:
        return None


def create_run_dir(script_dir: Path, dir_name: str = "sampling"):
    """Create a unique timestamped run directory under ``script_dir/runs``."""
    main_runs_dir = script_dir / "runs"
    current_time = datetime.datetime.now().strftime("%Y%m%d.%H%M%S%d")
    uid = uuid.uuid4().hex
    temp_dir = main_runs_dir / "_".join(["dakota", current_time, uid, dir_name])
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir
