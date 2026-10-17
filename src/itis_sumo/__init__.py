"""itis-sumo: headless core of the MetaModeling tools.

Aggregates the computational core of the mmux/vite flaskapi Dakota
modules plus recycled utilities from previous trials, as a standalone,
importable, headless package.
"""

from importlib.metadata import version as package_version

__version__ = package_version("itis-sumo")
from itis_sumo.core.capture import capture_to_file
from itis_sumo.core.dakota_object import DakotaObject, working_directory

__all__ = ["DakotaObject", "__version__", "capture_to_file", "working_directory"]
