"""itis-sumo: headless core of the MetaModeling tools.

Aggregates the computational core of the mmux/vite flaskapi Dakota
modules plus recycled utilities from previous trials, as a standalone,
importable, headless package.
"""

from itis_sumo.core.dakota_object import DakotaObject, working_directory
from itis_sumo.core.wiofiles import capture_to_file

__all__ = ["DakotaObject", "working_directory", "capture_to_file"]
