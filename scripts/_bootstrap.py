"""
Load the source package when legacy scripts run without an editable install.
"""

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """
    Add this repository's src directory to the module search path.
    """
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

