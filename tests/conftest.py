"""Test configuration for untargeted-metabolomics.

Adds the scripts/ directory to sys.path so tests can import functions
directly from pipeline scripts (which use a shared ``utils`` module rather
than a proper package layout).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from scripts/ (where utils.py and all step scripts live)
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
