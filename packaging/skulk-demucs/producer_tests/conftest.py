"""Expose standalone producer tools only while collecting their private tests."""

import sys
from pathlib import Path

# These build-time modules deliberately stay outside the installed Core package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
