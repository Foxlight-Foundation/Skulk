"""Pinned standalone audio.cpp Vulkan server with two music families."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SOURCE_REVISION = "4d88768fbcae4e6eb3352c6ab1422dabb7d90b58"
"""Exact upstream audio.cpp source used for the packaged executable."""

_PACKAGE_ROOT = Path(__file__).resolve().parent


def binary_path() -> Path:
    """Return the packaged server binary, or fail for an incomplete wheel."""
    binary = _PACKAGE_ROOT / "bin" / "audiocpp_server"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"audio.cpp server payload is missing: {binary}")
    return binary


def model_spec_directory() -> Path:
    """Return the directory containing the two pinned family model specs."""
    directory = _PACKAGE_ROOT / "model_specs"
    for family in ("ace_step", "minimax_music3"):
        if not (directory / f"{family}.json").is_file():
            raise FileNotFoundError(f"audio.cpp model spec is missing: {family}")
    return directory


def main() -> int:
    """Exec the bundled server with model management and UI disabled."""
    binary = binary_path()
    model_spec_directory()
    args = [str(binary), *sys.argv[1:]]
    if "--no-ui" not in args:
        args.append("--no-ui")
    os.execv(str(binary), args)
    return 1
