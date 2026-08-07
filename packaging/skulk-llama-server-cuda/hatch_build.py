"""Hatch build hook: tag the wheel as a platform binary.

The Python shim is pure, but the payload under ``bin/`` is a compiled Linux
x86_64 binary linked against glibc 2.35 (the ubuntu-22.04 build runner), so
the wheel must carry a platform tag rather than ``any``.
"""

from __future__ import annotations

from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Stamp the platform-specific wheel tag."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Mark the wheel non-pure with the manylinux tag of the payload."""
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-manylinux_2_35_x86_64"
