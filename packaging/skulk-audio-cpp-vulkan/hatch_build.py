"""Stamp the supported Linux platform tag on the Vulkan audio.cpp payload."""

from __future__ import annotations

import platform
import sys
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_LINUX_TAGS = {
    "x86_64": "py3-none-manylinux_2_35_x86_64",
    "aarch64": "py3-none-manylinux_2_35_aarch64",
}


def wheel_tag(system: str, machine: str) -> str:
    """Resolve the binary-compatible wheel tag for a supported Skulk host."""
    if system == "linux" and machine in _LINUX_TAGS:
        return _LINUX_TAGS[machine]
    raise ValueError(f"unsupported audio.cpp wheel target {system}/{machine}")


class CustomBuildHook(BuildHookInterface):
    """Mark the wheel as native to one OS and CPU architecture."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Set the tag and prevent a false pure-Python wheel."""
        build_data["pure_python"] = False
        build_data["tag"] = wheel_tag(sys.platform, platform.machine())
