"""Hatch build hook: tag the wheel as a platform binary.

The Python shim is pure, but the payload under ``bin/`` is a compiled Linux
x86_64 or aarch64 binary linked against glibc 2.35 (the Ubuntu 22.04 build
runner), so the wheel must carry a platform tag rather than ``any``.
"""

from __future__ import annotations

import platform
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_MANYLINUX_TAG_BY_MACHINE = {
    "aarch64": "py3-none-manylinux_2_35_aarch64",
    "arm64": "py3-none-manylinux_2_35_aarch64",
    "x86_64": "py3-none-manylinux_2_35_x86_64",
}


def wheel_tag(machine: str) -> str:
    """Return the binary wheel tag for a supported Linux build machine.

    Args:
        machine: Architecture reported by :func:`platform.machine`.

    Returns:
        A Python 3, ABI-independent manylinux tag for the compiled payload.

    Raises:
        ValueError: The workflow ran on an architecture Skulk does not publish.
    """
    try:
        return _MANYLINUX_TAG_BY_MACHINE[machine]
    except KeyError as error:
        raise ValueError(f"unsupported CUDA wheel architecture: {machine}") from error


class CustomBuildHook(BuildHookInterface):
    """Stamp the platform-specific wheel tag."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Mark the wheel non-pure with the manylinux tag of the payload."""
        build_data["pure_python"] = False
        build_data["tag"] = wheel_tag(platform.machine())
