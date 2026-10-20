"""Published-wheel entrypoints; probing provenance never initializes a GPU."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comfy_api.latest import ComfyExtension


def manifest_json() -> str:
    """Read the immutable package manifest without importing torch or Comfy."""
    return files(__package__).joinpath("manifest.json").read_text(encoding="utf-8")


async def comfy_entrypoint() -> ComfyExtension:
    """Load the trusted node set only inside the selected Comfy process."""
    from skulk_comfy_ltx.nodes import SkulkLTXExtension

    return SkulkLTXExtension()
