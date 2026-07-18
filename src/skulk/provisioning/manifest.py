"""Pinned engine artifact manifest (#614 Phase 3).

Skulk manages engine binaries the way it manages models: pinned known-good
versions, checksums recorded in-repo, fetched on demand and verified before
use. A new user never builds llama.cpp.

The pin is a specific upstream llama.cpp release tag whose official prebuilt
Linux artifacts we resolve by platform, architecture, and compute variant.
Upstream publishes Linux CPU, Vulkan, and ROCm builds (CUDA prebuilts are
Windows-only); the Vulkan build drives both AMD (RADV, fleet-proven) and
NVIDIA GPUs through their Vulkan ICDs, so it is the GPU default. macOS
provisions nothing (in-process MLX owns that platform), and vLLM remains the
first-class CUDA serving path.

Pinning beats "latest" for supply-chain and behavior stability both: the
checksums below make substitution loud, and an upstream behavior change (the
kind that broke pooled HTTP clients in newer builds) arrives only when the pin
is deliberately advanced and re-validated.
"""

from __future__ import annotations

from typing import Final, Literal, final

from pydantic import ConfigDict

from skulk.utils.pydantic_ext import CamelCaseModel

LLAMA_SERVER_PIN: Final = "b10068"
"""The pinned upstream llama.cpp release tag for managed llama-server builds.

Recent enough for ``--spec-type draft-mtp`` (>= b9196) and ``--list-devices``
(which the facts probe uses to verify what a managed binary can actually
drive). Advance deliberately, re-recording checksums and re-running the
fresh-box gauntlet.
"""

EngineVariant = Literal["cpu", "vulkan", "rocm"]
"""Compute variant of a managed llama-server build."""


@final
class EngineArtifact(CamelCaseModel):
    """One downloadable pinned engine build with its integrity checksum."""

    model_config = ConfigDict(frozen=True)

    asset_name: str
    """Release asset filename."""

    sha256: str
    """Hex SHA-256 of the archive; verification failure aborts provisioning."""

    def url(self) -> str:
        """The upstream release download URL for this artifact."""
        return (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{LLAMA_SERVER_PIN}/{self.asset_name}"
        )


# (machine, variant) -> artifact, for sys.platform == "linux". Checksums are
# the upstream release asset digests, recorded 2026-07-18.
LLAMA_SERVER_ARTIFACTS: Final[dict[tuple[str, EngineVariant], EngineArtifact]] = {
    ("x86_64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-x64.tar.gz",
        sha256="6bf3d20de562e4df230f1a7c54fb7a06a80c7ff40f5311c953e8255744be4eb2",
    ),
    ("x86_64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="713641920dce6c8efb953ebc9ffa309977e200cec5e182e6ad0e8b086203cdc3",
    ),
    ("x86_64", "rocm"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-rocm-7.2-x64.tar.gz",
        sha256="81735d049c50e18c89de2e6d88f4e2091bf3e148eafafb859deadc4ac977225b",
    ),
    ("aarch64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-arm64.tar.gz",
        sha256="2c0e4d3d5932e472b6c669090968fdc84a7f6a2940f2e8bb40fa03225bd01960",
    ),
    ("aarch64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-arm64.tar.gz",
        sha256="c3c49e6e124a574165ca28317be021b1a12a2ea06977e3eb7daee3eb443eb186",
    ),
}
