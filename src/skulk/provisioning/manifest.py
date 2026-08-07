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

LLAMA_SERVER_PIN: Final = "b10092"
"""The pinned upstream llama.cpp release tag for managed llama-server builds.

Recent enough for ``--spec-type draft-mtp`` (>= b9196), ``--list-devices``
(which the facts probe uses to verify what a managed binary can actually
drive), the Laguna 2 model family, and native DFlash speculative decoding
with sidecar auto-resolution (the b10068..b10092 window landed all four
DFlash pieces upstream, ending the fork-only era). Advance deliberately,
re-recording checksums and re-running the fresh-box gauntlet.
"""

EngineVariant = Literal["cpu", "vulkan", "rocm", "cuda"]
"""Compute variant of a managed llama-server build.

``cuda`` has no upstream Linux prebuilt; it is reserved for the
Foxlight-built artifact (see ``LLAMA_SERVER_ARTIFACTS``), which needs a
hosting decision before it can ship in the manifest. Until then, NVIDIA
nodes fall through the variant chain to ``vulkan`` (bare-metal drivers ship
a working Vulkan ICD; container GPU clouds do not, and there vLLM is the
serving path)."""


@final
class EngineArtifact(CamelCaseModel):
    """One downloadable pinned engine build with its integrity checksum."""

    model_config = ConfigDict(frozen=True)

    asset_name: str
    """Release asset filename."""

    sha256: str
    """Hex SHA-256 of the archive; verification failure aborts provisioning."""

    url_override: str | None = None
    """Full download URL for artifacts not hosted on the upstream release
    (e.g. the Foxlight-built Linux CUDA build, which upstream does not
    publish). ``None`` resolves against the upstream llama.cpp release."""

    def url(self) -> str:
        """The download URL for this artifact."""
        if self.url_override is not None:
            return self.url_override
        return (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{LLAMA_SERVER_PIN}/{self.asset_name}"
        )


# (machine, variant) -> artifact, for sys.platform == "linux". Checksums are
# the upstream release asset digests, recorded 2026-07-23.
LLAMA_SERVER_ARTIFACTS: Final[dict[tuple[str, EngineVariant], EngineArtifact]] = {
    ("x86_64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-x64.tar.gz",
        sha256="b047abca5eb5186afb8c6fe816b008b34063f484613c3453b27ebc5600f937fe",
    ),
    ("x86_64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="751811bf24857c9491749c3ea0c6be5680a035903ccae01c0facb5e10ca510cc",
    ),
    ("x86_64", "rocm"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-rocm-7.2-x64.tar.gz",
        sha256="0a7d613fcdf269a7df2d81aade4ff9ec4e0429feb77c41905de482199ee39813",
    ),
    ("aarch64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-arm64.tar.gz",
        sha256="fb3241aa451d502707727878008ed5ff8a9f130b920d48c7f06899992b46e3f4",
    ),
    ("aarch64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-arm64.tar.gz",
        sha256="381be7c7c9b1ec583be9219a0196341d683806aac16ef5a7dfee02af36c4f02b",
    ),
}
