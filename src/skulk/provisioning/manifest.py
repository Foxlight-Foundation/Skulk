"""Pinned engine artifact manifest (#614 Phase 3).

Skulk manages engine binaries the way it manages models: pinned known-good
versions, checksums recorded in-repo, fetched on demand and verified before
use. A new user never builds llama.cpp.

The pin is a specific upstream llama.cpp release tag whose official prebuilt
Linux artifacts we resolve by platform, architecture, and compute variant.
The pinned release publishes Linux CPU and Vulkan builds (its ROCm and CUDA
prebuilts are Windows-only); the Vulkan build drives both AMD (RADV,
fleet-proven) and NVIDIA GPUs through their Vulkan ICDs, so it is the GPU
default. macOS
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

LLAMA_SERVER_PIN: Final = "b10434"
"""The pinned upstream llama.cpp release tag for managed llama-server builds.

Includes the RPC tensor-memset protocol required by current DeepSeek V4
multi-node execution, Qwen 3.8 text and native long-context support, recurrent
state rollback, and served reasoning-effort plumbing. The RPC protocol changed
in this release window, so ``llama-server`` and every ``ggml-rpc-server`` donor
must always advance together. Advance deliberately, re-recording checksums and
re-running the fresh-box gauntlet.
"""

EngineVariant = Literal["cpu", "vulkan", "rocm", "cuda"]
"""Compute variant of a managed llama-server build.

``cuda`` has no upstream Linux prebuilt and is delivered by the Foxlight wheel.
The pinned release also has no Linux ROCm prebuilt; AMD nodes use the
fleet-qualified Vulkan wheel or archive. NVIDIA nodes fall through from the
CUDA wheel to Vulkan when necessary (bare-metal drivers ship a working Vulkan
ICD; compute-only container drivers generally do not)."""


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
# the upstream release asset digests, recorded 2026-08-15.
LLAMA_SERVER_ARTIFACTS: Final[dict[tuple[str, EngineVariant], EngineArtifact]] = {
    ("x86_64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-x64.tar.gz",
        sha256="2b1a1cca630c2211f87d6d590731ae7cbdd3bac6bf0de8ebf01dd2e16571eff9",
    ),
    ("x86_64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="54bd06e2f6a366494b91f7cc32a2ab47b48075292ac17fa1d061fbd2c2aa8b85",
    ),
    ("aarch64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-arm64.tar.gz",
        sha256="65b72db64637407cc3e3dbdb0f4184e700a5f0501748e62db96dcfb3f7aada6e",
    ),
    ("aarch64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-arm64.tar.gz",
        sha256="5a6fb7412311853415620a6da559a1ddf65e4664dfe35ba51495e0afbcc1c1cb",
    ),
}
