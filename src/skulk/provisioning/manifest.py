"""Pinned engine artifact manifest (#614 Phase 3).

Skulk manages engine binaries the way it manages models: pinned known-good
versions, checksums recorded in-repo, fetched on demand and verified before
use. A new user never builds llama.cpp.

The pin is a specific upstream llama.cpp release tag whose official prebuilt
Linux artifacts we resolve by platform, architecture, and compute variant.
The pinned release publishes Linux CPU, Vulkan, and ROCm builds (its CUDA
prebuilts remain Windows-only). Skulk consumes only CPU and Vulkan: the
Vulkan build drives both AMD (RADV, fleet-proven) and NVIDIA GPUs through
their Vulkan ICDs, so it is the GPU default, and the upstream Linux ROCm
archive stays unconsumed until a ROCm lane is qualified on real hardware. macOS
provisions nothing (in-process MLX owns that platform), and vLLM remains the
first-class CUDA serving path.

Pinning beats "latest" for supply-chain and behavior stability both: the
checksums below make substitution loud, and an upstream behavior change (the
kind that broke pooled HTTP clients in newer builds) arrives only when the pin
is deliberately advanced and re-validated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final, Literal, final

from pydantic import ConfigDict

from skulk.utils.pydantic_ext import CamelCaseModel

LLAMA_SERVER_PIN: Final = "b10753"
"""The pinned upstream llama.cpp release tag for managed llama-server builds.

Includes the RPC tensor-memset protocol required by current DeepSeek V4
multi-node execution, Qwen 3.8 text and native long-context support, recurrent
state rollback, and served reasoning-effort plumbing. Advancing from b10434
additionally picks up Kimi-K3 text, native MTP speculative decoding
(``--spec-type draft-mtp`` second generation), a gemma4-assistant model fix,
and MTP/next-n loader ordering fixes that matter to the served draft path.
The RPC protocol changed in the b10434 window, so ``llama-server`` and every
``ggml-rpc-server`` donor must always advance together. Advance deliberately,
re-recording checksums and re-running the fresh-box gauntlet.
"""

LLAMA_SERVER_CUDA_MIN_REVISION: Final = 1
"""Minimum CUDA packaging revision for the current engine pin.

Revision 1 fixes missing NCCL and build-host CPU instructions. Reset deliberately
when advancing the engine pin, after validating the new wheel's packaging.
"""

EngineVariant = Literal["cpu", "vulkan", "rocm", "cuda"]
"""Compute variant of a managed llama-server build.

``cuda`` has no upstream Linux prebuilt and is delivered by the Foxlight wheel.
Upstream does publish a Linux ROCm x86_64 archive as of this pin, but Skulk
does not consume it: AMD nodes stay on the fleet-qualified Vulkan wheel or
archive until a ROCm lane is qualified on real Strix hardware. NVIDIA nodes fall through from the
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
# the upstream release asset digests, recorded 2026-09-02 (verified by
# streaming each asset and comparing against the release API digest).
LLAMA_SERVER_ARTIFACTS: Final[dict[tuple[str, EngineVariant], EngineArtifact]] = {
    ("x86_64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-x64.tar.gz",
        sha256="a25f023c1c68bafb315ada095fa7780e286d5867783e5eebd7dfc1e36eb1a856",
    ),
    ("x86_64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="30362addb83f0d1275a608c2cc9521d2b2d9a3596704aacebaf1294f94aa91e3",
    ),
    ("aarch64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-arm64.tar.gz",
        sha256="4224302b9bdb52b3fdfbf2439c320d6f51e3a3afc48500d4c1a5f9a76623d2ae",
    ),
    ("aarch64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-arm64.tar.gz",
        sha256="cba2f4a533c77a0bc5e0bcc13d4ac1129f941ba784be915196acbb403c1b2ffa",
    ),
}


# --- ComfyUI (served video engine) --------------------------------------------

COMFY_PIN: Final = "40c4fcdf513a4523e39d54a9d391908af8df8171"
"""The pinned ComfyUI commit for managed installs: release v0.35.0 (2026-09-09).

Native MiniMax H3 support landed in v0.30.0; this release follows the H3
denoise-mask fix (421a1c2, 2026-09-08) and carries the model patch loader
the ControlNet union companion needs. Advance deliberately: bump the pin,
re-record the torch wheel set if it moves, and rerun the engine battery.
"""

COMFY_REPOSITORY: Final = "https://github.com/comfyanonymous/ComfyUI.git"
"""Upstream repository the managed checkout clones at the pin."""

COMFY_PYTHON: Final = "3.13"
"""Interpreter the managed environment is created with; the wheel set below
is recorded for its ``cp313`` tags."""


@final
class PinnedWheel(CamelCaseModel):
    """One exact wheel from a package index with its integrity checksum."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Distribution name."""

    version: str
    """Exact version including any local tag (``2.14.0+cu130``)."""

    filename: str
    """Wheel filename as published."""

    sha256: str
    """Hex SHA-256 of the wheel; passed to the installer as a required hash."""

    index: str
    """Directory URL the filename resolves against."""

    def url(self) -> str:
        """The download URL for this wheel."""
        return f"{self.index.rstrip('/')}/{self.filename}"

    def requirement(self) -> str:
        """A hashed direct-URL requirement line."""
        return f"{self.name} @ {self.url()} --hash=sha256:{self.sha256}"

    def constraint(self) -> str:
        """An exact version constraint for later resolver passes."""
        return f"{self.name}=={self.version}"


_PYTORCH_CU130_INDEX: Final = "https://download.pytorch.org/whl/cu130"
_PYTORCH_ROCM72_INDEX: Final = "https://download.pytorch.org/whl/rocm7.2"


def _pytorch_wheel(name: str, version: str, local: str, index: str, machine: str, sha256: str) -> PinnedWheel:
    return PinnedWheel(
        name=name,
        version=f"{version}+{local}",
        filename=f"{name}-{version}%2B{local}-cp313-cp313-manylinux_2_28_{machine}.whl",
        sha256=sha256,
        index=index,
    )


def _cu130(name: str, version: str, machine: str, sha256: str) -> PinnedWheel:
    return _pytorch_wheel(name, version, "cu130", _PYTORCH_CU130_INDEX, machine, sha256)


def _rocm72(name: str, version: str, machine: str, sha256: str) -> PinnedWheel:
    return _pytorch_wheel(name, version, "rocm7.2", _PYTORCH_ROCM72_INDEX, machine, sha256)


def wheel_set_digest(wheels: Sequence[PinnedWheel]) -> str:
    """Content identity of one wheel set: the SHA-256 over its wheel digests.

    A managed install is keyed by the ComfyUI pin and this digest together,
    so a wheel-set change without a pin change reprovisions instead of
    silently reusing an environment built on different torch builds.
    """
    return hashlib.sha256("".join(sorted(wheel.sha256 for wheel in wheels)).encode()).hexdigest()


# (machine, variant) -> the torch wheel set installed into the managed
# ComfyUI environment, for sys.platform == "linux" and cp313. Checksums are
# the index's own link digests, recorded 2026-09-10 from
# download.pytorch.org/whl/cu130 and download.pytorch.org/whl/rocm7.2. CUDA
# 13.0 wheels need a 580-series or newer driver. The ROCm 7.2 wheels bundle
# their own HIP runtime, so the host needs only the amdgpu kernel driver (a
# Strix Halo gfx1151 node runs them on an in-tree kernel 7 driver); they are
# published for x86_64 only. Both lanes sit on the same torch release so a
# card's behavior does not fork by vendor.
COMFY_TORCH_WHEELS: Final[dict[tuple[str, EngineVariant], tuple[PinnedWheel, ...]]] = {
    ("aarch64", "cuda"): (
        _cu130("torch", "2.14.0", "aarch64", "20ec4bb8944a847dee60e6d5536b415670dd5403342f36dbe08a6be5bf582e08"),
        _cu130("torchvision", "0.29.0", "aarch64", "bfecf363f1c2f99e273405d9f15b647d6d51afaa20174803452f9a1071af666c"),
        _cu130("torchaudio", "2.11.0", "aarch64", "23498b01097648e304e78d6495a9f5bdce8441a802afc3025e2561973d74c025"),
    ),
    ("x86_64", "cuda"): (
        _cu130("torch", "2.14.0", "x86_64", "745010695e0458d6f28accb697b5371b6fd245aa8696530165f972cd126bbd8d"),
        _cu130("torchvision", "0.29.0", "x86_64", "360f048dd21a2c23f12af210962d4bd23321da32410a67fac98ef1e22fd2bc21"),
        _cu130("torchaudio", "2.11.0", "x86_64", "e9c07cfdab691454092ff12d21dd1407a4bb8ad081d38f222cf6fcf6abcc18c8"),
    ),
    ("x86_64", "rocm"): (
        _rocm72("torch", "2.14.0", "x86_64", "fa22a1a85f4f92c4fb49dc9b117b3eb32b8c684968508f585df042e7016eb97f"),
        _rocm72("torchvision", "0.29.0", "x86_64", "a400ed6494ec9d1f007754492e7de714c7e0bfeedfe295b401435403bf2630fb"),
        _rocm72("torchaudio", "2.11.0", "x86_64", "9228e1c534d04d80fa7b46205f448da947fd06f5573ba8602f041b504edb3596"),
    ),
}
