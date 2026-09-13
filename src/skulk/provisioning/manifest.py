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

    channel: str | None = None
    """Simple-index root for later resolver passes when it differs from ``index``
    (AMD's channel keeps one directory per package; the PyTorch index is flat)."""

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
_AMD_ROCM_STABLE_INDEX: Final = "https://stable.repo.amd.com/rocm/whl-next"
"""AMD's stable ROCm release channel (TheRock multi-arch builds): torch is a
host wheel plus a per-architecture device package on top of the ``rocm``
runtime packages, all versioned together. gfx1151 is a supported target."""


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


def _amd(name: str, version: str, filename: str, sha256: str) -> PinnedWheel:
    """One artifact from AMD's stable channel; filenames follow no single pattern there."""
    return PinnedWheel(
        name=name,
        version=version,
        filename=filename,
        sha256=sha256,
        index=f"{_AMD_ROCM_STABLE_INDEX}/{name}",
        channel=_AMD_ROCM_STABLE_INDEX,
    )


def wheel_set_digest(wheels: Sequence[PinnedWheel]) -> str:
    """Content identity of one wheel set: the SHA-256 over its wheel digests.

    A managed install is keyed by the ComfyUI pin and this digest together,
    so a wheel-set change without a pin change reprovisions instead of
    silently reusing an environment built on different torch builds.
    """
    return hashlib.sha256("".join(sorted(wheel.sha256 for wheel in wheels)).encode()).hexdigest()


# (machine, variant) -> the torch wheel set installed into the managed
# ComfyUI environment, for sys.platform == "linux" and cp313. The CUDA
# lane's checksums are the PyTorch index's own link digests, recorded
# 2026-09-10 from download.pytorch.org/whl/cu130; CUDA 13.0 wheels need a
# 580-series or newer driver. The ROCm lane uses AMD's stable ROCm 10.0.0
# channel (digests recorded 2026-09-11 by downloading each artifact and
# hashing it; that index publishes no digests): torch plus its gfx1151 device
# packages, torchvision and torchaudio, triton, and the ``rocm`` runtime
# packages (core, libraries, and the gfx1151 device libraries), which bundle
# the HIP runtime so the host needs only the amdgpu kernel driver. The
# PyTorch-index rocm7.2 wheels were retired from this lane because their
# gfx1151 BLAS libraries lack GEMM solutions that image-conditioned prompts
# reach (recorded in the video initiative's M3 battery). Published for
# x86_64 only.
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
        _amd("rocm", "10.0.0", "rocm-10.0.0.tar.gz", "65b5982249612a310135f5e061b9f28909e8ad718a1ddb3f1abba92298b7216b"),
        _amd("rocm-bootstrap", "0.1.0", "rocm_bootstrap-0.1.0-py3-none-any.whl", "33e3d4dad086b1f0e06bc049a618ca874d4fb72f758fff00676e23d2a644b652"),
        _amd("rocm-sdk-core", "10.0.0", "rocm_sdk_core-10.0.0-py3-none-linux_x86_64.whl", "3bf4a72d11aa2a4ee1e90572c73630f937d38c1b7af4dda45b483b54655138a7"),
        _amd("rocm-sdk-libraries", "10.0.0", "rocm_sdk_libraries-10.0.0-py3-none-linux_x86_64.whl", "bbe83a550cb8ceeb306abf6d57df6617857a1b19384549526db2ec81a4374e57"),
        _amd("rocm-sdk-device-gfx1151", "10.0.0", "rocm_sdk_device_gfx1151-10.0.0-py3-none-linux_x86_64.whl", "ec705df531e523777a5cd47b2dfadb2272935927c46c122effb11fcdcedb73d0"),
        _amd("triton", "3.8.0+git4cff872c.rocm10.0.0", "triton-3.8.0%2Bgit4cff872c.rocm10.0.0-cp313-cp313-linux_x86_64.whl", "874f41046c50f84392c8481f074a2f96cfc1dc44a9b05e162246cb7f9175e588"),
        _amd("torch", "2.13.0+rocm10.0.0", "torch-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl", "28cbb1dda100903127a02b7cca39742972b57c383f79e60b103617a2d0b7a064"),
        _amd("amd-torch-device-gfx1151", "2.13.0+rocm10.0.0", "amd_torch_device_gfx1151-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl", "33dfb69b1d6d89127aec416d2891b205ed31d7ffb0a106aa45a54da7afd0c01d"),
        _amd("amd-torch-device-gfx115x", "2.13.0+rocm10.0.0", "amd_torch_device_gfx115x-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl", "44b27db9d540e589f2b17a78ca227a832f70c95ff7f476927507fbda92b524e5"),
        _amd("torchvision", "0.28.0+rocm10.0.0", "torchvision-0.28.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl", "2cacf4672d025c1c24871cb4f7bdd40cc0ed073cb0458b89d665e47c2cc251be"),
        _amd("amd-torchvision-device-gfx1151", "0.28.0+rocm10.0.0", "amd_torchvision_device_gfx1151-0.28.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl", "847738fbb85db7e3cd6e711565a314bd145e71bb15f3b9271004b9821d18f234"),
        _amd("torchaudio", "2.11.0.2+rocm10.0.0", "torchaudio-2.11.0.2%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl", "7b5a98831486138582b67127e50ecb21576acbf9d85d343205847c966d1ea3ab"),
    ),
}
