"""Prepare sealed native ACE libraries without weights or loaded-model claims.

Only independently pinned built-in packages can enter this cache. CPU/Metal,
CUDA and ROCm retain separate dependency profiles; original model artifacts are
downloaded through the model store and never enter an engine package.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self, final

import httpx
from packaging.tags import platform_tags
from pydantic import Field, model_validator

from skulk.provisioning.demucs_package import (
    BuildManifest,
    Checkpoint,
    Digest,
    NativePackageLimits,
    NativePackageVariant,
    inspect_candidate,
    native_package_limits,
)
from skulk.provisioning.demucs_runtime import (
    DemucsBuildMismatchError,
    PreparationPhase,
    SealedRuntimeCache,
    download_demucs_package,
)
from skulk.provisioning.native_install_process import NativeInstallCommandRunner
from skulk.shared.types.native_music import NativeMusicDevice, NativeMusicLibraryProbe
from skulk.utils.pydantic_ext import FrozenModel

AceStepPlatform = Literal[
    "macosx_14_0_arm64",
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "linux_x86_64",
]
_SOURCE_REVISION = "ca1e85fe9430179831e6bc6be790c332190a3866"
_FEATURES = frozenset(
    {
        "music.generate",
        "music.edit",
        "source_conditioning",
        "pcm24",
        "offline_loader",
        "bundled_python",
        "resident_model",
        "library_probe",
        "memory_probe",
    }
)


def _supports_device(
    variant: NativePackageVariant, platform: AceStepPlatform, device: NativeMusicDevice
) -> bool:
    if variant == "cpu":
        return device == "cpu" or device == "mps" and platform.startswith("macosx_")
    return device == variant


@final
class AceStepWheel(FrozenModel):
    """One exact engine-channel object, including its actual dependency profile.

    The raw Linux ROCm target deliberately makes no inferred manylinux/libc
    promise. Full ELF/loader qualification is required before publishing such a
    package. A real library/device probe remains mandatory on each installation.
    """

    variant: NativePackageVariant = Field(
        description="Built-in CPU/Metal, CUDA or ROCm dependency profile."
    )
    platform_tag: AceStepPlatform = Field(
        description="Package OS/CPU ABI; not a tested-device permission list."
    )
    filename: str = Field(description="Exact immutable engine-channel basename.")
    sha256: Digest = Field(description="Original complete engine package SHA-256.")
    payload_sha256: Digest = Field(
        description="Expected adapter/source/dependency payload identity."
    )
    size_bytes: int = Field(
        gt=0, le=4 * 1024**3, description="Exact bounded engine object length."
    )

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        """Keep filename, storage policy and actually assembled targets together."""
        if (
            self.filename
            != f"skulk_ace_step_{self.variant}-0.1.0-cp312-cp312-{self.platform_tag}.whl"
            or self.size_bytes > native_package_limits(self.variant).package_bytes
            or self.variant == "cpu"
            and self.platform_tag == "linux_x86_64"
            or self.variant == "cuda"
            and not self.platform_tag.startswith("manylinux_")
            or self.variant == "rocm"
            and self.platform_tag != "linux_x86_64"
        ):
            raise ValueError(
                "native ACE package identity differs from its built-in profile"
            )
        return self

    @property
    def url(self) -> str:
        """Return the fixed Foxlight package channel, never a caller-selected URL."""
        return "https://wheels.foxlight.ai/wheels/" + self.filename

    @property
    def build_identity(self) -> str:
        """Identify verified libraries without implying any loaded model."""
        return "ace_step@sha256:" + self.payload_sha256

    @property
    def required_platform_tags(self) -> frozenset[str]:
        """Include the known interpreter ABI without inventing a GPU-library floor."""
        if self.platform_tag == "linux_x86_64":
            return frozenset((self.platform_tag, "manylinux_2_28_x86_64"))
        return frozenset((self.platform_tag,))


@final
class PreparedAceStepRuntime(FrozenModel):
    """Independently verified installation and its actual weights-free probe."""

    wheel: AceStepWheel = Field(description="Selected complete package/payload pins.")
    generation: Path = Field(description="Stable private cache generation.")
    interpreter: Path = Field(description="Verified isolated native Python.")
    probe: NativeMusicLibraryProbe = Field(
        description="Measured native library/device policy, distinct from model ready."
    )

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        """Refuse foreign payloads, incompatible lanes and escaped private paths."""
        if (
            self.probe.payload_sha256 != self.wheel.payload_sha256
            or not _supports_device(
                self.wheel.variant, self.wheel.platform_tag, self.probe.device
            )
            or not self.generation.is_absolute()
            or not self.interpreter.is_relative_to(self.generation)
        ):
            raise ValueError(
                "native ACE preparation receipt does not bind its installation"
            )
        return self

    @property
    def device(self) -> NativeMusicDevice:
        """Return the actual requested native lane exercised by the probe."""
        return self.probe.device


def download_ace_step_package(
    wheel: AceStepWheel,
    destination: Path,
    checkpoint: Checkpoint,
    *,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Stream the pinned native object through the common bounded download path."""
    download_demucs_package(wheel, destination, checkpoint, transport=transport)


@final
class _AceStepPreparationPolicy:
    """Bind the common cache to one reviewed native music dependency profile."""

    label = "native ACE"

    def __init__(self, variant: NativePackageVariant) -> None:
        self.limits: NativePackageLimits = native_package_limits(variant)

    def validate(self, package: AceStepWheel, device: NativeMusicDevice) -> None:
        """Reject an actual package/host/lane mismatch before cache effects."""
        if (
            package.variant != self.limits.variant
            or not package.required_platform_tags.issubset(set(platform_tags()))
            or not _supports_device(package.variant, package.platform_tag, device)
        ):
            raise DemucsBuildMismatchError(
                "native ACE package cannot provide the requested host/device profile"
            )

    def inspect(
        self, path: Path, package: AceStepWheel, checkpoint: Checkpoint
    ) -> BuildManifest:
        """Verify original bytes and bind exact source/features before execution."""
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != package.size_bytes
        ):
            raise ValueError("native ACE package is missing, redirected or incomplete")
        manifest = inspect_candidate(
            path, package.sha256, checkpoint=checkpoint, limits=self.limits
        )
        if (
            manifest.source_repository != "https://github.com/ace-step/ACE-Step-1.5.git"
            or manifest.source_revision != _SOURCE_REVISION
            or manifest.variant != package.variant
            or manifest.platform_tag != package.platform_tag
            or manifest.payload_sha256 != package.payload_sha256
            or not _FEATURES.issubset(manifest.features)
        ):
            raise DemucsBuildMismatchError(
                "native ACE manifest differs from its pinned built-in build policy"
            )
        return manifest

    def probe(
        self,
        interpreter: Path,
        package: AceStepWheel,
        device: NativeMusicDevice,
        runner: NativeInstallCommandRunner,
    ) -> NativeMusicLibraryProbe:
        """Probe verified code under the install guardian without model paths.

        The install guardian already owns the complete private session and its
        inherited cache fence. The resident CLI requires a direct supervisor and
        a session leader, so invoking that CLI inside this guardian would violate
        its lifetime contract. Call the same closed probe under this owner.
        """
        code = (
            "import contextlib,json,sys\n"
            "with contextlib.redirect_stdout(sys.stderr):\n"
            " from importlib.resources import files\n"
            " from skulk_ace_step.policy import probe_libraries\n"
            " manifest=json.loads(files('skulk_ace_step').joinpath('build.json').read_text())\n"
            f" receipt=probe_libraries({device!r},manifest['payload_sha256'])\n"
            "print(receipt.model_dump_json())\n"
        )
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP", "LANG"}
        } | {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "MIOPEN_FIND_MODE": "FAST",
        }
        receipt = NativeMusicLibraryProbe.model_validate_json(
            runner(
                [str(interpreter), "-I", "-B", "-c", code],
                environment=environment,
                timeout_seconds=300,
                capture=True,
            )
        )
        if (receipt.payload_sha256, receipt.device) != (package.payload_sha256, device):
            raise DemucsBuildMismatchError(
                "native ACE probe selected a different payload or compute device"
            )
        if (
            receipt.memory_capacity is None
            or receipt.memory_capacity.sampled_at_unix_ms <= 0
        ):
            raise DemucsBuildMismatchError(
                "native ACE probe has no actual device memory observation"
            )
        return receipt

    def prepared(
        self,
        package: AceStepWheel,
        generation: Path,
        interpreter: Path,
        device: NativeMusicDevice,
        observation: NativeMusicLibraryProbe,
    ) -> PreparedAceStepRuntime:
        """Return the verified package and its correlated actual library receipt."""
        if observation.device != device:
            raise DemucsBuildMismatchError(
                "native ACE probe differs from requested lane"
            )
        return PreparedAceStepRuntime(
            wheel=package,
            generation=generation,
            interpreter=interpreter,
            probe=observation,
        )

    def download(
        self, package: AceStepWheel, destination: Path, checkpoint: Checkpoint
    ) -> None:
        """Download only this selected engine-channel object, never model weights."""
        download_ace_step_package(package, destination, checkpoint)


@final
class AceStepRuntimeCache:
    """Own all native music profiles under the common cache publication fence.

    Each profile retains its cleanup handles; the shared root serializes actual
    disk reservations across profiles/controllers. Published generations survive
    upgrades and cancellation. No engine inventory or model readiness is emitted.
    """

    def __init__(self, root: Path) -> None:
        """Select the private root without any filesystem, download or import effect."""
        self.root = root.absolute()
        self._guard = threading.Lock()
        self._caches: dict[
            NativePackageVariant,
            SealedRuntimeCache[
                AceStepWheel,
                NativeMusicDevice,
                PreparedAceStepRuntime,
                NativeMusicLibraryProbe,
            ],
        ] = {}

    def prepare(
        self,
        wheel: AceStepWheel,
        *,
        device: NativeMusicDevice,
        offline: bool,
        is_cancelled: Callable[[], bool],
        progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
        download: Callable[[AceStepWheel, Path, Checkpoint], None] | None = None,
    ) -> PreparedAceStepRuntime:
        """Verify or install the exact profile and execute its real device probe.

        Offline mode permits only verified cached original package bytes. Failure
        or cancellation never publishes a generation before probe and process
        drain; model weights/loading and ready capacity remain separate.
        """
        with self._guard:
            cache = self._caches.get(wheel.variant)
            if cache is None:
                cache = SealedRuntimeCache(
                    self.root, _AceStepPreparationPolicy(wheel.variant)
                )
                self._caches[wheel.variant] = cache
        return cache.prepare(
            wheel,
            device=device,
            offline=offline,
            is_cancelled=is_cancelled,
            progress=progress,
            timeout_seconds=timeout_seconds,
            download=download,
        )

    def close(self) -> None:
        """Retry retained cleanup after the worker has cancelled/joined preparation."""
        with self._guard:
            for cache in self._caches.values():
                cache.close()
