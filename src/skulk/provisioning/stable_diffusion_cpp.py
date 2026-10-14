"""Prepare immutable native image wheels only for accepted model placements."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast, final

import httpx
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from pydantic import ConfigDict, Field

from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.shared.constants import SKULK_ENGINES_DIR
from skulk.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from skulk.shared.types.node_facts import NodeFacts

SOURCE_REVISION: Final = "d637cfb935feb2254898a8f1524046c958f3df80"
SERVER_PATCH_SHA256: Final = (
    "5c97917794372067d71ef6a49083f0c1afcfa3bbf46a368d08a7d0591691769b"
)
BINARY_ENV: Final = "SKULK_STABLE_DIFFUSION_CPP_BIN"
ImageCompute = Literal["metal", "cuda", "vulkan", "rocm"]
CpuInstruction = Literal["avx", "avx2", "bmi2", "f16c", "fma", "sse4_2"]
_INSTALL_LOCK = threading.Lock()
_MAX_PACKAGE_BYTES = 1024 * 1024 * 1024
_MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
_DEVICE_COMPUTES: Final[dict[str, ImageCompute]] = {
    "MTL": "metal",
    "CUDA": "cuda",
    "VULKAN": "vulkan",
    "HIP": "rocm",
    "ROCM": "rocm",
}


@final
class StableDiffusionCppWheel(FrozenModel):
    """One published engine package, independently pinned before it is installable."""

    filename: str = Field(
        pattern=r"^skulk_stable_diffusion_cpp_(?:metal|cuda|vulkan|rocm)-[a-zA-Z0-9_.]+-py3-none-[a-zA-Z0-9_]+\.whl$",
        description="Immutable wheel filename on the Foxlight engine index.",
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Expected complete wheel digest."
    )
    binary_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Expected native executable digest."
    )
    payload_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$",
        description="Complete managed executable/runtime identity; absent for legacy candidates.",
    )
    size_bytes: int | None = Field(
        default=None, ge=1, le=3 * 1024**3,
        description="Exact bounded wheel size, including bundled GPU runtime libraries.",
    )
    expanded_size_bytes: int | None = Field(
        default=None, ge=1, le=8 * 1024**3,
        description="Exact bounded runtime payload size needed during installation.",
    )
    compute: ImageCompute = Field(
        description="Compiled GPU backend; CPU-only builds are not deliverable."
    )
    system: Literal["darwin", "linux"] = Field(
        description="Supported host operating system."
    )
    architecture: Literal["arm64", "aarch64", "x86_64"] = Field(
        description="Native host architecture."
    )
    hardware_classes: frozenset[str] = Field(
        min_length=1, description="Hardware classes this exact package has qualified."
    )
    required_cpu_features: frozenset[CpuInstruction] = Field(
        default=frozenset(),
        description="CPU instructions required by the package's compiled helper kernels.",
    )

    @property
    def prefix(self) -> str:
        """Return the wheel's fixed runtime payload directory."""
        return f"skulk_stable_diffusion_cpp_{self.compute}/"

    @property
    def build_identity(self) -> str:
        """Return executable-bound build truth, independent of user declarations."""
        return f"stable_diffusion_cpp@sha256:{self.payload_sha256 or self.binary_sha256}"

    @property
    def url(self) -> str:
        """Return the append-only package channel location."""
        return f"https://wheels.foxlight.ai/wheels/{self.filename}"


# These exact bytes were promoted from an attested build and qualified on the
# observed hardware classes below. Compiled kernel targets alone do not qualify
# another GPU; model feature support remains separately bound by registry claims.
PUBLISHED_WHEELS: Final[tuple[StableDiffusionCppWheel, ...]] = (
    StableDiffusionCppWheel(
        filename='skulk_stable_diffusion_cpp_metal-0.1.0.post1-py3-none-macosx_15_0_arm64.whl',
        sha256='d84428a1073f21a4f40e8eec7610eb44eae86dd45170047473d18a4c568ca939',
        binary_sha256='f0e876810448d30fe1a263bd8053f66b8b97ff20735f6aee9592492c2712abda',
        payload_sha256='201af2888ceeeaaa86a7523023610b91e6ef7644f6b63b3af740c145ff4ace5d',
        size_bytes=16435084,
        expanded_size_bytes=42336264,
        compute='metal',
        system='darwin',
        architecture='arm64',
        hardware_classes=frozenset(['apple:apple-m4']),
        required_cpu_features=frozenset([]),
    ),
    StableDiffusionCppWheel(
        filename='skulk_stable_diffusion_cpp_cuda-0.1.0.post1-py3-none-manylinux_2_35_aarch64.whl',
        sha256='e4f68af1b4bca2041cf40e93ccb40868ea4cf92a660497aa57120e08393f8cfd',
        binary_sha256='d569fe99973f3c4c3b3b6a0bd54593a21c60ec9338d5f62ffc06dc3376720b69',
        payload_sha256='4ecf8709af95fd6429dc64d8f32ef43165b71b4ccd48a571222c56173510c1e9',
        size_bytes=543371268,
        expanded_size_bytes=748018484,
        compute='cuda',
        system='linux',
        architecture='aarch64',
        hardware_classes=frozenset(['nvidia:nvidia-gb10']),
        required_cpu_features=frozenset([]),
    ),
    StableDiffusionCppWheel(
        filename='skulk_stable_diffusion_cpp_cuda-0.1.0.post1-py3-none-manylinux_2_35_x86_64.whl',
        sha256='6a055d454b7dd90edf2d74e948a4398df123e74129564f51ebd6b4d35a3610ad',
        binary_sha256='56c6f97815b103f6c0477d998ecbcc95f3dda60ef3965969ef9fffd59dbdd9c7',
        payload_sha256='0dcd9be9a5e5607f1edc1b3218e55b7b045ef8fe608dd93341d64a1fefdda4ab',
        size_bytes=613420003,
        expanded_size_bytes=926212077,
        compute='cuda',
        system='linux',
        architecture='x86_64',
        hardware_classes=frozenset(['nvidia:nvidia-a100-sxm4-80gb', 'nvidia:nvidia-geforce-rtx-4090']),
        required_cpu_features=frozenset(['avx', 'avx2', 'bmi2', 'f16c', 'fma', 'sse4_2']),
    ),
    StableDiffusionCppWheel(
        filename='skulk_stable_diffusion_cpp_vulkan-0.1.0.post1-py3-none-manylinux_2_35_x86_64.whl',
        sha256='dbbb89980b894a6ac00e7188f563e10bd01d7d10c9d9a00f88aaa1e57fd6fc96',
        binary_sha256='23d7cf1efeb5d6c966813e15cee4a1b2fbb5c145fd77abbe9b6a8a1f778b273a',
        payload_sha256='9b9984949802fbd0f2b33aa22044bc728c05f252b46ba21da471c4e86c550732',
        size_bytes=30791307,
        expanded_size_bytes=86390804,
        compute='vulkan',
        system='linux',
        architecture='x86_64',
        hardware_classes=frozenset(['amd:pci-1002-1586']),
        required_cpu_features=frozenset(['avx', 'avx2', 'bmi2', 'f16c', 'fma', 'sse4_2']),
    ),
    StableDiffusionCppWheel(
        filename='skulk_stable_diffusion_cpp_rocm-0.1.0.post1-py3-none-manylinux_2_35_x86_64.whl',
        sha256='45bec6e5e4df083cbf44eeb81d5f09be5214c323994a4800ae918129e3bb44ff',
        binary_sha256='67122895591407c0f1656bd9d8eb91c1e358e0ad6fe6ea77a154b0834e8ee3c4',
        payload_sha256='46bfe6624f77b43f24bcef1ee2df6ffb2cb461b6ff4fc5bedda4a302998b2759',
        size_bytes=2170695534,
        expanded_size_bytes=4644443682,
        compute='rocm',
        system='linux',
        architecture='x86_64',
        hardware_classes=frozenset(['amd:pci-1002-74b9']),
        required_cpu_features=frozenset(['avx', 'avx2', 'bmi2', 'f16c', 'fma', 'sse4_2']),
    ),
)


@final
class NativeImageProbe(FrozenModel):
    """Executable identity and actual devices from the hardened native server."""

    source_revision: str = Field(
        description="Source revision reported by the executable."
    )
    binary_sha256: str = Field(description="Measured executable digest.")
    payload_sha256: str | None = Field(
        default=None, description="Verified managed runtime payload identity, when present."
    )
    devices: tuple[str, ...] = Field(
        description="Native non-CPU device names, preserving runtime identifiers."
    )
    computes: tuple[ImageCompute, ...] = Field(
        description="Observed usable GPU compute lanes."
    )

    @property
    def build_identity(self) -> str:
        """Return the identity registry support claims bind."""
        return f"stable_diffusion_cpp@sha256:{self.payload_sha256 or self.binary_sha256}"


class ImageEngineInstallError(RuntimeError):
    """Fail a preparation at an operator-readable stage without advertising readiness."""

    def __init__(self, step: str, detail: str) -> None:
        """Retain the public stage and private diagnostic detail."""
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


def file_sha256(path: Path) -> str:
    """Hash a runtime file without holding its bytes in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@final
class NativeCpuRequirements(FrozenModel):
    """Read the compiled CPU requirements from a managed native build manifest."""

    # Other provenance is checked separately by the archive and executable probes.
    model_config = ConfigDict(extra="ignore")

    required_cpu_features: frozenset[CpuInstruction] = Field(
        default=frozenset(),
        description="Required x86 helper instructions; empty for other CPU targets.",
    )
    payload_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", description="Canonical member-digest mapping identity."
    )
    members: dict[str, str] = Field(
        default_factory=dict, max_length=10000,
        description="Managed file digests, including the executable and runtime code objects."
    )


def _host_cpu_features() -> frozenset[str]:
    if platform.system().lower() != "linux":
        return frozenset()
    try:
        cpu_info = Path("/proc/cpuinfo").read_text()
    except OSError:
        return frozenset()
    processors = [
        frozenset(value.split())
        for line in cpu_info.splitlines()
        if (parts := line.partition(":"))[0].strip() == "flags"
        and (value := parts[2])
    ]
    return processors[0].intersection(*processors[1:]) if processors else frozenset()


def probe_native_image(binary: Path) -> NativeImageProbe:
    """Verify compiled CPU requirements, source identity and a real GPU device."""
    metadata = binary.parent.parent / "build.json"
    payload_digest: str | None = None
    if metadata.is_file():
        requirements = NativeCpuRequirements.model_validate_json(metadata.read_text())
        if requirements.required_cpu_features and not (
            requirements.required_cpu_features <= _host_cpu_features()
        ):
            raise ValueError("native image engine requires unavailable CPU instructions")
        if requirements.payload_sha256 is not None:
            encoded = json.dumps(requirements.members, sort_keys=True, separators=(",", ":")).encode()
            if not requirements.members or hashlib.sha256(encoded).hexdigest() != requirements.payload_sha256:
                raise ValueError("native image payload manifest identity differs")
            for name, digest in requirements.members.items():
                relative = Path(name)
                if not relative.parts or relative.is_absolute() or ".." in relative.parts or relative.parts[0] != binary.parent.parent.name:
                    raise ValueError("native image payload member is unsafe")
                target = binary.parent.parent.parent / relative
                if not target.is_file() or file_sha256(target) != digest:
                    raise ValueError("native image payload member differs from its identity")
            if requirements.members.get(f"{binary.parent.parent.name}/bin/sd-server") != file_sha256(binary):
                raise ValueError("native image payload does not bind its executable")
            payload_digest = requirements.payload_sha256
    completed = subprocess.run(
        [str(binary), "--skulk-probe"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env=native_image_environment(binary),
    )
    raw = cast("object", json.loads(completed.stdout.strip()))
    if not isinstance(raw, dict):
        raise ValueError("native image probe is not an object")
    record = cast("dict[str, object]", raw)
    if record.get("protocol") != 1 or record.get("source_revision") != SOURCE_REVISION:
        raise ValueError(
            "native image probe reports an unrecognized protocol or source"
        )
    report = record.get("device_report")
    if not isinstance(report, str):
        raise ValueError("native image probe lacks a device report")
    devices: list[str] = []
    computes: list[ImageCompute] = []
    for line in report.splitlines():
        name = line.split("\t", 1)[0]
        # Vulkan software devices have GPU-like names; their prefix is not hardware proof.
        if any(
            marker in line.lower()
            for marker in (
                "llvmpipe",
                "lavapipe",
                "swiftshader",
                "softpipe",
                "software rasterizer",
            )
        ):
            continue
        prefix = re.sub(r"\d+$", "", name).upper()
        if (compute := _DEVICE_COMPUTES.get(prefix)) is not None:
            devices.append(name)
            if compute not in computes:
                computes.append(compute)
    if not devices:
        raise ValueError("native image engine reports no GPU device")
    return NativeImageProbe(
        source_revision=SOURCE_REVISION,
        binary_sha256=file_sha256(binary),
        payload_sha256=payload_digest,
        devices=tuple(devices),
        computes=tuple(computes),
    )


def native_image_environment(binary: Path) -> dict[str, str]:
    """Return an isolated managed-runtime environment without mutating the caller.

    Legacy raw builds retain their operator environment. Managed payloads select
    their verified libraries/code objects while system GPU drivers stay external.
    """
    environment = dict(os.environ)
    metadata = binary.parent.parent / "build.json"
    if not metadata.is_file() or NativeCpuRequirements.model_validate_json(metadata.read_text()).payload_sha256 is None:
        return environment
    libraries = binary.parent.parent / "lib"
    for key in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "ROCBLAS_TENSILE_LIBPATH", "HIPBLASLT_TENSILE_LIBPATH"):
        environment.pop(key, None)
    if platform.system().lower() == "linux":
        environment["LD_LIBRARY_PATH"] = str(libraries)
    for key, component in (("ROCBLAS_TENSILE_LIBPATH", "rocblas"), ("HIPBLASLT_TENSILE_LIBPATH", "hipblaslt")):
        directory = libraries / component / "library"
        if directory.is_dir():
            environment[key] = str(directory)
    return environment


def wheel_for_host(
    backend: str, hardware_classes: frozenset[str]
) -> StableDiffusionCppWheel | None:
    """Find an exact published package matching this host and qualified GPU class."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    machine = (
        {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
        if system == "linux"
        else machine
    )
    return next(
        (
            wheel
            for wheel in PUBLISHED_WHEELS
            if wheel.system == system
            and wheel.architecture == machine
            and backend == f"stable_diffusion_cpp-{wheel.compute}"
            and bool(wheel.hardware_classes & hardware_classes)
            and (
                not wheel.required_cpu_features
                or wheel.required_cpu_features <= _host_cpu_features()
            )
        ),
        None,
    )


def _root(wheel: StableDiffusionCppWheel) -> Path:
    return SKULK_ENGINES_DIR / "stable_diffusion_cpp" / SOURCE_REVISION / wheel.sha256


@contextmanager
def _package_lock(root: Path, stop: threading.Event) -> Iterator[None]:
    """Serialize independent worker/doctor processes while allowing cancellation."""
    lock = FileLock(str(root) + ".lock")
    while True:
        if stop.is_set():
            raise ImageEngineInstallError("the preparation", "installation cancelled")
        try:
            lock.acquire(timeout=0.25)
            break
        except FileLockTimeout:
            continue
    try:
        yield
    finally:
        lock.release()


def installable_image_builds(
    facts: NodeFacts, hardware_classes: frozenset[str], *, offline: bool
) -> dict[str, str]:
    """Advertise only published packages an online participating node may acquire."""
    if (
        offline
        or os.environ.get(AUTOPROVISION_OPT_OUT_ENV) == "1"
        or (os.environ.get("SKULK_NODE_PARTICIPATION", "full") != "full")
        or facts.stable_diffusion_cpp_binary.state != "not_configured"
    ):
        return {}
    builds: dict[str, str] = {}
    for compute in ("metal", "cuda", "rocm", "vulkan"):
        backend = f"stable_diffusion_cpp-{compute}"
        wheel = wheel_for_host(backend, hardware_classes)
        if wheel is not None:
            builds[backend] = wheel.build_identity
    return builds


def rehydrate_native_image(hardware_classes: frozenset[str]) -> Path | None:
    """Wire a verified cached GPU package on restart without downloading."""
    if os.environ.get(BINARY_ENV, "").strip():
        return None
    for compute in ("metal", "cuda", "rocm", "vulkan"):
        wheel = wheel_for_host(f"stable_diffusion_cpp-{compute}", hardware_classes)
        if wheel is None:
            continue
        binary = _verify_payload(_root(wheel), wheel)
        if binary is not None:
            os.environ[BINARY_ENV] = str(binary)
            return binary
    return None


def prepared_native_image(
    backend: str, expected_build: str, hardware_classes: frozenset[str]
) -> Path | None:
    """Resolve a verified installed package for a spawned runner without downloading.

    The runner may have started before its worker installed the engine, so its
    inherited environment cannot prove current cache readiness. Only the exact
    eligible package/build is returned; loading still probes the actual GPU.
    """
    wheel = wheel_for_host(backend, hardware_classes)
    if wheel is None or wheel.build_identity != expected_build:
        return None
    return _verify_payload(_root(wheel), wheel)


def _verify_payload(root: Path, wheel: StableDiffusionCppWheel) -> Path | None:
    archive_path = root / wheel.filename
    binary = root / wheel.prefix / "bin/sd-server"
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size > (wheel.size_bytes or _MAX_PACKAGE_BYTES)
        or (wheel.size_bytes is not None and archive_path.stat().st_size != wheel.size_bytes)
        or file_sha256(archive_path) != wheel.sha256
    ):
        return None
    if (
        not binary.is_file()
        or not os.access(binary, os.X_OK)
        or file_sha256(binary) != wheel.binary_sha256
    ):
        return None
    # Revalidate every extracted runtime file against independently pinned archive bytes.
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if not item.filename.startswith(wheel.prefix) or item.is_dir():
                continue
            target = root / item.filename
            if not target.is_file():
                return None
            with archive.open(item) as source:
                digest = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if file_sha256(target) != digest.hexdigest():
                return None
    return binary


def install_native_image(
    backend: str,
    expected_build: str,
    hardware_classes: frozenset[str],
    *,
    offline: bool,
    stop: threading.Event,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Prepare one pinned package, verify actual build/device, then wire the cache.

    Runs off the worker loop. Concurrent requests share a serialized install;
    interruption leaves no final cache. Offline mode only reuses verified bytes.
    """
    env = os.environ if environ is None else environ
    if explicit := env.get(BINARY_ENV, "").strip():
        binary = Path(explicit)
        observed = probe_native_image(binary)
        if (
            observed.build_identity != expected_build
            or len(observed.devices) != 1
            or backend.removeprefix("stable_diffusion_cpp-") not in observed.computes
        ):
            raise ImageEngineInstallError(
                "the engine check", "override build or GPU differs from placement"
            )
        return binary
    wheel = wheel_for_host(backend, hardware_classes)
    if wheel is None or wheel.build_identity != expected_build:
        raise ImageEngineInstallError(
            "the eligibility check",
            "no exact published image engine matches this placement",
        )
    root = _root(wheel)
    root.parent.mkdir(parents=True, exist_ok=True)
    with _INSTALL_LOCK, _package_lock(root, stop):
        if stop.is_set():
            raise ImageEngineInstallError("the preparation", "installation cancelled")
        binary = _verify_payload(root, wheel)
        if binary is None:
            if root.exists():
                raise ImageEngineInstallError(
                    "the integrity check",
                    "cached image engine is corrupt; refusing to replace potentially active files",
                )
            if offline or env.get(AUTOPROVISION_OPT_OUT_ENV) == "1":
                raise ImageEngineInstallError(
                    "the download",
                    "engine is not cached and automatic download is unavailable",
                )
            root.parent.mkdir(parents=True, exist_ok=True)
            package_limit = wheel.size_bytes or _MAX_PACKAGE_BYTES
            expanded_limit = wheel.expanded_size_bytes or _MAX_EXPANDED_BYTES
            if shutil.disk_usage(root.parent).free < package_limit + expanded_limit + 1024**3:
                raise ImageEngineInstallError(
                    "the disk-space check", "insufficient engine staging space"
                )
            with tempfile.TemporaryDirectory(
                prefix=".image-engine-", dir=root.parent
            ) as temporary:
                staging = Path(temporary)
                archive_path = staging / wheel.filename
                digest = hashlib.sha256()
                total = 0
                with (
                    httpx.Client(timeout=httpx.Timeout(60, read=30)) as client,
                    client.stream("GET", wheel.url, follow_redirects=False) as response,
                    archive_path.open("wb") as target,
                ):
                    response.raise_for_status()
                    for chunk in response.iter_bytes(1024 * 1024):
                        if stop.is_set():
                            raise ImageEngineInstallError(
                                "the download", "installation cancelled"
                            )
                        total += len(chunk)
                        if total > package_limit:
                            raise ImageEngineInstallError(
                                "the download",
                                "package exceeds the bounded download size",
                            )
                        digest.update(chunk)
                        target.write(chunk)
                if digest.hexdigest() != wheel.sha256:
                    raise ImageEngineInstallError(
                        "the integrity check", "wheel digest differs from the pin"
                    )
                with zipfile.ZipFile(archive_path) as archive:
                    entries = [
                        item
                        for item in archive.infolist()
                        if item.filename.startswith(wheel.prefix)
                    ]
                    expanded_size = sum(item.file_size for item in entries)
                    if expanded_size > expanded_limit or (wheel.expanded_size_bytes is not None and expanded_size != expanded_limit):
                        raise ImageEngineInstallError(
                            "the integrity check", "expanded package exceeds its bound"
                        )
                    for item in entries:
                        if stop.is_set():
                            raise ImageEngineInstallError(
                                "the extraction", "installation cancelled"
                            )
                        relative = Path(item.filename)
                        if (
                            relative.is_absolute()
                            or ".." in relative.parts
                            or (item.external_attr >> 16) & 0o170000 == 0o120000
                        ):
                            raise ImageEngineInstallError(
                                "the integrity check", "unsafe archive member"
                            )
                        destination = staging / relative
                        if item.is_dir():
                            destination.mkdir(parents=True, exist_ok=True)
                        else:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with (
                                archive.open(item) as source,
                                destination.open("wb") as target,
                            ):
                                shutil.copyfileobj(source, target)
                (staging / wheel.prefix / "bin/sd-server").chmod(0o755)
                binary = _verify_payload(staging, wheel)
                if binary is None:
                    raise ImageEngineInstallError(
                        "the integrity check",
                        "extracted package differs from its pinned bytes",
                    )
                observed = probe_native_image(binary)
                if (
                    observed.build_identity != expected_build
                    or len(observed.devices) != 1
                    or wheel.compute not in observed.computes
                    or stop.is_set()
                ):
                    raise ImageEngineInstallError(
                        "the engine check",
                        "prepared binary cannot serve the selected GPU",
                    )
                staging.rename(root)
                binary = root / wheel.prefix / "bin/sd-server"
        observed = probe_native_image(binary)
        if (
            observed.build_identity != expected_build
            or len(observed.devices) != 1
            or wheel.compute not in observed.computes
        ):
            raise ImageEngineInstallError(
                "the engine check", "cached build/device no longer matches placement"
            )
        os.environ[BINARY_ENV] = str(binary)
        return binary
