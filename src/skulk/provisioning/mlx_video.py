"""Install a Foxlight-published MLX video wheel into an isolated, verified runtime."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, final

import httpx
from filelock import FileLock, Timeout
from packaging.tags import sys_tags
from packaging.version import Version
from pydantic import Field

from skulk.provisioning.comfy import stoppable_runner
from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.shared.constants import SKULK_ENABLE_VIDEO_MODELS, SKULK_ENGINES_DIR
from skulk.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from skulk.shared.types.node_facts import NodeFacts

SOURCE_REVISION: Final = "9e4a1a54194460547a103334b5e06382724fbba6"
_MAX_EXPANDED_BYTES: Final = 1024**3
NativeVideoFeature = Literal["audio_conditioning", "retake", "extend", "prompt_relay"]
"""Advanced protocols bound to an exact published native engine build."""


@final
class MlxVideoWheel(FrozenModel):
    """One immutable published package; hardware benchmarks do not gate compatibility."""

    filename: str = Field(pattern=r"^skulk_mlx_video-[\w.]+-cp313-cp313-macosx_15_0_arm64\.whl$", description="Published Foxlight wheel filename.")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$", description="Complete package digest.")
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", description="Canonical package member identity.")
    size_bytes: int = Field(ge=1, le=512 * 1024**2, description="Exact bounded download size.")
    features: tuple[NativeVideoFeature, ...] = Field(default=(), description="Advanced local-job controls implemented by these exact owned bytes.")

    @property
    def build_identity(self) -> str:
        """Return the identity carried by placement and installed-node facts."""
        return f"mlx_video@sha256:{self.payload_sha256}"

    @property
    def url(self) -> str:
        """Return the append-only Foxlight distribution URL."""
        return f"https://wheels.foxlight.ai/wheels/{self.filename}"


# Candidate run 38631488818 was attested at initiative commit 48e11c3d,
# promoted without rebuilding by run 38631848725, and read back from the
# owned channel. Only these published bytes advertise installable capacity.
PUBLISHED_WHEELS: Final[tuple[MlxVideoWheel, ...]] = (
    MlxVideoWheel(
        filename="skulk_mlx_video-0.1.0-cp313-cp313-macosx_15_0_arm64.whl",
        sha256="e2ced0b7c5b176c9285b047eb908c5a63b1322c22b0ce5d8a07e504bf02bfdee",
        payload_sha256="839dcad5cb57cc6cf1a4215cd3dbd9422d97a2b2d21963dbbc1bf7d8bf14bc89",
        size_bytes=119949758,
    ),
)


class _BuildMetadata(FrozenModel):
    source_revision: str
    payload_sha256: str
    members: dict[str, str]
    features: tuple[NativeVideoFeature, ...] = ()


class _ProbeResult(FrozenModel):
    build_identity: str
    compute: str
    ready: bool
    features: tuple[NativeVideoFeature, ...] = ()


def published_video_features() -> frozenset[NativeVideoFeature]:
    """Return features provided by owned published builds, independent of this API host."""
    return frozenset(feature for wheel in PUBLISHED_WHEELS for feature in wheel.features)


def video_features_for_build(build: str | None) -> frozenset[NativeVideoFeature]:
    """Resolve one placed build's protocol without treating qualification as compatibility."""
    return next((frozenset(wheel.features) for wheel in PUBLISHED_WHEELS if wheel.build_identity == build), frozenset[NativeVideoFeature]())


class MlxVideoInstallError(RuntimeError):
    """Explain one installation stage without advertising a failed engine."""

    def __init__(self, step: str, detail: str) -> None:
        """Retain the stage for safe UI guidance and detail for private diagnostics."""
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_for_host(expected_build: str | None = None) -> MlxVideoWheel | None:
    """Match actual macOS/ARM64/interpreter ABI, without a tested-GPU allowlist."""
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        return None
    tags = {str(tag) for tag in sys_tags()}
    return next((wheel for wheel in PUBLISHED_WHEELS if "-".join(wheel.filename.removesuffix(".whl").split("-")[-3:]) in tags and (expected_build is None or wheel.build_identity == expected_build)), None)


def runtime_root(wheel: MlxVideoWheel) -> Path:
    """Return the versioned cache location for an exact package."""
    # A compatible CPython upgrade needs a fresh venv instead of treating an
    # older, possibly active runtime as corrupt and refusing installation.
    foundation = f"{Path(sys.executable).resolve()}\n{sys.version.split()[0]}"
    interpreter_identity = hashlib.sha256(foundation.encode()).hexdigest()[:16]
    return SKULK_ENGINES_DIR / "mlx_video" / SOURCE_REVISION / wheel.sha256 / interpreter_identity


def runtime_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Prevent inherited Python/library injection and independent weight downloads."""
    environment = dict(os.environ if environ is None else environ)
    for key in tuple(environment):
        if key.startswith(("PYTHON", "DYLD_", "HF_", "HUGGING_FACE_", "UV_", "PIP_", "LTX2_", "LTX_")) or key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "IMAGEIO_FFMPEG_EXE"}:
            environment.pop(key)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
    return environment


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    entries = archive.infolist()
    if len({entry.filename for entry in entries}) != len(entries) or sum(entry.file_size for entry in entries) > _MAX_EXPANDED_BYTES:
        raise ValueError("duplicate or oversized runtime archive")
    for entry in entries:
        name = PurePosixPath(entry.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in entry.filename or (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("unsafe runtime archive member")
    return entries


def _installed_name(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if not parts or name.endswith("/"):
        return None
    if parts[0].endswith(".dist-info") and len(parts) == 2 and parts[1] == "RECORD":
        return None
    if parts[0].endswith(".data"):
        if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
            return None
        return PurePosixPath(*parts[2:]).as_posix()
    return name


def verified_interpreter(root: Path, wheel: MlxVideoWheel) -> Path | None:
    """Verify installed code against the trusted archive and bundled dependency bytes.

    Mutable installed RECORD/build files cannot become their own trust root.
    Extra importable files and altered libraries fail restart/load verification.
    """
    interpreter = root / "venv/bin/python"
    package = root / wheel.filename
    libraries = root / "venv/lib/python3.13/site-packages"
    if not interpreter.is_file() or not package.is_file() or not libraries.is_dir() or _sha256(package) != wheel.sha256:
        return None
    try:
        configuration = dict(line.split(" = ", 1) for line in (root / "venv/pyvenv.cfg").read_text().splitlines() if " = " in line)
        if interpreter.resolve() != Path(sys.executable).resolve() or configuration.get("include-system-site-packages") != "false" or configuration.get("implementation") != "CPython" or Version(configuration.get("version_info", "")).release[:2] != (3, 13):
            return None
        if Path(configuration.get("home", "")).resolve() != Path(sys.executable).resolve().parent:
            return None
        with zipfile.ZipFile(package) as archive:
            _safe_members(archive)
            metadata = _BuildMetadata.model_validate_json(archive.read("skulk_mlx_video/build.json"))
            if metadata.source_revision != SOURCE_REVISION or metadata.payload_sha256 != wheel.payload_sha256 or metadata.features != wheel.features:
                return None
            members = metadata.members
            if hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != wheel.payload_sha256:
                return None
            for name, digest in members.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    return None
            expected: dict[str, bytes] = {}
            for entry in _safe_members(archive):
                name = _installed_name(entry.filename)
                if name is not None:
                    expected[name] = archive.read(entry)
            for name in members:
                if name.startswith("skulk_mlx_video/wheels/"):
                    with archive.open(name) as stream, zipfile.ZipFile(stream) as dependency:
                        for entry in _safe_members(dependency):
                            installed = _installed_name(entry.filename)
                            if installed is not None:
                                if installed in expected:
                                    raise ValueError("overlapping runtime package members")
                                expected[installed] = dependency.read(entry)
        for name, data in expected.items():
            path = libraries / name
            if path.is_symlink() or not path.is_file() or _sha256(path) != hashlib.sha256(data).hexdigest():
                return None
        for path in libraries.rglob("*"):
            relative = path.relative_to(libraries)
            if path.is_symlink():
                return None
            if path.is_file() and relative.as_posix() not in expected:
                if len(relative.parts) == 2 and relative.parts[0].endswith(".dist-info") and relative.parts[1] in {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"}:
                    continue
                return None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return interpreter


def probe_runtime(interpreter: Path, expected_build: str) -> None:
    """Run a weightless Metal operation and bind the isolated engine to its build."""
    result = subprocess.run([str(interpreter), "-I", "-B", "-m", "skulk_mlx_video", "--probe"], capture_output=True, timeout=30, check=True, env=runtime_environment())
    expected_features = next((wheel.features for wheel in PUBLISHED_WHEELS if wheel.build_identity == expected_build), None)
    if expected_features is None or len(result.stdout) > 4096 or _ProbeResult.model_validate_json(result.stdout) != _ProbeResult(build_identity=expected_build, compute="metal", ready=True, features=expected_features):
        raise ValueError("native video probe did not verify its build and GPU")


def prepared_runtime(expected_build: str) -> Path | None:
    """Resolve verified installed capacity without downloading or trusting declarations."""
    wheel = wheel_for_host(expected_build)
    return verified_interpreter(runtime_root(wheel), wheel) if wheel is not None else None


def installable_video_builds(facts: NodeFacts, *, offline: bool) -> dict[str, str]:
    """Offer only a published compatible runtime to participating online GPU nodes."""
    wheel = wheel_for_host()
    if wheel is None or offline or not SKULK_ENABLE_VIDEO_MODELS or facts.platform != "darwin" or not facts.gpus_of("apple") or facts.mlx_video_build_identity is not None or os.environ.get(AUTOPROVISION_OPT_OUT_ENV) == "1" or os.environ.get("SKULK_NODE_PARTICIPATION", "full") != "full" or shutil.which("uv") is None:
        return {}
    return {"mlx_video-metal": wheel.build_identity}


def installed_video_runtime() -> tuple[Path, str] | None:
    """Rehydrate a verified published package and test its actual GPU on restart."""
    wheel = wheel_for_host()
    if wheel is None:
        return None
    interpreter = verified_interpreter(runtime_root(wheel), wheel)
    if interpreter is None:
        return None
    probe_runtime(interpreter, wheel.build_identity)
    return interpreter, wheel.build_identity


def install_mlx_video(backend: str, expected_build: str, *, offline: bool, stop: threading.Event) -> Path:
    """Download verified owned bytes, install locally and atomically publish readiness.

    Cancellation leaves no final runtime. Concurrent workers share a file lock;
    offline retries reuse only a fully verified cache. No package index or source
    checkout is consulted during placement.
    """
    wheel = wheel_for_host(expected_build)
    if backend != "mlx_video-metal" or wheel is None:
        raise MlxVideoInstallError("the eligibility check", "no published GPU video package matches the placement")
    root = runtime_root(wheel)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root) + ".lock")
    while not stop.is_set():
        try:
            lock.acquire(timeout=0.25)
            break
        except Timeout:
            continue
    else:
        raise MlxVideoInstallError("the preparation", "installation cancelled")
    try:
        if stop.is_set():
            raise MlxVideoInstallError("the preparation", "installation cancelled")
        # A killed worker cannot run TemporaryDirectory cleanup. Only discard
        # staging older than a full day, beyond this installer's command/network
        # budgets; never replace a final runtime that may still be in use.
        for abandoned in root.parent.glob(".mlx-video-*"):
            if abandoned.is_dir() and not abandoned.is_symlink() and time.time() - abandoned.stat().st_mtime > 24 * 60 * 60:
                shutil.rmtree(abandoned)
        cached = verified_interpreter(root, wheel)
        if cached is not None:
            probe_runtime(cached, expected_build)
            return cached
        if root.exists():
            raise MlxVideoInstallError("the integrity check", "cached runtime is corrupt; refusing to replace active files")
        if offline or os.environ.get(AUTOPROVISION_OPT_OUT_ENV) == "1":
            raise MlxVideoInstallError("the download", "engine is not cached and automatic download is unavailable")
        if shutil.disk_usage(root.parent).free < wheel.size_bytes + 2 * _MAX_EXPANDED_BYTES:
            raise MlxVideoInstallError("the disk-space check", "insufficient runtime staging space")
        with tempfile.TemporaryDirectory(prefix=".mlx-video-", dir=root.parent) as temporary:
            staging = Path(temporary)
            package = staging / wheel.filename
            total = 0
            with httpx.Client(timeout=httpx.Timeout(60, read=30)) as client, client.stream("GET", wheel.url, follow_redirects=False) as response, package.open("wb") as target:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024**2):
                    if stop.is_set():
                        raise MlxVideoInstallError("the download", "installation cancelled")
                    total += len(chunk)
                    if total > wheel.size_bytes:
                        raise MlxVideoInstallError("the download", "package exceeds its pinned size")
                    target.write(chunk)
            if total != wheel.size_bytes or _sha256(package) != wheel.sha256:
                raise MlxVideoInstallError("the integrity check", "package differs from its pinned bytes")
            wheelhouse = staging / "wheelhouse"
            wheelhouse.mkdir()
            with zipfile.ZipFile(package) as archive:
                for entry in _safe_members(archive):
                    if entry.filename.startswith("skulk_mlx_video/wheels/") and entry.filename.endswith(".whl"):
                        (wheelhouse / PurePosixPath(entry.filename).name).write_bytes(archive.read(entry))
            run = stoppable_runner(stop)
            environment = runtime_environment()
            run(["uv", "venv", "--no-config", "--no-project", "--python", sys.executable, str(staging / "venv")], env=environment, check=True, capture_output=True, timeout=60)
            interpreter = staging / "venv/bin/python"
            # This runtime never builds packages. Remove uv's build-time .pth
            # hook so every importable file has a wheel-backed trust root.
            for name in ("_virtualenv.py", "_virtualenv.pth"):
                (staging / "venv/lib/python3.13/site-packages" / name).unlink(missing_ok=True)
            packages = [package, *sorted(wheelhouse.glob("*.whl"))]
            requirements = staging / "install.lock"
            requirements.write_text("\n".join(f"{path.as_uri()} --hash=sha256:{_sha256(path)}" for path in packages) + "\n")
            run(["uv", "pip", "install", "--no-config", "--python", str(interpreter), "--no-index", "--no-deps", "--require-hashes", "-r", str(requirements)], env=environment, check=True, capture_output=True, timeout=300)
            shutil.rmtree(wheelhouse)
            requirements.unlink()
            verified = verified_interpreter(staging, wheel)
            if verified is None:
                raise MlxVideoInstallError("the integrity check", "installed code differs from its package")
            probe_runtime(verified, expected_build)
            if stop.is_set():
                raise MlxVideoInstallError("the preparation", "installation cancelled")
            # venv entrypoints are unused: the runner executes its relocated
            # interpreter directly. pyvenv.cfg keeps the trusted base interpreter.
            staging.rename(root)
        return root / "venv/bin/python"
    except (OSError, ValueError, subprocess.SubprocessError, httpx.HTTPError) as error:
        raise MlxVideoInstallError("the engine preparation", str(error)) from error
    finally:
        lock.release()
