"""Fetch a pinned audio.cpp wheel only when a music mount prepares this node.

The package lives in the existing Foxlight engine wheel channel. Skulk extracts
only its fixed server, model specs, and license into an engine cache; model
weights remain separate. No installable package is added to NodeResources until
the extracted binary passes the normal Node Facts probes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast, final

import httpx
from pydantic import BaseModel, ConfigDict, Field

from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.shared.backends import AUDIO_CPP_BIN_ENV, AUDIO_CPP_SPECS_DIR_ENV
from skulk.shared.constants import SKULK_ENGINES_DIR

AUDIO_CPP_SOURCE_REVISION: Final = "4d88768fbcae4e6eb3352c6ab1422dabb7d90b58"
AUDIO_CPP_PACKAGE_VERSION: Final = "0.8.2.post1"
_MAX_WHEEL_BYTES: Final = 256 * 1024 * 1024
_MAX_EXPANDED_BYTES: Final = 512 * 1024 * 1024
_PACKAGE_PREFIX: Final = "skulk_audio_cpp_cpu/"
_BINARY_MEMBER: Final = _PACKAGE_PREFIX + "bin/audiocpp_server"
_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        _BINARY_MEMBER,
        _PACKAGE_PREFIX + "model_specs/ace_step.json",
        _PACKAGE_PREFIX + "model_specs/minimax_music3.json",
        _PACKAGE_PREFIX + "licenses/AUDIO_CPP_LICENSE.txt",
        *(
            _PACKAGE_PREFIX + "licenses/" + name
            for name in (
                "ABSL_LICENSE.txt",
                "CJSON_LICENSE.txt",
                "CPP_HTTPLIB_LICENSE.txt",
                "DARTS_CLONE_LICENSE.txt",
                "ESAXX_LICENSE.txt",
                "GGML_LICENSE.txt",
                "LIBYAML_LICENSE.txt",
                "PROTOBUF_LITE_LICENSE.txt",
                "SENTENCEPIECE_LICENSE.txt",
                "THIRD_PARTY_NOTICES.txt",
            )
        ),
    }
)
_INSTALL_LOCK = threading.Lock()
_MODEL_SPEC_SHA256: Final[dict[str, str]] = {
    "ace_step.json": "e71d84adebb00d607e4d6f97a0c6043e20721c1913f5716e1274e54079a146a9",
    "minimax_music3.json": "233f0b1e1f9601003739200b3451919925c65ecf2524f4160fa26346bb7c2fe5",
}


@final
class AudioCppWheel(BaseModel):
    """One immutable platform package from the Foxlight engine channel."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    filename: str = Field(pattern=r"^skulk_audio_cpp_cpu-0\.8\.2\.post1-.+\.whl$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def url(self) -> str:
        """Return the immutable wheel URL; the installer verifies its digest."""
        return "https://wheels.foxlight.ai/wheels/" + self.filename


# Built from audio.cpp 4d88768 by the attested three-platform workflow and
# published together at https://wheels.foxlight.ai/simple/skulk-audio-cpp-cpu/.
# Engine availability here grants no model support claim by itself.
AUDIO_CPP_CPU_WHEELS: Final[dict[tuple[str, str], AudioCppWheel]] = {
    ("darwin", "arm64"): AudioCppWheel(
        filename="skulk_audio_cpp_cpu-0.8.2.post1-py3-none-macosx_15_0_arm64.whl",
        sha256="6a5dc4118818c3c3e3be5259c1f189f114703b6ce68ade5a1cd11e8d89064baa",
    ),
    ("linux", "x86_64"): AudioCppWheel(
        filename="skulk_audio_cpp_cpu-0.8.2.post1-py3-none-manylinux_2_35_x86_64.whl",
        sha256="89a73bc202ed79f06ec85f26a127b62723179f637b32c2ebb5eef0093bc2f315",
    ),
    ("linux", "aarch64"): AudioCppWheel(
        filename="skulk_audio_cpp_cpu-0.8.2.post1-py3-none-manylinux_2_35_aarch64.whl",
        sha256="858b98f45cb7cfdc5b4e8f19f74b37f937dd9060706d5b8b4281770b92a3ce9e",
    ),
}


def audio_cpp_wheel_for_host(
    *, system: str | None = None, machine: str | None = None
) -> AudioCppWheel:
    """Return the exact CPU-capable wheel for a supported Skulk architecture."""
    target = (system or sys.platform, (machine or platform.machine()).lower())
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    if target[0] == "darwin":
        target = ("darwin", "arm64" if target[1] in {"arm64", "aarch64"} else target[1])
    else:
        target = (target[0], aliases.get(target[1], target[1]))
    wheel = AUDIO_CPP_CPU_WHEELS.get(target)
    if wheel is None:
        raise RuntimeError(
            "No SHA-256-pinned audio.cpp engine package has been published for "
            f"{target[0]}/{target[1]}; the music model cannot mount on this node"
        )
    return wheel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_root(wheel: AudioCppWheel) -> Path:
    return SKULK_ENGINES_DIR / "audio_cpp" / AUDIO_CPP_SOURCE_REVISION / wheel.sha256


def _cached_binary(root: Path, wheel: AudioCppWheel) -> Path | None:
    """Verify cached runtime files against the independently pinned wheel."""
    binary = root / _BINARY_MEMBER
    wheel_path = root / wheel.filename
    try:
        raw = cast("object", json.loads((root / "provisioned.json").read_text()))
        if not isinstance(raw, dict):
            return None
        record = cast("dict[str, object]", raw)
        if (
            record.get("source_revision") != AUDIO_CPP_SOURCE_REVISION
            or record.get("wheel_sha256") != wheel.sha256
            or not wheel_path.is_file()
            or wheel_path.stat().st_size > _MAX_WHEEL_BYTES
            or _sha256(wheel_path) != wheel.sha256
        ):
            return None
        if not binary.is_file() or not os.access(binary, os.X_OK):
            return None
        member_hashes = record.get("member_sha256")
        if not isinstance(member_hashes, dict):
            return None
        hashes = cast("dict[str, object]", member_hashes)
        with zipfile.ZipFile(wheel_path) as archive:
            members = {item.filename: item for item in archive.infolist()}
            if not members.keys() >= _REQUIRED_MEMBERS or sum(
                members[member].file_size for member in _REQUIRED_MEMBERS
            ) > _MAX_EXPANDED_BYTES:
                return None
            for member in _REQUIRED_MEMBERS:
                expected = hashes.get(member)
                path = root / member
                if not isinstance(expected, str) or not path.is_file():
                    return None
                archive_digest = hashlib.sha256()
                with archive.open(members[member]) as stream:
                    while chunk := stream.read(1024 * 1024):
                        archive_digest.update(chunk)
                if expected != archive_digest.hexdigest() or _sha256(path) != expected:
                    return None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return binary


def verified_cached_audio_cpp_binary(binary: Path) -> bool:
    """Return whether this executable belongs to this host's pinned wheel cache."""
    try:
        wheel = audio_cpp_wheel_for_host()
    except RuntimeError:
        return False
    root = _cache_root(wheel)
    return binary == root / _BINARY_MEMBER and _cached_binary(root, wheel) == binary


def audio_cpp_model_specs(
    binary: Path, *, environ: Mapping[str, str] | None = None
) -> Path:
    """Find the pinned specs for a wheel or standalone binary override."""
    env = os.environ if environ is None else environ
    configured = env.get(AUDIO_CPP_SPECS_DIR_ENV, "").strip()
    directory = Path(configured) if configured else binary.parent.parent / "model_specs"
    if not binary.is_absolute() or not directory.is_absolute():
        raise RuntimeError("audio.cpp binary and model specs paths must be absolute")
    for name, expected_digest in _MODEL_SPEC_SHA256.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected_digest:
            raise RuntimeError(
                f"audio.cpp model spec {name} is missing or differs from the pin "
                f"in {directory}; set {AUDIO_CPP_SPECS_DIR_ENV} to the v0.8.2 specs directory"
            )
    return directory


def rehydrate_cached_audio_cpp(
    *, environ: Mapping[str, str] | None = None
) -> Path | None:
    """Restore a previously verified package without network or a new download."""
    env = os.environ if environ is None else environ
    if env.get(AUDIO_CPP_BIN_ENV, "").strip():
        return None  # an explicit operator path remains authoritative
    try:
        wheel = audio_cpp_wheel_for_host()
        cached = _cached_binary(_cache_root(wheel), wheel)
        if cached is None:
            return None
    except (OSError, RuntimeError):
        return None
    os.environ[AUDIO_CPP_BIN_ENV] = str(cached)
    return cached


def _download_wheel(wheel: AudioCppWheel, destination: Path) -> None:
    """Stream a bounded wheel from the engine channel and verify SHA-256."""
    total = 0
    digest = hashlib.sha256()
    with (
        httpx.Client(timeout=httpx.Timeout(60.0, read=300.0)) as client,
        client.stream("GET", wheel.url, follow_redirects=False) as response,
        destination.open("wb") as stream,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes(1024 * 1024):
            total += len(chunk)
            if total > _MAX_WHEEL_BYTES:
                raise RuntimeError("audio.cpp engine package exceeds the download limit")
            digest.update(chunk)
            stream.write(chunk)
    if digest.hexdigest() != wheel.sha256:
        raise RuntimeError("audio.cpp engine package SHA-256 differs from the pin")


def _extract_wheel(wheel_path: Path, staging: Path) -> str:
    """Extract only the fixed runtime payload from a verified wheel."""
    with zipfile.ZipFile(wheel_path) as archive:
        members = {item.filename: item for item in archive.infolist()}
        if not members.keys() >= _REQUIRED_MEMBERS:
            raise RuntimeError("audio.cpp engine package lacks required runtime files")
        wanted = {
            name: item
            for name, item in members.items()
            if name.startswith(_PACKAGE_PREFIX)
            and name in _REQUIRED_MEMBERS
        }
        if sum(item.file_size for item in wanted.values()) > _MAX_EXPANDED_BYTES:
            raise RuntimeError("audio.cpp engine package expands beyond its limit")
        for name, item in wanted.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    binary = staging / _BINARY_MEMBER
    binary.chmod(0o755)
    return _sha256(binary)


def prepare_audio_cpp(
    *, allow_download: bool,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Wire an explicit binary or install one verified package for this host.

    This synchronous operation is called in a worker thread by the mount
    preparation handler. Offline mode may use an intact cached package but
    never contacts the package channel. A failed preparation does not change
    the node's advertised backends or health.
    """
    env = os.environ if environ is None else environ
    explicit = env.get(AUDIO_CPP_BIN_ENV, "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(
                f"{AUDIO_CPP_BIN_ENV} names no executable audio.cpp server: {explicit}"
            )
        wheel = audio_cpp_wheel_for_host()
        root = _cache_root(wheel)
        if path == root / _BINARY_MEMBER and _cached_binary(root, wheel) is None:
            raise RuntimeError("the cached audio.cpp package failed integrity verification")
        audio_cpp_model_specs(path, environ=env)
        return path
    with _INSTALL_LOCK:
        return _prepare_pinned_audio_cpp(allow_download=allow_download, env=env)


def _prepare_pinned_audio_cpp(*, allow_download: bool, env: Mapping[str, str]) -> Path:
    """Serialize cache verification and installation within one worker process."""

    wheel = audio_cpp_wheel_for_host()
    root = _cache_root(wheel)
    cached = _cached_binary(root, wheel)
    if cached is not None:
        os.environ[AUDIO_CPP_BIN_ENV] = str(cached)
        return cached
    if not allow_download or env.get(AUTOPROVISION_OPT_OUT_ENV) == "1":
        raise RuntimeError(
            "audio.cpp is not in the verified local cache; this node is offline "
            "or engine auto-provisioning is disabled"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audio-cpp-", dir=root.parent) as temporary:
        staging = Path(temporary)
        wheel_path = staging / wheel.filename
        _download_wheel(wheel, wheel_path)
        binary_sha256 = _extract_wheel(wheel_path, staging)
        # Offline revalidation needs the original pinned bytes. A mutable
        # provisioned.json beside the payload cannot establish integrity.
        (staging / "provisioned.json").write_text(
            json.dumps(
                {
                    "source_revision": AUDIO_CPP_SOURCE_REVISION,
                    "wheel_sha256": wheel.sha256,
                    "binary_sha256": binary_sha256,
                    "member_sha256": {
                        member: _sha256(staging / member)
                        for member in sorted(_REQUIRED_MEMBERS)
                    },
                }
            )
        )
        if root.exists():
            shutil.rmtree(root)
        staging.rename(root)
    cached = _cached_binary(root, wheel)
    if cached is None:
        raise RuntimeError("audio.cpp engine package failed post-install verification")
    os.environ[AUDIO_CPP_BIN_ENV] = str(cached)
    return cached
