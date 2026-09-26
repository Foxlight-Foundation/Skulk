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
from typing import Final, Literal, cast, final

import httpx
from pydantic import BaseModel, ConfigDict, Field

from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.shared.backends import (
    AUDIO_CPP_BIN_ENV,
    AUDIO_CPP_SPECS_DIR_ENV,
    AUDIO_CPP_VULKAN_BIN_ENV,
)
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

    filename: str = Field(pattern=r"^skulk_audio_cpp_(?:cpu|vulkan)-0\.8\.2\.post1-.+\.whl$")
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

# The Vulkan package is separate so preparing it cannot replace an in-use CPU
# executable. Its exact wheel pin is added after the CI artifact passes separate
# target-GPU qualification; no unpinned URL is a preparation candidate.
AUDIO_CPP_VULKAN_WHEELS: Final[dict[tuple[str, str], AudioCppWheel]] = {
    ("linux", "x86_64"): AudioCppWheel(
        filename="skulk_audio_cpp_vulkan-0.8.2.post1-py3-none-manylinux_2_35_x86_64.whl",
        sha256="72f0a600cff38d65640254e24c7c7b4271effbb49bf9e948ecdebcb2bb210930",
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


def audio_cpp_vulkan_wheel_for_host(
    *, system: str | None = None, machine: str | None = None
) -> AudioCppWheel:
    """Return the exact Linux Vulkan wheel for a supported host architecture."""
    target = (system or sys.platform, (machine or platform.machine()).lower())
    if target[0] == "linux" and target[1] == "amd64":
        target = ("linux", "x86_64")
    wheel = AUDIO_CPP_VULKAN_WHEELS.get(target)
    if wheel is None:
        raise RuntimeError(
            "No SHA-256-pinned audio.cpp Vulkan engine package has been published "
            f"for {target[0]}/{target[1]}"
        )
    return wheel


def _wheel_for_variant(variant: Literal["cpu", "vulkan"]) -> AudioCppWheel:
    """Resolve the immutable package for the requested compute variant."""
    return (
        audio_cpp_wheel_for_host()
        if variant == "cpu"
        else audio_cpp_vulkan_wheel_for_host()
    )


def _package_prefix(wheel: AudioCppWheel) -> str:
    """Return the fixed payload root selected by the pinned distribution name."""
    return (
        "skulk_audio_cpp_vulkan/"
        if wheel.filename.startswith("skulk_audio_cpp_vulkan-")
        else _PACKAGE_PREFIX
    )


def _required_members(wheel: AudioCppWheel) -> frozenset[str]:
    """Return the complete pinned runtime payload for one package variant."""
    prefix = _package_prefix(wheel)
    return frozenset(member.replace(_PACKAGE_PREFIX, prefix, 1) for member in _REQUIRED_MEMBERS)


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
    members_required = _required_members(wheel)
    binary = root / _package_prefix(wheel) / "bin/audiocpp_server"
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
            if not members.keys() >= members_required or sum(
                members[member].file_size for member in members_required
            ) > _MAX_EXPANDED_BYTES:
                return None
            for member in members_required:
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
    for variant in ("cpu", "vulkan"):
        try:
            wheel = _wheel_for_variant(variant)
        except RuntimeError:
            continue
        root = _cache_root(wheel)
        if (
            binary == root / _package_prefix(wheel) / "bin/audiocpp_server"
            and _cached_binary(root, wheel) == binary
        ):
            return True
    return False


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
    selected: Path | None = None
    variants: tuple[tuple[Literal["cpu", "vulkan"], str], ...] = (
        ("cpu", AUDIO_CPP_BIN_ENV),
        ("vulkan", AUDIO_CPP_VULKAN_BIN_ENV),
    )
    for variant, variable in variants:
        if env.get(variable, "").strip():
            continue  # an explicit operator path remains authoritative
        try:
            wheel = _wheel_for_variant(variant)
            cached = _cached_binary(_cache_root(wheel), wheel)
        except (OSError, RuntimeError):
            continue
        if cached is not None:
            os.environ[variable] = str(cached)
            if selected is None:
                selected = cached
    return selected


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
    wheel = AudioCppWheel(filename=wheel_path.name, sha256=_sha256(wheel_path))
    required = _required_members(wheel)
    prefix = _package_prefix(wheel)
    with zipfile.ZipFile(wheel_path) as archive:
        members = {item.filename: item for item in archive.infolist()}
        if not members.keys() >= required:
            raise RuntimeError("audio.cpp engine package lacks required runtime files")
        wanted = {
            name: item
            for name, item in members.items()
            if name.startswith(prefix)
            and name in required
        }
        if sum(item.file_size for item in wanted.values()) > _MAX_EXPANDED_BYTES:
            raise RuntimeError("audio.cpp engine package expands beyond its limit")
        for name, item in wanted.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    binary = staging / prefix / "bin/audiocpp_server"
    binary.chmod(0o755)
    return _sha256(binary)


def prepare_audio_cpp(
    *, allow_download: bool,
    variant: Literal["cpu", "vulkan"] = "cpu",
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Wire an explicit binary or install one verified package for this host.

    This synchronous operation is called in a worker thread by the mount
    preparation handler. Offline mode may use an intact cached package but
    never contacts the package channel. A failed preparation does not change
    the node's advertised backends or health.
    """
    env = os.environ if environ is None else environ
    variable = AUDIO_CPP_BIN_ENV if variant == "cpu" else AUDIO_CPP_VULKAN_BIN_ENV
    explicit = env.get(variable, "").strip()
    if variant == "vulkan" and not explicit:
        # Before the dedicated wheel, operators could supply a pinned Vulkan
        # build through the primary override. Preserve that route when its
        # executable and device probe still pass; managed CPU cache paths do
        # not take precedence over the dedicated Vulkan package.
        primary = env.get(AUDIO_CPP_BIN_ENV, "").strip()
        primary_path = Path(primary) if primary else None
        if (
            primary_path is not None
            and primary_path.is_file()
            and os.access(primary_path, os.X_OK)
            and not verified_cached_audio_cpp_binary(primary_path)
        ):
            try:
                audio_cpp_model_specs(primary_path, environ=env)
                from skulk.facts.probe import probe_audio_cpp

                if "vulkan" in probe_audio_cpp(str(primary_path)).computes:
                    return primary_path
            except RuntimeError:
                # An invalid primary override must not prevent the pinned
                # Vulkan package from being prepared for a valid claim.
                pass
    if explicit:
        path = Path(explicit)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(
                f"{variable} names no executable audio.cpp server: {explicit}"
            )
        wheel = _wheel_for_variant(variant)
        root = _cache_root(wheel)
        if path == root / _package_prefix(wheel) / "bin/audiocpp_server" and _cached_binary(root, wheel) is None:
            raise RuntimeError("the cached audio.cpp package failed integrity verification")
        audio_cpp_model_specs(path, environ=env)
        return path
    with _INSTALL_LOCK:
        return _prepare_pinned_audio_cpp(allow_download=allow_download, env=env, variant=variant)


def _prepare_pinned_audio_cpp(
    *, allow_download: bool, env: Mapping[str, str], variant: Literal["cpu", "vulkan"]
) -> Path:
    """Serialize cache verification and installation within one worker process."""

    wheel = _wheel_for_variant(variant)
    variable = AUDIO_CPP_BIN_ENV if variant == "cpu" else AUDIO_CPP_VULKAN_BIN_ENV
    root = _cache_root(wheel)
    cached = _cached_binary(root, wheel)
    if cached is not None:
        os.environ[variable] = str(cached)
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
                        for member in sorted(_required_members(wheel))
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
    os.environ[variable] = str(cached)
    return cached
