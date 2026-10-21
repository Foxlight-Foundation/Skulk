"""Verified Foxlight wheel overlays for the existing Comfy video engine.

An overlay never alters the base interpreter or checkout. Placement binds its
composite build, installation uses the normal InstallEngine task, and a runner
whitelists only the verified package directory. Unpublished candidates never
create installation offers.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Final, Literal, final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from skulk.provisioning.manifest import COMFY_PIN, COMFY_TORCH_WHEELS
from skulk.shared.constants import SKULK_ENGINES_DIR

if TYPE_CHECKING:
    from skulk.provisioning.comfy import Runner
    from skulk.shared.types.node_facts import NodeFacts

ComfyExtensionFeature = Literal["res2s", "retake", "extend", "dfr", "prompt_relay", "union_control", "motion_tracks", "inpaint"]
"""Explicit node features; publication does not imply model qualification."""
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_PACKAGE: Final = "skulk_comfy_ltx"
_EXTENSION_MARKER: Final = "/extension@sha256:"
_MAX_ARCHIVE_BYTES: Final = 4 * 1024 * 1024
_MAX_PAYLOAD_BYTES: Final = 16 * 1024 * 1024
LTX_REFERENCE_REVISION: Final = "e98f6d927d83c92ef2be6aec76c8440b19306e39"


@final
class ComfyExtensionWheel(BaseModel):
    """One immutable published overlay, selected without package resolution."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    filename: str = Field(pattern=r"^skulk_comfy_ltx-[0-9]+\.[0-9]+\.[0-9]+-py3-none-any\.whl$", description="Published Foxlight wheel filename.")
    sha256: Digest = Field(description="Digest of the complete published archive.")
    payload_sha256: Digest = Field(description="Digest of the canonical package file manifest.")
    size_bytes: int = Field(gt=0, le=_MAX_ARCHIVE_BYTES, description="Exact published archive size.")
    comfy_revision: str = Field(pattern=r"^[0-9a-f]{40}$", description="Exact base Comfy API revision supported by these issued bytes.")
    ltx_revision: str = Field(pattern=r"^[0-9a-f]{40}$", description="Immutable upstream algorithm provenance retained per issued wheel.")
    features: tuple[ComfyExtensionFeature, ...] = Field(min_length=1, max_length=8, description="Node features implemented by these exact bytes.")

    @property
    def url(self) -> str:
        """Return the owned immutable distribution URL; arbitrary hosts are forbidden."""
        return f"https://wheels.foxlight.ai/wheels/{self.filename}"

    def build_identity(self, base_build: str) -> str:
        """Bind the overlay payload to the base Comfy commit and torch identity."""
        if not valid_base_build(base_build) or not base_build.startswith(f"comfy@{self.comfy_revision}/torch@"):
            raise ValueError("extension requires an exact compatible Comfy/torch build")
        return f"{base_build}{_EXTENSION_MARKER}{self.payload_sha256}"


@final
class ComfyExtensionManifest(BaseModel):
    """Archive-authenticated package members and protocol provenance."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    protocol: Literal[1] = Field(description="Trusted overlay manifest protocol.")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$", description="Immutable wheel version.")
    comfy_revision: str = Field(pattern=r"^[0-9a-f]{40}$", description="Exact supported base Comfy source.")
    ltx_revision: str = Field(pattern=r"^[0-9a-f]{40}$", description="Reference pipeline provenance.")
    features: tuple[ComfyExtensionFeature, ...] = Field(min_length=1, max_length=8, description="Implemented node features.")
    payload_sha256: Digest = Field(description="Canonical member-manifest identity.")
    members: dict[str, Digest] = Field(min_length=1, max_length=128, description="Every installed package file except this manifest, by exact SHA-256.")

    @field_validator("members")
    @classmethod
    def validate_members(cls, members: dict[str, str]) -> dict[str, str]:
        """Reject escaping, ambiguous and cached-bytecode package member names."""
        for name in members:
            path = PurePosixPath(name)
            if path.is_absolute() or name != path.as_posix() or ".." in path.parts or "\\" in name or len(path.parts) < 2 or path.parts[0] != _PACKAGE or "__pycache__" in path.parts or path.suffix == ".pyc":
                raise ValueError("extension members must be canonical source-package paths")
        return members


@final
class _ComfyExtensionReceipt(BaseModel):
    """Archive-backed interpreter and base-build binding written after installation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    base_build: str = Field(description="Exact base Comfy and torch identity.")
    interpreter: Digest = Field(description="Selected interpreter path and executable-byte identity.")
    payload_sha256: Digest = Field(description="Verified overlay payload identity.")


@final
class ComfyExtensionProbeReply(BaseModel):
    """Bounded protocol and actual torch identity returned by the selected interpreter."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    torch: str = Field(min_length=1, max_length=128, description="Observed selected torch build.")
    manifest: ComfyExtensionManifest = Field(description="Manifest read by the actual isolated package import.")


# Descriptors are added only after attested promotion and owned-URL read-back.
# A locally built candidate is deliberately absent from normal node offers.
PUBLISHED_EXTENSIONS: Final[tuple[ComfyExtensionWheel, ...]] = (
    ComfyExtensionWheel(
        filename="skulk_comfy_ltx-0.1.5-py3-none-any.whl",
        sha256="c73705262534fa78258b6eff472f7c78acd87bd62ffbc7f5d680b4b93a5f6cc3",
        payload_sha256="58a546e3197521f8a3b3a90b516814df39bf13173a365f57c4c52c7d18eb073b",
        size_bytes=86391,
        comfy_revision="40c4fcdf513a4523e39d54a9d391908af8df8171",
        ltx_revision="e98f6d927d83c92ef2be6aec76c8440b19306e39",
        features=("res2s", "retake", "extend", "dfr", "prompt_relay", "union_control", "motion_tracks"),
    ),
    ComfyExtensionWheel(
        filename="skulk_comfy_ltx-0.1.4-py3-none-any.whl",
        sha256="bcafacf142ec1d782f192e5f7f0305e91203f589017db6f89d6424c17534f909",
        payload_sha256="9cb0a7990934537f50daaea9573aaf3f35860bf06e95486159236c5a8e35962c",
        size_bytes=65552,
        comfy_revision="40c4fcdf513a4523e39d54a9d391908af8df8171",
        ltx_revision="e98f6d927d83c92ef2be6aec76c8440b19306e39",
        features=("res2s", "retake", "extend", "dfr", "prompt_relay", "union_control"),
    ),
    ComfyExtensionWheel(
        filename="skulk_comfy_ltx-0.1.3-py3-none-any.whl",
        sha256="dade727b886fbcf3980f11c991ac35c70f1f5f98896c63ecf6a7cd06dac27b5e",
        payload_sha256="6b6a6b719952f292252fe78c1d02b851ad334cca8defc6dc2fd5878f49b4a7fd",
        size_bytes=63228,
        comfy_revision="40c4fcdf513a4523e39d54a9d391908af8df8171",
        ltx_revision="e98f6d927d83c92ef2be6aec76c8440b19306e39",
        features=("res2s", "retake", "extend", "dfr", "prompt_relay"),
    ),
    ComfyExtensionWheel(
        filename="skulk_comfy_ltx-0.1.2-py3-none-any.whl",
        sha256="6be37e297325d9ae7da2b185c0034f98e945745c6b9989f58507505f6ab7d842",
        payload_sha256="91edc941109ab83734c55a65c34eb47a6040ef502dd868457c41fa867b901184",
        size_bytes=51797,
        comfy_revision="40c4fcdf513a4523e39d54a9d391908af8df8171",
        ltx_revision="e98f6d927d83c92ef2be6aec76c8440b19306e39",
        features=("res2s", "retake", "extend", "dfr"),
    ),
    ComfyExtensionWheel(
        filename="skulk_comfy_ltx-0.1.1-py3-none-any.whl",
        sha256="0b6c28318b7f0aeaded710dea7862cfb1700cda3dc421354f62de65c9f78ea4a",
        payload_sha256="df3189f3232051717b249cec0087d728f0808463183f866ee32e0b4414cd3b16",
        size_bytes=48855,
        comfy_revision="40c4fcdf513a4523e39d54a9d391908af8df8171",
        ltx_revision="e98f6d927d83c92ef2be6aec76c8440b19306e39",
        features=("res2s", "retake", "extend"),
    ),
    ComfyExtensionWheel(
        filename="skulk_comfy_ltx-0.1.0-py3-none-any.whl",
        sha256="3ff6a292521de798de59d171474a8ec041e741d3fcd4f59b0a51aa4c7e8879e3",
        payload_sha256="345d060988431db5496b9916e329e856f13b3f55883c2dc5052b6fc85a35fedb",
        size_bytes=48584,
        comfy_revision="40c4fcdf513a4523e39d54a9d391908af8df8171",
        ltx_revision="e98f6d927d83c92ef2be6aec76c8440b19306e39",
        features=("res2s", "retake", "extend"),
    ),
)


def published_extensions() -> tuple[ComfyExtensionWheel, ...]:
    """Return issued immutable descriptors; candidates never enter this inventory."""
    return PUBLISHED_EXTENSIONS


def valid_base_build(build: str) -> bool:
    """Require an exact Comfy/torch identity; each wheel checks its own API pin."""
    return re.fullmatch(r"comfy@[0-9a-f]{40}/torch@[A-Za-z0-9.+_-]+", build) is not None


def extension_payload(build: str | None) -> str | None:
    """Read a well-formed overlay identity; malformed or baseline builds have none."""
    if build is None:
        return None
    base, marker, payload = build.partition(_EXTENSION_MARKER)
    return payload if marker and valid_base_build(base) and re.fullmatch(r"[0-9a-f]{64}", payload) else None


def extension_for_build(build: str | None) -> ComfyExtensionWheel | None:
    """Resolve only an issued descriptor, never an arbitrary caller-supplied wheel."""
    payload = extension_payload(build)
    return next((wheel for wheel in PUBLISHED_EXTENSIONS if wheel.payload_sha256 == payload and build is not None and build.startswith(f"comfy@{wheel.comfy_revision}/torch@")), None)


def extension_features_for_build(build: str | None) -> frozenset[ComfyExtensionFeature]:
    """Return capabilities of one exact published overlay; unknown builds add none."""
    wheel = extension_for_build(build)
    return frozenset(wheel.features) if wheel is not None else frozenset()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _interpreter_identity(interpreter: Path) -> str:
    resolved = interpreter.resolve(strict=True)
    return hashlib.sha256(f"{resolved}\0{_sha256(resolved)}".encode()).hexdigest()


def extension_root(wheel: ComfyExtensionWheel, base_build: str, interpreter: Path) -> Path:
    """Locate an overlay isolated by payload, base runtime and actual interpreter bytes."""
    identity = hashlib.sha256(f"{base_build}\0{_interpreter_identity(interpreter)}".encode()).hexdigest()
    return SKULK_ENGINES_DIR / "comfy_extensions" / wheel.payload_sha256 / identity


def _read_manifest(package: Path, wheel: ComfyExtensionWheel) -> ComfyExtensionManifest:
    path = package / "manifest.json"
    if path.is_symlink() or path.stat().st_size > 131072:
        raise ValueError("extension manifest exceeds its bound or is a symlink")
    manifest = ComfyExtensionManifest.model_validate_json(path.read_bytes())
    expected_version = wheel.filename.removeprefix("skulk_comfy_ltx-").removesuffix("-py3-none-any.whl")
    digest = hashlib.sha256(json.dumps(manifest.members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.comfy_revision != wheel.comfy_revision or manifest.ltx_revision != wheel.ltx_revision or manifest.version != expected_version or manifest.features != wheel.features or manifest.payload_sha256 != wheel.payload_sha256 or digest != wheel.payload_sha256:
        raise ValueError("extension manifest differs from its published descriptor")
    required = {f"{_PACKAGE}/{name}" for name in ("__init__.py", "nodes.py", "comfy_sampler.py", "res2s.py", "guidance.py", "layout.py", "launcher.py")}
    if {"retake", "extend"} & set(manifest.features):
        required.update(f"{_PACKAGE}/{name}" for name in ("editing.py", "editing_nodes.py"))
    if "dfr" in manifest.features:
        required.update(f"{_PACKAGE}/{name}" for name in ("dfr.py", "dfr_nodes.py"))
    if "prompt_relay" in manifest.features:
        required.update(f"{_PACKAGE}/{name}" for name in ("relay.py", "relay_encoding.py", "relay_geometry.py", "relay_binding.py", "relay_nodes.py"))
    if {"union_control", "motion_tracks"} & set(manifest.features):
        required.update(f"{_PACKAGE}/{name}" for name in ("union_control.py", "union_nodes.py"))
    if "motion_tracks" in manifest.features:
        required.update(f"{_PACKAGE}/{name}" for name in ("motion_geometry.py", "motion_tracks.py", "motion_nodes.py"))
    if "inpaint" in manifest.features:
        required.update(f"{_PACKAGE}/{name}" for name in ("editing.py", "editing_nodes.py", "inpaint_pixels.py", "inpaint_blend.py", "inpaint_audio.py", "inpaint_control.py", "inpaint_nodes.py"))
    if not required <= manifest.members.keys():
        raise ValueError("extension lacks its required implementation members")
    return manifest


def verify_extension_root(root: Path, wheel: ComfyExtensionWheel, base_build: str, interpreter: Path) -> Path | None:
    """Verify installed code against the issued archive and interpreter binding.

    Returns the sole allowlisted package directory, or None for missing,
    incomplete, modified or unexpectedly expanded installations. This is file
    integrity proof; the installer additionally performs its actual GPU probe.
    """
    try:
        archive = root / wheel.filename
        library = root / "lib"
        package = library / _PACKAGE
        receipt = root / "installed.json"
        if root.is_symlink() or library.is_symlink() or package.is_symlink() or archive.is_symlink() or receipt.is_symlink() or archive.stat().st_size != wheel.size_bytes or _sha256(archive) != wheel.sha256:
            return None
        manifest = _read_manifest(package, wheel)
        if receipt.stat().st_size > 16384:
            return None
        installed = _ComfyExtensionReceipt.model_validate_json(receipt.read_bytes())
        if installed.base_build != base_build or installed.interpreter != _interpreter_identity(interpreter) or installed.payload_sha256 != wheel.payload_sha256:
            return None
        files: set[str] = set()
        for path in library.rglob("*"):
            if path.is_symlink():
                return None
            if path.is_file():
                files.add(path.relative_to(library).as_posix())
        if files != {*manifest.members, f"{_PACKAGE}/manifest.json"}:
            return None
        if any(_sha256(library / name) != digest for name, digest in manifest.members.items()):
            return None
        return package
    except (OSError, ValueError):
        return None


def _extract_package(archive_path: Path, stage: Path, wheel: ComfyExtensionWheel) -> None:
    library = stage / "lib"
    library.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 160 or sum(member.file_size for member in members) > _MAX_PAYLOAD_BYTES:
            raise ValueError("extension archive exceeds its member or expanded-byte limit")
        if len({member.filename for member in members}) != len(members):
            raise ValueError("extension archive repeats a path")
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or member.filename != path.as_posix() or ".." in path.parts or "\\" in member.filename or (mode & 0o170000) not in (0, 0o100000):
                raise ValueError("extension archive contains a nonregular or escaping path")
            if not path.parts or path.parts[0] != _PACKAGE:
                if not path.parts or path.parts[0] != wheel.filename.removesuffix("-py3-none-any.whl") + ".dist-info":
                    raise ValueError("extension archive contains an unexpected namespace")
                continue
            destination = library / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=65536)
            destination.chmod(0o444)
    _read_manifest(library / _PACKAGE, wheel)


Download = Callable[[str, Path, int, threading.Event], None]
"""A bounded, cancellation-aware owned-URL download effect, injectable in tests."""
Probe = Callable[[Path, Path, str, str, threading.Event], None]
"""Verify the package protocol and actual selected GPU before advertising readiness."""


def install_extension_payload(
    wheel: ComfyExtensionWheel,
    base_build: str,
    interpreter: Path,
    backend: str,
    *,
    offline: bool,
    stop: threading.Event,
    download: Download,
    probe: Probe,
) -> Path:
    """Stage and verify a selected descriptor using injected download/probe effects.

    Production calls select an issued descriptor in install_comfy_extension;
    this effect boundary also permits integrity and cancellation tests without
    downloading a wheel or initializing a GPU. Returns the verified package.
    """
    if not valid_base_build(base_build) or not base_build.startswith(f"comfy@{wheel.comfy_revision}/torch@"):
        raise ValueError("overlay base build is incompatible with its pinned API")
    root = extension_root(wheel, base_build, interpreter)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Base Comfy uses its own install lock; this lock isolates an immutable
    # overlay and remains interruptible while another instance installs it.
    import fcntl

    with root.with_suffix(".lock").open("a+b") as lock:
        while True:
            if stop.is_set():
                raise InterruptedError("Comfy extension installation cancelled")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                _ = stop.wait(0.1)
        try:
            existing = verify_extension_root(root, wheel, base_build, interpreter)
            if existing is not None:
                probe(interpreter, existing, base_build, backend, stop)
                return existing
            if offline:
                raise ValueError("the exact Comfy extension is unavailable or modified offline")
            stage = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
            try:
                archive = stage / wheel.filename
                download(wheel.url, archive, wheel.size_bytes, stop)
                if stop.is_set():
                    raise InterruptedError("Comfy extension installation cancelled")
                if archive.stat().st_size != wheel.size_bytes or _sha256(archive) != wheel.sha256:
                    raise ValueError("downloaded Comfy extension differs from its published digest or size")
                _extract_package(archive, stage, wheel)
                record = {"base_build": base_build, "interpreter": _interpreter_identity(interpreter), "payload_sha256": wheel.payload_sha256}
                _ = (stage / "installed.json").write_text(json.dumps(record, sort_keys=True) + "\n")
                package = verify_extension_root(stage, wheel, base_build, interpreter)
                if package is None:
                    raise ValueError("staged Comfy extension failed archive-backed integrity")
                probe(interpreter, package, base_build, backend, stop)
                if stop.is_set():
                    raise InterruptedError("Comfy extension installation cancelled")
                if root.exists():
                    shutil.rmtree(root)
                os.replace(stage, root)
                installed = verify_extension_root(root, wheel, base_build, interpreter)
                if installed is None:
                    raise ValueError("installed Comfy extension lost its verified payload")
                return installed
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _download(url: str, destination: Path, expected_size: int, stop: threading.Event) -> None:
    import httpx

    if not url.startswith("https://wheels.foxlight.ai/wheels/"):
        raise ValueError("Comfy extension download must use the owned wheel channel")
    transferred = 0
    with httpx.stream("GET", url, follow_redirects=False, timeout=30.0) as response:
        response.raise_for_status()
        with destination.open("xb") as output:
            for chunk in response.iter_bytes(chunk_size=65536):
                if stop.is_set():
                    raise InterruptedError("Comfy extension download cancelled")
                transferred += len(chunk)
                if transferred > expected_size or transferred > _MAX_ARCHIVE_BYTES:
                    raise ValueError("Comfy extension download exceeds its published size")
                _ = output.write(chunk)
    if transferred != expected_size:
        raise ValueError("Comfy extension download is truncated")


_PROBE_SCRIPT: Final = """
import json, sys
sys.path.insert(0, sys.argv[1])
from skulk_comfy_ltx import manifest_json
manifest = json.loads(manifest_json())
import torch
if sys.version_info[:2] != (3, 13):
    raise RuntimeError('Comfy extension requires the selected CPython 3.13 runtime')
if not torch.cuda.is_available():
    raise RuntimeError('Comfy extension requires an actual GPU; CPU rendering is unavailable')
if bool(torch.version.hip) != (sys.argv[2] == 'comfy-rocm'):
    raise RuntimeError('selected torch GPU stack differs from its placement backend')
value = torch.arange(4, dtype=torch.float64, device='cuda').sum().item()
if value != 6:
    raise RuntimeError('Comfy extension float64 GPU probe failed')
print('_SKULK_COMFY_EXTENSION_' + json.dumps({'torch': torch.__version__, 'manifest': manifest}))
"""


def _probe_with_runner(interpreter: Path, package: Path, base_build: str, backend: str, stop: threading.Event, run: Runner, wheel: ComfyExtensionWheel) -> None:
    if stop.is_set():
        raise InterruptedError("Comfy extension probe cancelled")
    completed = run([str(interpreter), "-I", "-B", "-c", _PROBE_SCRIPT, str(package.parent), backend], capture_output=True, text=True, timeout=60.0, check=True)
    if completed.returncode != 0:
        raise ValueError("the Comfy extension GPU probe failed")
    replies = [line.removeprefix("_SKULK_COMFY_EXTENSION_") for line in completed.stdout.splitlines() if line.startswith("_SKULK_COMFY_EXTENSION_")]
    if len(replies) != 1 or len(replies[0]) > 131072:
        raise ValueError("the Comfy extension probe returned an invalid bounded reply")
    result = ComfyExtensionProbeReply.model_validate_json(replies[0])
    if result.torch != base_build.partition("/torch@")[2] or result.manifest != _read_manifest(package, wheel):
        raise ValueError("the Comfy extension probe differs from the placed torch build")
    # The archive-backed manifest is checked before executing this trusted
    # module. Parsing its reply does not grant publication or model support.
    if stop.is_set():
        raise InterruptedError("Comfy extension probe cancelled")


def install_comfy_extension(
    expected_build: str,
    interpreter: Path,
    backend: str,
    *,
    offline: bool,
    stop: threading.Event,
    run: Runner,
) -> Path:
    """Install only the exact issued overlay requested by normal model placement.

    Args:
        expected_build: Composite base/torch/overlay identity stamped by master.
        interpreter: Actual selected Comfy interpreter, never the API environment.
        backend: GPU lane used for the real torch device probe.
        offline: Permit only a complete archive-backed cached installation.
        stop: Shared installation cancellation signal.
        run: Bounded stoppable subprocess effect owned by the worker.

    Returns:
        Verified package folder to whitelist for this exact runner.

    Raises:
        ValueError: The requested build is unissued or fails integrity/probing.
        InterruptedError: Worker cancellation interrupts preparation.
    """
    wheel = extension_for_build(expected_build)
    if wheel is None or backend not in ("comfy-cuda", "comfy-rocm"):
        raise ValueError("the requested Comfy extension build or GPU backend is unavailable")
    base_build = expected_build.partition(_EXTENSION_MARKER)[0]

    def probe(selected: Path, package: Path, base: str, compute: str, stopping: threading.Event) -> None:
        _probe_with_runner(selected, package, base, compute, stopping, run, wheel)

    return install_extension_payload(wheel, base_build, interpreter, backend, offline=offline, stop=stop, download=_download, probe=probe)


def comfy_extension_inventory(
    facts: NodeFacts, *, offline: bool, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Return installable and integrity-verified installed overlays for this snapshot.

    An installed base keeps its backend advertised as installed while a new
    overlay is available. Offline and opted-out nodes can reuse a verified
    cache, but cannot promise a download. Explicit base overrides are measured
    and never replaced with a managed base to make an overlay fit.
    """
    published = published_extensions()
    if not published or facts.platform != "linux":
        return {}, {}
    from skulk.facts.derive import derive_node_backends
    from skulk.facts.inventory import comfy_base_build
    from skulk.provisioning.comfy import comfy_on_demand_variants
    from skulk.shared.constants import SKULK_ENABLE_VIDEO_MODELS

    env = os.environ if environ is None else environ
    if not SKULK_ENABLE_VIDEO_MODELS or env.get("SKULK_NODE_PARTICIPATION", "full").strip().lower() not in ("", "full"):
        return {}, {}
    can_download = not offline and env.get("SKULK_NO_AUTO_PROVISION", "").strip() != "1"
    bases: dict[str, str] = {}
    interpreter: Path | None = None
    if facts.comfy_binary.state == "ok" and facts.comfy_binary.configured_path is not None and facts.comfy_root_state == "ok":
        base = comfy_base_build(facts)
        if base is not None and valid_base_build(base):
            interpreter = Path(facts.comfy_binary.configured_path)
            bases = {backend: base for backend in derive_node_backends(facts).installed_backends if backend in ("comfy-cuda", "comfy-rocm")}
    else:
        for variant in comfy_on_demand_variants(facts, offline=offline, environ=env):
            wheels = COMFY_TORCH_WHEELS.get((platform.machine(), variant), ())
            torch_wheel = next((wheel for wheel in wheels if wheel.name == "torch"), None)
            if torch_wheel is not None:
                bases[f"comfy-{variant}"] = f"comfy@{COMFY_PIN}/torch@{torch_wheel.version}"
    offers: dict[str, str] = {}
    installed: dict[str, str] = {}
    for backend, base in bases.items():
        for wheel in published:
            if not base.startswith(f"comfy@{wheel.comfy_revision}/torch@"):
                continue
            build = wheel.build_identity(base)
            cached = interpreter is not None and verify_extension_root(extension_root(wheel, base, interpreter), wheel, base, interpreter) is not None
            if cached:
                installed.setdefault(backend, build)
            if can_download or cached:
                offers.setdefault(backend, build)
    return offers, installed


def prepare_comfy_extension(
    facts: NodeFacts,
    expected_build: str,
    backend: str,
    *,
    offline: bool,
    stop: threading.Event,
    run: Runner,
) -> Path:
    """Prepare an exact overlay through InstallEngine, retaining the chosen base.

    A missing base follows the existing Comfy installer. An explicit valid
    override is used in place; a broken or different override fails visibly.
    Only issued overlays may reach either installation path.
    """
    import subprocess

    import httpx

    from skulk.provisioning.comfy import (
        ComfyInstallError,
        comfy_checkout,
        comfy_interpreter,
        install_comfy_on_demand,
    )
    from skulk.utils.git_checkout import read_checkout_head

    wheel = extension_for_build(expected_build)
    if wheel is None:
        raise ComfyInstallError("the extension check", "requested Comfy extension has no published descriptor")
    if facts.comfy_binary.state == "ok" and facts.comfy_binary.configured_path is not None and facts.comfy_root_state == "ok" and facts.comfy_root is not None:
        interpreter, checkout = Path(facts.comfy_binary.configured_path), Path(facts.comfy_root)
    elif facts.comfy_binary.state != "not_configured" or facts.comfy_root is not None:
        raise ComfyInstallError("the extension check", "explicit Comfy override is incomplete; refusing replacement")
    else:
        if wheel.comfy_revision != COMFY_PIN:
            raise ComfyInstallError("the extension check", "the placed extension requires a compatible retained base; refusing to install a different base")
        root = install_comfy_on_demand(facts, offline=offline, run=run)
        interpreter, checkout = comfy_interpreter(root), comfy_checkout(root)
    if read_checkout_head(checkout) != wheel.comfy_revision:
        raise ComfyInstallError("the extension check", "selected Comfy checkout differs from the overlay API pin")
    try:
        return install_comfy_extension(expected_build, interpreter, backend, offline=offline, stop=stop, run=run)
    except InterruptedError:
        raise
    except (OSError, ValueError, httpx.HTTPError, subprocess.SubprocessError) as error:
        raise ComfyInstallError("the renderer update", str(error)) from error


def verified_comfy_extension(expected_build: str | None, interpreter: Path, checkout: Path) -> Path | None:
    """Return one placed overlay's verified package, refusing missing or modified bytes.

    Baseline placements return None and load no extension. An overlay placement
    must have completed InstallEngine before this read-only runner check.
    """
    if extension_payload(expected_build) is None:
        return None
    from skulk.utils.git_checkout import read_checkout_head

    wheel = extension_for_build(expected_build)
    if wheel is None or expected_build is None or read_checkout_head(checkout) != wheel.comfy_revision:
        raise ValueError("placed Comfy overlay is unissued or its base checkout changed")
    base = expected_build.partition(_EXTENSION_MARKER)[0]
    package = verify_extension_root(extension_root(wheel, base, interpreter), wheel, base, interpreter)
    if package is None:
        raise ValueError("placed Comfy overlay is missing or modified; prepare its exact engine first")
    return package
