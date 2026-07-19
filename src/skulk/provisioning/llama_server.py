"""Managed llama-server provisioning: fetch, verify, and wire the pinned build.

The provisioning contract (#614 Phase 3):

- ``SKULK_LLAMA_SERVER_BIN`` always overrides: an operator's custom build wins
  and provisioning never runs. An *invalid* override is deliberately NOT
  papered over with a managed binary; it stays a loud
  ``invalid_engine_binary`` conflict, because silently substituting a
  different binary would mask the config error.
- Otherwise, on Linux, Skulk downloads the pinned upstream build for this
  machine's architecture and GPU shape into ``SKULK_ENGINES_DIR``, verifies
  its SHA-256 against the in-repo manifest, and exports
  ``SKULK_LLAMA_SERVER_BIN`` for this process (runner subprocesses inherit
  it). The facts probe then validates the managed binary exactly like any
  other: ``--list-devices`` ground truth, loud conflicts when the build
  cannot drive visible hardware.
- macOS provisions nothing (in-process MLX owns Apple GPUs).

Opt out with ``SKULK_NO_ENGINE_AUTOPROVISION=1`` (node-local launch policy).
"""

from __future__ import annotations

import hashlib
import os
import platform as platform_module
import sys
import tarfile
import tempfile
from importlib.util import find_spec
from pathlib import Path

import httpx
from loguru import logger

from skulk.provisioning.manifest import (
    LLAMA_SERVER_ARTIFACTS,
    LLAMA_SERVER_PIN,
    EngineArtifact,
    EngineVariant,
)
from skulk.shared.backends import LLAMA_SERVER_BIN_ENV, RPC_SERVER_BIN_ENV
from skulk.shared.constants import SKULK_ENGINES_DIR
from skulk.shared.types.node_facts import NodeFacts

AUTOPROVISION_OPT_OUT_ENV = "SKULK_NO_ENGINE_AUTOPROVISION"
"""Set to ``1`` to disable engine auto-provisioning on this node."""

_DOWNLOAD_TIMEOUT_SECONDS = 300.0


def select_variant_chain(facts: NodeFacts) -> tuple[EngineVariant, ...]:
    """Pick the managed build variants to try, most capable first.

    An NVIDIA GPU prefers the Foxlight-built CUDA build (container GPU clouds
    inject compute-only driver stacks where the Vulkan ICD cannot create an
    instance, so Vulkan-on-NVIDIA only works on bare metal with a full driver
    install), falling back to Vulkan where the CUDA artifact is unavailable
    for this architecture. An AMD GPU uses the Vulkan build (RADV is the
    fleet-proven path). No GPU means the CPU build. Empty off Linux: macOS
    serves through in-process MLX and provisions nothing.
    """
    if facts.platform != "linux":
        return ()
    if facts.gpus_of("nvidia"):
        return ("cuda", "vulkan")
    if facts.gpus_of("amd"):
        return ("vulkan",)
    return ("cpu",)


def select_variant(facts: NodeFacts) -> EngineVariant | None:
    """The preferred managed build variant for this node (first of the chain)."""
    chain = select_variant_chain(facts)
    return chain[0] if chain else None


# The CUDA wheel is compiled for SM 80/86/89/90 (+PTX from 90); a pre-Ampere
# GPU (T4 is 7.5, V100 is 7.0) still ENUMERATES under --list-devices, so
# selection must gate on compute capability or the failure surfaces only at
# model load. Kept in lockstep with CUDA_ARCHITECTURES in engine-wheel.yml.
CUDA_WHEEL_MIN_COMPUTE_CAPABILITY = (8, 0)

# The pip-installable engine wheels, per GPU vendor: (distribution name,
# importable module, server shim, RPC donor shim). NVIDIA tries the CUDA
# wheel then the Vulkan one (bare-metal NVIDIA drives Vulkan fine); AMD uses
# the Vulkan wheel.
_WHEELS_BY_VENDOR: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "nvidia": (
        (
            "skulk-llama-server-cuda",
            "skulk_llama_server_cuda",
            "llama-server-cuda",
            "ggml-rpc-server-cuda",
        ),
        (
            "skulk-llama-server-vulkan",
            "skulk_llama_server_vulkan",
            "llama-server-vulkan",
            "ggml-rpc-server-vulkan",
        ),
    ),
    "amd": (
        (
            "skulk-llama-server-vulkan",
            "skulk_llama_server_vulkan",
            "llama-server-vulkan",
            "ggml-rpc-server-vulkan",
        ),
    ),
}


def _wheel_version_matches_pin(distribution: str) -> bool:
    """Whether the installed engine wheel packages the pinned build.

    The wheel version scheme is ``0.<llama.cpp build>.<packaging rev>``; a
    wheel packaging a different build than :data:`LLAMA_SERVER_PIN` is
    ignored (with a log line) rather than silently overriding the validated
    pin, since ``ensure_llama_server`` prefers wheels over the
    checksum-verified tarball path.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version(distribution)
    except PackageNotFoundError:
        return False
    expected_prefix = f"0.{LLAMA_SERVER_PIN.removeprefix('b')}."
    if installed.startswith(expected_prefix):
        return True
    logger.warning(
        f"installed {distribution} {installed} does not package the pinned "
        f"llama.cpp build {LLAMA_SERVER_PIN}; ignoring the wheel (install "
        f"{distribution}=={expected_prefix}* to use it)"
    )
    return False


def _cuda_capability_ok(facts: NodeFacts) -> bool:
    """Whether an observed NVIDIA GPU meets the CUDA wheel's compiled SM floor.

    An unknown capability (NVML degraded) is treated as not-ok: preferring a
    wheel that may have no kernels for the silicon would fail at model load,
    while the Vulkan fallback fails loudly at probe time if it fails at all.
    """
    for gpu in facts.gpus_of("nvidia"):
        if gpu.compute_capability is None:
            continue
        try:
            major, minor = (int(part) for part in gpu.compute_capability.split("."))
        except ValueError:
            continue
        if (major, minor) >= CUDA_WHEEL_MIN_COMPUTE_CAPABILITY:
            return True
    return False


def wheel_llama_server(
    vendor: str, facts: NodeFacts
) -> tuple[Path, Path | None] | None:
    """The pip-installed engine wheel's shims for one GPU vendor, or ``None``.

    The engine wheels (packaging/skulk-llama-server-*) carry the
    Foxlight-built binaries behind console shims that wire the loader path
    and exec, so Skulk treats a shim exactly like any other llama-server
    binary (the facts probe validates it via ``--list-devices``). Installed
    via ordinary pip/uv, a wheel is the preferred managed source: standard
    tooling, ecosystem-verified hashes, works offline once installed.

    Args:
        vendor: ``"nvidia"`` or ``"amd"``.
        facts: The facts snapshot (gates the CUDA wheel on compute capability).

    Returns:
        ``(server_shim, rpc_shim_or_None)`` for the first installed wheel of
        the vendor's preference order whose version matches the pin.
    """
    bin_dir = Path(sys.executable).resolve().parent
    for distribution, module, server_shim, rpc_shim in _WHEELS_BY_VENDOR.get(
        vendor, ()
    ):
        if find_spec(module) is None:
            continue
        if distribution == "skulk-llama-server-cuda" and not _cuda_capability_ok(
            facts
        ):
            continue
        if not _wheel_version_matches_pin(distribution):
            continue
        server = bin_dir / server_shim
        if not server.is_file() or not os.access(server, os.X_OK):
            continue
        rpc = bin_dir / rpc_shim
        rpc_usable = rpc.is_file() and os.access(rpc, os.X_OK)
        return server, (rpc if rpc_usable else None)
    return None


def managed_llama_server_path() -> Path | None:
    """Return the already-provisioned pinned binary, or ``None``."""
    root = SKULK_ENGINES_DIR / "llama-server" / LLAMA_SERVER_PIN
    if not root.is_dir():
        return None
    for candidate in sorted(root.rglob("llama-server")):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _verify_sha256(path: Path, expected: str) -> None:
    """Raise when the downloaded archive does not match the pinned checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"checksum mismatch for {path.name}: expected {expected}, got "
            f"{actual}; refusing to install an unverified engine binary"
        )


def _download(artifact: EngineArtifact, destination: Path) -> None:
    """Stream one release artifact to disk."""
    with httpx.stream(
        "GET",
        artifact.url(),
        follow_redirects=True,
        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a verified archive, refusing path-traversal members."""
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")


def provision_llama_server(variant: EngineVariant) -> Path:
    """Download, verify, and install the pinned llama-server build.

    Args:
        variant: The compute variant to install.

    Returns:
        The path of the installed ``llama-server`` binary.

    Raises:
        RuntimeError: On unsupported architecture, checksum mismatch, or an
            archive missing the binary.
    """
    machine = platform_module.machine()
    artifact = LLAMA_SERVER_ARTIFACTS.get((machine, variant))
    if artifact is None:
        raise RuntimeError(
            f"no pinned llama-server artifact for machine={machine} "
            f"variant={variant}; set {LLAMA_SERVER_BIN_ENV} to a custom build"
        )
    target_dir = SKULK_ENGINES_DIR / "llama-server" / LLAMA_SERVER_PIN / variant
    existing = _binary_in(target_dir)
    if existing is not None:
        return existing

    logger.info(
        f"provisioning llama-server {LLAMA_SERVER_PIN} ({variant}) from "
        f"{artifact.url()}"
    )
    # Download, verify, and extract into a staging directory, then rename it
    # into place: the variant directory either exists complete or not at all,
    # so an interrupted install can never satisfy the fast path above with a
    # partial (unverified) tree on the next start.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target_dir.parent) as staging:
        archive = Path(staging) / artifact.asset_name
        _download(artifact, archive)
        _verify_sha256(archive, artifact.sha256)
        extracted = Path(staging) / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        if _binary_in(extracted) is None:
            raise RuntimeError(
                f"pinned archive {artifact.asset_name} contained no "
                "llama-server binary; report this as a skulk bug"
            )
        if not target_dir.exists():
            extracted.rename(target_dir)
    binary = _binary_in(target_dir)
    if binary is None:
        raise RuntimeError(
            f"pinned archive {artifact.asset_name} contained no llama-server "
            "binary; report this as a skulk bug"
        )
    binary.chmod(binary.stat().st_mode | 0o111)
    rpc_sibling = binary.parent / "ggml-rpc-server"
    if rpc_sibling.is_file():
        # Same umask/extraction guard as the server binary: a donor spawn
        # must not fail on a stripped exec bit after a successful provision.
        rpc_sibling.chmod(rpc_sibling.stat().st_mode | 0o111)
    logger.info(f"provisioned llama-server at {binary}")
    return binary


def _binary_in(target_dir: Path) -> Path | None:
    """Find an executable llama-server under one installed variant dir."""
    if not target_dir.is_dir():
        return None
    for candidate in sorted(target_dir.rglob("llama-server")):
        if candidate.is_file():
            return candidate
    return None


def ensure_llama_server(
    facts: NodeFacts, *, allow_download: bool = True
) -> Path | None:
    """Ensure a usable llama-server for this node, honoring overrides.

    Called at node startup (before the first serving decision) and by
    ``skulk doctor --fix``. Exports ``SKULK_LLAMA_SERVER_BIN`` for this
    process when a managed binary is used, so the served runner and every
    downstream consumer see one consistent path.

    Args:
        facts: The current facts snapshot (decides variant and applicability).
        allow_download: When ``False`` (an ``--offline`` node), never touch
            the network: an already-provisioned managed install still wires,
            so offline restarts keep their served GGUF capability, but a
            missing build is simply absent.

    Returns:
        The managed binary path when provisioning happened or an existing
        managed install was wired, else ``None`` (override present, opted
        out, non-Linux, nothing on disk while offline, or provisioning
        failed).
    """
    if os.environ.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() == "1":
        return None
    if facts.llama_server_binary.state != "not_configured":
        # An explicit override (valid or not) wins; invalid ones stay loud
        # via the invalid_engine_binary conflict rather than being masked.
        return None
    if not select_variant_chain(facts):
        return None
    vendor = (
        "nvidia"
        if facts.gpus_of("nvidia")
        else "amd"
        if facts.gpus_of("amd")
        else None
    )
    if vendor is not None:
        # A pip-installed engine wheel outranks tarball provisioning: it is
        # the standard-tooling path and already on disk (works offline too).
        wheel = wheel_llama_server(vendor, facts)
        if wheel is not None:
            server_shim, rpc_shim = wheel
            os.environ[LLAMA_SERVER_BIN_ENV] = str(server_shim)
            # The RPC donor binary is bundled in the wheel too, behind its
            # own loader-wiring shim; without this, rpc_server_binary()'s
            # sibling search next to the console script finds nothing and a
            # multi-node donor spawn fails (#615 review).
            rpc_override = os.environ.get(RPC_SERVER_BIN_ENV, "").strip()
            if not rpc_override and rpc_shim is not None:
                os.environ[RPC_SERVER_BIN_ENV] = str(rpc_shim)
            return server_shim
    if not allow_download:
        # Prefer the same variant order a download would use: with multiple
        # variants on disk (a cpu install from a pre-GPU run plus a later
        # vulkan/cuda one), an offline restart must not wire the weakest
        # build just because it sorts first.
        for variant in select_variant_chain(facts):
            existing = _binary_in(
                SKULK_ENGINES_DIR / "llama-server" / LLAMA_SERVER_PIN / variant
            )
            if existing is not None and os.access(existing, os.X_OK):
                os.environ[LLAMA_SERVER_BIN_ENV] = str(existing)
                return existing
        return None
    binary: Path | None = None
    last_error: Exception | None = None
    for variant in select_variant_chain(facts):
        try:
            binary = provision_llama_server(variant)
            break
        except Exception as error:  # noqa: BLE001 - try the next variant
            last_error = error
    if binary is None:
        if last_error is not None:
            # A node must start without network; provisioning failure
            # degrades to "no served engine", never a crash.
            logger.warning(
                f"engine provisioning unavailable ({last_error}); the node "
                "serves GGUF models only if another engine is configured"
            )
        return None
    os.environ[LLAMA_SERVER_BIN_ENV] = str(binary)
    return binary
