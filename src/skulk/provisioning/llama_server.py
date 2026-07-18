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
import tarfile
import tempfile
from pathlib import Path

import httpx
from loguru import logger

from skulk.provisioning.manifest import (
    LLAMA_SERVER_ARTIFACTS,
    LLAMA_SERVER_PIN,
    EngineArtifact,
    EngineVariant,
)
from skulk.shared.backends import LLAMA_SERVER_BIN_ENV
from skulk.shared.constants import SKULK_ENGINES_DIR
from skulk.shared.types.node_facts import NodeFacts

AUTOPROVISION_OPT_OUT_ENV = "SKULK_NO_ENGINE_AUTOPROVISION"
"""Set to ``1`` to disable engine auto-provisioning on this node."""

_DOWNLOAD_TIMEOUT_SECONDS = 300.0


def select_variant(facts: NodeFacts) -> EngineVariant | None:
    """Pick the managed build variant for this node's observed hardware.

    A visible NVIDIA or AMD GPU selects the Vulkan build (RADV is the
    fleet-proven AMD path, and NVIDIA's driver ships a Vulkan ICD); no GPU
    selects the CPU build. Returns ``None`` off Linux: macOS serves through
    in-process MLX and provisions nothing.
    """
    if facts.platform != "linux":
        return None
    return "vulkan" if facts.has_serving_gpu else "cpu"


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
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target_dir.parent) as staging:
        archive = Path(staging) / artifact.asset_name
        _download(artifact, archive)
        _verify_sha256(archive, artifact.sha256)
        _safe_extract(archive, target_dir)
    binary = _binary_in(target_dir)
    if binary is None:
        raise RuntimeError(
            f"pinned archive {artifact.asset_name} contained no llama-server "
            "binary; report this as a skulk bug"
        )
    binary.chmod(binary.stat().st_mode | 0o111)
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


def ensure_llama_server(facts: NodeFacts) -> Path | None:
    """Ensure a usable llama-server for this node, honoring overrides.

    Called at node startup (before the first serving decision) and by
    ``skulk doctor --fix``. Exports ``SKULK_LLAMA_SERVER_BIN`` for this
    process when a managed binary is used, so the served runner and every
    downstream consumer see one consistent path.

    Args:
        facts: The current facts snapshot (decides variant and applicability).

    Returns:
        The managed binary path when provisioning happened or an existing
        managed install was wired, else ``None`` (override present, opted
        out, non-Linux, or provisioning failed).
    """
    if os.environ.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() == "1":
        return None
    if facts.llama_server_binary.state != "not_configured":
        # An explicit override (valid or not) wins; invalid ones stay loud
        # via the invalid_engine_binary conflict rather than being masked.
        return None
    variant = select_variant(facts)
    if variant is None:
        return None
    try:
        binary = provision_llama_server(variant)
    except Exception as error:  # noqa: BLE001 - a node must start without network
        logger.warning(
            f"engine provisioning unavailable ({error}); the node serves "
            "GGUF models only if another engine is configured"
        )
        return None
    os.environ[LLAMA_SERVER_BIN_ENV] = str(binary)
    return binary
