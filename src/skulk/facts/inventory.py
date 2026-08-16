"""Open engine-build and hardware-class inventory derived from node facts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import cast

from skulk.shared.backends import EngineType, engine_of
from skulk.shared.types.node_facts import EngineBinaryFact, NodeFacts

_ENGINE_DISTRIBUTIONS: dict[EngineType, tuple[str, str]] = {
    "mlx": ("mlx", "mlx"),
    "mlx_audio": ("mlx-audio", "mlx-audio"),
    "llama_cpp": ("llama-cpp-python", "llama-cpp-python"),
    "vllm": ("vllm", "vllm"),
}
_HARDWARE_TOKEN = re.compile(r"[^a-z0-9.]+")
_VERSION_TOKEN = re.compile(r"(?i)(?:vllm\s+)?(\d[0-9A-Za-z.+_-]*)")


@lru_cache(maxsize=None)
def _distribution_build(engine: EngineType) -> str | None:
    """Return the installed distribution identity for one known Python engine."""
    distribution = _ENGINE_DISTRIBUTIONS.get(engine)
    if distribution is None:
        return None
    package, label = distribution
    try:
        return f"{label}@{importlib.metadata.version(package)}"
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=8)
def _binary_digest(configured_path: str) -> str | None:
    """Hash one configured executable once for the lifetime of this process."""
    path = Path(configured_path)
    try:
        digest_builder = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
    except OSError:
        return None
    return digest


def _binary_build(label: str, binary: EngineBinaryFact) -> str | None:
    """Return a content-bound build identity for one configured executable."""
    if binary.state != "ok" or binary.configured_path is None:
        return None
    digest = _binary_digest(binary.configured_path)
    if digest is None:
        return None
    return f"{label}@sha256:{digest}"


@lru_cache(maxsize=8)
def _binary_version(configured_path: str) -> str | None:
    """Read one configured CLI's bounded version output once per process."""
    try:
        completed = subprocess.run(  # noqa: S603 - operator-configured engine path
            [configured_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip().splitlines()
    if not output:
        return None
    match = _VERSION_TOKEN.fullmatch(output[0].strip())
    return match.group(1) if match is not None else None


def _vllm_build(binary: EngineBinaryFact) -> str | None:
    """Return the configured vLLM environment's own distribution version."""
    if binary.state != "ok" or binary.configured_path is None:
        return None
    version = _binary_version(binary.configured_path)
    return f"vllm@{version}" if version is not None else None


def _declared_engine_builds(environ: Mapping[str, str]) -> dict[str, str]:
    """Parse optional open engine/tag build overrides from JSON configuration."""
    raw = environ.get("SKULK_ENGINE_BUILDS", "{}").strip() or "{}"
    parsed = cast("object", json.loads(raw))
    if not isinstance(parsed, dict):
        raise ValueError("SKULK_ENGINE_BUILDS must be a JSON object of strings")
    entries = cast("dict[object, object]", parsed)
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in entries.items()
    ):
        raise ValueError("SKULK_ENGINE_BUILDS must be a JSON object of strings")
    return {str(key).strip(): str(value).strip() for key, value in entries.items()}


def engine_build_inventory(
    backends: frozenset[str],
    facts: NodeFacts,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return exact build identities keyed by every advertised engine/backend tag.

    Python packages use their installed distribution version. Configured served
    binaries use their SHA-256 digest. ``SKULK_ENGINE_BUILDS`` can provide an
    exact identity for a new engine or override a known engine/tag when its
    canonical upstream build identifier is more useful than a local file hash.
    """
    declared = _declared_engine_builds(os.environ if environ is None else environ)
    discovered: dict[EngineType, str] = {}
    for backend in backends:
        engine = engine_of(backend)
        if engine is None or engine in discovered:
            continue
        build = _distribution_build(engine)
        if engine == "llama_server":
            build = _binary_build("llama.cpp", facts.llama_server_binary)
        elif engine == "vllm":
            # vLLM normally lives in its own managed virtual environment, so
            # the Skulk environment's package metadata is not its build truth.
            build = _vllm_build(facts.vllm_binary)
        if build is not None:
            discovered[engine] = build

    inventory: dict[str, str] = {}
    for backend in backends:
        engine = engine_of(backend)
        if engine is None:
            continue
        build = declared.get(backend) or declared.get(engine) or discovered.get(engine)
        if build is None:
            continue
        inventory[backend] = build
        inventory.setdefault(engine, build)
    return dict(sorted(inventory.items()))


def hardware_class_inventory(facts: NodeFacts) -> frozenset[str]:
    """Return open stable hardware identifiers derived from observed node facts."""
    classes = {f"platform:{facts.platform.lower()}"}
    for gpu in facts.gpus:
        vendor = gpu.vendor.lower()
        classes.add(vendor)
        name = _HARDWARE_TOKEN.sub("-", gpu.name.lower()).strip("-")
        if name and name != "unknown":
            classes.add(f"{vendor}:{name}")
        if vendor == "nvidia" and gpu.compute_capability:
            classes.add(f"nvidia:sm-{gpu.compute_capability}")
    return frozenset(classes)
