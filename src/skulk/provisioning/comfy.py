"""Managed ComfyUI provisioning for the ``comfy`` video engine.

ComfyUI ships as a git repository, not a wheel, and needs a torch build that
matches the node's GPU stack, so this is the first provisioner that creates
its own environment: a checkout at the pinned commit, a virtual environment
on the pinned interpreter, the pinned torch wheel set installed by hash, and
ComfyUI's own requirements resolved against constraints that keep that wheel
set in place. Everything lands under one directory keyed by pin and variant,
built in a staging directory and renamed into place, so a half-finished
install is never adopted.

The same gates as the llama-server provisioner apply (opt-out, explicit
override wins, Linux only). Because the torch wheel set is several gigabytes
and most nodes never render video, a node does not install ComfyUI when it
starts: it advertises that it can install the engine, and the worker installs
it the first time a video model is placed there (``install_comfy_on_demand``),
alongside the model download. Node startup only wires an install already on
disk.
"""

from __future__ import annotations

import importlib
import json
import os
import platform as platform_module
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from loguru import logger

from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.provisioning.manifest import (
    COMFY_INSTALL_FREE_BYTES,
    COMFY_PIN,
    COMFY_PYTHON,
    COMFY_REPOSITORY,
    COMFY_ROCM_PCI_DEVICE_IDS,
    COMFY_TORCH_WHEELS,
    EngineVariant,
    PinnedWheel,
    wheel_set_digest,
)
from skulk.shared.backends import COMFY_BIN_ENV, COMFY_ROOT_ENV
from skulk.shared.constants import SKULK_ENABLE_VIDEO_MODELS, SKULK_ENGINES_DIR
from skulk.shared.types.node_facts import NodeFacts

_CLONE_TIMEOUT_SECONDS: Final = 600.0
_VENV_TIMEOUT_SECONDS: Final = 300.0
# The torch wheel set is several gigabytes and ComfyUI's requirements pull a
# large transitive set. An on-demand install also shares the link with the
# video model's own download (tens of gigabytes), so the one-time install gets
# a budget that a slow connection can still meet.
_INSTALL_TIMEOUT_SECONDS: Final = 14400.0
_GPU_CHECK_TIMEOUT_SECONDS: Final = 300.0
# Older than any live install could be: two pip-install budgets.
_ABANDONED_STAGING_SECONDS: Final = 2 * _INSTALL_TIMEOUT_SECONDS
_PYPI_INDEX: Final = "https://pypi.org/simple/"
_DIGEST_KEY_LENGTH: Final = 12

CHECKOUT_DIRNAME: Final = "ComfyUI"
VENV_DIRNAME: Final = "venv"
RECORD_FILENAME: Final = "provisioned.json"
"""Written last; its presence is what marks an install complete."""

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
"""Shape of ``subprocess.run`` as the provisioner calls it (injectable for tests)."""


def sanitized_index_environment() -> dict[str, str]:
    """The process environment without any package-index overrides.

    A host-level mirror configured through uv or pip variables could serve a
    different torch build despite the explicit pins, so resolution must use
    exactly the indexes named on the command line (the same discipline as
    the CUDA engine wheel install).
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_INDEX", "UV_EXTRA_INDEX", "UV_DEFAULT_INDEX"))
        and key
        not in (
            "UV_FIND_LINKS",
            "UV_NO_INDEX",
            "UV_CONFIG_FILE",
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_FIND_LINKS",
            "PIP_NO_INDEX",
        )
    }


def select_comfy_variant_chain(facts: NodeFacts) -> tuple[EngineVariant, ...]:
    """Managed ComfyUI variants to try, most capable first.

    Only variants with a recorded wheel set for this machine count: an AMD
    node returns nothing until the ROCm wheel set is recorded, rather than
    provisioning an environment that cannot drive its GPU.
    """
    if facts.platform != "linux":
        return ()
    machine = platform_module.machine()
    wanted: tuple[EngineVariant, ...]
    if facts.gpus_of("nvidia"):
        wanted = ("cuda",)
    elif any(
        gpu.pci_device_id in COMFY_ROCM_PCI_DEVICE_IDS for gpu in facts.gpus_of("amd")
    ):
        # The ROCm wheel set is built for one GPU architecture; another AMD
        # GPU would install gigabytes it cannot use.
        wanted = ("rocm",)
    else:
        return ()
    return tuple(variant for variant in wanted if (machine, variant) in COMFY_TORCH_WHEELS)


def managed_comfy_root(variant: EngineVariant, machine: str | None = None) -> Path:
    """Directory holding one managed install (checkout plus environment).

    Keyed by the ComfyUI pin, the variant, and the wheel set's digest, so an
    install built on other torch wheels is never reused for this set.
    Returns ``None``-free paths only for recorded wheel sets; an unrecorded
    machine and variant pair keys under the variant alone, where nothing is
    ever provisioned.
    """
    wheels = COMFY_TORCH_WHEELS.get((machine or platform_module.machine(), variant))
    key = variant if wheels is None else f"{variant}-{wheel_set_digest(wheels)[:_DIGEST_KEY_LENGTH]}"
    return SKULK_ENGINES_DIR / "comfy" / COMFY_PIN / key


def comfy_interpreter(root: Path) -> Path:
    """The managed environment's interpreter under ``root``."""
    return root / VENV_DIRNAME / "bin" / "python"


def comfy_checkout(root: Path) -> Path:
    """The ComfyUI checkout under ``root``."""
    return root / CHECKOUT_DIRNAME


def _install_complete(root: Path) -> bool:
    interpreter = comfy_interpreter(root)
    return (
        (root / RECORD_FILENAME).is_file()
        and (comfy_checkout(root) / "main.py").is_file()
        and interpreter.is_file()
        and os.access(interpreter, os.X_OK)
    )


def _legacy_comfy_root(variant: EngineVariant) -> Path:
    """The pre-digest layout, ``<pin>/<variant>``, that earlier installs used."""
    return SKULK_ENGINES_DIR / "comfy" / COMFY_PIN / variant


def _record_matches(root: Path, wheels: Sequence[PinnedWheel]) -> bool:
    """Whether an install's record names exactly this wheel set by hash."""
    try:
        loaded = cast(object, json.loads((root / RECORD_FILENAME).read_text()))
    except (OSError, ValueError):
        return False
    if not isinstance(loaded, dict):
        return False
    record = cast(dict[str, object], loaded)
    recorded = record.get("wheels")
    if not isinstance(recorded, list):
        return False
    hashes: set[object] = set()
    for entry in cast(list[object], recorded):
        if isinstance(entry, dict):
            hashes.add(cast(dict[str, object], entry).get("sha256"))
    return hashes == {wheel.sha256 for wheel in wheels}


def _existing_install(variant: EngineVariant, machine: str) -> Path | None:
    """A complete install for this wheel set: the digest-keyed root, or a legacy
    root whose record proves it was built on exactly these wheels. An install
    made before roots carried the digest is reused rather than rebuilt, so an
    upgraded node (offline or not) keeps the engine it already has."""
    root = managed_comfy_root(variant, machine)
    if _install_complete(root):
        return root
    wheels = COMFY_TORCH_WHEELS.get((machine, variant))
    legacy = _legacy_comfy_root(variant)
    if wheels is not None and legacy != root and _install_complete(legacy) and _record_matches(legacy, wheels):
        return legacy
    return None


def managed_comfy_install(facts: NodeFacts) -> Path | None:
    """The complete managed install this node would use, if one is on disk."""
    machine = platform_module.machine()
    for variant in select_comfy_variant_chain(facts):
        existing = _existing_install(variant, machine)
        if existing is not None:
            return existing
    return None


def _uv_executable() -> str | None:
    """Locate uv: on PATH, from the ``uv`` package Skulk depends on, or beside Python.

    A packaged runtime has no uv on PATH, but its environment carries the
    ``uv`` distribution, whose ``find_uv_bin`` names the bundled binary.
    """
    found = shutil.which("uv")
    if found is not None:
        return found
    try:
        module = importlib.import_module("uv")
        find_uv_bin = cast(Callable[[], str], module.find_uv_bin)
        located = find_uv_bin()
    except (ImportError, AttributeError, FileNotFoundError):
        located = None
    if located is not None and os.access(located, os.X_OK):
        return located
    sibling = Path(sys.executable).parent / "uv"
    return str(sibling) if sibling.is_file() and os.access(sibling, os.X_OK) else None


def _tool_path(name: str) -> str | None:
    return _uv_executable() if name == "uv" else shutil.which(name)


def _require_tool(name: str) -> str:
    path = _tool_path(name)
    if path is None:
        raise RuntimeError(f"{name} is not on PATH; it is required to provision ComfyUI")
    return path


def _run(run: Runner, args: Sequence[str], *, timeout: float, what: str, env: dict[str, str] | None = None) -> str:
    try:
        completed = run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{what} timed out after {timeout:.0f}s") from error
    except OSError as error:
        raise RuntimeError(f"{what} could not start: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {completed.returncode}): {completed.stderr.strip()[-800:]}")
    return completed.stdout


def _write_requirements(root: Path, wheels: Sequence[PinnedWheel]) -> tuple[Path, Path]:
    """Write the hashed torch requirement file and the matching constraints."""
    requirements = root / "torch-requirements.txt"
    constraints = root / "torch-constraints.txt"
    requirements.write_text("".join(f"{wheel.requirement()}\n" for wheel in wheels))
    constraints.write_text("".join(f"{wheel.constraint()}\n" for wheel in wheels))
    return requirements, constraints


def provision_comfy(variant: EngineVariant, *, run: Runner = subprocess.run) -> Path:
    """Provision the pinned ComfyUI install for ``variant``; returns its root.

    Idempotent: a complete install returns immediately. Otherwise the checkout,
    environment, hashed torch wheels, and ComfyUI requirements are built in a
    staging directory beside the target and renamed into place, with the
    record file written last. Any failure leaves the target absent.
    """
    machine = platform_module.machine()
    wheels = COMFY_TORCH_WHEELS.get((machine, variant))
    if wheels is None:
        raise RuntimeError(f"no ComfyUI torch wheel set is recorded for {machine}/{variant}")
    # The resolver pass for ComfyUI's requirements sees the channel the wheel
    # set came from, so its constraints resolve there.
    torch_index = wheels[0].channel or wheels[0].index
    existing = _existing_install(variant, machine)
    if existing is not None:
        return existing
    target = managed_comfy_root(variant, machine)
    git = _require_tool("git")
    uv = _require_tool("uv")
    target.parent.mkdir(parents=True, exist_ok=True)
    env = sanitized_index_environment()
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}-") as staging:
        root = Path(staging) / target.name
        root.mkdir()
        checkout = comfy_checkout(root)
        _run(
            run,
            [git, "clone", "--quiet", "--no-checkout", "--filter=blob:none", COMFY_REPOSITORY, str(checkout)],
            timeout=_CLONE_TIMEOUT_SECONDS,
            what="ComfyUI clone",
        )
        _run(
            run,
            [git, "-C", str(checkout), "checkout", "--quiet", "--detach", COMFY_PIN],
            timeout=_CLONE_TIMEOUT_SECONDS,
            what="ComfyUI checkout at the pinned commit",
        )
        head = _run(run, [git, "-C", str(checkout), "rev-parse", "HEAD"], timeout=60, what="ComfyUI HEAD check").strip()
        if head != COMFY_PIN:
            raise RuntimeError(f"ComfyUI checkout is at {head}, not the pinned {COMFY_PIN}; refusing to install")
        if not (checkout / "main.py").is_file():
            raise RuntimeError("ComfyUI checkout has no main.py; refusing to install")
        venv = root / VENV_DIRNAME
        _run(
            run,
            [uv, "venv", "--quiet", "--python", COMFY_PYTHON, str(venv)],
            timeout=_VENV_TIMEOUT_SECONDS,
            what="ComfyUI virtual environment",
            env=env,
        )
        interpreter = comfy_interpreter(root)
        requirements, constraints = _write_requirements(root, wheels)
        # The torch wheel set goes in first, by exact URL and hash, with no
        # resolution: this is the supply-chain pin. Its own dependencies
        # arrive with ComfyUI's requirements below, constrained so the
        # resolver keeps exactly these torch builds.
        _run(
            run,
            [uv, "pip", "install", "--no-config", "--quiet", "--python", str(interpreter), "--no-deps", "--require-hashes", "-r", str(requirements)],
            timeout=_INSTALL_TIMEOUT_SECONDS,
            what="pinned torch wheel install",
            env=env,
        )
        _run(
            run,
            [
                uv, "pip", "install", "--no-config", "--quiet", "--python", str(interpreter),
                "--index-url", _PYPI_INDEX, "--extra-index-url", torch_index,
                "-c", str(constraints), "-r", str(checkout / "requirements.txt"),
            ],
            timeout=_INSTALL_TIMEOUT_SECONDS,
            what="ComfyUI requirements install",
            env=env,
        )
        freeze = _run(
            run,
            [uv, "pip", "freeze", "--no-config", "--python", str(interpreter)],
            timeout=_VENV_TIMEOUT_SECONDS,
            what="ComfyUI environment freeze",
            env=env,
        )
        (root / "freeze.txt").write_text(freeze)
        (root / RECORD_FILENAME).write_text(
            json.dumps(
                {
                    "pin": COMFY_PIN,
                    "variant": variant,
                    "machine": machine,
                    "python": COMFY_PYTHON,
                    "wheelSetDigest": wheel_set_digest(wheels),
                    "wheels": [{"filename": wheel.filename, "sha256": wheel.sha256} for wheel in wheels],
                },
                indent=2,
            )
        )
        try:
            root.rename(target)
        except OSError:
            # Two provisioners racing (node startup and doctor --fix): the
            # loser falls through to the winner's completed install.
            if not _install_complete(target):
                raise
    return target


def _video_models_enabled() -> bool:
    return SKULK_ENABLE_VIDEO_MODELS


def _gates_pass(facts: NodeFacts, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    if env.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() == "1":
        return False
    if facts.comfy_binary.state != "not_configured" or facts.comfy_root is not None:
        # An explicit override (valid, invalid, or half-set) wins; broken ones
        # stay loud through the invalid_engine_binary conflict rather than
        # being papered over by a managed install.
        return False
    if not _video_models_enabled():
        return False
    return bool(select_comfy_variant_chain(facts))


def dormant_comfy(facts: NodeFacts) -> Path | None:
    """The managed install startup would wire, without wiring it (doctor)."""
    if not _gates_pass(facts):
        return None
    return managed_comfy_install(facts)


def _export(root: Path) -> Path:
    os.environ[COMFY_BIN_ENV] = str(comfy_interpreter(root))
    os.environ[COMFY_ROOT_ENV] = str(comfy_checkout(root))
    return root


def ensure_comfy(facts: NodeFacts) -> Path | None:
    """Wire the managed ComfyUI install already on disk, honoring the gates.

    Exports ``SKULK_COMFY_BIN`` and ``SKULK_COMFY_ROOT`` for this process
    when a complete managed install exists. Returns its root, or ``None``
    when nothing applies (override present, opted out, video models
    disabled, non-Linux or no wheel set for this hardware, or nothing on
    disk). Never downloads: the engine is installed by
    ``install_comfy_on_demand`` when a video model is placed on the node or
    when ``skulk doctor --fix`` runs.
    """
    existing = dormant_comfy(facts)
    return None if existing is None else _export(existing)


def comfy_on_demand_variants(
    facts: NodeFacts,
    *,
    offline: bool,
    environ: Mapping[str, str] | None = None,
) -> tuple[EngineVariant, ...]:
    """Variants this node may install when a video model is placed on it.

    Empty unless every gate holds: the node fully participates, managed
    provisioning is not opted out, no explicit ``SKULK_COMFY_BIN`` or
    ``SKULK_COMFY_ROOT`` override is set, video models are enabled, the node
    is online, git and uv are available, and a recorded wheel set matches this
    hardware. A node advertises the video engine before installing it only
    when this is non-empty, so it never promises an install it cannot do.

    Args:
        facts: This node's facts.
        offline: Whether the node runs in offline mode.
        environ: Environment to read (the process environment by default).

    Returns:
        The installable variants, most capable first.
    """
    env = os.environ if environ is None else environ
    if env.get("SKULK_NODE_PARTICIPATION", "").strip().lower() not in ("", "full"):
        return ()
    if not _gates_pass(facts, environ=env):
        return ()
    if offline or _tool_path("git") is None or _tool_path("uv") is None:
        return ()
    return select_comfy_variant_chain(facts)


class ComfyInstallError(RuntimeError):
    """An on-demand ComfyUI install failed at a named step.

    ``step`` is a fixed phrase safe to show an operator; ``detail`` may carry
    installer output and belongs in logs only.
    """

    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


_INSTALL_LOCK: Final = threading.Lock()


def _remove_abandoned_staging(variants: Sequence[EngineVariant]) -> None:
    """Delete staging trees an interrupted install left beside its target.

    A node restarted mid-install leaves a staging tree of several gigabytes.
    Only trees older than any live install could be are removed, so a
    concurrent install in another process (``skulk doctor --fix``) keeps its own.
    """
    machine = platform_module.machine()
    cutoff = time.time() - _ABANDONED_STAGING_SECONDS
    for variant in variants:
        target = managed_comfy_root(variant, machine)
        if not target.parent.is_dir():
            continue
        for staging in target.parent.glob(f".{target.name}-*"):
            try:
                abandoned = staging.is_dir() and staging.stat().st_mtime < cutoff
            except OSError:
                continue
            if abandoned:
                logger.info(f"Removing an interrupted video engine install at {staging}")
                shutil.rmtree(staging, ignore_errors=True)


def _verify_gpu(root: Path, run: Runner) -> None:
    """Fail an install whose torch cannot see this node's GPU (an old driver)."""
    probe = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"
    _run(
        run,
        [str(comfy_interpreter(root)), "-c", probe],
        timeout=_GPU_CHECK_TIMEOUT_SECONDS,
        what="ComfyUI GPU check",
    )


def install_comfy_on_demand(
    facts: NodeFacts, *, offline: bool, run: Runner = subprocess.run
) -> Path:
    """Install the managed ComfyUI engine because a video model was placed here.

    Serialized per process, so concurrent placements share one install. An
    install already on disk is wired without downloading. Otherwise the node's
    eligibility is checked again, free disk space is required, each eligible
    variant is provisioned in turn, and the new environment must see the GPU
    before ``SKULK_COMFY_BIN`` and ``SKULK_COMFY_ROOT`` are exported; an
    install that fails the GPU check is removed. ``skulk doctor --fix`` uses
    the same path.

    Args:
        facts: This node's current facts.
        offline: Whether the node runs in offline mode.
        run: Subprocess runner (injectable for tests).

    Returns:
        The install root.

    Raises:
        ComfyInstallError: The install could not complete; ``step`` names where.
    """
    with _INSTALL_LOCK:
        existing = managed_comfy_install(facts)
        if existing is not None:
            return _export(existing)
        if offline:
            raise ComfyInstallError("the download", "this node runs offline")
        variants = comfy_on_demand_variants(facts, offline=offline)
        if not variants:
            raise ComfyInstallError(
                "the eligibility check", "this node cannot install the video engine"
            )
        SKULK_ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        _remove_abandoned_staging(variants)
        free = shutil.disk_usage(SKULK_ENGINES_DIR).free
        if free < COMFY_INSTALL_FREE_BYTES:
            raise ComfyInstallError(
                "the disk-space check",
                f"{free / 1024**3:.1f} GB free, "
                f"{COMFY_INSTALL_FREE_BYTES / 1024**3:.0f} GB needed",
            )
        last_error: Exception | None = None
        for variant in variants:
            logger.info(
                f"Installing the video engine (ComfyUI {COMFY_PIN[:8]}, {variant}) "
                "because a video model was placed on this node"
            )
            try:
                root = provision_comfy(variant, run=run)
            except Exception as error:  # noqa: BLE001 - try the next variant
                logger.warning(f"ComfyUI {variant} install failed: {error}")
                last_error = error
                continue
            try:
                _verify_gpu(root, run)
            except RuntimeError as error:
                # Startup wires any complete install on disk without checking
                # it again, so an engine that cannot see the GPU must not stay.
                shutil.rmtree(root, ignore_errors=True)
                raise ComfyInstallError(
                    "the GPU check",
                    "the installed engine cannot use this node's GPU",
                ) from error
            logger.info(f"Video engine installed at {root}")
            return _export(root)
        raise ComfyInstallError(
            "the engine download and install",
            str(last_error) if last_error is not None else "no variant installed",
        ) from last_error


def installed_managed_comfy(resolved_backend: str | None) -> tuple[Path, Path] | None:
    """The interpreter and checkout of a managed install already on disk.

    The comfy runner process starts before an on-demand install finishes, so
    it cannot rely on the worker's exported environment and looks the install
    up on disk instead. ``comfy-cuda`` and ``comfy-rocm`` name their variant;
    any other value tries both.

    Args:
        resolved_backend: The shard's stamped backend tag, if any.

    Returns:
        ``(interpreter, checkout)`` of a complete install, or ``None``.
    """
    machine = platform_module.machine()
    variants: tuple[EngineVariant, ...]
    if resolved_backend == "comfy-cuda":
        variants = ("cuda",)
    elif resolved_backend == "comfy-rocm":
        variants = ("rocm",)
    else:
        variants = ("cuda", "rocm")
    for variant in variants:
        root = _existing_install(variant, machine)
        if root is not None:
            return comfy_interpreter(root), comfy_checkout(root)
    return None
