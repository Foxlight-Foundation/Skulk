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
override wins, Linux only) plus one more: nothing is provisioned unless
video models are enabled on the node, because the torch wheel set is several
gigabytes and most nodes never render video.
"""

from __future__ import annotations

import json
import os
import platform as platform_module
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from loguru import logger

from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.provisioning.manifest import (
    COMFY_PIN,
    COMFY_PYTHON,
    COMFY_REPOSITORY,
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
# large transitive set; give the one-time install a generous budget.
_INSTALL_TIMEOUT_SECONDS: Final = 3600.0
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
    elif facts.gpus_of("amd"):
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


def managed_comfy_install(facts: NodeFacts) -> Path | None:
    """The complete managed install this node would use, if one is on disk."""
    for variant in select_comfy_variant_chain(facts):
        root = managed_comfy_root(variant)
        if _install_complete(root):
            return root
    return None


def _require_tool(name: str) -> str:
    path = shutil.which(name)
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
    torch_index = wheels[0].index
    target = managed_comfy_root(variant, machine)
    if _install_complete(target):
        return target
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


def _gates_pass(facts: NodeFacts) -> bool:
    if os.environ.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() == "1":
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


def ensure_comfy(facts: NodeFacts, *, allow_download: bool = True) -> Path | None:
    """Ensure a managed ComfyUI install for this node, honoring the gates.

    Exports ``SKULK_COMFY_BIN`` and ``SKULK_COMFY_ROOT`` for this process
    when a managed install is used. Returns the install root, or ``None``
    when nothing applies (override present, opted out, video models
    disabled, non-Linux or no wheel set for this hardware, nothing on disk
    while offline, or provisioning failed). Node startup and ``skulk doctor
    --fix`` both call this; the gates decide, not the caller.
    """
    if not _gates_pass(facts):
        return None
    existing = managed_comfy_install(facts)
    if existing is not None:
        return _export(existing)
    if not allow_download:
        return None
    last_error: Exception | None = None
    for variant in select_comfy_variant_chain(facts):
        try:
            return _export(provision_comfy(variant))
        except Exception as error:  # noqa: BLE001 - try the next variant, then degrade
            last_error = error
    if last_error is not None:
        # A node must start without network or with a failed install;
        # provisioning failure degrades to "no video engine", never a crash.
        logger.warning(
            f"ComfyUI provisioning unavailable ({last_error}); this node serves "
            "no video models until `skulk doctor --fix` succeeds or "
            f"{COMFY_BIN_ENV} and {COMFY_ROOT_ENV} point at an install"
        )
    return None
