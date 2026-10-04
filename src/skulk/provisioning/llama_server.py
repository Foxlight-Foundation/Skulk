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
import html
import importlib
import os
import platform as platform_module
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from importlib.metadata import distributions
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import unquote, urljoin

import httpx
from loguru import logger
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

from skulk.provisioning.manifest import (
    LLAMA_SERVER_ARTIFACTS,
    LLAMA_SERVER_CUDA_MIN_REVISION,
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

# The CUDA engine wheel is large (hundreds of MB) and pip resolution adds
# overhead on slow pipes; give its one-time install a generous budget before
# degrading to the Vulkan/tarball chain.
_WHEEL_INSTALL_TIMEOUT_SECONDS = 900.0

# The first native arm64 CUDA wheel is deliberately GB10-specific: unlike the
# x86_64 wheel, its CI build carries only an sm_121 real-code image. Do not let
# package-tag compatibility masquerade as GPU-kernel compatibility on GH200 or
# another arm64 NVIDIA system; those nodes retain the verified Vulkan fallback.
_ARM64_CUDA_WHEEL_COMPUTE_CAPABILITIES = frozenset({(12, 1)})

# Mirrors install.sh's ENGINE_INDEX_FLAGS: the Foxlight PEP 503 index is the
# source of truth for engine wheels (the CUDA wheel exceeds PyPI's size
# limit), with the default index pinned explicitly so a host exporting
# UV_INDEX_URL cannot redirect resolution to its own mirror.
FOXLIGHT_WHEEL_INDEX = "https://wheels.foxlight.ai/simple/"
_PYPI_INDEX = "https://pypi.org/simple/"


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


def _wheel_version_matches_pin(distribution: str, *, quiet: bool = False) -> bool:
    """Whether the installed engine wheel packages the pinned build.

    The wheel version scheme is ``0.<llama.cpp build>.<packaging rev>``; a
    wheel packaging a different build than :data:`LLAMA_SERVER_PIN` is
    ignored (with a log line) rather than silently overriding the validated
    pin, since ``ensure_llama_server`` prefers wheels over the
    checksum-verified tarball path.

    Args:
        distribution: The engine wheel's distribution name.
        quiet: Suppress the mismatch warning. The install-decision probe
            (:func:`_cuda_wheel_usable`) runs alongside the resolver's own
            check, which already warns; a second identical line per startup
            is noise (PR #665 review).
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version(distribution)
    except PackageNotFoundError:
        return False
    constraint = _pin_constraint(distribution)
    if _version_satisfies_pin(installed, distribution):
        return True
    if not quiet:
        logger.warning(
            f"installed {distribution} {installed} does not package the pinned "
            f"llama.cpp build {LLAMA_SERVER_PIN} at a supported packaging revision; "
            f"ignoring the wheel (install {distribution}{constraint} to use it)"
        )
    return False


def _pin_constraint(distribution: str) -> str:
    """The version range an engine wheel must fall in to package the pin.

    The wheel version scheme is ``0.<llama.cpp build>.<packaging rev>``; the
    CUDA wheel also needs at least :data:`LLAMA_SERVER_CUDA_MIN_REVISION`.
    """
    expected_prefix = f"0.{LLAMA_SERVER_PIN.removeprefix('b')}."
    minimum_revision = (
        LLAMA_SERVER_CUDA_MIN_REVISION
        if distribution == "skulk-llama-server-cuda"
        else 0
    )
    return f"=={expected_prefix}*,>={expected_prefix}{minimum_revision}"


def _version_satisfies_pin(installed: str, distribution: str) -> bool:
    """Whether an engine wheel version packages the pinned llama.cpp build."""
    try:
        version = Version(installed)
    except InvalidVersion:
        return False
    # Explicit because packaging releases disagree on whether a plain range
    # admits pre-releases, and no engine pre-release is ever qualified.
    return SpecifierSet(_pin_constraint(distribution)).contains(
        version, prereleases=False
    )


def _cuda_capability_ok(facts: NodeFacts) -> bool:
    """Whether an observed NVIDIA GPU matches this host wheel's compiled SMs.

    An unknown capability (NVML degraded) is treated as not-ok: preferring a
    wheel that may have no kernels for the silicon would fail at model load,
    while the Vulkan fallback fails loudly at probe time if it fails at all.
    The x86_64 wheel retains its established Ampere-or-newer policy. The first
    aarch64 wheel is narrower and accepts only the GB10 ``sm_121`` target it
    actually carries.
    """
    machine = platform_module.machine().lower()
    for gpu in facts.gpus_of("nvidia"):
        if gpu.compute_capability is None:
            continue
        try:
            major, minor = (int(part) for part in gpu.compute_capability.split("."))
        except ValueError:
            continue
        capability = (major, minor)
        if machine in {"aarch64", "arm64"}:
            if capability in _ARM64_CUDA_WHEEL_COMPUTE_CAPABILITIES:
                return True
            continue
        if machine in {"x86_64", "amd64"} and (
            capability >= CUDA_WHEEL_MIN_COMPUTE_CAPABILITY
        ):
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
    # sysconfig's scripts path is the venv's own bin directory; resolving
    # sys.executable would follow the venv symlink to uv's cached base
    # interpreter, whose bin dir does not contain the wheel shims.
    bin_dir = Path(sysconfig.get_path("scripts"))
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


def _cuda_wheel_usable() -> bool:
    """Whether the pinned CUDA engine wheel is installed and pin-matched.

    Distinct from :func:`wheel_llama_server` returning something: on an
    NVIDIA node with only the VULKAN wheel installed (install.sh fell back,
    or a bare-metal environment reused in a compute-only container), the
    resolver happily returns the Vulkan shim, but that shim cannot drive the
    GPU where no Vulkan ICD exists. The on-demand CUDA install must key on
    the CUDA wheel itself, not on any-wheel-resolved (PR #665 review).
    """
    return find_spec("skulk_llama_server_cuda") is not None and (
        _wheel_version_matches_pin("skulk-llama-server-cuda", quiet=True)
    )


_CUDA_DISTRIBUTION = "skulk-llama-server-cuda"

#: Shell launchers written next to a user-installed CUDA engine. They put the
#: NVIDIA runtime libraries on llama-server's loader path and exec it, the job
#: the wheel's console shim does inside a writable environment.
_USER_CUDA_LAUNCHERS = {
    "llama-server-cuda": "llama-server",
    "ggml-rpc-server-cuda": "ggml-rpc-server",
}


def _user_cuda_engine_dir() -> Path:
    """Where this user's copy of the pinned CUDA engine wheel is installed.

    An operating-system package installs Skulk read-only and ships no ``uv``,
    so the CUDA engine cannot join that environment; it is installed for the
    running user under the engines directory instead, one directory per pin.
    """
    return SKULK_ENGINES_DIR / "wheels" / _CUDA_DISTRIBUTION / LLAMA_SERVER_PIN


def _installed_version_in(directory: Path, distribution: str) -> str | None:
    """The version of ``distribution`` installed under ``directory``, if any.

    A damaged install reads as absent, so startup reinstalls it instead of
    failing: an empty METADATA (a power loss before the install reached the
    disk) has no fields, and undecodable bytes raise ``UnicodeDecodeError``.
    """
    if not directory.is_dir():
        return None
    try:
        for installed in distributions(path=[str(directory)]):
            metadata = installed.metadata
            if "Name" not in metadata or "Version" not in metadata:
                continue
            if canonicalize_name(metadata["Name"]) == distribution:
                return metadata["Version"]
    except (OSError, ValueError):
        return None
    return None


def user_cuda_llama_server(facts: NodeFacts) -> tuple[Path, Path | None] | None:
    """The CUDA engine installed for this user, as ``(server, rpc_or_None)``.

    The same gates as an environment wheel: the GPU must match the wheel's
    compiled kernels and the installed version must package the pin.

    Args:
        facts: The facts snapshot (gates on the CUDA wheel's SM floor).

    Returns:
        The launcher paths, or ``None`` when no usable user install exists.
    """
    if not _cuda_capability_ok(facts):
        return None
    directory = _user_cuda_engine_dir()
    server = directory / "launchers" / "llama-server-cuda"
    if not server.is_file() or not os.access(server, os.X_OK):
        return None
    version = _installed_version_in(directory, _CUDA_DISTRIBUTION)
    if version is None or not _version_satisfies_pin(version, _CUDA_DISTRIBUTION):
        return None
    rpc = directory / "launchers" / "ggml-rpc-server-cuda"
    rpc_usable = rpc.is_file() and os.access(rpc, os.X_OK)
    return server, (rpc if rpc_usable else None)


def _environment_writable() -> bool:
    """Whether engine wheels can be installed into the running environment."""
    return os.access(sysconfig.get_path("purelib"), os.W_OK)


def _foxlight_cuda_wheel() -> tuple[str, str] | None:
    """The newest pin-matched CUDA engine wheel for this platform on the index.

    Returns:
        ``(url, sha256)`` from the Foxlight PEP 503 page, or ``None`` when the
        index is unreachable or lists no compatible wheel.
    """
    page = urljoin(FOXLIGHT_WHEEL_INDEX, f"{_CUDA_DISTRIBUTION}/")
    try:
        response = httpx.get(
            page, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        logger.warning(f"could not read the CUDA engine index at {page}: {error}")
        return None
    supported = {str(tag) for tag in sys_tags()}
    best: tuple[Version, str, str] | None = None
    hrefs: list[str] = re.findall(r'href="([^"]+)"', response.text)
    for href in hrefs:
        url, _, fragment = urljoin(page, html.unescape(href)).partition("#")
        if not fragment.startswith("sha256="):
            continue
        try:
            name, version, _, tags = parse_wheel_filename(
                unquote(url.rsplit("/", 1)[-1])
            )
        except InvalidWheelFilename:
            continue
        if name != _CUDA_DISTRIBUTION or not _version_satisfies_pin(
            str(version), _CUDA_DISTRIBUTION
        ):
            continue
        if not any(str(tag) in supported for tag in tags):
            continue
        if best is None or version > best[0]:
            best = (version, url, fragment.removeprefix("sha256="))
    if best is None:
        logger.warning(
            f"the CUDA engine index lists no {_CUDA_DISTRIBUTION}"
            f"{_pin_constraint(_CUDA_DISTRIBUTION)} wheel for this platform"
        )
        return None
    return best[1], best[2]


def _write_cuda_launchers(tree: Path) -> None:
    """Write the shell launchers for a CUDA engine installed under ``tree``.

    Each launcher resolves its own directory at run time, so the install can
    be moved into place after it is written.
    """
    library_dirs = sorted(
        str(lib.relative_to(tree)) for lib in tree.glob("nvidia/*/lib") if lib.is_dir()
    )
    library_dirs.append("skulk_llama_server_cuda/bin")
    loader_path = ":".join(f"$root/{relative}" for relative in library_dirs)
    launchers = tree / "launchers"
    launchers.mkdir()
    for launcher, binary in _USER_CUDA_LAUNCHERS.items():
        target = tree / "skulk_llama_server_cuda" / "bin" / binary
        if not target.is_file():
            continue
        script = launchers / launcher
        script.write_text(
            "#!/bin/sh\n"
            "# Written by Skulk: runs the CUDA engine installed for this user with\n"
            "# NVIDIA's runtime libraries on the loader path.\n"
            'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
            f'LD_LIBRARY_PATH="{loader_path}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n'
            "export LD_LIBRARY_PATH\n"
            f'exec "$root/skulk_llama_server_cuda/bin/{binary}" "$@"\n'
        )
        script.chmod(0o755)


def _pip_available() -> bool:
    """Whether this interpreter can run ``python -m pip``."""
    return find_spec("pip") is not None


def _install_cuda_wheel_for_user(facts: NodeFacts) -> bool:
    """Install the pinned CUDA engine into this user's engines directory.

    For runtimes the CUDA wheel cannot join, such as an operating-system
    package that installs Skulk read-only without ``uv``. The engine wheel is
    taken by exact URL from the Foxlight index and checked against the
    index's SHA-256, the same trust the ``uv`` path places in that index.
    This interpreter's ``pip`` then installs it into a private directory,
    resolving NVIDIA's runtime wheels from PyPI only. Every failure degrades
    to the Vulkan/tarball chain; node startup never fails here.

    Args:
        facts: The facts snapshot (gates on the CUDA wheel's SM floor).

    Returns:
        ``True`` when a usable CUDA engine is installed for this user.
    """
    link = _foxlight_cuda_wheel()
    if link is None:
        return False
    url, sha256 = link
    destination = _user_cuda_engine_dir()
    logger.info(
        f"installing the CUDA engine for this user from {url} into "
        f"{destination}; multi-GB, one-time"
    )
    try:
        failure = _install_cuda_engine_tree(url, sha256, destination)
    except OSError as error:
        failure = f"filesystem error ({error})"
    if failure is None and user_cuda_llama_server(facts) is None:
        failure = "the installed engine is not usable"
    if failure is not None:
        logger.warning(
            f"CUDA engine install for this user failed: {failure}; the node "
            "degrades to the Vulkan/tarball chain"
        )
        return False
    return True


def _install_cuda_engine_tree(url: str, sha256: str, destination: Path) -> str | None:
    """Download, verify, and pip-install the CUDA engine at ``destination``.

    The install is staged beside ``destination`` and moved into place only
    when complete, so an interrupted install never leaves a partial engine
    where startup would find it.

    Args:
        url: The engine wheel's exact URL on the Foxlight index.
        sha256: The digest the index publishes for that wheel.
        destination: The per-pin user engine directory.

    Returns:
        ``None`` on success, otherwise why the install failed.

    Raises:
        OSError: When the engines directory cannot be written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        wheel = staging / unquote(url.rsplit("/", 1)[-1])
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                with wheel.open("wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        handle.write(chunk)
            _verify_sha256(wheel, sha256)
        except (httpx.HTTPError, RuntimeError) as error:
            return f"no verified engine wheel ({error})"
        tree = staging / "tree"
        # --isolated already ignores pip configuration files and PIP_*
        # variables; stripping them too keeps a host mirror out of the
        # resolution of NVIDIA's runtime wheels even if that changes. pip
        # stages a --target install under TMPDIR, which may be a small tmpfs,
        # so the gigabytes of runtime libraries stay on the engines volume.
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PIP_")
        }
        environment["TMPDIR"] = str(staging)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--isolated",
                    "--no-input",
                    "--disable-pip-version-check",
                    "--no-warn-script-location",
                    # The engine is installed once per pin; a cached copy of
                    # NVIDIA's wheels would only double its disk use.
                    "--no-cache-dir",
                    "--only-binary",
                    ":all:",
                    "--index-url",
                    _PYPI_INDEX,
                    "--target",
                    str(tree),
                    str(wheel),
                ],
                capture_output=True,
                text=True,
                timeout=_WHEEL_INSTALL_TIMEOUT_SECONDS,
                check=False,
                env=environment,
            )
        except (subprocess.TimeoutExpired, OSError) as error:
            return f"could not run pip ({error})"
        if completed.returncode != 0:
            return (
                f"pip exited {completed.returncode}: {completed.stderr.strip()[-500:]}"
            )
        _write_cuda_launchers(tree)
        if destination.exists():
            shutil.rmtree(destination)
        tree.rename(destination)
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _cuda_lane_ready(facts: NodeFacts) -> bool:
    """Whether a pin-matched CUDA engine is installed, here or for this user."""
    return _cuda_wheel_usable() or user_cuda_llama_server(facts) is not None


def _installed_engine(vendor: str, facts: NodeFacts) -> tuple[Path, Path | None] | None:
    """The installed engine to wire, preferring a CUDA build on NVIDIA.

    Order: the environment's engine wheel when it is the CUDA one, then a CUDA
    engine installed for this user, then any other environment wheel.
    """
    wheel = wheel_llama_server(vendor, facts)
    if vendor != "nvidia" or _cuda_wheel_usable():
        return wheel
    return user_cuda_llama_server(facts) or wheel


def try_install_cuda_wheel(facts: NodeFacts) -> bool:
    """Install the pinned CUDA engine wheel from the Foxlight index.

    Completes the CUDA lane for nodes that never ran ``install.sh``'s engine
    step (bare ``uv sync`` checkouts, GPU-cloud containers, #661): the
    tarball chain has no CUDA artifact by design (no upstream Linux CUDA
    prebuilt exists), so without the wheel an NVIDIA container degrades
    through Vulkan (no ICD in compute-only driver stacks) to a CPU-tagged
    engine while the facts probe plainly sees the GPU. Uses ``uv`` (the
    canonical Skulk runtime path) into the current interpreter's
    environment. A runtime without ``uv`` or with a read-only environment,
    such as the operating-system packages, installs the engine for the
    running user with its own ``pip`` instead; with neither tool the manual
    remediation is logged and the node degrades.

    Args:
        facts: The facts snapshot (gates on the CUDA wheel's SM floor).

    Returns:
        ``True`` when the wheel is installed and importable afterward.
    """
    if not _cuda_capability_ok(facts):
        return False
    uv = shutil.which("uv")
    if (uv is None or not _environment_writable()) and _pip_available():
        # The operating-system packages install Skulk read-only and ship pip
        # but no uv, so the wheel cannot join their environment.
        return _install_cuda_wheel_for_user(facts)
    pin = LLAMA_SERVER_PIN.removeprefix("b")
    specifier = (
        f"skulk-llama-server-cuda==0.{pin}.*,"
        f">=0.{pin}.{LLAMA_SERVER_CUDA_MIN_REVISION}"
    )
    # The remediation mirrors the automated uv invocation exactly, because
    # the resolution semantics are load-bearing: uv consults extra indexes
    # BEFORE the default index, so the Foxlight wheel wins, while pip's
    # --extra-index-url picks the best version across ALL indexes, letting a
    # same-name package on PyPI or a mirror satisfy the 0.<pin>.* specifier
    # and replace the pinned engine binary (PR #665 review). The env -u
    # prefix reproduces the sanitized subprocess environment: UV_INDEX and
    # friends outrank CLI flags, so without it the copied command is not
    # equivalent to the automated one on a host with a corporate mirror
    # configured. Quoted and targeting THIS interpreter for copy/paste
    # safety.
    manual_command = (
        "env -u UV_INDEX -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL "
        "-u UV_DEFAULT_INDEX -u UV_FIND_LINKS -u UV_CONFIG_FILE "
        f"uv pip install --no-config --python {sys.executable} '{specifier}' "
        f"--extra-index-url {FOXLIGHT_WHEEL_INDEX} --index-url {_PYPI_INDEX}"
    )
    if uv is None:
        logger.warning(
            "no uv on PATH to install the CUDA engine wheel; install uv "
            "(https://docs.astral.sh/uv/getting-started/installation/) and "
            f"run: {manual_command}"
        )
        return False
    logger.info(
        f"installing the CUDA engine wheel ({specifier}) from "
        f"{FOXLIGHT_WHEEL_INDEX}; multi-GB, one-time"
    )
    # Env-provided index settings outrank CLI flags in uv's resolution
    # (UV_INDEX is consulted before --extra-index-url, and the default
    # first-index strategy stops at the first index carrying the package),
    # so a host-level corporate mirror or cache index could serve a stale
    # or non-Foxlight engine build despite the explicit pins. Strip every
    # index/find-links override so resolution uses exactly the two indexes
    # named on the command line (PR #665 review).
    sanitized_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_INDEX", "UV_EXTRA_INDEX", "UV_DEFAULT_INDEX"))
        and key
        not in (
            "UV_FIND_LINKS",
            "UV_CONFIG_FILE",
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_FIND_LINKS",
        )
    }
    try:
        completed = subprocess.run(
            [
                uv,
                "pip",
                "install",
                # --no-config: a discovered pyproject.toml/uv.toml (or an
                # operator UV_CONFIG_FILE) can carry [[index]]/find-links
                # settings that outrank the CLI flags; the automated install
                # must resolve from exactly the two indexes named below
                # (PR #665 review).
                "--no-config",
                "--python",
                sys.executable,
                specifier,
                "--extra-index-url",
                FOXLIGHT_WHEEL_INDEX,
                "--index-url",
                _PYPI_INDEX,
            ],
            capture_output=True,
            text=True,
            timeout=_WHEEL_INSTALL_TIMEOUT_SECONDS,
            check=False,
            env=sanitized_env,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "CUDA engine wheel install timed out; the node degrades to the "
            "Vulkan/tarball chain (retry via `skulk doctor --fix` or install "
            f"manually: {manual_command})"
        )
        return False
    except OSError as error:
        # A which()-found uv that cannot actually execute (stale path,
        # stripped exec bit, exec-format error) must degrade like every
        # other installer failure, never crash node startup (PR #665
        # review).
        logger.warning(
            f"could not execute uv for the CUDA engine wheel install "
            f"({error}); the node degrades to the Vulkan/tarball chain "
            f"(install manually: {manual_command})"
        )
        return False
    if completed.returncode != 0:
        logger.warning(
            "CUDA engine wheel install failed (exit "
            f"{completed.returncode}): {completed.stderr.strip()[-500:]}; the "
            f"node degrades to the Vulkan/tarball chain ({manual_command})"
        )
        return False
    # find_spec caches negative lookups from before the install.
    importlib.invalidate_caches()
    if not _cuda_wheel_usable():
        # A zero exit that did not land a usable pin-matched wheel in THIS
        # environment (uv resolved a different interpreter, a broken wheel)
        # must not read as success: the caller would skip the Vulkan/tarball
        # chain believing the CUDA lane is live (PR #665 review).
        logger.warning(
            "CUDA engine wheel install reported success but the wheel is "
            "not importable and pin-matched in this environment; the node "
            f"degrades to the Vulkan/tarball chain ({manual_command})"
        )
        return False
    return True


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
            try:
                extracted.rename(target_dir)
            except OSError:
                # Two provisioners racing (node startup + doctor --fix): the
                # loser falls through to the winner's completed install.
                if not target_dir.exists():
                    raise
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


def dormant_llama_server(facts: NodeFacts) -> Path | None:
    """The managed llama-server that node startup would wire, without wiring it.

    Read-only twin of :func:`ensure_llama_server`'s no-network paths, for
    diagnostics: ``skulk doctor`` without ``--fix`` must not export
    environment variables or download anything, but it also must not report
    an engine-less node when startup will wire an already-installed managed
    engine (#628). Same gates and the same preference order (engine wheel
    shim first, then an already-provisioned managed install on disk); no
    environment mutation, no network.

    Args:
        facts: The current facts snapshot (decides applicability and variant).

    Returns:
        The managed binary startup would adopt, or ``None`` when nothing on
        disk would wire (override present, opted out, non-Linux, or no
        installed wheel/build).
    """
    if os.environ.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() == "1":
        return None
    if facts.llama_server_binary.state != "not_configured":
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
        wheel = _installed_engine(vendor, facts)
        if wheel is not None:
            return wheel[0]
    for variant in select_variant_chain(facts):
        existing = _binary_in(
            SKULK_ENGINES_DIR / "llama-server" / LLAMA_SERVER_PIN / variant
        )
        if existing is not None and os.access(existing, os.X_OK):
            return existing
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
        wheel = _installed_engine(vendor, facts)
        if (
            vendor == "nvidia"
            and not _cuda_lane_ready(facts)
            and allow_download
            and try_install_cuda_wheel(facts)
        ):
            # Completing the CUDA lane on demand (#661): the tarball chain
            # below has no CUDA artifact by design, and in a compute-only
            # container the Vulkan fallback sees no devices — the node would
            # advertise a CPU engine while the facts probe plainly reports
            # the GPU. install.sh's engine step covers curl-installed nodes;
            # this covers every other entry path (bare checkouts, GPU-cloud
            # containers) and self-heals a stale wheel after a pin advance.
            # Keyed on the CUDA wheel itself, not any-wheel-resolved: an
            # installed VULKAN wheel must not shadow the attempt, since its
            # shim cannot drive the GPU in a compute-only container
            # (PR #665 review).
            wheel = _installed_engine(vendor, facts)
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
