"""Engine packages a node can install when a model needs one.

A node that already holds a package offers it from disk
(``skulk.engines.inventory``). This module covers the step before that: a
package the node does not hold yet but could install. Such a package is
offered the way the built-in on-demand engines are, so a placement can choose
it, the launch says the engine installs first, and the worker installs it
beside the model download.

What a node offers is read from the package list it last verified and kept,
with no network access, so gathering node facts stays quick and works when
the store cannot be reached. Fetching a newer list and installing a package
are the only steps here that use the network, and each is a separate call.

Every gate that stops a built-in engine from being installed on demand stops
a package too: the node is offline, installs are turned off
(``SKULK_NO_ENGINE_AUTOPROVISION=1``), the node does not run models, or
engine packages are turned off (``SKULK_ENGINE_PACKAGES_DISABLE=1``).
"""

import os
import platform
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, final

import httpx

from skulk.engines.protocol import PROTOCOL_VERSIONS_SUPPORTED
from skulk.extensions.capability_store import (
    StoreTrustClient,
    StoreTrustUnavailableError,
)
from skulk.extensions.runtime_artifacts import RuntimeTrust, current_platform
from skulk.provisioning.engine_index import (
    ENGINE_STORE_URL,
    EngineChannel,
    EngineIndexClient,
    EngineIndexEntry,
    EngineIndexRefusedError,
    VerifiedEngineIndex,
    installable_engine_packages,
)
from skulk.provisioning.engine_install import (
    ENGINE_PACKAGES_DIR,
    HttpPackageSource,
    InstalledEngine,
    activate_engine_generation,
    install_engine_package,
)
from skulk.provisioning.engine_packages import (
    MAX_ENGINE_MANIFEST_BYTES,
    EngineHost,
    EngineLane,
    EnginePackageRefusedError,
    check_engine_host,
    verify_engine_manifest,
)
from skulk.provisioning.engine_trust import engine_trust_client, load_engine_trust
from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.shared.backends import engine_packages_disabled
from skulk.shared.types.node_facts import NodeFacts
from skulk.shared.version import get_skulk_version

ENGINE_PACKAGE_CHANNEL_ENV: Final = "SKULK_ENGINE_PACKAGE_CHANNEL"
"""Set to ``candidate`` to make a node also install package builds that have
not been promoted to ``stable``. Unset, or any other value, follows ``stable``
only. A launch switch for test segments; it is not a fleet setting."""

_MANIFEST_NAME: Final = "manifest.json"
_FETCH_SECONDS: Final = 8.0


@final
class EnginePackageInstallError(Exception):
    """An on-demand package install stopped at a named step.

    Attributes:
        step: The step that failed, phrased for an operator.
        detail: The reason, for this node's log.
    """

    def __init__(self, step: str, detail: str) -> None:
        """Record the failed step and its reason."""
        super().__init__(f"{step}: {detail}")
        self.step: Final = step
        self.detail: Final = detail


def followed_channels(
    environ: Mapping[str, str] | None = None,
) -> frozenset[EngineChannel]:
    """The package channels this node installs from."""
    env = os.environ if environ is None else environ
    if env.get(ENGINE_PACKAGE_CHANNEL_ENV, "").strip().lower() == "candidate":
        return frozenset({"stable", "candidate"})
    return frozenset({"stable"})


def on_demand_gates_pass(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this node may install an engine package at all.

    The same switches that stop a built-in engine from installing on demand,
    plus the engine-package off switch. Offline mode is passed separately by
    each caller, as it is for the built-in engines.
    """
    env = os.environ if environ is None else environ
    return (
        not engine_packages_disabled()
        and env.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() != "1"
        and env.get("SKULK_NODE_PARTICIPATION", "").strip().lower() in ("", "full")
    )


def engine_host_of(
    facts: NodeFacts, *, platform_name: str | None = None
) -> EngineHost | None:
    """Describe this node the way a package's requirements are judged.

    The accelerator classes and exact device targets come from the node's
    observed hardware. An undetected device target is left out, which a
    package's requirements treat as unknown rather than as a mismatch.

    Args:
        facts: The node's observed facts.
        platform_name: The artifact family, for tests; the running host's by
            default.

    Returns:
        The host description, or ``None`` on an operating system no package
        is built for.
    """
    try:
        family = platform_name if platform_name is not None else current_platform()
    except ValueError:
        return None
    nvidia = facts.gpus_of("nvidia")
    amd = facts.gpus_of("amd")
    lanes: set[EngineLane] = {"cpu"}
    if facts.platform == "darwin":
        lanes.add("metal")
    else:
        if nvidia:
            lanes.add("cuda")
        if amd:
            lanes.add("rocm")
        if nvidia or amd or any(gpu.vendor == "intel" for gpu in facts.gpus):
            lanes.add("vulkan")
    targets = {
        gpu.gpu_architecture for gpu in amd if gpu.gpu_architecture is not None
    } | {
        "sm_" + gpu.compute_capability.replace(".", "")
        for gpu in nvidia
        if gpu.compute_capability is not None
    }
    return EngineHost(
        platform=family,
        lanes=frozenset(lanes),
        compute_targets=frozenset(targets),
        skulk_version=get_skulk_version(),
        control_protocols=(
            min(PROTOCOL_VERSIONS_SUPPORTED),
            max(PROTOCOL_VERSIONS_SUPPORTED),
        ),
        python_version=platform.python_version(),
    )


def _kept_index(
    trust_client: StoreTrustClient | None,
    index_client: EngineIndexClient | None,
    now: int,
) -> VerifiedEngineIndex | None:
    """The kept package list under the kept engine trust; no network."""
    trust_reader = trust_client if trust_client is not None else engine_trust_client()
    index_reader = index_client if index_client is not None else EngineIndexClient()
    try:
        trust = trust_reader.last_known_good(now=now)
        return index_reader.kept(trust, now=now)
    except (StoreTrustUnavailableError, EngineIndexRefusedError):
        return None


def engine_package_offers(
    facts: NodeFacts,
    *,
    offline: bool,
    environ: Mapping[str, str] | None = None,
    trust_client: StoreTrustClient | None = None,
    index_client: EngineIndexClient | None = None,
    platform_name: str | None = None,
    now: int | None = None,
) -> dict[str, EngineIndexEntry]:
    """The package builds this node would install on demand, by backend tag.

    Reads only the kept package list. A tag the node already holds a package
    for is not offered again: moving an installed engine to a newer build is
    an explicit action, never a side effect of launching a model. A packaged
    Skulk cannot run a Python adapter, so it is offered none.

    Args:
        facts: The node's observed facts, including the packages it holds.
        offline: The node must not reach the network, so it can install
            nothing and offers nothing.
        environ: Environment to read switches from; the process's by default.
        trust_client: Engine trust reader, for tests.
        index_client: Package list reader, for tests.
        platform_name: Artifact family, for tests.
        now: Unix time for expiry checks; the current time by default.

    Returns:
        The chosen build for each backend tag. Empty when nothing is offered,
        for any reason; a problem with the optional package list never stops
        a node from gathering its facts.
    """
    if offline or not on_demand_gates_pass(environ):
        return {}
    host = engine_host_of(facts, platform_name=platform_name)
    if host is None:
        return {}
    current = now if now is not None else int(time.time())
    try:
        index = _kept_index(trust_client, index_client, current)
    except (OSError, ValueError):
        return {}
    if index is None:
        return {}
    packaged = bool(getattr(sys, "frozen", False))
    return {
        tag: entry
        for tag, entry in installable_engine_packages(
            index, host, channels=followed_channels(environ)
        ).items()
        if tag not in facts.engine_package_builds
        and not (packaged and entry.adapter_python_requires is not None)
    }


def installable_engine_package_builds(
    facts: NodeFacts, *, offline: bool
) -> dict[str, str]:
    """Backend tag to build identity for each package offered on demand."""
    return {
        tag: entry.manifest_digest
        for tag, entry in sorted(
            engine_package_offers(facts, offline=offline).items()
        )
    }


def published_engine_package(
    manifest_digest: str,
    *,
    trust_client: StoreTrustClient | None = None,
    index_client: EngineIndexClient | None = None,
    now: int | None = None,
) -> EngineIndexEntry | None:
    """Look up one published build in the kept package list, by identity.

    Used to quote a download size for a build another node will install. It
    reads this node's kept list, so an unknown build is ``None``, not an
    error.
    """
    current = now if now is not None else int(time.time())
    try:
        index = _kept_index(trust_client, index_client, current)
    except (OSError, ValueError):
        return None
    if index is None:
        return None
    return next(
        (
            entry
            for entry in index.claims.entries
            if entry.manifest_digest == manifest_digest
        ),
        None,
    )


_KNOWN_DOWNLOAD_BYTES: Final[dict[str, int]] = {}
_MAX_KNOWN_DOWNLOADS: Final = 256


def engine_package_download_bytes(manifest_digest: str | None) -> int:
    """Bytes an install of one published build fetches, or zero when unknown.

    A build's size never changes, so a size found once is remembered for the
    life of the process and the kept list is not read again for it.
    """
    if manifest_digest is None:
        return 0
    known = _KNOWN_DOWNLOAD_BYTES.get(manifest_digest)
    if known is not None:
        return known
    entry = published_engine_package(manifest_digest)
    if entry is None:
        return 0
    if len(_KNOWN_DOWNLOAD_BYTES) < _MAX_KNOWN_DOWNLOADS:
        _KNOWN_DOWNLOAD_BYTES[manifest_digest] = entry.download_bytes
    return entry.download_bytes


def refresh_engine_package_list(
    *,
    offline: bool,
    environ: Mapping[str, str] | None = None,
    trust_client: StoreTrustClient | None = None,
    index_client: EngineIndexClient | None = None,
    now: int | None = None,
) -> bool:
    """Ask the store for a newer engine trust and package list, and keep them.

    Does nothing on a node that may not install packages. Every failure is
    quiet: the kept copies stay in force and the node offers what it offered
    before.

    Returns:
        True when a verified package list is in force afterwards.

    Side effects:
        Network access to the store; writes the kept trust and package list.
    """
    if offline or not on_demand_gates_pass(environ):
        return False
    try:
        trust = load_engine_trust(offline=False, now=now, client=trust_client)
        reader = index_client if index_client is not None else EngineIndexClient()
        reader.load(trust, offline=False, now=now)
    except (EnginePackageRefusedError, EngineIndexRefusedError, OSError, ValueError):
        return False
    return True


def _fetch_manifest(
    address: str, transport: httpx.BaseTransport | None
) -> bytes:
    """Read one signed manifest, stopping at the manifest size bound."""
    received = bytearray()
    with (
        httpx.Client(
            transport=transport, timeout=_FETCH_SECONDS, follow_redirects=False
        ) as client,
        client.stream("GET", address) as response,
    ):
        if response.status_code != 200:
            raise ValueError(f"the store answered HTTP {response.status_code}")
        for block in response.iter_bytes(65536):
            received.extend(block)
            if len(received) > MAX_ENGINE_MANIFEST_BYTES:
                raise ValueError("the manifest document is too large")
    return bytes(received)


def install_engine_package_on_demand(
    backend_tag: str,
    expected_build: str,
    facts: NodeFacts,
    *,
    offline: bool,
    stop: threading.Event,
    environ: Mapping[str, str] | None = None,
    trust_client: StoreTrustClient | None = None,
    index_client: EngineIndexClient | None = None,
    transport: httpx.BaseTransport | None = None,
    store_url: str = ENGINE_STORE_URL,
    root: Path = ENGINE_PACKAGES_DIR,
    platform_name: str | None = None,
    now: int | None = None,
) -> InstalledEngine:
    """Install and activate the exact package build a placement was stamped with.

    The build is found in the package list by its identity, its signed
    manifest is fetched and verified under the engine trust, and the manifest
    must have exactly that identity. Only then are its files fetched, each
    checked against the signed size and digest. The finished generation is
    made the active one, so the node offers the engine from disk afterwards.

    Args:
        backend_tag: The backend the placement resolved to.
        expected_build: The build identity the placement was stamped with.
        facts: The node's observed facts.
        offline: The node must not reach the network.
        stop: Set to cancel; a partial installation is removed.
        environ: Environment to read switches from; the process's by default.
        trust_client: Engine trust reader, for tests.
        index_client: Package list reader, for tests.
        transport: Network transport, for tests.
        store_url: The engine store directory.
        root: Root of installed engine packages.
        platform_name: Artifact family, for tests.
        now: Unix time for expiry checks; the current time by default.

    Returns:
        The installed, active generation.

    Raises:
        EnginePackageInstallError: Naming the step that failed. Nothing is
            activated in that case, and an engine that was active stays so.

    Side effects:
        Network access to the store; writes under ``root`` and the kept
        trust and package list.
    """
    if offline:
        raise EnginePackageInstallError(
            "the network check", "this node is offline and cannot fetch a package"
        )
    if not on_demand_gates_pass(environ):
        raise EnginePackageInstallError(
            "the install check", "engine package installs are turned off on this node"
        )
    host = engine_host_of(facts, platform_name=platform_name)
    if host is None:
        raise EnginePackageInstallError(
            "the host check", "no engine package is built for this operating system"
        )
    current = now if now is not None else int(time.time())
    try:
        trust: RuntimeTrust = load_engine_trust(
            offline=False, now=current, client=trust_client
        )
    except EnginePackageRefusedError as refused:
        raise EnginePackageInstallError("the engine trust check", str(refused)) from None
    reader = index_client if index_client is not None else EngineIndexClient()
    try:
        index = reader.load(trust, offline=False, now=current)
    except EngineIndexRefusedError as refused:
        raise EnginePackageInstallError("the package list check", str(refused)) from None
    entry = next(
        (
            candidate
            for candidate in index.claims.entries
            if candidate.manifest_digest == expected_build
            and candidate.backend_tag == backend_tag
        ),
        None,
    )
    if entry is None:
        raise EnginePackageInstallError(
            "the package list check",
            f"the build this placement names for {backend_tag} is no longer published",
        )
    directory = store_url.rstrip("/") + "/" + entry.directory + "/"
    try:
        document = _fetch_manifest(directory + _MANIFEST_NAME, transport)
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise EnginePackageInstallError(
            "the manifest download", f"{type(error).__name__}: {error}"
        ) from None
    try:
        package = verify_engine_manifest(document, trust, now=current)
        if (
            package.digest != expected_build
            or package.claims.backend_tag != backend_tag
            or package.claims.engine_id != entry.engine_id
        ):
            raise EnginePackageInstallError(
                "the manifest check",
                "the store served a manifest that is not the build the placement names",
            )
        check_engine_host(package, host)
    except EnginePackageRefusedError as refused:
        raise EnginePackageInstallError(
            "the manifest check", f"{refused.code}: {refused}"
        ) from None
    try:
        install_engine_package(
            package,
            HttpPackageSource(directory, transport=transport),
            root=root,
            stop=stop,
        )
        return activate_engine_generation(
            package.claims.engine_id, package.claims.lane, package.digest, root=root
        )
    except EnginePackageRefusedError as refused:
        raise EnginePackageInstallError(
            "the package download", f"{refused.code}: {refused}"
        ) from None
