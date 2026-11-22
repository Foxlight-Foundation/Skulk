"""Which publishers may sign engine packages, and where a node learns it.

The built-in store's TUF repository signs one trust document for capability
publishers and a separate one for engine publishers, under the same root.
This module reads only the engine document. A key that may sign a capability
release is therefore not, by that fact, a key that may sign an engine: the
two lists are different signed targets with separate retained copies and
separate revision floors.

Engine trust is consulted when a package is installed and at no other time.
A package that is already installed and verified starts without it, so an
unreachable store or an expired trust document never stops a node from
serving with the engines it has. It does stop a new installation.
"""

from pathlib import Path
from typing import Final

from tuf.ngclient.fetcher import FetcherInterface

from skulk.extensions.capability_store import (
    EMBEDDED_CAPABILITY_STORE_ROOT,
    StoreTrustClient,
    StoreTrustUnavailableError,
)
from skulk.extensions.runtime_artifacts import RuntimeTrust
from skulk.provisioning.engine_packages import EnginePackageRefusedError
from skulk.shared.constants import SKULK_DATA_HOME

ENGINE_TRUST_TARGET: Final = "v1/engine-trust.json"
"""The signed target that lists engine publishers. Fixed in host code: no
manifest, request or setting can point an installation at another list."""

ENGINE_TRUST_DIR: Final = SKULK_DATA_HOME / "engine-trust"
"""Where this node keeps the engine trust it last verified. It is separate
from every capability trust directory."""


def engine_trust_client(
    cache_dir: Path = ENGINE_TRUST_DIR,
    *,
    embedded_root: Path = EMBEDDED_CAPABILITY_STORE_ROOT,
    fetcher: FetcherInterface | None = None,
) -> StoreTrustClient:
    """Build the client that reads the engine trust target.

    Construction touches neither the network nor the disk.

    Args:
        cache_dir: Owner-only directory for the retained engine trust.
        embedded_root: The store's TUF root shipped with this build.
        fetcher: Network access for TUF, injectable for tests.

    Returns:
        A store trust client bound to the engine target and nothing else.
    """
    return StoreTrustClient(
        cache_dir,
        embedded_root=embedded_root,
        fetcher=fetcher,
        trust_target=ENGINE_TRUST_TARGET,
    )


def load_engine_trust(
    *,
    offline: bool,
    now: int | None = None,
    client: StoreTrustClient | None = None,
) -> RuntimeTrust:
    """The engine trust in force, for deciding an installation.

    Online, the store is asked first and a newer verified document replaces
    the retained one. If the store cannot be reached, or offers nothing
    acceptable, the retained document is used while it is still in force.
    Offline, only the retained document is read. A cluster that never reaches
    the store receives that document as a file and retains it the same way.

    Call this when installing a package. Do not call it to start an engine
    that is already installed.

    Args:
        offline: Never reach the network; use the retained document.
        now: Unix time for expiry checks; the current time by default.
        client: The client to use; the node's own engine client by default.

    Returns:
        Verified engine trust whose expiry is still ahead.

    Raises:
        EnginePackageRefusedError: ``trust_unavailable`` when this build has
            no store root, or no verified engine trust is in force. The
            capability publisher trust is never used in its place.

    Side effects:
        Writes TUF metadata and the retained document under the client's
        cache directory.
    """
    reader = client if client is not None else engine_trust_client()
    if reader.trust_target != ENGINE_TRUST_TARGET:
        # A client for any other target holds a different authority's keys.
        raise EnginePackageRefusedError(
            "trust_unavailable", "the trust client is not bound to engine trust"
        )
    try:
        return reader.load(offline=offline, now=now)
    except StoreTrustUnavailableError:
        raise EnginePackageRefusedError(
            "trust_unavailable",
            "no verified engine trust is in force on this node; an "
            "installation needs the store to be reachable once, or the "
            "engine trust document imported as a file",
        ) from None
