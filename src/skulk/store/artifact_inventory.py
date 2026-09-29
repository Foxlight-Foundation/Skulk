"""Node-local artifact discovery shared by APIs and telemetry publishers."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from skulk.shared import constants as shared_constants
from skulk.shared.models.model_cards import ModelCard
from skulk.store.installed_cards import (
    InstalledCardRecord,
    VerifiedDetachedInstalledCardCache,
    ensure_installed_cards,
)
from skulk.store.staging_eviction import StagedModelInfo, list_staged_models


def installed_artifact_roots(staging_root: Path | None) -> tuple[Path, ...]:
    """Return every node-local root that may contain launchable artifacts.

    Args:
        staging_root: Configured cache root for this node, when staging is enabled.

    Returns:
        Deduplicated launchable roots, preserving their operator-facing paths.
    """

    candidates = [
        *((staging_root,) if staging_root is not None else ()),
        shared_constants.SKULK_MODELS_DIR,
        *(shared_constants.SKULK_MODELS_PATH or ()),
    ]
    roots_by_path: dict[Path, Path] = {}
    for candidate in candidates:
        expanded = candidate.expanduser()
        roots_by_path.setdefault(expanded.resolve(), expanded)
    return tuple(roots_by_path.values())


def inventory_installed_artifacts(
    roots: Iterable[Path],
    cards: Iterable[ModelCard],
    in_use_model_ids: frozenset[str] = frozenset(),
    canonical_store_root: Path | None = None,
    verified_detached_cache: VerifiedDetachedInstalledCardCache | None = None,
    materialized: list[InstalledCardRecord] | None = None,
) -> list[StagedModelInfo]:
    """Inventory complete and unresolved artifacts across local roots.

    Args:
        roots: Launchable local roots to inspect.
        cards: Current cards used to recover safe installed identities.
        in_use_model_ids: Models protected by a live runner on this node.
        canonical_store_root: Authoritative root whose entries are store-local
            rather than node-cache copies.
        verified_detached_cache: Optional process-local cache that prevents
            periodic operator scans from rehashing unchanged read-only roots.
        materialized: Optional list that receives every card record written
            while associating legacy artifacts. The scan runs in a worker
            thread; the caller registers these on its event loop so the live
            catalog lists the associated models at once.

    Returns:
        Deduplicated local artifacts with explicit location provenance.
    """

    inventory_by_directory: dict[Path, StagedModelInfo] = {}
    card_list = tuple(cards)
    canonical_root = (
        canonical_store_root.expanduser().resolve()
        if canonical_store_root is not None
        else None
    )
    for root in roots:
        written = ensure_installed_cards(root, card_list, verified_detached_cache)
        if materialized is not None:
            materialized.extend(written)
        for item in list_staged_models(
            root,
            in_use_model_ids,
            verified_detached_cache,
        ):
            directory = Path(item.directory).resolve()
            location_kind: Literal["store_local", "node_cache"] = (
                "store_local"
                if canonical_root is not None and directory.is_relative_to(canonical_root)
                else "node_cache"
            )
            inventory_by_directory.setdefault(
                directory,
                item.model_copy(update={"location_kind": location_kind}),
            )
    if verified_detached_cache is not None:
        verified_detached_cache.retain(inventory_by_directory)
    return list(inventory_by_directory.values())


def associate_installed_artifacts(
    roots: Iterable[Path],
    cards: Iterable[ModelCard],
    verified_detached_cache: VerifiedDetachedInstalledCardCache | None = None,
) -> tuple[InstalledCardRecord, ...]:
    """Give every complete legacy artifact under ``roots`` its card record.

    Runs on every node, with or without a model store, so a model downloaded
    before card records existed gets one whenever a card that recognizes it
    is available.

    Args:
        roots: Launchable local roots to inspect.
        cards: Cards that may associate artifacts, as
            ``get_association_cards()`` returns them.
        verified_detached_cache: Optional cache for detached records on
            read-only roots.

    Returns:
        The records written, for the caller to register on its event loop.
    """

    card_list = tuple(cards)
    written: list[InstalledCardRecord] = []
    for root in roots:
        written.extend(ensure_installed_cards(root, card_list, verified_detached_cache))
    return tuple(written)

