"""Node-local artifact discovery shared by APIs and telemetry publishers."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from skulk.shared import constants as shared_constants
from skulk.shared.models.model_cards import ModelCard
from skulk.store.installed_cards import ensure_installed_cards
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
) -> list[StagedModelInfo]:
    """Inventory complete and unresolved artifacts across local roots.

    Args:
        roots: Launchable local roots to inspect.
        cards: Current cards used to recover safe installed identities.
        in_use_model_ids: Models protected by a live runner on this node.
        canonical_store_root: Authoritative root whose entries are store-local
            rather than node-cache copies.

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
        ensure_installed_cards(root, card_list)
        for item in list_staged_models(root, in_use_model_ids):
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
    return list(inventory_by_directory.values())
