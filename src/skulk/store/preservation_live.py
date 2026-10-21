"""Publish a prepared full closure through the existing canonical store owner."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import anyio
from foxlight_model_preservation.models import PreservationError, TrustProfile
from foxlight_model_preservation.registry_proof import ProofMode
from foxlight_model_preservation.signing import verify_manifest
from pydantic import TypeAdapter

from skulk.shared.models.model_cards import (
    ModelCard,
    add_to_card_cache,
    register_installed_card_record,
)
from skulk.store.installed_cards import read_installed_card, verify_installed_card
from skulk.store.model_store import (
    ModelStore,
    StoreModelEntry,
    StoreRegistryIndex,
)
from skulk.store.preservation import inspect_verified_manifest, read_recovery_metadata


def recovered_index(home: Path) -> StoreRegistryIndex:
    """Read only the known worker's bounded isolated generation index."""
    return StoreRegistryIndex.model_validate_json(
        read_recovery_metadata(home / "model-store" / "registry.json")
    )


async def publish_recovered_home(
    store: ModelStore,
    home: Path,
    generation_id: str,
    custom_cards: Path,
    *,
    manifest_id: str,
    trust: TrustProfile,
    mode: ProofMode,
    deletions: dict[str, str],
    existing_companions: dict[str, str],
) -> None:
    """Publish every member in one index update, then its new receiver admission.

    The caller holds this actual owner's mutation lease. Existing aliases and
    deletion tombstones require their exact reviewed consent. Matching owned
    companions can move to the complete generation after byte verification;
    an exact already committed closure may finish admission after an interrupted prior attempt. No existing bytes are replaced.
    """
    index = recovered_index(home)
    aliases = set(index.entries)
    cards = [entry.installed_card for entry in index.entries.values()]
    base = next(
        (
            record
            for record in cards
            if record is not None and record.artifact_role == "base"
        ),
        None,
    )
    if base is None or any(record is None for record in cards):
        raise PreservationError("prepared recovery lacks complete installed records")
    original = verify_manifest(
        read_recovery_metadata(home / "preservation" / "envelope.json"),
        trust,
        manifest_id,
    )
    # The worker's isolated-home publication is preparation. Freshness and
    # projection must still hold at the actual canonical store commit boundary.
    review, projected = inspect_verified_manifest(original, manifest_id, mode)
    if set(review.members) != aliases or projected != base.model_card:
        raise PreservationError("prepared recovery differs from its signed projection")
    custom_cards.mkdir(mode=0o700, parents=True, exist_ok=True)
    admission = custom_cards / (base.model_card.model_id.normalize() + ".toml")
    if admission.exists() or admission.is_symlink():
        prior = (await ModelCard.load_from_path(anyio.Path(str(admission)))).model_copy(
            update={"is_custom": True}
        )
        if prior != base.model_card:
            raise PreservationError(
                "existing local card admission differs from recovery"
            )
    published = store.publish_recovery_generation(
        home,
        generation_id,
        index,
        deletions=deletions,
        existing_companions=existing_companions,
    )
    if base.model_card.is_custom:
        # The complete closure is already indexed before a selectable custom
        # card can appear. An interrupted admission is completed by explicit retry.
        candidate = published / "custom_model_cards" / admission.name
        if not admission.exists():
            pending = custom_cards / (".recovery-admission-" + uuid.uuid4().hex)
            try:
                descriptor = os.open(
                    pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
                )
                with os.fdopen(descriptor, "wb") as output:
                    _ = output.write(candidate.read_bytes())
                    output.flush()
                    os.fsync(output.fileno())
                os.link(pending, admission, follow_symlinks=False)
                parent_descriptor = os.open(custom_cards, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)
            finally:
                pending.unlink(missing_ok=True)
        add_to_card_cache(base.model_card)
    for entry in index.entries.values():
        committed = store.get_entry(entry.model_id)
        record = committed.installed_card if committed is not None else None
        if record is None:
            raise PreservationError("missing recovered record")
        register_installed_card_record(record)


def validate_prepared_home(home: Path) -> None:
    """Hash a complete isolated preparation before the worker reports success."""
    for entry in recovered_index(home).entries.values():
        path = home / "model-store" / entry.store_path
        if (
            read_installed_card(path) != entry.installed_card
            or entry.installed_card is None
            or not verify_installed_card(path, entry.installed_card, verify_hashes=True)
        ):
            raise PreservationError(
                "prepared recovery failed normal installed verification"
            )


def validate_existing_recovery(
    home: Path, store_root: Path, existing_companions: tuple[str, ...] = ()
) -> None:
    """Check any exact retry against the real current bytes under the host lease."""
    current = store_root / "registry.json"
    if not current.exists():
        return
    adapter: TypeAdapter[StoreRegistryIndex | dict[str, StoreModelEntry]] = TypeAdapter(
        StoreRegistryIndex | dict[str, StoreModelEntry]
    )
    value = adapter.validate_json(current.read_bytes())
    entries = value.entries if isinstance(value, StoreRegistryIndex) else value
    for alias, entry in recovered_index(home).entries.items():
        previous = entries.get(alias)
        if previous is None:
            continue
        record = previous.installed_card
        from skulk.store.preservation_compatibility import compatible_companion

        if (
            alias in existing_companions
            and record is not None
            and entry.installed_card is not None
            and compatible_companion(record, entry.installed_card)
        ):
            if not verify_installed_card(
                store_root / previous.store_path, record, verify_hashes=True
            ):
                raise PreservationError("existing companion bytes failed verification")
            continue
        if (
            record is None
            or entry.installed_card is None
            or record.files != entry.installed_card.files
            or record.model_card != entry.installed_card.model_card
            or record.preservation != entry.installed_card.preservation
        ):
            raise PreservationError("existing alias differs from recovery")
        if not verify_installed_card(
            store_root / previous.store_path, record, verify_hashes=True
        ):
            raise PreservationError("existing recovery bytes failed verification")
