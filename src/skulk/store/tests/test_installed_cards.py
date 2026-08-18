# pyright: reportPrivateUsage=false
"""Installed model cards remain bound to local artifact bytes."""

import asyncio
from pathlib import Path

import pytest

import skulk.store.installed_cards as installed_cards
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import (
    InstalledCardRecord,
    VerifiedDetachedInstalledCardCache,
    associate_installed_card,
    build_installed_card_record,
    installed_card_matches,
    read_installed_card,
    read_installed_card_with_fallback,
    require_registry_installed_artifact,
    verify_installed_file,
    write_installed_card,
    write_installed_card_with_fallback,
)
from skulk.store.model_store import (
    ModelStore,
    StoreDownloadStatus,
    StoreRegistryIndex,
)


def _card(*, registry: bool = True, custom: bool = False) -> ModelCard:
    return ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision="a" * 40,
        registry_card_id=(f"card_{'a' * 52}" if registry else None),
        registry_snapshot_id=("snapshot_1_test" if registry else None),
        registry_provenance=("foxlight" if registry else None),
        is_custom=custom,
    )


def _artifact(tmp_path: Path, *, revision_marker: bool = True) -> Path:
    artifact = tmp_path / "org--model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors").write_bytes(b"weights")
    if revision_marker:
        (artifact / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    return artifact


def test_verified_registry_record_round_trips(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    card = _card()

    record = build_installed_card_record(
        artifact,
        card,
        artifact_format="mlx",
    )
    write_installed_card(artifact, record)

    assert record.verification == "registry_verified"
    assert record.installed_identity == card.registry_card_id
    assert read_installed_card(artifact) == record
    assert installed_card_matches(artifact, card)
    assert all(entry.path != ".skulk-source-revision" for entry in record.files)


def test_pinned_companion_verification_detects_corruption(tmp_path: Path) -> None:
    """Runner-bound companion verification checks path, size, and exact digest."""

    artifact = _artifact(tmp_path)
    projector = artifact / "mmproj-F16.gguf"
    projector.write_bytes(b"projector")
    record = build_installed_card_record(artifact, _card())

    assert verify_installed_file(
        artifact,
        record,
        "mmproj-F16.gguf",
        expected_size=len(b"projector"),
    )
    projector.write_bytes(b"corrupt!!")
    assert not verify_installed_file(
        artifact,
        record,
        "mmproj-F16.gguf",
        expected_size=len(b"projector"),
    )


def test_unmarked_registry_bytes_remain_local_legacy(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, revision_marker=False)

    record = build_installed_card_record(artifact, _card())

    assert record.verification == "local_legacy"
    assert record.installed_identity.startswith("local_")
    assert record.model_card.registry_card_id is not None


def test_runner_boundary_accepts_exact_verified_registry_artifact(
    tmp_path: Path,
) -> None:
    """Pinned sidecar, revision marker and signed card agree at model load."""
    artifact = _artifact(tmp_path)
    card = _card()
    write_installed_card(artifact, build_installed_card_record(artifact, card))

    require_registry_installed_artifact(artifact, card)


@pytest.mark.parametrize("failure", ["legacy", "revision", "card"])
def test_runner_boundary_rejects_mismatched_registry_artifact(
    tmp_path: Path,
    failure: str,
) -> None:
    """Signed provenance never authorizes different or unproven local bytes."""
    artifact = _artifact(tmp_path, revision_marker=failure != "legacy")
    card = _card()
    record = build_installed_card_record(artifact, card)
    write_installed_card(artifact, record)
    requested = card
    if failure == "revision":
        (artifact / ".skulk-source-revision").write_text(f"{'b' * 40}\n")
    elif failure == "card":
        requested = card.model_copy(
            update={"registry_card_id": f"card_{'b' * 52}"}
        )

    with pytest.raises(PermissionError, match="signed card"):
        require_registry_installed_artifact(artifact, requested)


def test_revision_qualified_base_directory_associates_signed_card(
    tmp_path: Path,
) -> None:
    """Canonical pinned generations retain their signed registry identity."""

    card = _card()
    assert card.source_revision is not None
    artifact = tmp_path / f"org--model--revision-{card.source_revision}"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")

    record = associate_installed_card(artifact, [card])

    assert record is not None
    assert record.verification == "registry_verified"
    assert record.installed_identity == card.registry_card_id


def test_custom_card_keeps_custom_verification(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    record = build_installed_card_record(
        artifact,
        _card(registry=False, custom=True),
    )

    assert record.verification == "custom"


def test_custom_card_metadata_change_invalidates_installed_match(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    card = _card(registry=False, custom=True)
    write_installed_card(artifact, build_installed_card_record(artifact, card))
    changed_card = card.model_copy(update={"hidden_size": 2})

    assert not installed_card_matches(artifact, changed_card)


def test_sidecar_is_not_part_of_its_own_manifest(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    first = build_installed_card_record(artifact, _card())
    write_installed_card(artifact, first)
    second = build_installed_card_record(artifact, _card())

    assert first.manifest_sha256 == second.manifest_sha256


def test_record_rejects_unknown_fields(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    record = build_installed_card_record(artifact, _card())
    payload = record.model_dump(mode="json")
    payload["unexpected"] = True

    try:
        InstalledCardRecord.model_validate(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("strict installed record accepted an unknown field")


def test_manifest_rejects_unsafe_relative_paths(tmp_path: Path) -> None:
    record = build_installed_card_record(_artifact(tmp_path), _card())
    payload = record.model_dump(mode="json")
    payload["files"][0]["path"] = "../outside"

    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        InstalledCardRecord.model_validate(payload, strict=False)


def test_companion_association_retains_owning_full_card(tmp_path: Path) -> None:
    payload = _card().model_dump(mode="json")
    payload["gguf_file"] = "base.gguf"
    payload["runtime"] = {
        "mtp_sidecar_repo": "org/model-mtp",
        "mtp_sidecar_revision": "b" * 40,
    }
    owner = ModelCard.model_validate(payload)
    companion = tmp_path / "org--model-mtp"
    companion.mkdir()
    (companion / "weights.safetensors").write_bytes(b"mtp")

    record = associate_installed_card(companion, [owner])

    assert record is not None
    assert record.artifact_role == "mtp_sidecar"
    assert record.owner_model_id == str(owner.model_id)
    assert record.owner_card_id == owner.registry_card_id
    assert record.model_card == owner
    assert record.artifact_file is None


def test_revision_qualified_companion_associates_owning_card(tmp_path: Path) -> None:
    """Pinned companion generations recognize the canonical directory layout."""

    payload = _card().model_dump(mode="json")
    revision = "b" * 40
    payload["runtime"] = {
        "mtp_sidecar_repo": "org/model-mtp",
        "mtp_sidecar_revision": revision,
    }
    owner = ModelCard.model_validate(payload)
    companion = tmp_path / f"org--model-mtp--revision-{revision}"
    companion.mkdir()
    (companion / "weights.safetensors").write_bytes(b"mtp")
    (companion / ".skulk-source-revision").write_text(f"{revision}\n")

    record = associate_installed_card(companion, [owner])

    assert record is not None
    assert record.artifact_role == "mtp_sidecar"
    assert record.verification == "registry_verified"
    assert record.owner_card_id == owner.registry_card_id


def test_served_draft_record_selects_its_own_gguf(tmp_path: Path) -> None:
    """A separate served draft cannot inherit its owner's base quant file."""

    payload = _card().model_dump(mode="json")
    payload["gguf_file"] = "base.gguf"
    payload["runtime"] = {
        "served_spec_type": "draft_dflash",
        "served_spec_draft_repo": "org/model-draft",
        "served_spec_draft_revision": "b" * 40,
        "served_spec_draft_file": "draft.gguf",
    }
    owner = ModelCard.model_validate(payload)
    companion = tmp_path / "org--model-draft"
    companion.mkdir()
    (companion / "draft.gguf").write_bytes(b"draft")

    record = build_installed_card_record(
        companion,
        owner,
        artifact_role="served_draft",
        artifact_model_id="org/model-draft",
        owner_model_id=str(owner.model_id),
        owner_card_id=owner.registry_card_id,
        artifact_repository="org/model-draft",
        artifact_revision="b" * 40,
    )

    assert record.artifact_file == "draft.gguf"


def test_mutable_companion_does_not_inherit_base_revision(tmp_path: Path) -> None:
    """Explicit companion main remains distinct from its owner's pinned base."""

    owner = _card()
    companion = tmp_path / "org--model-mtp"
    companion.mkdir()
    (companion / "weights.safetensors").write_bytes(b"mtp")

    record = build_installed_card_record(
        companion,
        owner,
        artifact_role="mtp_sidecar",
        artifact_model_id="org/model-mtp",
        owner_model_id=str(owner.model_id),
        owner_card_id=owner.registry_card_id,
        artifact_repository="org/model-mtp",
        artifact_revision=None,
    )

    assert owner.source_revision is not None
    assert record.artifact_revision is None


def test_incomplete_legacy_companion_is_not_associated_by_directory_name(
    tmp_path: Path,
) -> None:
    payload = _card().model_dump(mode="json")
    payload["runtime"] = {
        "mtp_sidecar_repo": "org/model-mtp",
        "mtp_sidecar_revision": "b" * 40,
    }
    owner = ModelCard.model_validate(payload)
    companion = tmp_path / "org--model-mtp"
    companion.mkdir()
    (companion / "config.json").write_text("{}")

    assert associate_installed_card(companion, [owner]) is None


def test_incomplete_legacy_base_is_not_associated_by_directory_name(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "org--model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")

    assert associate_installed_card(artifact, [_card()]) is None


def test_conflicting_legacy_revision_is_not_associated(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    (artifact / ".skulk-source-revision").write_text(f"{'c' * 40}\n")

    assert associate_installed_card(artifact, [_card()]) is None


def test_read_only_artifact_uses_path_and_manifest_bound_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    record = build_installed_card_record(artifact, _card())
    fallback_root = tmp_path / "data" / "installed-cards"

    def _deny_adjacent_write(_directory: Path, _record: InstalledCardRecord) -> Path:
        raise PermissionError("read-only model root")

    monkeypatch.setattr(installed_cards, "write_installed_card", _deny_adjacent_write)
    written = write_installed_card_with_fallback(
        artifact,
        record,
        fallback_root=fallback_root,
    )

    assert written.parent == fallback_root
    assert read_installed_card(artifact) is None
    assert (
        read_installed_card_with_fallback(artifact, fallback_root=fallback_root)
        == record
    )


def test_detached_record_cache_rehashes_only_after_file_metadata_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic inventory reuses trust only while artifact stats are unchanged."""

    artifact = _artifact(tmp_path)
    record = build_installed_card_record(artifact, _card())
    fallback_root = tmp_path / "data" / "installed-cards"

    def _deny_adjacent_write(_directory: Path, _record: InstalledCardRecord) -> Path:
        raise PermissionError("read-only model root")

    monkeypatch.setattr(installed_cards, "write_installed_card", _deny_adjacent_write)
    write_installed_card_with_fallback(
        artifact,
        record,
        fallback_root=fallback_root,
    )
    original_sha256_file = installed_cards._sha256_file
    hashed_paths: list[Path] = []

    def _record_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr(installed_cards, "_sha256_file", _record_hash)
    cache = VerifiedDetachedInstalledCardCache()

    assert (
        read_installed_card_with_fallback(
            artifact,
            fallback_root=fallback_root,
            verified_detached_cache=cache,
        )
        == record
    )
    first_hash_count = len(hashed_paths)
    assert first_hash_count == len(record.files)
    assert (
        read_installed_card_with_fallback(
            artifact,
            fallback_root=fallback_root,
            verified_detached_cache=cache,
        )
        == record
    )
    assert len(hashed_paths) == first_hash_count

    (artifact / "model.safetensors").write_bytes(b"WEIGHTS")

    assert (
        read_installed_card_with_fallback(
            artifact,
            fallback_root=fallback_root,
            verified_detached_cache=cache,
        )
        is None
    )
    assert len(hashed_paths) > first_hash_count


async def test_existing_companion_store_entry_retains_full_owner_card(
    tmp_path: Path,
) -> None:
    payload = _card().model_dump(mode="json")
    payload["runtime"] = {
        "mtp_sidecar_repo": "org/model-mtp",
        "mtp_sidecar_revision": "b" * 40,
        "mtp_heads": True,
    }
    owner = ModelCard.model_validate(payload)
    store = ModelStore(tmp_path)
    companion = tmp_path / "org--model-mtp"
    companion.mkdir()
    (companion / "mtp.safetensors").write_bytes(b"mtp")
    store.register_model(
        "org/model-mtp",
        companion,
        ["mtp.safetensors"],
        3,
        source_revision="b" * 40,
    )

    status = await store.request_download(
        "org/model-mtp",
        source_revision="b" * 40,
        model_card=owner,
        artifact_role="mtp_sidecar",
        owner_model_id=str(owner.model_id),
        owner_card_id=owner.registry_card_id,
    )

    entry = store.get_entry("org/model-mtp")
    assert status.status == "complete"
    assert entry is not None
    assert entry.installed_card is not None
    assert entry.installed_card.model_card == owner
    assert entry.installed_card.artifact_role == "mtp_sidecar"
    assert entry.installed_card.owner_card_id == owner.registry_card_id


def test_corrupt_registry_rebuilds_from_installed_sidecar(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    store_root.mkdir()
    artifact = _artifact(store_root)
    record = build_installed_card_record(artifact, _card())
    write_installed_card(artifact, record)
    (store_root / "registry.json").write_text("{torn")

    entry = ModelStore(store_root).get_entry(str(record.model_card.model_id))

    assert entry is not None
    assert entry.installed_card == record
    index = StoreRegistryIndex.model_validate_json(
        (store_root / "registry.json").read_bytes(), strict=False
    )
    assert index.schema_version == 1


@pytest.mark.anyio
async def test_running_store_adopts_new_canonical_sidecar_in_place(
    tmp_path: Path,
) -> None:
    """A pre-upgrade canonical entry is not imported back into itself."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    store = ModelStore(store_root)
    artifact = _artifact(store_root)
    store.register_model(
        "org/model",
        artifact,
        [
            str(path.relative_to(artifact))
            for path in artifact.rglob("*")
            if path.is_file()
        ],
        sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file()),
        source_revision="a" * 40,
    )
    assert store.get_entry("org/model") is not None
    assert store.get_entry("org/model").installed_card is None  # type: ignore[union-attr]

    record = build_installed_card_record(artifact, _card())
    write_installed_card(artifact, record)

    await store.recover_installed_cards_serialized()

    recovered = store.get_entry("org/model")
    assert recovered is not None
    assert recovered.store_path == artifact.name
    assert recovered.installed_card == record


def test_registry_rebuild_ignores_incomplete_installed_artifact(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    store_root.mkdir()
    artifact = _artifact(store_root)
    record = build_installed_card_record(artifact, _card())
    write_installed_card(artifact, record)
    (artifact / "model.safetensors").unlink()

    entry = ModelStore(store_root).get_entry(str(record.model_card.model_id))

    assert entry is None


def test_registry_rebuild_honors_operator_deletion_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed byte cleanup cannot resurrect an operator-deleted alias."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    artifact = _artifact(store_root)
    record = build_installed_card_record(artifact, _card())
    write_installed_card(artifact, record)
    store = ModelStore(store_root)
    store.register_model(
        "org/model",
        artifact,
        [entry.path for entry in record.files],
        sum(entry.size_bytes for entry in record.files),
        installed_card=record,
    )

    def retain_directory(_path: str | Path, ignore_errors: bool = False) -> None:
        del ignore_errors

    monkeypatch.setattr("shutil.rmtree", retain_directory)

    assert store.delete_model("org/model")
    assert artifact.is_dir()
    assert ModelStore(store_root).get_entry("org/model") is None


def test_registry_rebuild_removes_corrupted_indexed_generation(
    tmp_path: Path,
) -> None:
    """A stale index cannot hide a healthy peer from reconciliation."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    store = ModelStore(store_root)
    artifact = _artifact(store_root)
    record = build_installed_card_record(artifact, _card())
    write_installed_card(artifact, record)
    store.register_model(
        "org/model",
        artifact,
        [entry.path for entry in record.files],
        sum(entry.size_bytes for entry in record.files),
        source_revision=record.artifact_revision,
        installed_card=record,
    )
    (artifact / "model.safetensors").write_bytes(b"truncated")

    recovered = ModelStore(store_root)

    assert recovered.get_entry("org/model") is None
    index = StoreRegistryIndex.model_validate_json(
        (store_root / "registry.json").read_bytes(), strict=False
    )
    assert "org/model" not in index.entries


def test_registry_rebuild_preserves_previously_selected_generation(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    store = ModelStore(store_root)
    store_root.mkdir()
    current_path = store_root / "a-current"
    stale_path = store_root / "z-stale"
    for artifact in (current_path, stale_path):
        artifact.mkdir()
        (artifact / "model.safetensors").write_bytes(artifact.name.encode())
        (artifact / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    current_card = _card()
    stale_card = current_card.model_copy(
        update={"registry_card_id": f"card_{'b' * 52}"}
    )
    current_record = build_installed_card_record(current_path, current_card)
    stale_record = build_installed_card_record(stale_path, stale_card)
    write_installed_card(current_path, current_record)
    write_installed_card(stale_path, stale_record)
    store.register_model(
        "org/model",
        current_path,
        [entry.path for entry in current_record.files],
        sum(entry.size_bytes for entry in current_record.files),
        source_revision=current_card.source_revision,
        installed_card=current_record,
    )

    recovered = ModelStore(store_root).get_entry("org/model")

    assert recovered is not None
    assert recovered.store_path == "a-current"
    assert recovered.installed_card == current_record


async def test_registry_rebuild_reselects_current_signed_generation(
    tmp_path: Path,
) -> None:
    """Post-TUF recovery replaces a deterministic pre-registry selection."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    current_path = store_root / "z-current"
    stale_path = store_root / "a-stale"
    for artifact in (current_path, stale_path):
        artifact.mkdir()
        (artifact / "model.safetensors").write_bytes(artifact.name.encode())
        (artifact / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    current_card = _card()
    stale_card = current_card.model_copy(
        update={"registry_card_id": f"card_{'b' * 52}"}
    )
    current_record = build_installed_card_record(current_path, current_card)
    stale_record = build_installed_card_record(stale_path, stale_card)
    write_installed_card(current_path, current_record)
    write_installed_card(stale_path, stale_record)

    recovered = ModelStore(store_root)
    initial = recovered.get_entry("org/model")
    assert initial is not None
    assert initial.installed_card == stale_record

    await recovered.refresh_recovered_generations([current_card])

    selected = recovered.get_entry("org/model")
    assert selected is not None
    assert selected.store_path == "z-current"
    assert selected.installed_card == current_record


async def test_registry_rebuild_reselects_companion_for_current_owner_card(
    tmp_path: Path,
) -> None:
    """Companion recovery ranks generations under their owning base alias."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    current_path = store_root / "z-current-sidecar"
    stale_path = store_root / "a-stale-sidecar"
    for artifact in (current_path, stale_path):
        artifact.mkdir()
        (artifact / "mtp.safetensors").write_bytes(artifact.name.encode())
    current_card = _card()
    stale_card = current_card.model_copy(
        update={"registry_card_id": f"card_{'b' * 52}"}
    )
    current_record = build_installed_card_record(
        current_path,
        current_card,
        artifact_role="mtp_sidecar",
        artifact_model_id="org/model-mtp",
        owner_model_id="org/model",
        owner_card_id=current_card.registry_card_id,
        artifact_repository="org/model-mtp",
        artifact_revision="c" * 40,
    )
    stale_record = build_installed_card_record(
        stale_path,
        stale_card,
        artifact_role="mtp_sidecar",
        artifact_model_id="org/model-mtp",
        owner_model_id="org/model",
        owner_card_id=stale_card.registry_card_id,
        artifact_repository="org/model-mtp",
        artifact_revision="d" * 40,
    )
    write_installed_card(current_path, current_record)
    write_installed_card(stale_path, stale_record)

    recovered = ModelStore(store_root)
    initial = recovered.get_entry("org/model-mtp")
    assert initial is not None
    assert initial.installed_card == stale_record

    await recovered.refresh_recovered_generations([current_card])

    selected = recovered.get_entry("org/model-mtp")
    assert selected is not None
    assert selected.store_path == "z-current-sidecar"
    assert selected.installed_card == current_record


async def test_registry_refresh_waits_for_publication_lock(tmp_path: Path) -> None:
    """Post-TUF recovery cannot race a download, import, or deletion publish."""

    store = ModelStore(tmp_path)
    await store._download_transfer_lock.acquire()
    refresh = asyncio.create_task(store.refresh_recovered_generations([]))
    await asyncio.sleep(0)

    assert not refresh.done()

    store._download_transfer_lock.release()
    await refresh


async def test_completed_download_refreshes_card_without_rehashing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ModelStore(tmp_path)
    artifact = _artifact(tmp_path)
    old_card = _card()
    old_record = build_installed_card_record(artifact, old_card)
    write_installed_card(artifact, old_record)
    store.register_model(
        "org/model",
        artifact,
        [entry.path for entry in old_record.files],
        sum(entry.size_bytes for entry in old_record.files),
        source_revision=old_card.source_revision,
        source_repository=str(old_card.artifact_repository),
        installed_card=old_record,
    )
    store._active_downloads["org/model"] = StoreDownloadStatus(  # noqa: SLF001
        model_id="org/model",
        source_revision=old_card.source_revision,
        source_repository=str(old_card.artifact_repository),
        status="complete",
        progress=1.0,
        model_card=old_card,
    )
    new_card = old_card.model_copy(update={"registry_card_id": f"card_{'b' * 52}"})

    def reject_rehash(_directory: Path) -> object:
        raise AssertionError("metadata-only refresh rehashed artifact bytes")

    monkeypatch.setattr(installed_cards, "build_file_manifest", reject_rehash)

    status = await store.request_download(
        "org/model",
        source_revision=new_card.source_revision,
        source_repository=str(new_card.artifact_repository),
        model_card=new_card,
    )

    refreshed = store.get_entry("org/model")
    assert status.model_card == new_card
    assert refreshed is not None and refreshed.installed_card is not None
    assert refreshed.installed_card.model_card == new_card
    assert refreshed.installed_card.files == old_record.files


async def test_completed_legacy_import_with_matching_revision_becomes_verified(
    tmp_path: Path,
) -> None:
    """A trusted card request upgrades exact marked bytes without downloading."""

    store = ModelStore(tmp_path / "store")
    artifact = tmp_path / "store" / "org--model"
    artifact.mkdir(parents=True)
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors").write_bytes(b"weights")
    card = _card()
    legacy = build_installed_card_record(artifact, card)
    write_installed_card(artifact, legacy)
    (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")
    store.register_model(
        "org/model",
        artifact,
        [entry.path for entry in legacy.files],
        sum(entry.size_bytes for entry in legacy.files),
        source_revision=card.source_revision,
        source_repository=str(card.artifact_repository),
        installed_card=legacy,
    )

    status = await store.request_download(
        "org/model",
        source_revision=card.source_revision,
        source_repository=str(card.artifact_repository),
        model_card=card,
    )

    refreshed = store.get_entry("org/model")
    assert status.status == "complete"
    assert refreshed is not None and refreshed.installed_card is not None
    assert refreshed.installed_card.verification == "registry_verified"
    assert refreshed.installed_card.installed_identity == card.registry_card_id
    assert refreshed.installed_card.manifest_sha256 == legacy.manifest_sha256
