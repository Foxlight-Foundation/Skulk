"""Installed model cards remain bound to local artifact bytes."""

from pathlib import Path

import pytest

import skulk.store.installed_cards as installed_cards
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import (
    InstalledCardRecord,
    associate_installed_card,
    build_installed_card_record,
    installed_card_matches,
    read_installed_card,
    read_installed_card_with_fallback,
    write_installed_card,
    write_installed_card_with_fallback,
)
from skulk.store.model_store import ModelStore, StoreRegistryIndex


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


def test_unmarked_registry_bytes_remain_local_legacy(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, revision_marker=False)

    record = build_installed_card_record(artifact, _card())

    assert record.verification == "local_legacy"
    assert record.installed_identity.startswith("local_")
    assert record.model_card.registry_card_id is not None


def test_custom_card_keeps_custom_verification(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    record = build_installed_card_record(
        artifact,
        _card(registry=False, custom=True),
    )

    assert record.verification == "custom"


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
