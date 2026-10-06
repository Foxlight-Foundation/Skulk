# pyright: reportPrivateUsage=false
"""The load-time trust check verifies the copy the runner opens."""

from pathlib import Path

import pytest

import skulk.worker.main as worker_main
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    VisionCardConfig,
)
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import (
    InstalledCardRecord,
    build_installed_card_record,
    read_installed_card,
    require_registry_installed_artifact,
    write_installed_card,
)

_OLD_CARD_ID = f"card_{'a' * 52}"
_NEW_CARD_ID = f"card_{'b' * 52}"


def _card(card_id: str = _OLD_CARD_ID) -> ModelCard:
    return ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision="a" * 40,
        registry_card_id=card_id,
        registry_snapshot_id="snapshot_1_test",
        registry_provenance="foxlight",
    )


def _staged_copy_from_previous_card(tmp_path: Path) -> Path:
    """Stage one artifact whose sidecar names the previous signed card."""
    artifact = tmp_path / "org--model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    write_installed_card(artifact, build_installed_card_record(artifact, _card()))
    return artifact


async def test_load_adopts_a_replacement_card_for_unchanged_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store host's older staged copy must not refuse a card whose bytes it holds.

    The registry replaced the card (new identity, same repository, revision and
    files). The coordinator vouched for another copy, but the runner opens this
    one, so the load check adopts the current card for it before verifying.
    """
    artifact = _staged_copy_from_previous_card(tmp_path)
    requested = _card(_NEW_CARD_ID)
    registered: list[InstalledCardRecord] = []
    monkeypatch.setattr(
        worker_main, "register_installed_card_record", registered.append
    )
    _current_catalog_card(monkeypatch, requested)

    with pytest.raises(PermissionError):
        require_registry_installed_artifact(artifact, requested)

    await worker_main._require_installed_artifact_for_load(artifact, requested)

    record = read_installed_card(artifact)
    assert record is not None
    assert record.installed_identity == _NEW_CARD_ID
    assert [entry.installed_identity for entry in registered] == [_NEW_CARD_ID]
    require_registry_installed_artifact(artifact, requested)


async def test_load_still_refuses_a_card_that_selects_new_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement card that needs bytes the copy lacks stays a trust refusal."""
    artifact = _staged_copy_from_previous_card(tmp_path)
    requested = _card(_NEW_CARD_ID).model_copy(
        update={
            "vision": VisionCardConfig(
                projector_file="mmproj-F16.gguf",
                projector_size=9,
            ),
        }
    )
    registered: list[InstalledCardRecord] = []
    monkeypatch.setattr(
        worker_main, "register_installed_card_record", registered.append
    )
    _current_catalog_card(monkeypatch, requested)

    with pytest.raises(PermissionError):
        await worker_main._require_installed_artifact_for_load(artifact, requested)

    _assert_unchanged(artifact, registered)


async def test_load_never_adopts_a_superseded_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A load stamped with an older card must not roll a copy back to it.

    The copy names one card, the catalog has since moved to another, and an
    instance placed before that move loads with a third, superseded card. The
    proof of identical bytes holds, but adopting would rewrite the copy and the
    process-wide installed record to stale truth.
    """
    artifact = _staged_copy_from_previous_card(tmp_path)
    superseded = _card(_NEW_CARD_ID)
    registered: list[InstalledCardRecord] = []
    monkeypatch.setattr(
        worker_main, "register_installed_card_record", registered.append
    )
    _current_catalog_card(monkeypatch, _card(f"card_{'c' * 52}"))

    with pytest.raises(PermissionError):
        await worker_main._require_installed_artifact_for_load(artifact, superseded)

    _assert_unchanged(artifact, registered)


async def test_load_refuses_when_the_adopted_sidecar_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed sidecar write keeps the trust refusal instead of escaping."""
    artifact = _staged_copy_from_previous_card(tmp_path)
    requested = _card(_NEW_CARD_ID)
    registered: list[InstalledCardRecord] = []
    monkeypatch.setattr(
        worker_main, "register_installed_card_record", registered.append
    )
    _current_catalog_card(monkeypatch, requested)

    def _disk_full(_directory: Path, _card: ModelCard) -> InstalledCardRecord:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(
        worker_main, "refresh_registry_installed_card_if_same_artifact", _disk_full
    )

    with pytest.raises(PermissionError):
        await worker_main._require_installed_artifact_for_load(artifact, requested)

    _assert_unchanged(artifact, registered)


def _current_catalog_card(
    monkeypatch: pytest.MonkeyPatch, current: ModelCard
) -> None:
    """Pin the signed catalog's current card for the test model."""

    def _lookup(model_id: ModelId) -> ModelCard | None:
        return current if model_id == current.model_id else None

    monkeypatch.setattr(worker_main, "get_current_registry_card", _lookup)


def _assert_unchanged(artifact: Path, registered: list[InstalledCardRecord]) -> None:
    """The copy still names its original card and nothing was registered."""

    record = read_installed_card(artifact)
    assert record is not None
    assert record.installed_identity == _OLD_CARD_ID
    assert registered == []
