# pyright: reportPrivateUsage=false
"""Cluster model listings project authoritative installed-card state."""

from pathlib import Path
from typing import cast

import pytest

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import (
    InstalledCardRecord,
    build_installed_card_record,
)
from skulk.store.model_store_client import ModelStoreClient


class _RegistryStoreClient:
    """Minimal store client returning a controlled registry snapshot."""

    def __init__(self, entries: list[dict[str, object]]) -> None:
        """Retain the entries returned by ``fetch_registry``."""

        self.entries = entries
        self.fetch_count = 0

    async def fetch_registry(self) -> list[dict[str, object]]:
        """Return one synthetic authoritative-store snapshot."""

        self.fetch_count += 1
        return self.entries


def _card(identity_character: str) -> ModelCard:
    """Create one immutable signed card for the projection tests."""

    return ModelCard(
        model_id=ModelId("example/model@q4"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision=identity_character * 40,
        registry_card_id=f"card_{identity_character * 52}",
        registry_snapshot_id="snapshot_test",
        registry_provenance="foxlight",
        quantization="Q4_K_M",
    )


def _installed_record(tmp_path: Path, card: ModelCard) -> InstalledCardRecord:
    """Build a revision-verified installed record for ``card``."""

    assert card.registry_card_id is not None
    artifact = tmp_path / card.registry_card_id
    artifact.mkdir()
    (artifact / "model.gguf").write_bytes(b"weights")
    assert card.source_revision is not None
    (artifact / ".skulk-source-revision").write_text(
        f"{card.source_revision}\n",
        encoding="utf-8",
    )
    return build_installed_card_record(artifact, card, artifact_format="gguf")


def _custom_installed_record(tmp_path: Path) -> InstalledCardRecord:
    """Build a custom installed record sharing the registry test alias."""

    card = ModelCard(
        model_id=ModelId("example/model@q4"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=True,
        quantization="operator",
    )
    artifact = tmp_path / "custom"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}", encoding="utf-8")
    return build_installed_card_record(artifact, card, artifact_format="custom")


def _configure_model_list_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    card: ModelCard,
    local_record: InstalledCardRecord | None,
) -> None:
    """Install deterministic catalog and local-record effects for one test."""

    async def model_cards() -> list[ModelCard]:
        return [card]

    def cluster_approvals(_api: API) -> frozenset[str]:
        return frozenset()

    def intelligent_fabric_enabled(_api: API) -> bool:
        return False

    def local_record_lookup(_model_id: ModelId) -> InstalledCardRecord | None:
        return local_record

    def current_registry_card_id(_model_id: ModelId) -> str | None:
        return card.registry_card_id

    def current_registry_card(_model_id: ModelId) -> ModelCard:
        return card

    def model_advisories(_card: ModelCard) -> tuple[()]:
        return ()

    monkeypatch.setattr(api_main, "get_model_cards", model_cards)
    monkeypatch.setattr(api_main, "get_installed_card_record", local_record_lookup)
    monkeypatch.setattr(
        api_main,
        "get_current_registry_card_id",
        current_registry_card_id,
    )
    monkeypatch.setattr(
        api_main,
        "get_current_registry_card",
        current_registry_card,
    )
    monkeypatch.setattr(api_main, "get_model_advisories", model_advisories)
    monkeypatch.setattr(API, "_cluster_remote_code_approvals", cluster_approvals)
    monkeypatch.setattr(
        API,
        "_intelligent_fabric_enabled",
        intelligent_fabric_enabled,
    )


async def test_models_use_cluster_store_installed_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every API node should expose the central store's active generation."""

    card = _card("a")
    record = _installed_record(tmp_path, card)
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(card.model_id),
                "installed_card": record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(monkeypatch, card=card, local_record=None)
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    response = await api.get_models(status=None)

    assert store_client.fetch_count == 1
    assert len(response.data) == 1
    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == card.registry_card_id
    assert response.data[0].installed_verification == "registry_verified"
    assert response.data[0].update_available is False


async def test_models_fall_back_to_local_installed_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing store record must not erase a usable local generation."""

    installed_card = _card("a")
    current_card = _card("b")
    local_record = _installed_record(tmp_path, installed_card)
    store_client = _RegistryStoreClient([])
    _configure_model_list_test(
        monkeypatch,
        card=current_card,
        local_record=local_record,
    )
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    response = await api.get_models(status=None)

    assert store_client.fetch_count == 1
    assert len(response.data) == 1
    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == installed_card.registry_card_id
    assert response.data[0].installed_verification == "registry_verified"
    assert response.data[0].current_registry_identity == current_card.registry_card_id
    assert response.data[0].update_available is True


async def test_custom_card_keeps_local_installed_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A same-alias store card must not override an operator's custom card."""

    registry_card = _card("a")
    store_record = _installed_record(tmp_path, registry_card)
    local_record = _custom_installed_record(tmp_path)
    custom_card = local_record.model_card
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(registry_card.model_id),
                "installed_card": store_record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        card=custom_card,
        local_record=local_record,
    )
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    response = await api.get_models(status=None)

    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == local_record.installed_identity
    assert response.data[0].installed_verification == "custom"


def test_store_projection_rejects_mismatched_alias(
    tmp_path: Path,
) -> None:
    """An index key cannot assert installation for a different signed card alias."""

    card = _card("a")
    record = _installed_record(tmp_path, card)

    records = API._store_installed_records(
        [
            {
                "model_id": "example/other@q4",
                "installed_card": record.model_dump(mode="json"),
            }
        ]
    )

    assert records == {}
