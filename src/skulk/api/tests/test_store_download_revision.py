# pyright: reportPrivateUsage=false
"""Model-store API requests preserve qualified card revisions."""

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from skulk.api import main as api_main
from skulk.api.main import API, StoreDownloadRequest
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.store.model_store_client import ModelStoreClient
from skulk.utils.channels import channel

_MODEL_ID = "google/gemma-4-31B-it-qat-q4_0-gguf"
_QUALIFIED_REVISION = "3374b395f6a01379f0dd4997b37aacaab77a3596"


class _RecordingStoreClient:
    def __init__(self) -> None:
        self.requests: list[
            tuple[str, str | None, str | None, str | None, str | None]
        ] = []
        self.cancelled_models: list[str] = []
        self.cancellation_succeeds = True

    async def request_store_download(
        self,
        model_id: str,
        gguf_file: str | None = None,
        source_revision: str | None = None,
        source_repository: str | None = None,
        registry_card_id: str | None = None,
    ) -> dict[str, object]:
        self.requests.append(
            (
                model_id,
                gguf_file,
                source_revision,
                source_repository,
                registry_card_id,
            )
        )
        return {"status": "pending"}

    async def cancel_store_download(self, model_id: str) -> bool:
        self.cancelled_models.append(model_id)
        return self.cancellation_succeeds


def _build_api() -> API:
    """Create a minimal API instance for store route matching."""

    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("store-cancel-test-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


async def test_store_download_inherits_bundled_card_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def qualified_card(_model_id: ModelId) -> SimpleNamespace:
        return SimpleNamespace(
            gguf_file="gemma-4-31B_q4_0-it.gguf",
            source_revision=_QUALIFIED_REVISION,
            artifact_repository=ModelId(_MODEL_ID),
            registry_card_id=None,
        )

    monkeypatch.setattr(
        api_main,
        "get_card",
        qualified_card,
    )
    store_client = _RecordingStoreClient()
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    await api.request_store_download(_MODEL_ID)

    assert store_client.requests == [
        (
            _MODEL_ID,
            "gemma-4-31B_q4_0-it.gguf",
            _QUALIFIED_REVISION,
            _MODEL_ID,
            None,
        )
    ]


async def test_store_download_populates_card_cache_before_inheriting_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = SimpleNamespace(
        gguf_file="gemma-4-31B_q4_0-it.gguf",
        source_revision=_QUALIFIED_REVISION,
        artifact_repository=ModelId(_MODEL_ID),
        registry_card_id=None,
    )
    lookups = 0

    def cold_then_warm(_model_id: ModelId) -> SimpleNamespace | None:
        nonlocal lookups
        lookups += 1
        return None if lookups == 1 else card

    async def populate_cards() -> list[object]:
        return []

    monkeypatch.setattr(api_main, "get_card", cold_then_warm)
    monkeypatch.setattr(api_main, "get_model_cards", populate_cards)
    store_client = _RecordingStoreClient()
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    await api.request_store_download(_MODEL_ID)

    assert lookups == 2
    assert store_client.requests == [
        (
            _MODEL_ID,
            "gemma-4-31B_q4_0-it.gguf",
            _QUALIFIED_REVISION,
            _MODEL_ID,
            None,
        )
    ]


async def test_store_download_selects_current_registry_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed generation does not hide the current downloadable card."""

    installed = ModelCard(
        model_id=ModelId(_MODEL_ID),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="old.gguf",
        source_revision="a" * 40,
        registry_card_id=f"card_{'a' * 52}",
        registry_snapshot_id="snapshot_1_old",
        registry_provenance="foxlight",
    )
    current = ModelCard(
        model_id=ModelId(_MODEL_ID),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="current.gguf",
        source_revision="b" * 40,
        registry_card_id=f"card_{'b' * 52}",
        registry_snapshot_id="snapshot_2_current",
        registry_provenance="foxlight",
    )

    def installed_card(_model_id: ModelId) -> ModelCard:
        return installed

    def current_card(_model_id: ModelId) -> ModelCard:
        return current

    monkeypatch.setattr(api_main, "get_card", installed_card)
    monkeypatch.setattr(
        api_main,
        "get_current_registry_card",
        current_card,
    )
    store_client = _RecordingStoreClient()
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    await api.request_store_download(
        _MODEL_ID,
        StoreDownloadRequest(registry_card_id=current.registry_card_id),
    )

    assert store_client.requests == [
        (
            _MODEL_ID,
            "current.gguf",
            "b" * 40,
            _MODEL_ID,
            current.registry_card_id,
        )
    ]


async def test_store_download_cancellation_uses_the_canonical_store_client() -> None:
    store_client = _RecordingStoreClient()
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    response = await api.cancel_store_download(_MODEL_ID)

    assert response.status_code == 200
    assert store_client.cancelled_models == [_MODEL_ID]


async def test_store_download_cancellation_rejects_inactive_work() -> None:
    store_client = _RecordingStoreClient()
    store_client.cancellation_succeeds = False
    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    with pytest.raises(HTTPException) as raised:
        await api.cancel_store_download(_MODEL_ID)

    assert raised.value.status_code == 409


def test_store_download_cancellation_route_precedes_model_deletion() -> None:
    store_client = _RecordingStoreClient()
    api = _build_api()
    api._store_client = cast(ModelStoreClient, cast(object, store_client))

    response = TestClient(api.app).delete(
        "/store/models/google/gemma-4-31B-it-qat-q4_0-gguf/download"
    )

    assert response.status_code == 200
    assert store_client.cancelled_models == [_MODEL_ID]
