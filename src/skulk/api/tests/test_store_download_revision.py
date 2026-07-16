# pyright: reportPrivateUsage=false
"""Model-store API requests preserve qualified card revisions."""

from types import SimpleNamespace
from typing import cast

import pytest

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.shared.models.model_cards import ModelId
from skulk.store.model_store_client import ModelStoreClient

_MODEL_ID = "google/gemma-4-31B-it-qat-q4_0-gguf"
_QUALIFIED_REVISION = "3374b395f6a01379f0dd4997b37aacaab77a3596"


class _RecordingStoreClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str | None, str | None]] = []

    async def request_store_download(
        self,
        model_id: str,
        gguf_file: str | None = None,
        source_revision: str | None = None,
    ) -> dict[str, object]:
        self.requests.append((model_id, gguf_file, source_revision))
        return {"status": "pending"}


async def test_store_download_inherits_bundled_card_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def qualified_card(_model_id: ModelId) -> SimpleNamespace:
        return SimpleNamespace(
            gguf_file="gemma-4-31B_q4_0-it.gguf",
            source_revision=_QUALIFIED_REVISION,
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
        (_MODEL_ID, "gemma-4-31B_q4_0-it.gguf", _QUALIFIED_REVISION)
    ]


async def test_store_download_populates_card_cache_before_inheriting_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = SimpleNamespace(
        gguf_file="gemma-4-31B_q4_0-it.gguf",
        source_revision=_QUALIFIED_REVISION,
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
        (_MODEL_ID, "gemma-4-31B_q4_0-it.gguf", _QUALIFIED_REVISION)
    ]
