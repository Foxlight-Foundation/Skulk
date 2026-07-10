"""Store-client download requests preserve a selected GGUF file."""

from types import TracebackType
from typing import Self

import pytest

from skulk.store import model_store_client
from skulk.store.model_store_client import ModelStoreClient


class _Response:
    status = 200

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    async def json(self) -> object:
        return {"status": "pending"}


class _Session:
    def __init__(self) -> None:
        self.requests: list[tuple[str, object | None]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def post(self, url: str, *, json: object | None = None) -> _Response:
        self.requests.append((url, json))
        return _Response()


@pytest.mark.anyio
async def test_store_client_sends_requested_gguf_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()

    def fake_session_factory(**_kwargs: object) -> _Session:
        return session

    monkeypatch.setattr(
        model_store_client,
        "create_http_session",
        fake_session_factory,
    )
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    result = await client.request_store_download(
        "org/model",
        gguf_file="model-IQ3_XXS.gguf",
    )

    assert result == {"status": "pending"}
    assert session.requests == [
        (
            "http://store.local:58080/models/org%2Fmodel/download",
            {"gguf_file": "model-IQ3_XXS.gguf"},
        )
    ]
