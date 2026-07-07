# pyright: reportPrivateUsage=false
"""Store staging retries transient HTTP failures and resumes partial files."""

from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import pytest

from skulk.store import model_store_client
from skulk.store.model_store_client import ModelStoreClient


class _FakeContent:
    def __init__(
        self,
        chunks: list[bytes],
        error: aiohttp.ClientError | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error

    async def iter_chunked(self, _chunk_size: int) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


class _FakeFileResponse:
    def __init__(
        self,
        status: int,
        chunks: list[bytes],
        error: aiohttp.ClientError | None = None,
    ) -> None:
        self.status = status
        self.content = _FakeContent(chunks, error)

    async def __aenter__(self) -> "_FakeFileResponse":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _FakeClientSession:
    def __init__(self, factory: "_FakeClientSessionFactory") -> None:
        self._factory = factory

    async def __aenter__(self) -> "_FakeClientSession":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def get(self, _url: str, *, headers: dict[str, str]) -> _FakeFileResponse:
        self._factory.requests.append(headers.copy())
        return self._factory.responses.pop(0)


class _FakeClientSessionFactory:
    def __init__(self, responses: list[_FakeFileResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, str]] = []

    def __call__(self, *_args: object, **_kwargs: object) -> _FakeClientSession:
        return _FakeClientSession(self)


async def _no_sleep(_delay: float) -> None:
    return None


@pytest.mark.anyio
async def test_store_http_retry_covers_minute_scale_route_flaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < model_store_client._STORE_HTTP_RETRY_ATTEMPTS:
            raise aiohttp.ClientError("route temporarily unavailable")
        return "ok"

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(model_store_client.asyncio, "sleep", record_sleep)

    result = await model_store_client._retry_store_http(
        flaky_operation,
        description="availability probe for org/model",
    )

    assert result == "ok"
    assert attempts == model_store_client._STORE_HTTP_RETRY_ATTEMPTS
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]


@pytest.mark.anyio
async def test_store_file_download_retries_and_resumes_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeClientSessionFactory(
        [
            _FakeFileResponse(
                200,
                [b"ab"],
                aiohttp.ClientPayloadError("route dropped mid-transfer"),
            ),
            _FakeFileResponse(206, [b"cd"]),
        ]
    )
    monkeypatch.setattr(model_store_client.aiohttp, "ClientSession", factory)
    monkeypatch.setattr(model_store_client.asyncio, "sleep", _no_sleep)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    written = await client._download_store_file(
        "org/model",
        "weights.safetensors",
        tmp_path,
        on_progress=None,
        total_bytes_offset=0,
        grand_total=4,
    )

    assert written == 4
    assert (tmp_path / "weights.safetensors").read_bytes() == b"abcd"
    assert not (tmp_path / "weights.safetensors.partial").exists()
    assert factory.requests == [{}, {"Range": "bytes=2-"}]


@pytest.mark.anyio
async def test_store_file_download_restarts_when_range_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "weights.safetensors.partial"
    partial.write_bytes(b"ab")
    factory = _FakeClientSessionFactory(
        [
            _FakeFileResponse(200, [b"abcd"]),
        ]
    )
    monkeypatch.setattr(model_store_client.aiohttp, "ClientSession", factory)
    monkeypatch.setattr(model_store_client.asyncio, "sleep", _no_sleep)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    written = await client._download_store_file(
        "org/model",
        "weights.safetensors",
        tmp_path,
        on_progress=None,
        total_bytes_offset=0,
        grand_total=4,
    )

    assert written == 4
    assert (tmp_path / "weights.safetensors").read_bytes() == b"abcd"
    assert not partial.exists()
    assert factory.requests == [{"Range": "bytes=2-"}]
