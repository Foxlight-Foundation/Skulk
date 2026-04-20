"""Tests for static browser-tool helpers."""

import socket
from dataclasses import dataclass
from typing import Any

import pytest

from exo.tools.web_search import DefaultBrowserToolProvider


@dataclass(frozen=True)
class _FetchedUrlFixture:
    url: str
    final_url: str
    status_code: int
    content_type: str | None
    body: bytes
    encoding: str | None


@dataclass(frozen=True)
class _FakeStreamResponse:
    url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    encoding: str | None = "utf-8"

    @property
    def is_redirect(self) -> bool:
        return 300 <= self.status_code < 400

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def aiter_bytes(self):
        yield self.body


class _FakeAsyncClient:
    def __init__(self, responses: list[_FakeStreamResponse]) -> None:
        self._responses = responses
        self.requested_urls: list[str] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def stream(self, method: str, url: str) -> _FakeStreamResponse:
        assert method == "GET"
        self.requested_urls.append(url)
        if not self._responses:
            raise AssertionError(f"No fake response queued for {url}")
        return self._responses.pop(0)


class _CapturingAsyncClientFactory:
    def __init__(self, client: _FakeAsyncClient) -> None:
        self.client = client
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _FakeAsyncClient:
        self.calls.append(kwargs)
        return self.client


@pytest.mark.anyio
async def test_open_url_returns_redirected_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()

    async def fake_fetch_url(_url: str) -> _FetchedUrlFixture:
        return _FetchedUrlFixture(
            url="https://example.com/start",
            final_url="https://example.com/final",
            status_code=200,
            content_type="text/html",
            body=b"<html><head><title>Example title</title></head><body>Hello</body></html>",
            encoding="utf-8",
        )

    monkeypatch.setattr(provider, "_fetch_url", fake_fetch_url)

    response = await provider.open_url("https://example.com/start")

    assert response.final_url == "https://example.com/final"
    assert response.title == "Example title"
    assert response.status_code == 200
    assert response.content_type == "text/html"


@pytest.mark.anyio
async def test_extract_page_returns_bounded_readable_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()

    html_body = (
        "<html><head><title>Readable page</title></head><body>"
        "<main><article><h1>Heading</h1><p>"
        "This is a readable paragraph with enough content to exercise truncation."
        "</p><p>Second paragraph with more useful text.</p></article></main>"
        "</body></html>"
    ).encode("utf-8")

    async def fake_fetch_url(_url: str) -> _FetchedUrlFixture:
        return _FetchedUrlFixture(
            url="https://example.com/start",
            final_url="https://example.com/final",
            status_code=200,
            content_type="text/html",
            body=html_body,
            encoding="utf-8",
        )

    monkeypatch.setattr(provider, "_fetch_url", fake_fetch_url)

    response = await provider.extract_page("https://example.com/start", max_chars=40)

    assert response.title == "Readable page"
    assert response.final_url == "https://example.com/final"
    assert response.text.startswith("Heading")
    assert response.truncated is True
    assert len(response.text) == 40


@pytest.mark.anyio
async def test_extract_page_pretty_prints_json_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()

    async def fake_fetch_url(_url: str) -> _FetchedUrlFixture:
        return _FetchedUrlFixture(
            url="https://example.com/data.json",
            final_url="https://example.com/data.json",
            status_code=200,
            content_type="application/json",
            body=b'{"hello":"world","count":2}',
            encoding="utf-8",
        )

    monkeypatch.setattr(provider, "_fetch_url", fake_fetch_url)

    response = await provider.extract_page(
        "https://example.com/data.json", max_chars=500
    )

    assert '"hello": "world"' in response.text
    assert response.truncated is False


@pytest.mark.anyio
async def test_browser_tools_reject_unsupported_schemes() -> None:
    provider = DefaultBrowserToolProvider()

    with pytest.raises(ValueError, match="http:// and https://"):
        await provider.open_url("file:///tmp/nope")

    with pytest.raises(ValueError, match="http:// and https://"):
        await provider.extract_page("ftp://example.com/nope", max_chars=500)


@pytest.mark.anyio
async def test_browser_tools_reject_private_ip_literal_targets() -> None:
    provider = DefaultBrowserToolProvider()

    with pytest.raises(
        ValueError, match="Private, loopback, and link-local targets"
    ):
        await provider.open_url("http://127.0.0.1:8080/private")


@pytest.mark.anyio
async def test_browser_tools_reject_loopback_dns_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()

    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[Any, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", 443),
            )
        ]

    monkeypatch.setattr("exo.tools.web_search.socket.getaddrinfo", fake_getaddrinfo)

    with pytest.raises(
        ValueError, match="Private, loopback, and link-local targets"
    ):
        await provider.open_url("https://example.com/internal")


@pytest.mark.anyio
async def test_browser_tools_fail_closed_when_dns_validation_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()

    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[Any, ...]]:
        raise socket.gaierror("lookup failed")

    monkeypatch.setattr("exo.tools.web_search.socket.getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError, match="Could not resolve URL host during validation"):
        await provider.open_url("https://example.com/unresolved")


@pytest.mark.anyio
async def test_fetch_url_keeps_non_2xx_status_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()
    fake_client = _FakeAsyncClient(
        [
            _FakeStreamResponse(
                url="https://example.com/missing",
                status_code=404,
                headers={"content-type": "text/html"},
                body=b"<html><head><title>Missing</title></head><body>not found</body></html>",
            )
        ]
    )

    def fake_async_client(**_kwargs: object) -> _FakeAsyncClient:
        return fake_client

    monkeypatch.setattr(
        "exo.tools.web_search.httpx.AsyncClient",
        fake_async_client,
    )

    response = await provider.open_url("https://example.com/missing")

    assert response.status_code == 404
    assert response.title == "Missing"
    assert response.final_url == "https://example.com/missing"


@pytest.mark.anyio
async def test_fetch_url_disables_proxy_env_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()
    fake_client = _FakeAsyncClient(
        [
            _FakeStreamResponse(
                url="https://example.com/ok",
                status_code=200,
                headers={"content-type": "text/html"},
                body=b"<html><body>ok</body></html>",
            )
        ]
    )
    capturing_factory = _CapturingAsyncClientFactory(fake_client)

    monkeypatch.setattr(
        "exo.tools.web_search.httpx.AsyncClient",
        capturing_factory,
    )

    response = await provider.open_url("https://example.com/ok")

    assert response.status_code == 200
    assert capturing_factory.calls
    assert capturing_factory.calls[0]["trust_env"] is False


@pytest.mark.anyio
async def test_fetch_url_rejects_private_redirect_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DefaultBrowserToolProvider()
    fake_client = _FakeAsyncClient(
        [
            _FakeStreamResponse(
                url="https://example.com/start",
                status_code=302,
                headers={"location": "http://127.0.0.1:8080/admin"},
                body=b"",
            )
        ]
    )

    def fake_async_client(**_kwargs: object) -> _FakeAsyncClient:
        return fake_client

    monkeypatch.setattr(
        "exo.tools.web_search.httpx.AsyncClient",
        fake_async_client,
    )

    with pytest.raises(
        ValueError, match="Private, loopback, and link-local targets"
    ):
        await provider.open_url("https://example.com/start")

    assert fake_client.requested_urls == ["https://example.com/start"]
