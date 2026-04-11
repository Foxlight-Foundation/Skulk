"""Tests for static browser-tool helpers."""

from dataclasses import dataclass

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
