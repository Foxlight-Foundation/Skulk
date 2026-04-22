"""Tests for the generic web-search tool endpoint."""

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types import (
    ExtractPageToolRequest,
    OpenUrlToolRequest,
    WebSearchToolRequest,
)
from exo.api.types.api import (
    ExtractPageToolResponse,
    OpenUrlToolResponse,
    WebSearchResult,
)


class _FakeProvider:
    @property
    def provider_name(self) -> str:
        return "fake-search"

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title=f"Result for {query}",
                url="https://example.com/result",
                snippet=f"top_k={top_k}",
            )
        ]

    async def open_url(self, url: str) -> OpenUrlToolResponse:
        return OpenUrlToolResponse(
            url=url,
            final_url="https://example.com/final",
            title="Example title",
            status_code=200,
            content_type="text/html",
            provider=self.provider_name,
        )

    async def extract_page(
        self, url: str, *, max_chars: int
    ) -> ExtractPageToolResponse:
        return ExtractPageToolResponse(
            url=url,
            final_url="https://example.com/final",
            title="Example title",
            text=f"max_chars={max_chars}",
            truncated=False,
            provider=self.provider_name,
        )


class _FailingProvider:
    @property
    def provider_name(self) -> str:
        return "failing-search"

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        raise RuntimeError(f"provider blew up for {query} / {top_k}")

    async def open_url(self, url: str) -> OpenUrlToolResponse:
        raise RuntimeError(f"open_url blew up for {url}")

    async def extract_page(
        self, url: str, *, max_chars: int
    ) -> ExtractPageToolResponse:
        raise RuntimeError(f"extract_page blew up for {url} / {max_chars}")


class _InvalidUrlProvider:
    @property
    def provider_name(self) -> str:
        return "invalid"

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        return []

    async def open_url(self, url: str) -> OpenUrlToolResponse:
        raise ValueError(f"bad url: {url}")

    async def extract_page(
        self, url: str, *, max_chars: int
    ) -> ExtractPageToolResponse:
        raise ValueError(f"bad url: {url}")


@pytest.mark.anyio
async def test_web_search_returns_structured_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("exo.api.main.default_browser_tool_provider", lambda: _FakeProvider())

    api = object.__new__(API)
    response = await api.web_search(WebSearchToolRequest(query="skulk", top_k=3))

    assert response.provider == "fake-search"
    assert response.query == "skulk"
    assert len(response.results) == 1
    assert response.results[0].snippet == "top_k=3"


@pytest.mark.anyio
async def test_web_search_wraps_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo.api.main.default_browser_tool_provider", lambda: _FailingProvider()
    )

    api = object.__new__(API)

    with pytest.raises(HTTPException) as exc_info:
        await api.web_search(WebSearchToolRequest(query="skulk", top_k=3))

    assert exc_info.value.status_code == 502
    assert "Web search failed" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_open_url_returns_structured_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("exo.api.main.default_browser_tool_provider", lambda: _FakeProvider())

    api = object.__new__(API)
    response = await api.open_url(OpenUrlToolRequest(url="https://example.com/start"))

    assert response.final_url == "https://example.com/final"
    assert response.status_code == 200
    assert response.content_type == "text/html"
    assert response.provider == "fake-search"


@pytest.mark.anyio
async def test_extract_page_returns_structured_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("exo.api.main.default_browser_tool_provider", lambda: _FakeProvider())

    api = object.__new__(API)
    response = await api.extract_page(
        ExtractPageToolRequest(url="https://example.com/start", max_chars=2048)
    )

    assert response.final_url == "https://example.com/final"
    assert response.text == "max_chars=2048"
    assert response.truncated is False
    assert response.provider == "fake-search"


@pytest.mark.anyio
async def test_open_url_wraps_invalid_urls_as_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo.api.main.default_browser_tool_provider", lambda: _InvalidUrlProvider()
    )

    api = object.__new__(API)

    with pytest.raises(HTTPException) as exc_info:
        await api.open_url(OpenUrlToolRequest(url="file:///tmp/nope"))

    assert exc_info.value.status_code == 400
    assert "bad url" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_extract_page_wraps_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo.api.main.default_browser_tool_provider", lambda: _FailingProvider()
    )

    api = object.__new__(API)

    with pytest.raises(HTTPException) as exc_info:
        await api.extract_page(
            ExtractPageToolRequest(url="https://example.com/start", max_chars=2048)
        )

    assert exc_info.value.status_code == 502
    assert "Extract page failed" in str(exc_info.value.detail)
