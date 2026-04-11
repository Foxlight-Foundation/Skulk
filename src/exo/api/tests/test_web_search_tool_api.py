"""Tests for the generic web-search tool endpoint."""

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types import WebSearchToolRequest
from exo.api.types.api import WebSearchResult


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


class _FailingProvider:
    @property
    def provider_name(self) -> str:
        return "failing-search"

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        raise RuntimeError(f"provider blew up for {query} / {top_k}")


@pytest.mark.anyio
async def test_web_search_returns_structured_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("exo.api.main.default_web_search_provider", lambda: _FakeProvider())

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
        "exo.api.main.default_web_search_provider", lambda: _FailingProvider()
    )

    api = object.__new__(API)

    with pytest.raises(HTTPException) as exc_info:
        await api.web_search(WebSearchToolRequest(query="skulk", top_k=3))

    assert exc_info.value.status_code == 502
    assert "Web search failed" in str(exc_info.value.detail)
