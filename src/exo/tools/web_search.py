"""Generic web-search tool support for model-assisted browsing.

This module keeps search retrieval generic and reusable while remaining
deliberately small-scope: it returns structured search results only and does
not attempt interactive browsing, page navigation, or browser session state.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Iterable, Mapping
from typing import Protocol, Self, cast, final

from exo.api.types import WebSearchResult


class WebSearchProvider(Protocol):
    """Abstract provider contract for structured web search."""

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        """Return structured search results for one query."""
        ...

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier for API/debug output."""
        ...


class _DdgsClient(Protocol):
    """Minimal DDGS client interface used by the generic search provider."""

    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    def text(self, query: str, *, max_results: int) -> Iterable[Mapping[str, object]]: ...


class _DdgsFactory(Protocol):
    """Factory protocol for constructing DDGS client instances."""

    def __call__(self) -> _DdgsClient: ...


@final
class DdgsWebSearchProvider:
    """DDGS-backed provider using the open-source ``ddgs`` package.

    The package wraps multiple search backends behind a lightweight Python API
    and keeps Skulk out of the browser-automation business for the first
    browsing release.
    """

    @property
    def provider_name(self) -> str:
        """Return the provider identifier exposed to API clients."""

        return "ddgs"

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        """Run a web search off the event loop and normalize the results."""

        return await asyncio.to_thread(self._search_sync, query, top_k)

    def _search_sync(self, query: str, top_k: int) -> list[WebSearchResult]:
        """Execute a blocking DDGS text search and normalize the payload."""

        ddgs_module = importlib.import_module("ddgs")
        ddgs_factory = cast(_DdgsFactory, ddgs_module.DDGS)

        results: list[WebSearchResult] = []
        with ddgs_factory() as client:
            raw_result_items = client.text(query, max_results=top_k)
            raw_results = list(raw_result_items)

        for raw in raw_results:
            title = str(raw.get("title") or raw.get("headline") or "").strip()
            url = str(raw.get("href") or raw.get("url") or "").strip()
            snippet = str(raw.get("body") or raw.get("snippet") or "").strip()
            if not title or not url:
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                )
            )

        return results


def default_web_search_provider() -> WebSearchProvider:
    """Return the default provider for the generic web-search tool."""

    return DdgsWebSearchProvider()
