"""Reusable static browser tools for model-assisted web inspection.

These helpers intentionally stay on the small, reliable side of the spectrum:
search, URL metadata inspection, and bounded text extraction. They do not
attempt JavaScript execution, browser sessions, or interactive navigation.
"""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import json
import re
import socket
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol, Self, cast, final
from urllib.parse import urljoin, urlparse

import httpx

from exo.api.types import (
    ExtractPageToolResponse,
    OpenUrlToolResponse,
    WebSearchResult,
)

_MAX_FETCH_BYTES = 1_000_000
_MAX_REDIRECTS = 5
_DEFAULT_USER_AGENT = (
    "SkulkBrowserTools/1.0 (+https://github.com/Foxlight-Foundation/Skulk)"
)
_WHITESPACE_RE = re.compile(r"\s+")
_BLOCK_TAGS = {"article", "main", "section", "div", "p", "li", "pre", "blockquote", "h1", "h2", "h3", "h4"}
_NOISE_TAGS = {"script", "style", "noscript", "svg", "canvas", "template", "header", "footer", "nav", "aside", "form"}
_PREFERRED_TAGS = {"article", "main"}


class _ReadableHtmlParser(HTMLParser):
    """Small HTML parser that extracts best-effort title and readable text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.og_title: str | None = None
        self.twitter_title: str | None = None
        self.title_text: str | None = None
        self._title_parts: list[str] = []
        self._fallback_parts: list[str] = []
        self._current_parts: list[str] = []
        self._preferred_blocks: list[str] = []
        self._body_blocks: list[str] = []
        self._skip_depth = 0
        self._preferred_depth = 0
        self._in_title = False
        self._current_target_preferred = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Track HTML structure and begin metadata/content regions."""

        attr_map = {key.lower(): value for key, value in attrs}

        if tag == "meta":
            property_name = (attr_map.get("property") or "").lower()
            meta_name = (attr_map.get("name") or "").lower()
            content = _normalize_whitespace(attr_map.get("content") or "")
            if not content:
                return
            if property_name == "og:title" and self.og_title is None:
                self.og_title = content
            elif meta_name == "twitter:title" and self.twitter_title is None:
                self.twitter_title = content
            return

        if tag in _NOISE_TAGS:
            self._flush_block()
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True
            return

        if tag in _PREFERRED_TAGS or (attr_map.get("role") or "").lower() == "main":
            self._flush_block()
            self._preferred_depth += 1

        if tag in _BLOCK_TAGS:
            self._flush_block()
            self._current_target_preferred = self._preferred_depth > 0

        if tag == "br" and self._skip_depth == 0:
            self._current_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Close metadata/content regions and flush accumulated text."""

        if tag == "title":
            self._in_title = False
            title = _normalize_whitespace("".join(self._title_parts))
            if title:
                self.title_text = title
            self._title_parts.clear()
            return

        if tag in _NOISE_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if tag in _BLOCK_TAGS:
            self._flush_block()

        if tag in _PREFERRED_TAGS:
            self._flush_block()
            if self._preferred_depth > 0:
                self._preferred_depth -= 1

    def handle_data(self, data: str) -> None:
        """Collect title text and visible body text."""

        if self._in_title:
            self._title_parts.append(data)
            return

        if self._skip_depth > 0:
            return

        self._fallback_parts.append(data)
        self._current_parts.append(data)

    def _flush_block(self) -> None:
        """Commit one visible block of text into the preferred or body buffers."""

        text = _normalize_whitespace("".join(self._current_parts))
        self._current_parts.clear()
        if not text:
            return
        target = self._preferred_blocks if self._current_target_preferred else self._body_blocks
        if not target or target[-1] != text:
            target.append(text)

    def extracted_title(self) -> str | None:
        """Return the best available title found while parsing."""

        return self.og_title or self.twitter_title or self.title_text

    def extracted_text(self) -> str:
        """Return best-effort readable text with a preferred-content fallback."""

        self._flush_block()
        preferred = "\n\n".join(self._preferred_blocks).strip()
        if len(preferred) >= 120:
            return preferred
        body = "\n\n".join(self._body_blocks).strip()
        if body:
            return body
        return _normalize_whitespace("".join(self._fallback_parts))


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while keeping words separated."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_content_type(content_type: str | None) -> str | None:
    """Strip any charset suffix from a Content-Type header."""

    if content_type is None:
        return None
    normalized = content_type.split(";", 1)[0].strip().lower()
    return normalized or None


def _validate_http_url(url: str) -> str:
    """Validate that a model-supplied URL uses HTTP or HTTPS."""

    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute http:// and https:// URLs are supported.")
    return url.strip()


def _is_blocked_ip_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether one resolved IP should be rejected for browser-tool fetches."""

    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def _validate_public_http_target(url: str) -> str:
    """Reject loopback and private-network targets before any outbound fetch."""

    validated_url = _validate_http_url(url)
    parsed = urlparse(validated_url)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("Only absolute http:// and https:// URLs are supported.")

    normalized_host = hostname.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise ValueError("Private, loopback, and link-local targets are not allowed.")

    try:
        direct_address = ipaddress.ip_address(normalized_host)
    except ValueError:
        direct_address = None

    if direct_address is not None:
        if _is_blocked_ip_address(direct_address):
            raise ValueError("Private, loopback, and link-local targets are not allowed.")
        return validated_url

    try:
        address_infos = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return validated_url

    for family, _, _, _, sockaddr in address_infos:
        raw_address = sockaddr[0]
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        resolved_address = ipaddress.ip_address(raw_address)
        if _is_blocked_ip_address(resolved_address):
            raise ValueError("Private, loopback, and link-local targets are not allowed.")

    return validated_url


def _decode_body(body: bytes, encoding: str | None) -> str:
    """Decode response bytes with a best-effort encoding strategy."""

    if not body:
        return ""
    if encoding:
        try:
            return body.decode(encoding, errors="replace")
        except LookupError:
            pass
    return body.decode("utf-8", errors="replace")


def _extract_text_from_html(body_text: str) -> tuple[str | None, str]:
    """Extract a best-effort title and readable text from one HTML payload."""

    if not body_text.strip():
        return None, ""

    parser = _ReadableHtmlParser()
    parser.feed(body_text)
    parser.close()
    return parser.extracted_title(), parser.extracted_text()


def _extract_text_by_content_type(
    *,
    content_type: str | None,
    body: bytes,
    encoding: str | None,
) -> tuple[str | None, str]:
    """Extract a title and readable text according to one response type."""

    decoded = _decode_body(body, encoding)
    if not decoded:
        return None, ""

    if content_type in {"text/html", "application/xhtml+xml"}:
        return _extract_text_from_html(decoded)

    if content_type == "application/json":
        try:
            parsed = cast(object, json.loads(decoded))
        except json.JSONDecodeError:
            return None, decoded
        return None, json.dumps(parsed, indent=2, ensure_ascii=True)

    return None, decoded


@dataclass(frozen=True)
class _FetchedUrl:
    """Internal normalized fetch result shared by URL-open and extraction tools."""

    url: str
    final_url: str
    status_code: int
    content_type: str | None
    body: bytes
    encoding: str | None


class BrowserToolProvider(Protocol):
    """Provider contract for generic dashboard/browser tools."""

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        """Return structured search results for one natural-language query."""
        ...

    async def open_url(self, url: str) -> OpenUrlToolResponse:
        """Inspect one URL and return redirect-aware metadata."""
        ...

    async def extract_page(
        self, url: str, *, max_chars: int
    ) -> ExtractPageToolResponse:
        """Fetch one URL and return bounded readable text."""
        ...

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier for API/debug output."""
        ...


class _DdgsClient(Protocol):
    """Minimal DDGS client interface used by the generic search provider."""

    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    def text(
        self, query: str, *, max_results: int
    ) -> Iterable[Mapping[str, object]]: ...


class _DdgsFactory(Protocol):
    """Factory protocol for constructing DDGS client instances."""

    def __call__(self) -> _DdgsClient: ...


@final
class DefaultBrowserToolProvider:
    """Static browser-tool provider for search, metadata inspection, and extraction."""

    def __init__(self) -> None:
        self._headers = {"User-Agent": _DEFAULT_USER_AGENT}

    @property
    def provider_name(self) -> str:
        """Return the provider identifier exposed to API clients."""

        return "ddgs+httpx"

    async def search(self, query: str, *, top_k: int) -> list[WebSearchResult]:
        """Run a web search off the event loop and normalize the results."""

        return await asyncio.to_thread(self._search_sync, query, top_k)

    async def open_url(self, url: str) -> OpenUrlToolResponse:
        """Fetch one URL and return redirect-aware metadata."""

        fetched = await self._fetch_url(url)
        title = None
        if fetched.content_type in {"text/html", "application/xhtml+xml"}:
            title, _ = _extract_text_from_html(
                _decode_body(fetched.body, fetched.encoding)
            )

        return OpenUrlToolResponse(
            url=fetched.url,
            final_url=fetched.final_url,
            title=title,
            status_code=fetched.status_code,
            content_type=fetched.content_type,
            provider=self.provider_name,
        )

    async def extract_page(
        self, url: str, *, max_chars: int
    ) -> ExtractPageToolResponse:
        """Fetch one URL and return bounded readable text."""

        fetched = await self._fetch_url(url)
        extracted_title, extracted_text = _extract_text_by_content_type(
            content_type=fetched.content_type,
            body=fetched.body,
            encoding=fetched.encoding,
        )
        normalized_text = extracted_text.strip()
        truncated = len(normalized_text) > max_chars
        bounded_text = normalized_text[:max_chars] if truncated else normalized_text

        return ExtractPageToolResponse(
            url=fetched.url,
            final_url=fetched.final_url,
            title=extracted_title,
            text=bounded_text,
            truncated=truncated,
            provider=self.provider_name,
        )

    def _search_sync(self, query: str, top_k: int) -> list[WebSearchResult]:
        """Execute a blocking DDGS text search and normalize the payload."""

        ddgs_module = importlib.import_module("ddgs")
        ddgs_factory = cast(_DdgsFactory, ddgs_module.DDGS)

        results: list[WebSearchResult] = []
        with ddgs_factory() as client:
            raw_results = list(client.text(query, max_results=top_k))

        for raw in raw_results:
            title = str(raw.get("title") or raw.get("headline") or "").strip()
            url = str(raw.get("href") or raw.get("url") or "").strip()
            snippet = str(raw.get("body") or raw.get("snippet") or "").strip()
            if not title or not url:
                continue
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))

        return results

    async def _fetch_url(self, url: str) -> _FetchedUrl:
        """Fetch one URL with redirect following and a bounded response body."""

        current_url = url
        async with httpx.AsyncClient(
            follow_redirects=False,
            headers=self._headers,
            timeout=httpx.Timeout(15.0, connect=10.0),
            trust_env=False,
        ) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                validated_url = await _validate_public_http_target(current_url)
                async with client.stream("GET", validated_url) as response:
                    redirect_location = cast(
                        str | None, response.headers.get("location")
                    )
                    if response.is_redirect and redirect_location:
                        current_url = urljoin(str(response.url), redirect_location)
                        continue

                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        remaining = _MAX_FETCH_BYTES - len(body)
                        if remaining <= 0:
                            break
                        body.extend(chunk[:remaining])

                    return _FetchedUrl(
                        url=_validate_http_url(url),
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=_normalize_content_type(
                            cast(str | None, response.headers.get("content-type"))
                        ),
                        body=bytes(body),
                        encoding=response.encoding,
                    )

        raise ValueError(
            f"Too many redirects while fetching URL (limit: {_MAX_REDIRECTS})."
        )


def default_browser_tool_provider() -> BrowserToolProvider:
    """Return the default provider for generic dashboard/browser tools."""

    return DefaultBrowserToolProvider()


def default_web_search_provider() -> BrowserToolProvider:
    """Compatibility wrapper for older call sites that only need search."""

    return default_browser_tool_provider()
