# pyright: reportPrivateUsage=false
"""Stored-image URL construction must match the registered fetch route."""

from types import SimpleNamespace
from typing import cast

from starlette.requests import Request

from skulk.api.main import API
from skulk.shared.types.common import Id


def _fake_request(host: str = "node.example:52415", scheme: str = "http") -> Request:
    return cast(
        "Request",
        cast(
            object,
            SimpleNamespace(
                headers={"host": host}, url=SimpleNamespace(scheme=scheme)
            ),
        ),
    )


def _fake_api() -> API:
    return cast("API", cast(object, SimpleNamespace(port=52415)))


def test_build_image_url_matches_the_registered_route() -> None:
    # The fetch route is GET /images/{image_id}; a /v1 prefix here returned
    # URLs that always 404ed (found by the 1.5.0 docs review).
    url = API._build_image_url(_fake_api(), _fake_request(), Id("img-123"))
    assert url == "http://node.example:52415/images/img-123"


def test_build_image_url_preserves_https() -> None:
    url = API._build_image_url(_fake_api(), _fake_request(scheme="https"), Id("img-9"))
    assert url.startswith("https://")


async def test_fetch_image_url_rejects_loopback_and_tailscale_targets() -> None:
    """Remote image fetching refuses loopback and Tailscale/CGNAT SSRF targets."""
    import pytest
    from fastapi import HTTPException

    from skulk.api.adapters.chat_completions import fetch_image_url

    for blocked in (
        "http://127.0.0.1:52415/state",
        "http://100.68.140.20:52415/state",
        "file:///etc/passwd",
    ):
        with pytest.raises(HTTPException) as exc_info:
            await fetch_image_url(blocked)
        assert exc_info.value.status_code == 400


async def test_fetch_image_url_rejects_oversized_responses(
    monkeypatch: object,
) -> None:
    """Remote image fetching enforces a hard byte cap while streaming."""
    import pytest
    from fastapi import HTTPException

    import skulk.api.adapters.chat_completions as adapter

    mp = cast("pytest.MonkeyPatch", monkeypatch)

    class _FakeContent:
        async def iter_chunked(self, _size: int):
            yield b"a" * 64
            yield b"b" * 64

    class _FakeResponse:
        status = 200
        headers: dict[str, str] = {}
        content = _FakeContent()

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self) -> "_FakeResponse":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    class _FakeSession:
        def get(
            self, _url: str, *, headers: dict[str, str], allow_redirects: bool
        ) -> _FakeResponse:
            assert allow_redirects is False
            assert "User-Agent" in headers
            return _FakeResponse()

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    async def _allow_public(
        url: str,
    ) -> tuple[str, tuple[tuple[int, int, str, int], ...]]:
        import socket

        return url, ((int(socket.AF_INET), 0, "93.184.216.34", 443),)

    captured_pin: list[tuple[str, tuple[tuple[int, int, str, int], ...]]] = []

    def _fake_session(
        expected_host: str,
        endpoints: tuple[tuple[int, int, str, int], ...],
    ) -> _FakeSession:
        captured_pin.append((expected_host, endpoints))
        return _FakeSession()

    mp.setattr("skulk.tools.web_search.resolve_public_http_target", _allow_public)
    mp.setattr(adapter, "_create_pinned_image_session", _fake_session)
    mp.setattr(adapter, "_MAX_REMOTE_IMAGE_BYTES", 100)

    with pytest.raises(HTTPException) as exc_info:
        await adapter.fetch_image_url("https://93.184.216.34/big.png")
    assert exc_info.value.status_code == 413
    assert captured_pin == [
        ("93.184.216.34", ((2, 0, "93.184.216.34", 443),))
    ]


async def test_pinned_host_resolver_and_session_disable_proxy_and_rebinding() -> None:
    """Image fetch session pins validated IPs and disables proxy/DNS cache."""
    import socket

    import aiohttp
    import pytest

    import skulk.api.adapters.chat_completions as adapter

    resolver = adapter._PinnedHostResolver(
        "images.example",
        ((int(socket.AF_INET), 0, "93.184.216.34", 443),),
    )
    resolved = await resolver.resolve("images.example", 443, socket.AF_INET)
    assert len(resolved) == 1
    assert resolved[0]["host"] == "93.184.216.34"
    assert resolved[0]["hostname"] == "images.example"

    with pytest.raises(OSError, match="Unexpected host"):
        await resolver.resolve("127.0.0.1", 443, socket.AF_INET)

    # Internationalized (Unicode) URL hosts — including IDNA2008 sharp-s — must
    # match aiohttp/yarl's IDNA2008-encoded raw_host consistently.
    for unicode_host, ascii_host in (
        ("bücher.example", "xn--bcher-kva.example"),
        ("faß.example", "xn--fa-hia.example"),
    ):
        idn_resolver = adapter._PinnedHostResolver(
            unicode_host,
            ((int(socket.AF_INET), 0, "93.184.216.34", 443),),
        )
        idn_resolved = await idn_resolver.resolve(ascii_host, 443, socket.AF_INET)
        assert len(idn_resolved) == 1
        assert idn_resolved[0]["host"] == "93.184.216.34"
        assert idn_resolved[0]["hostname"] == ascii_host
        with pytest.raises(OSError, match="Unexpected host"):
            await idn_resolver.resolve("other.example", 443, socket.AF_INET)

    session = adapter._create_pinned_image_session(
        "images.example",
        ((int(socket.AF_INET), 0, "93.184.216.34", 443),),
    )
    try:
        assert session._trust_env is False
        assert isinstance(session.connector, aiohttp.TCPConnector)
        assert session.connector._use_dns_cache is False
    finally:
        await session.close()


async def test_resolve_public_http_target_uses_idna2008_for_dns_lookup(
    monkeypatch: object,
) -> None:
    """Initial DNS lookup must use the same IDNA2008 hostname aiohttp sends."""
    import socket

    import pytest

    from skulk.tools.web_search import resolve_public_http_target

    mp = cast("pytest.MonkeyPatch", monkeypatch)
    seen_hosts: list[str] = []

    def _fake_getaddrinfo(
        host: str,
        port: int,
        family: int = 0,
        socktype: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        _ = (family, socktype)
        seen_hosts.append(host)
        return [(int(socket.AF_INET), int(socket.SOCK_STREAM), 6, "", ("93.184.216.34", port))]

    mp.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    validated, endpoints = await resolve_public_http_target("https://faß.example/img.png")
    assert validated == "https://faß.example/img.png"
    assert seen_hosts == ["xn--fa-hia.example"]
    assert endpoints == ((int(socket.AF_INET), 6, "93.184.216.34", 443),)
