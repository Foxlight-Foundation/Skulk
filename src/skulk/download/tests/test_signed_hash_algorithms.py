# pyright: reportPrivateUsage=false
"""Exercise signed content identities against real HTTP Git-blob metadata."""

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web

from skulk.download import download_utils
from skulk.shared.types.common import ModelId


def _git_blob_digest(content: bytes) -> str:
    """Calculate a Git blob identity, including its size-prefixed object header."""
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


@asynccontextmanager
async def _serve_file(
    content: bytes,
    etag: str,
    port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[list[str]]:
    """Serve fixed bytes locally without ever forwarding operator credentials."""
    methods: list[str] = []

    async def reply(request: web.Request) -> web.Response:
        methods.append(request.method)
        return web.Response(body=content, headers={"ETag": f'"{etag}"'})

    async def no_credentials() -> dict[str, str]:
        return {}

    application = web.Application()
    application.router.add_get("/{path:.*}", reply)
    runner = web.AppRunner(application)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
        monkeypatch.setattr(
            download_utils, "get_hf_endpoint", lambda: f"http://127.0.0.1:{port}"
        )
        monkeypatch.setattr(download_utils, "get_download_headers", no_credentials)
        yield methods
    finally:
        await runner.cleanup()


async def test_sha256_pin_accepts_a_regular_git_blob_download(
    tmp_path: Path, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hub Git metadata and a signed SHA-256 may identify the same exact bytes."""
    content = b"Signed model README.\n"
    async with _serve_file(
        content, _git_blob_digest(content), unused_tcp_port, monkeypatch
    ) as methods:
        result = await download_utils._download_file(
            ModelId("example/model"),
            "a" * 40,
            "README.md",
            tmp_path,
            expected_size=len(content),
            expected_object_id=f"sha256:{hashlib.sha256(content).hexdigest()}",
        )
    assert result.read_bytes() == content
    assert methods == ["HEAD", "GET"]


async def test_sha256_pin_rejects_remote_bytes_even_when_git_metadata_matches(
    tmp_path: Path, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid Hub blob checksum cannot replace the independent signed digest."""
    expected = b"good bytes\n"
    actual = b"evil bytes\n"
    assert len(actual) == len(expected)
    async with _serve_file(
        actual, _git_blob_digest(actual), unused_tcp_port, monkeypatch
    ) as methods:
        with pytest.raises(ValueError, match="[Ss]igned artifact bundle"):
            await download_utils._download_file(
                ModelId("example/model"),
                "a" * 40,
                "README.md",
                tmp_path,
                expected_size=len(expected),
                expected_object_id=f"sha256:{hashlib.sha256(expected).hexdigest()}",
            )
    assert methods == ["HEAD", "GET"]
    assert not (tmp_path / "README.md").exists()
    assert not (tmp_path / "README.md.partial").exists()


async def test_same_algorithm_metadata_disagreement_refuses_the_body(
    tmp_path: Path, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comparable but different immutable identities still fail before download."""
    content = b"Signed model README.\n"
    async with _serve_file(
        content, _git_blob_digest(content), unused_tcp_port, monkeypatch
    ) as methods:
        with pytest.raises(ValueError, match="object mismatch"):
            await download_utils._download_file(
                ModelId("example/model"),
                "a" * 40,
                "README.md",
                tmp_path,
                expected_size=len(content),
                expected_object_id=f"git-sha1:{'0' * 40}",
            )
    assert methods == ["HEAD"]


async def test_cross_algorithm_metadata_reuses_only_verified_cached_bytes(
    tmp_path: Path, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correct cached SHA-256 file needs no body transfer for Git metadata."""
    content = b"Signed model README.\n"
    target = tmp_path / "README.md"
    target.write_bytes(content)
    async with _serve_file(
        content, _git_blob_digest(content), unused_tcp_port, monkeypatch
    ) as methods:
        result = await download_utils._download_file(
            ModelId("example/model"),
            "a" * 40,
            "README.md",
            tmp_path,
            expected_size=len(content),
            expected_object_id=f"sha256:{hashlib.sha256(content).hexdigest()}",
        )
    assert result == target
    assert result.read_bytes() == content
    assert methods == ["HEAD"]


async def test_same_size_cached_corruption_is_redownloaded_against_the_pin(
    tmp_path: Path, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different metadata algorithms must never turn size into cache authority."""
    content = b"good bytes\n"
    target = tmp_path / "README.md"
    target.write_bytes(b"evil bytes\n")
    async with _serve_file(
        content, _git_blob_digest(content), unused_tcp_port, monkeypatch
    ) as methods:
        result = await download_utils._download_file(
            ModelId("example/model"),
            "a" * 40,
            "README.md",
            tmp_path,
            expected_size=len(content),
            expected_object_id=f"sha256:{hashlib.sha256(content).hexdigest()}",
        )
    assert result.read_bytes() == content
    assert methods == ["HEAD", "HEAD", "GET"]


async def test_offline_pin_mismatch_preserves_but_refuses_cached_bytes(
    tmp_path: Path,
) -> None:
    """An offline caller cannot qualify corrupt pinned content by its size."""
    content = b"good bytes\n"
    target = tmp_path / "README.md"
    target.write_bytes(b"evil bytes\n")
    with pytest.raises(ValueError, match="[Cc]ached.*(identity|mismatch)"):
        await download_utils._download_file(
            ModelId("example/model"),
            "a" * 40,
            "README.md",
            tmp_path,
            skip_internet=True,
            expected_size=len(content),
            expected_object_id=f"sha256:{hashlib.sha256(content).hexdigest()}",
        )
    assert target.read_bytes() == b"evil bytes\n"
