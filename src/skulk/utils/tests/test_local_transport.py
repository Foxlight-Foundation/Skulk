"""The local transport seam, and the rule that nothing else opens a local socket."""

import ast
import asyncio
import os
import shutil
import socket
import stat
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

import skulk
from skulk.utils import local_transport
from skulk.utils.local_transport import (
    connect,
    discard,
    endpoint_directory,
    endpoint_is_private,
    serve,
)

_DIRECT_SOCKET_CALLS = frozenset(
    {
        "AF_UNIX",
        "create_unix_connection",
        "create_unix_server",
        "open_unix_connection",
        "start_unix_server",
    }
)
_ENDPOINT_DIRECTORY_PREFIXES = ("skulk-control-", "skulk-manager-")

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="the only transport so far is a Unix socket"
)


@pytest.fixture
def directory() -> Iterator[Path]:
    """A private directory short enough for a Unix socket address."""
    path = Path(tempfile.mkdtemp(prefix="skulk-lt-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_only_the_seam_opens_or_names_a_local_socket() -> None:
    """Another platform's transport must have exactly one place to go.

    A direct ``asyncio`` Unix-socket call, an ``AF_UNIX`` socket or a
    hand-built endpoint directory anywhere else in the package would bypass
    the seam and break on a platform without Unix sockets.
    """
    package_root = Path(skulk.__file__).parent
    seam = Path(local_transport.__file__).resolve()
    offenders: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        if "tests" in source.parts or source.resolve() == seam:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            found: str | None = None
            if isinstance(node, ast.Attribute) and node.attr in _DIRECT_SOCKET_CALLS:
                found = node.attr
            elif isinstance(node, ast.Name) and node.id in _DIRECT_SOCKET_CALLS:
                found = node.id
            elif isinstance(node, ast.alias) and node.name in _DIRECT_SOCKET_CALLS:
                found = node.name
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(_ENDPOINT_DIRECTORY_PREFIXES)
            ):
                found = node.value
            if found is not None:
                offenders.append(f"{source.relative_to(package_root)}: {found}")

    assert offenders == []


@posix_only
def test_an_endpoint_directory_is_named_for_the_account_and_the_root() -> None:
    """Owners and peers derive the same address independently, so it is fixed."""
    assert endpoint_directory("control", "abc123") == Path(
        f"/tmp/skulk-control-{os.getuid()}-abc123"
    )
    assert endpoint_directory("manager", "abc123") == Path(
        f"/tmp/skulk-manager-{os.getuid()}-abc123"
    )


@posix_only
async def test_a_served_endpoint_is_private_and_carries_a_stream(
    directory: Path,
) -> None:
    address = directory / "endpoint.sock"

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write((await reader.readline()).upper())
        await writer.drain()
        writer.close()

    server = await serve(address, echo, limit=1024)
    try:
        assert stat.S_IMODE(address.lstat().st_mode) == 0o600
        assert endpoint_is_private(address)

        reader, writer = await connect(address, limit=1024)
        writer.write(b"hello\n")
        await writer.drain()
        assert await reader.readline() == b"HELLO\n"
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()
        discard(address)

    assert not address.exists()
    discard(address)


@posix_only
async def test_serving_replaces_what_an_earlier_listener_left(directory: Path) -> None:
    address = directory / "endpoint.sock"
    address.write_bytes(b"left behind")

    async def ignore(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        del reader
        writer.close()

    server = await serve(address, ignore, limit=1024)
    try:
        assert endpoint_is_private(address)
    finally:
        server.close()
        await server.wait_closed()
        discard(address)


@posix_only
async def test_connect_applies_the_callers_read_limit(directory: Path) -> None:
    address = directory / "endpoint.sock"

    async def flood(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        del reader
        writer.write(b"x" * 4096 + b"\n")
        await writer.drain()
        writer.close()

    server = await serve(address, flood, limit=1024)
    try:
        reader, writer = await connect(address, limit=64)
        with pytest.raises(ValueError):
            await reader.readline()
        writer.close()
    finally:
        server.close()
        await server.wait_closed()
        discard(address)


@posix_only
def test_an_endpoint_others_can_reach_is_not_private(directory: Path) -> None:
    address = directory / "endpoint.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(address))
        address.chmod(0o660)
        assert not endpoint_is_private(address)
        address.chmod(0o600)
        assert endpoint_is_private(address)
    finally:
        listener.close()


@posix_only
def test_only_a_socket_is_an_endpoint(directory: Path) -> None:
    regular = directory / "endpoint.sock"
    regular.write_bytes(b"")
    regular.chmod(0o600)
    assert not endpoint_is_private(regular)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        real = directory / "real.sock"
        listener.bind(str(real))
        real.chmod(0o600)
        link = directory / "link.sock"
        link.symlink_to(real)
        assert not endpoint_is_private(link)
    finally:
        listener.close()


@posix_only
def test_a_missing_endpoint_cannot_be_examined(directory: Path) -> None:
    with pytest.raises(FileNotFoundError):
        endpoint_is_private(directory / "absent.sock")
