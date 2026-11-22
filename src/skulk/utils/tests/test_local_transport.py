"""The local transport seam, and the rule that nothing else opens a local endpoint."""

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

_DIRECT_ENDPOINT_CALLS = frozenset(
    {
        "AF_UNIX",
        "CreateNamedPipeW",
        "PipeServer",
        "create_pipe_connection",
        "create_unix_connection",
        "create_unix_server",
        "open_unix_connection",
        "start_serving_pipe",
        "start_unix_server",
    }
)
_ENDPOINT_DIRECTORY_PREFIXES = ("skulk-control-", "skulk-manager-")

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="examines a Unix socket's file"
)
windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="examines a named pipe"
)


@pytest.fixture
def directory() -> Iterator[Path]:
    """A private directory with a unique name, short enough for a socket address."""
    path = Path(
        tempfile.mkdtemp(
            prefix="skulk-lt-", dir=None if sys.platform == "win32" else "/tmp"
        )
    )
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    writer.write((await reader.readline()).upper())
    await writer.drain()
    writer.close()


def test_only_the_seam_opens_or_names_a_local_endpoint() -> None:
    """Each platform's transport has exactly one place to live.

    A direct Unix-socket or named-pipe call, or a hand-built endpoint
    directory, anywhere else in the package would bypass the seam: it would
    break on the other platform and skip the seam's protection of the
    endpoint.
    """
    package_root = Path(skulk.__file__).parent
    seam_module = Path(local_transport.__file__).resolve()
    seam = {seam_module, seam_module.with_name("_local_transport_windows.py")}
    assert all(module.is_file() for module in seam)
    offenders: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        if "tests" in source.parts or source.resolve() in seam:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            found: str | None = None
            if isinstance(node, ast.Attribute) and node.attr in _DIRECT_ENDPOINT_CALLS:
                found = node.attr
            elif isinstance(node, ast.Name) and node.id in _DIRECT_ENDPOINT_CALLS:
                found = node.id
            elif isinstance(node, ast.alias) and node.name in _DIRECT_ENDPOINT_CALLS:
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


@windows_only
def test_a_windows_endpoint_directory_is_under_the_accounts_own_data() -> None:
    base = Path(os.environ["LOCALAPPDATA"]) / "Skulk" / "endpoints"
    assert endpoint_directory("control", "abc123") == base / "skulk-control-abc123"
    assert endpoint_directory("manager", "abc123") == base / "skulk-manager-abc123"


async def test_a_served_endpoint_carries_a_stream_and_is_released(
    directory: Path,
) -> None:
    address = directory / "endpoint.sock"

    server = await serve(address, _echo, limit=1024)
    try:
        assert endpoint_is_private(address)

        for word in (b"hello\n", b"again\n"):
            reader, writer = await connect(address, limit=1024)
            writer.write(word)
            await writer.drain()
            assert await reader.readline() == word.upper()
            writer.close()
            await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()
        discard(address)

    with pytest.raises(OSError):
        endpoint_is_private(address)
    with pytest.raises(OSError):
        await connect(address, limit=1024)
    discard(address)


async def test_two_clients_are_served_at_once(directory: Path) -> None:
    address = directory / "endpoint.sock"
    release = asyncio.Event()

    async def hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        await release.wait()
        writer.write(line.upper())
        await writer.drain()
        writer.close()

    server = await serve(address, hold, limit=1024)
    try:
        first = await connect(address, limit=1024)
        second = await connect(address, limit=1024)
        for (_, writer), word in ((first, b"one\n"), (second, b"two\n")):
            writer.write(word)
            await writer.drain()
        release.set()
        assert await first[0].readline() == b"ONE\n"
        assert await second[0].readline() == b"TWO\n"
        first[1].close()
        second[1].close()
    finally:
        server.close()
        await server.wait_closed()
        discard(address)


async def test_an_endpoint_can_be_served_again_after_it_closes(directory: Path) -> None:
    address = directory / "endpoint.sock"
    for _ in range(2):
        server = await serve(address, _echo, limit=1024)
        try:
            reader, writer = await connect(address, limit=1024)
            writer.write(b"round\n")
            await writer.drain()
            assert await reader.readline() == b"ROUND\n"
            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
            discard(address)


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


async def test_a_server_applies_its_read_limit_to_each_client(directory: Path) -> None:
    address = directory / "endpoint.sock"
    outcome: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def read(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()
            outcome.set_result("accepted")
        except ValueError:
            outcome.set_result("refused")
        writer.close()

    server = await serve(address, read, limit=64)
    try:
        _, writer = await connect(address, limit=1024)
        writer.write(b"x" * 4096 + b"\n")
        await writer.drain()
        assert await asyncio.wait_for(outcome, 5) == "refused"
        writer.close()
    finally:
        server.close()
        await server.wait_closed()
        discard(address)


def test_a_missing_endpoint_cannot_be_examined(directory: Path) -> None:
    with pytest.raises(FileNotFoundError):
        endpoint_is_private(directory / "absent.sock")


@posix_only
async def test_a_served_socket_is_owner_only(directory: Path) -> None:
    address = directory / "endpoint.sock"
    server = await serve(address, _echo, limit=1024)
    try:
        assert stat.S_IMODE(address.lstat().st_mode) == 0o600
    finally:
        server.close()
        await server.wait_closed()
        discard(address)
    assert not address.exists()


@posix_only
async def test_serving_replaces_what_an_earlier_listener_left(directory: Path) -> None:
    address = directory / "endpoint.sock"
    address.write_bytes(b"left behind")

    server = await serve(address, _echo, limit=1024)
    try:
        assert endpoint_is_private(address)
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
