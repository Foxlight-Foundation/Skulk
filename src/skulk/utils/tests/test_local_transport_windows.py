"""What the Windows transport guarantees about the named pipe behind an endpoint."""

import asyncio
import contextlib
import ctypes
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from skulk.utils.local_transport import connect, serve

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="named pipes exist on Windows only"
)

_OWNER_AND_ACCESS_LIST = 0x1 | 0x4
_KERNEL_OBJECT = 6


@pytest.fixture
def directory() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="skulk-lt-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    writer.write((await reader.readline()).upper())
    await writer.drain()
    writer.close()


def _protection(handle: int) -> str:
    """The owner and access list of an open pipe, in Windows' own text form."""
    assert sys.platform == "win32"
    import ctypes.wintypes as wt

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.GetSecurityInfo.restype = wt.DWORD
    advapi32.GetSecurityInfo.argtypes = (
        wt.HANDLE,
        ctypes.c_int,
        wt.DWORD,
        wt.LPVOID,
        wt.LPVOID,
        wt.LPVOID,
        wt.LPVOID,
        ctypes.POINTER(wt.LPVOID),
    )
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = wt.BOOL
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = (
        wt.LPVOID,
        wt.DWORD,
        wt.DWORD,
        ctypes.POINTER(wt.LPWSTR),
        wt.LPVOID,
    )
    kernel32.LocalFree.argtypes = (wt.LPVOID,)
    descriptor = wt.LPVOID()
    status = advapi32.GetSecurityInfo(
        handle,
        _KERNEL_OBJECT,
        _OWNER_AND_ACCESS_LIST,
        None,
        None,
        None,
        None,
        ctypes.byref(descriptor),
    )
    assert status == 0, status
    text = wt.LPWSTR()
    assert advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
        descriptor, 1, _OWNER_AND_ACCESS_LIST, ctypes.byref(text), None
    )
    try:
        return text.value or ""
    finally:
        kernel32.LocalFree(text)
        kernel32.LocalFree(descriptor)


async def test_a_pipe_is_owned_by_and_open_to_this_account_and_system_only(
    directory: Path,
) -> None:
    assert sys.platform == "win32"
    from skulk.utils import _local_transport_windows as transport

    account = transport._this_account()
    address = directory / "endpoint.sock"
    server = await serve(address, _echo, limit=1024)
    try:
        _, writer = await connect(address, limit=1024)
        pipe = writer.get_extra_info("pipe")
        # Full access for SYSTEM and for this account, nothing inherited, no
        # other entry: no other account, administrators included, is listed.
        assert _protection(pipe.fileno()) == (
            f"O:{account}D:P(A;;FA;;;SY)(A;;FA;;;{account})"
        )
        writer.close()
    finally:
        server.close()
        await server.wait_closed()


async def test_a_served_name_cannot_be_taken_by_another_server(directory: Path) -> None:
    assert sys.platform == "win32"
    from asyncio import windows_events

    from skulk.utils import _local_transport_windows as transport

    address = directory / "endpoint.sock"
    server = await serve(address, _echo, limit=1024)
    try:
        with pytest.raises(PermissionError):
            await serve(address, _echo, limit=1024)
        with pytest.raises(PermissionError):
            windows_events.PipeServer(transport._pipe_name(address))

        reader, writer = await connect(address, limit=1024)
        writer.write(b"still ours\n")
        await writer.drain()
        assert await reader.readline() == b"STILL OURS\n"
        writer.close()
    finally:
        server.close()
        await server.wait_closed()


async def test_a_name_held_by_someone_else_is_not_served(directory: Path) -> None:
    """Serving fails closed when the name is taken first, whoever took it."""
    assert sys.platform == "win32"
    from asyncio import windows_events

    from skulk.utils import _local_transport_windows as transport

    address = directory / "endpoint.sock"
    squatter = windows_events.PipeServer(transport._pipe_name(address))
    try:
        with pytest.raises(PermissionError):
            await serve(address, _echo, limit=1024)
    finally:
        squatter.close()


async def test_a_pipe_served_by_another_account_is_refused(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sys.platform == "win32"
    from skulk.utils import _local_transport_windows as transport

    address = directory / "endpoint.sock"
    closed = asyncio.Event()

    async def watch(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # A pipe closed before any exchange can end as an error, not an
        # end of stream; either way the server learns the client left.
        try:
            with contextlib.suppress(OSError):
                await reader.read()
        finally:
            closed.set()
            writer.close()

    server = await serve(address, watch, limit=1024)
    try:
        # Let the server start waiting for a client, so that this one is
        # accepted and its departure is observed.
        await asyncio.sleep(0.05)

        def someone_else(handle: int) -> str:
            del handle
            return "S-1-5-21-1-2-3-1001"

        monkeypatch.setattr(transport, "_server_account", someone_else)
        with pytest.raises(PermissionError, match="another account"):
            await connect(address, limit=1024)
        # The refused connection is closed, not left open to the other side.
        await asyncio.wait_for(closed.wait(), 5)
    finally:
        server.close()
        await server.wait_closed()


def test_a_pipe_name_comes_from_the_address_and_the_account() -> None:
    assert sys.platform == "win32"
    from skulk.utils import _local_transport_windows as transport

    account = transport._this_account()
    assert account.startswith("S-1-")
    assert transport._pipe_name(Path("C:/anywhere/skulk-control-abc/control.sock")) == (
        rf"\\.\pipe\skulk-control-abc-{account}-control.sock"
    )
    assert transport._pipe_name(
        Path("D:/elsewhere/skulk-control-abc/control.sock")
    ) == transport._pipe_name(Path("C:/anywhere/skulk-control-abc/control.sock"))
    with pytest.raises(ValueError):
        transport._pipe_name(Path("control.sock"))


async def test_a_pipe_has_no_half_close(directory: Path) -> None:
    """Recorded because stream protocols that end input by half-close need another signal."""
    address = directory / "endpoint.sock"
    server = await serve(address, _echo, limit=1024)
    try:
        _, writer = await connect(address, limit=1024)
        assert not writer.can_write_eof()
        writer.close()
    finally:
        server.close()
        await server.wait_closed()


async def test_a_client_that_leaves_before_it_is_accepted_is_not_a_fault(
    directory: Path,
) -> None:
    assert sys.platform == "win32"
    from skulk.utils import _local_transport_windows as transport

    address = directory / "endpoint.sock"
    faults: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, context: faults.append(context))
    try:
        server = await serve(address, _echo, limit=1024)
        try:
            # Connect and leave before the server has run at all.
            handle = transport._kernel32.CreateFileW(
                transport._pipe_name(address), 0xC0000000, 0, None, 3, 0x40000000, None
            )
            assert transport._kernel32.CloseHandle(handle)
            await asyncio.sleep(0.2)

            reader, writer = await connect(address, limit=1024)
            writer.write(b"next\n")
            await writer.drain()
            assert await reader.readline() == b"NEXT\n"
            writer.close()
        finally:
            server.close()
            await server.wait_closed()
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(None)

    assert faults == []
