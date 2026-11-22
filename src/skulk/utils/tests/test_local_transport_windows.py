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


def _security_library() -> tuple[ctypes.CDLL, ctypes.CDLL]:
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
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wt.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        wt.LPCWSTR,
        wt.DWORD,
        ctypes.POINTER(wt.LPVOID),
        wt.LPVOID,
    )
    kernel32.LocalFree.argtypes = (wt.LPVOID,)
    return advapi32, kernel32


def _as_windows_writes_it(descriptor: object) -> str:
    """A security descriptor's owner and access list in Windows' own text form."""
    assert sys.platform == "win32"
    import ctypes.wintypes as wt

    advapi32, kernel32 = _security_library()
    text = wt.LPWSTR()
    assert advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
        descriptor, 1, _OWNER_AND_ACCESS_LIST, ctypes.byref(text), None
    )
    try:
        return text.value or ""
    finally:
        kernel32.LocalFree(text)


def _protection(handle: int) -> str:
    """The owner and access list of an open pipe."""
    assert sys.platform == "win32"
    import ctypes.wintypes as wt

    advapi32, kernel32 = _security_library()
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
    try:
        return _as_windows_writes_it(descriptor)
    finally:
        kernel32.LocalFree(descriptor)


def _normalized(protection: str) -> str:
    """The same protection as Windows writes it back.

    Windows abbreviates well-known accounts when it writes a descriptor (the
    built-in administrator becomes ``LA``), so two descriptors are compared
    after both have been through the same conversion.
    """
    assert sys.platform == "win32"
    import ctypes.wintypes as wt

    advapi32, kernel32 = _security_library()
    descriptor = wt.LPVOID()
    assert advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        protection, 1, ctypes.byref(descriptor), None
    )
    try:
        return _as_windows_writes_it(descriptor)
    finally:
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
        assert _protection(pipe.fileno()) == _normalized(
            f"O:{account}D:P(A;;FA;;;SY)(A;;FA;;;{account})"
        )
        # And that is not what an unprotected pipe looks like.
        assert _protection(pipe.fileno()) != _normalized(
            f"O:{account}D:(A;;FA;;;SY)(A;;FA;;;{account})(A;;FR;;;WD)"
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


async def test_clients_that_leave_early_do_not_accumulate_pipe_instances(
    directory: Path,
) -> None:
    """An instance whose client left before it was accepted is released at once."""
    assert sys.platform == "win32"
    from skulk.utils import _local_transport_windows as transport

    address = directory / "endpoint.sock"
    server = await serve(address, _echo, limit=1024)
    try:
        for _ in range(25):
            handle = None
            for _ in range(200):
                handle = transport._kernel32.CreateFileW(
                    transport._pipe_name(address),
                    0xC0000000,
                    0,
                    None,
                    3,
                    0x40000000,
                    None,
                )
                if handle not in (None, transport._INVALID_HANDLE):
                    break
                await asyncio.sleep(0.005)
            assert handle not in (None, transport._INVALID_HANDLE)
            assert transport._kernel32.CloseHandle(handle)
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)

        # Only the instance being waited on and the next one remain.
        assert len(server._waiting) <= 2
        assert all(pipe.handle is not None for pipe in server._waiting)

        reader, writer = await connect(address, limit=1024)
        writer.write(b"still serving\n")
        await writer.drain()
        assert await reader.readline() == b"STILL SERVING\n"
        writer.close()
    finally:
        server.close()
        await server.wait_closed()
    assert server._waiting == set()


async def test_a_wait_that_fails_later_releases_its_pipe_instance(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wait can also fail after it was registered; that instance is released too."""
    assert sys.platform == "win32"
    from asyncio import windows_utils

    address = directory / "endpoint.sock"
    loop = asyncio.get_running_loop()
    proactor = loop._proactor  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    real_accept = proactor.accept_pipe  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    failed: list[windows_utils.PipeHandle] = []

    def fail_the_first_wait(
        pipe: windows_utils.PipeHandle,
    ) -> asyncio.Future[windows_utils.PipeHandle]:
        if failed:
            return real_accept(pipe)  # pyright: ignore[reportUnknownVariableType]
        failed.append(pipe)
        broken: asyncio.Future[windows_utils.PipeHandle] = loop.create_future()
        loop.call_soon(broken.set_exception, BrokenPipeError(232, "left"))
        return broken

    monkeypatch.setattr(proactor, "accept_pipe", fail_the_first_wait)
    server = await serve(address, _echo, limit=1024)
    try:
        await asyncio.sleep(0.1)
        assert len(failed) == 1
        assert failed[0].handle is None
        assert failed[0] not in server._waiting

        reader, writer = await connect(address, limit=1024)
        writer.write(b"after a failed wait\n")
        await writer.drain()
        assert await reader.readline() == b"AFTER A FAILED WAIT\n"
        writer.close()
    finally:
        server.close()
        await server.wait_closed()
    assert server._waiting == set()
