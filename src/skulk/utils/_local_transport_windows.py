"""Windows side of :mod:`skulk.utils.local_transport`: one named pipe per endpoint.

Python on Windows has no Unix sockets, so an endpoint is a named pipe. The
guarantees are the ones the POSIX side gives, obtained differently:

- **Private to the account.** The pipe is created with an explicit security
  descriptor: owned by this account, reachable by this account and SYSTEM
  only, and closed to remote clients. A default pipe would let every account
  read it.
- **The name cannot be taken while it is served.** The first instance is
  created with ``FILE_FLAG_FIRST_PIPE_INSTANCE``, so a second server on the
  name, from any account, is refused. If another account holds the name first,
  serving fails instead of sharing it.
- **A client never trusts another account's server.** A pipe's protection can
  only be read from an open connection, so the check that POSIX makes on the
  socket path before connecting is made here on the connection itself:
  :func:`connect` refuses a pipe whose server process belongs to another
  account. It also connects at identification level, so a server cannot act
  as the client.

There is no file to remove afterwards: a pipe's name disappears with its last
handle.

Streams are ordinary ``asyncio`` streams over the event loop's proactor. The
standard library exposes pipes with fixed, permissive security only, so this
module creates the handles itself and gives them to the proactor through the
two internal calls ``asyncio`` uses for its own pipes (``accept_pipe`` and
``_make_duplex_pipe_transport``). Those are private to CPython; Windows nodes
are limited to CPython 3.13 at dependency resolution, and the transport tests
exercise every path on Windows.

A pipe has no half-close: ``StreamWriter.can_write_eof()`` is false here.
"""

# ruff: noqa: E402
# The platform assertion has to come before the Windows-only imports: it is
# what tells the type checker to skip this module on other platforms.
import sys

assert sys.platform == "win32"

import asyncio
import ctypes
import ctypes.wintypes as wt
import os
from asyncio import windows_utils
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol, cast, final

# What ``asyncio`` itself accepts for a stream server: a coroutine function or
# a plain function that starts its own task.
type ClientConnected = Callable[
    [asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None] | None
]

_ERROR_FILE_NOT_FOUND = 2
_ERROR_SEM_TIMEOUT = 121
_ERROR_PIPE_BUSY = 231

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_FILE_FLAG_OVERLAPPED = 0x40000000
_FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
# The server may learn who the client is but may not act as it.
_SECURITY_SQOS_PRESENT = 0x00100000
_SECURITY_IDENTIFICATION = 0x00010000

_PIPE_ACCESS_DUPLEX = 0x00000003
_PIPE_TYPE_BYTE = 0x00000000
_PIPE_READMODE_BYTE = 0x00000000
_PIPE_WAIT = 0x00000000
_PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
_PIPE_UNLIMITED_INSTANCES = 255
_PIPE_BUFFER_BYTES = 65536

_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SDDL_REVISION_1 = 1

# How long a client waits between attempts while every instance is busy. The
# caller's own deadline bounds the total.
_BUSY_RETRY_FIRST_SECONDS = 0.001
_BUSY_RETRY_LONGEST_SECONDS = 0.1


class _SecurityAttributes(ctypes.Structure):
    _fields_ = (
        ("nLength", wt.DWORD),
        ("lpSecurityDescriptor", wt.LPVOID),
        ("bInheritHandle", wt.BOOL),
    )


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

_kernel32.GetCurrentProcess.restype = wt.HANDLE
_kernel32.GetCurrentProcess.argtypes = ()
_kernel32.OpenProcess.restype = wt.HANDLE
_kernel32.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
_kernel32.CloseHandle.restype = wt.BOOL
_kernel32.CloseHandle.argtypes = (wt.HANDLE,)
_kernel32.LocalFree.restype = wt.LPVOID
_kernel32.LocalFree.argtypes = (wt.LPVOID,)
_kernel32.CreateNamedPipeW.restype = wt.HANDLE
_kernel32.CreateNamedPipeW.argtypes = (
    wt.LPCWSTR,
    wt.DWORD,
    wt.DWORD,
    wt.DWORD,
    wt.DWORD,
    wt.DWORD,
    wt.DWORD,
    ctypes.POINTER(_SecurityAttributes),
)
_kernel32.CreateFileW.restype = wt.HANDLE
_kernel32.CreateFileW.argtypes = (
    wt.LPCWSTR,
    wt.DWORD,
    wt.DWORD,
    wt.LPVOID,
    wt.DWORD,
    wt.DWORD,
    wt.HANDLE,
)
_kernel32.WaitNamedPipeW.restype = wt.BOOL
_kernel32.WaitNamedPipeW.argtypes = (wt.LPCWSTR, wt.DWORD)
_kernel32.GetNamedPipeServerProcessId.restype = wt.BOOL
_kernel32.GetNamedPipeServerProcessId.argtypes = (wt.HANDLE, ctypes.POINTER(wt.ULONG))
_advapi32.OpenProcessToken.restype = wt.BOOL
_advapi32.OpenProcessToken.argtypes = (wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.HANDLE))
_advapi32.GetTokenInformation.restype = wt.BOOL
_advapi32.GetTokenInformation.argtypes = (
    wt.HANDLE,
    ctypes.c_int,
    wt.LPVOID,
    wt.DWORD,
    ctypes.POINTER(wt.DWORD),
)
_advapi32.ConvertSidToStringSidW.restype = wt.BOOL
_advapi32.ConvertSidToStringSidW.argtypes = (wt.LPVOID, ctypes.POINTER(wt.LPWSTR))
_advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wt.BOOL
_advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
    wt.LPCWSTR,
    wt.DWORD,
    ctypes.POINTER(wt.LPVOID),
    ctypes.POINTER(wt.ULONG),
)

_INVALID_HANDLE = wt.HANDLE(-1).value

# ctypes calls return untyped values. These views give each foreign function
# the result type its ``restype`` above produces.
_get_current_process: Callable[..., int] = _kernel32.GetCurrentProcess
_open_process: Callable[..., int | None] = _kernel32.OpenProcess
_close_handle: Callable[..., int] = _kernel32.CloseHandle
_local_free: Callable[..., object] = _kernel32.LocalFree
_create_named_pipe: Callable[..., int | None] = _kernel32.CreateNamedPipeW
_create_file: Callable[..., int | None] = _kernel32.CreateFileW
_wait_named_pipe: Callable[..., int] = _kernel32.WaitNamedPipeW
_get_server_process_id: Callable[..., int] = _kernel32.GetNamedPipeServerProcessId
_open_process_token: Callable[..., int] = _advapi32.OpenProcessToken
_get_token_information: Callable[..., int] = _advapi32.GetTokenInformation
_sid_to_text: Callable[..., int] = _advapi32.ConvertSidToStringSidW
_text_to_descriptor: Callable[..., int] = (
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
)


class _Proactor(Protocol):
    """The part of the event loop's proactor this module drives."""

    def accept_pipe(
        self, pipe: windows_utils.PipeHandle
    ) -> asyncio.Future[windows_utils.PipeHandle]: ...


class _ProactorLoop(Protocol):
    """The internal pipe entry points of ``asyncio.ProactorEventLoop``."""

    _proactor: _Proactor

    def _make_duplex_pipe_transport(
        self,
        sock: windows_utils.PipeHandle,
        protocol: asyncio.StreamReaderProtocol,
        waiter: asyncio.Future[None] | None = None,
        extra: dict[str, str] | None = None,
    ) -> asyncio.Transport: ...


def _proactor_loop() -> _ProactorLoop:
    """The running loop, which on Windows is the proactor loop."""
    loop = asyncio.get_running_loop()
    if not isinstance(loop, asyncio.ProactorEventLoop):
        raise RuntimeError("local endpoints need the proactor event loop")
    return cast("_ProactorLoop", cast(object, loop))


def _last_error(action: str) -> OSError:
    """The calling thread's last Windows error as the matching ``OSError``."""
    return ctypes.WinError(ctypes.get_last_error(), action)


def _process_account(process: int) -> str:
    """The security identifier of the account a process runs as, as a string."""
    token = wt.HANDLE()
    if not _open_process_token(process, _TOKEN_QUERY, ctypes.byref(token)):
        raise _last_error("OpenProcessToken")
    try:
        size = wt.DWORD()
        _get_token_information(token, _TOKEN_USER, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not _get_token_information(
            token, _TOKEN_USER, buffer, size, ctypes.byref(size)
        ):
            raise _last_error("GetTokenInformation")
        # TOKEN_USER begins with a pointer to the account's identifier.
        identifier = ctypes.c_void_p.from_buffer(buffer).value
        text = wt.LPWSTR()
        if not _sid_to_text(identifier, ctypes.byref(text)):
            raise _last_error("ConvertSidToStringSid")
        try:
            value = text.value
        finally:
            _local_free(text)
        if not value:
            raise OSError("account identifier is empty")
        return value
    finally:
        _close_handle(token)


_account: str | None = None


def _this_account() -> str:
    """This process's account identifier; it cannot change while it runs."""
    global _account
    if _account is None:
        _account = _process_account(_get_current_process())
    return _account


def _server_account(handle: int) -> str:
    """The account of the process serving the pipe ``handle`` is connected to."""
    process_id = wt.ULONG()
    if not _get_server_process_id(handle, ctypes.byref(process_id)):
        raise _last_error("GetNamedPipeServerProcessId")
    process = _open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, process_id.value)
    if not process:
        raise _last_error("OpenProcess")
    try:
        return _process_account(process)
    finally:
        _close_handle(process)


def _pipe_name(path: Path) -> str:
    """Map an endpoint address to the pipe that carries it.

    The name is built from the endpoint's directory name, this account and the
    endpoint's own name, so it does not depend on where the directory lives
    and two accounts never collide. A pipe name cannot contain a backslash.
    """
    directory, name = path.parent.name, path.name
    if not directory or not name or "\\" in directory + name:
        raise ValueError("local endpoint address is not valid")
    return rf"\\.\pipe\{directory}-{_this_account()}-{name}"


def endpoint_directory(kind: Literal["control", "manager"], digest: str) -> Path:
    """Name the per-account directory that identifies one root's endpoints.

    On Windows nothing is stored in it: it gives each endpoint a stable
    address under the account's own application data, which callers protect
    exactly as they do the POSIX directory.

    Args:
        kind: Which side owns the endpoints: an installed plugin owner
            (``control``) or the plugin manager (``manager``).
        digest: The caller's digest of the root the endpoints belong to.

    Returns:
        The directory path.
    """
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "Skulk" / "endpoints" / f"skulk-{kind}-{digest}"


def endpoint_is_private(path: Path) -> bool:
    """Whether an endpoint is being served at ``path``.

    A pipe's protection cannot be read without connecting to it, so on Windows
    this only establishes that the endpoint exists. :func:`connect` then
    refuses a pipe served by another account, so a caller that checks and then
    connects still never talks to another account's endpoint.

    Args:
        path: The endpoint address.

    Returns:
        True when a server holds the endpoint's pipe.

    Raises:
        OSError: When nothing is serving there.
    """
    if _wait_named_pipe(_pipe_name(path), 1):
        return True
    error = ctypes.get_last_error()
    if error == _ERROR_SEM_TIMEOUT:
        # It exists and every instance is busy for the moment.
        return True
    raise ctypes.WinError(error, "local endpoint")


async def connect(
    path: Path, *, limit: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a stream to a local endpoint served by this account.

    Args:
        path: The endpoint address.
        limit: The reader's buffer limit in bytes.

    Returns:
        The stream's reader and writer.

    Raises:
        PermissionError: When the endpoint is served by another account.
        OSError: When nothing is serving there.
    """
    name = _pipe_name(path)
    loop = _proactor_loop()
    delay = _BUSY_RETRY_FIRST_SECONDS
    while True:
        handle = _create_file(
            name,
            _GENERIC_READ | _GENERIC_WRITE,
            0,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_OVERLAPPED | _SECURITY_SQOS_PRESENT | _SECURITY_IDENTIFICATION,
            None,
        )
        if handle is not None and handle != _INVALID_HANDLE:
            break
        error = ctypes.get_last_error()
        if error != _ERROR_PIPE_BUSY:
            raise ctypes.WinError(error, "local endpoint")
        await asyncio.sleep(delay)
        delay = min(delay * 2, _BUSY_RETRY_LONGEST_SECONDS)
    try:
        if _server_account(handle) != _this_account():
            raise PermissionError("local endpoint is served by another account")
        pipe = windows_utils.PipeHandle(handle)
    except BaseException:
        _close_handle(handle)
        raise
    reader = asyncio.StreamReader(limit=limit)
    protocol = asyncio.StreamReaderProtocol(reader)
    try:
        transport = loop._make_duplex_pipe_transport(  # pyright: ignore[reportPrivateUsage]
            pipe, protocol, extra={"addr": name}
        )
    except BaseException:
        pipe.close()
        raise
    return reader, asyncio.StreamWriter(
        transport, protocol, reader, asyncio.get_running_loop()
    )


@final
class _PipeListener:
    """A served endpoint: its pipe instances and the loop that accepts clients.

    This is the standard library's pipe server and accept loop
    (``asyncio.windows_events.PipeServer`` and
    ``ProactorEventLoop.start_serving_pipe``) with one difference: every
    instance is created with this module's security descriptor.
    """

    def __init__(
        self, name: str, client_connected: ClientConnected, limit: int
    ) -> None:
        self._name: str | None = name
        self._client_connected = client_connected
        self._limit = limit
        self._loop = _proactor_loop()
        # Instances not yet handed to a client; closing the listener closes them.
        self._waiting: set[windows_utils.PipeHandle] = set()
        self._accepting: asyncio.Future[windows_utils.PipeHandle] | None = None
        self._next: windows_utils.PipeHandle | None = self._instance(first=True)

    def _instance(self, *, first: bool) -> windows_utils.PipeHandle | None:
        """Create one more instance of the pipe, private to this account."""
        if self._name is None:
            return None
        account = _this_account()
        descriptor = wt.LPVOID()
        if not _text_to_descriptor(
            # Owned by this account; full access for it and for SYSTEM; no
            # inherited or default entries.
            f"O:{account}D:P(A;;GA;;;SY)(A;;GA;;;{account})",
            _SDDL_REVISION_1,
            ctypes.byref(descriptor),
            None,
        ):
            raise _last_error("security descriptor")
        try:
            attributes = _SecurityAttributes(
                ctypes.sizeof(_SecurityAttributes), descriptor, False
            )
            handle = _create_named_pipe(
                self._name,
                _PIPE_ACCESS_DUPLEX
                | _FILE_FLAG_OVERLAPPED
                | (_FILE_FLAG_FIRST_PIPE_INSTANCE if first else 0),
                _PIPE_TYPE_BYTE
                | _PIPE_READMODE_BYTE
                | _PIPE_WAIT
                | _PIPE_REJECT_REMOTE_CLIENTS,
                _PIPE_UNLIMITED_INSTANCES,
                _PIPE_BUFFER_BYTES,
                _PIPE_BUFFER_BYTES,
                0,
                ctypes.byref(attributes),
            )
            if handle is None or handle == _INVALID_HANDLE:
                raise _last_error("local endpoint")
        finally:
            _local_free(descriptor)
        pipe = windows_utils.PipeHandle(handle)
        self._waiting.add(pipe)
        return pipe

    def start(self) -> None:
        """Begin accepting clients on the running loop."""
        asyncio.get_running_loop().call_soon(self._accept)

    def _accept(
        self, accepted: asyncio.Future[windows_utils.PipeHandle] | None = None
    ) -> None:
        loop = asyncio.get_running_loop()
        pipe: windows_utils.PipeHandle | None = None
        try:
            if accepted is not None:
                pipe = accepted.result()
                self._waiting.discard(pipe)
                if self._name is None:
                    pipe.close()
                    return
                protocol = asyncio.StreamReaderProtocol(
                    asyncio.StreamReader(limit=self._limit), self._client_connected
                )
                self._loop._make_duplex_pipe_transport(  # pyright: ignore[reportPrivateUsage]
                    pipe, protocol, extra={"addr": self._name}
                )
            # Keep one unconnected instance ready before waiting on this one,
            # so the name is never without an instance while it is served.
            pipe, self._next = self._next, self._instance(first=False)
            if pipe is None:
                return
            pending = self._loop._proactor.accept_pipe(pipe)  # pyright: ignore[reportPrivateUsage]
        except BrokenPipeError:
            # A client that connected and left before it was accepted: an
            # ordinary departure. Drop that instance and keep serving.
            if pipe is not None and pipe.fileno() != -1:
                pipe.close()
            loop.call_soon(self._accept)
        except OSError as error:
            if pipe is not None and pipe.fileno() != -1:
                loop.call_exception_handler(
                    {
                        "message": "Local endpoint accept failed",
                        "exception": error,
                    }
                )
                pipe.close()
            loop.call_soon(self._accept)
        except asyncio.CancelledError:
            if pipe is not None:
                pipe.close()
        else:
            self._accepting = pending
            pending.add_done_callback(self._accept)

    def close(self) -> None:
        """Stop accepting clients and release the endpoint's name."""
        if self._accepting is not None:
            self._accepting.cancel()
            self._accepting = None
        if self._name is not None:
            self._name = None
            for pipe in self._waiting:
                pipe.close()
            self._waiting.clear()
            self._next = None

    async def wait_closed(self) -> None:
        """Return once the endpoint is released; closing is immediate."""
        return


async def serve(
    path: Path, client_connected: ClientConnected, *, limit: int
) -> _PipeListener:
    """Listen on a local endpoint that only this account can reach.

    Args:
        path: The endpoint address.
        client_connected: Called with a reader and a writer for each client.
        limit: Each client reader's buffer limit in bytes.

    Returns:
        The listening server.

    Raises:
        OSError: When the endpoint's name is already held, by any account.
    """
    listener = _PipeListener(_pipe_name(path), client_connected, limit)
    listener.start()
    return listener


def discard(path: Path) -> None:
    """Nothing to remove: a pipe's name disappears with its last handle."""
    del path
