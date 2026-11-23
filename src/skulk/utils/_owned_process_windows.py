# ruff: noqa: E402
# The platform assertion has to come before the Windows-only imports: it is
# what tells the type checker to skip this module on other platforms.
"""Windows side of :mod:`skulk.utils.owned_process`: a Job Object per child.

A Job Object is the Windows way to treat a process and everything it starts
as one thing. This module creates one per child with *kill on close* set:

- ``TerminateJobObject`` ends every process in the tree at once.
- When the last handle to the job closes, Windows ends every process still in
  it. This process holds the only handle, so the tree cannot outlive it,
  however it dies.

The child is a member of the job from the instant it exists: the job is named
in the attribute list ``CreateProcessW`` is given, so there is no moment at
which the child runs, or even exists, outside it. Python's own process
classes cannot pass that attribute, which is why this module calls
``CreateProcessW`` itself and supplies the output stream and the exit wait
that ``asyncio`` would otherwise provide. Jobs nest, so this works when this
process is itself inside a job (a service, a CI runner, a remote session).

Releasing a child waits for its tree to be gone, not merely told to go: an
ended process goes on closing what it held for a moment, and until then its
files cannot be replaced or removed. The job's own member count drops too
early to show that, so a handle is taken on every member before it is ended
and those handles are what is waited on (``_end_all``).

A file lock on Windows belongs to one process and is not shared with a child,
so the directory's lock cannot be passed down as it is on POSIX. The job alone
does not make up for that: when this process dies, Windows frees its lock and
ends its job in two separate steps, and for a millisecond or two in between
the lock is free while the tree still runs (measured: about 1 run in 130 saw a
late write). So the child is also started holding the lock's *children fence*
open with no sharing (:func:`skulk.utils._runtime_files_windows.hold_children_fence`).
The lock refuses a new holder while that file is held, which is until the
child has ended.

What that covers exactly: the fence is held by the child, not by what the
child starts. When this process dies, one operation ends the whole job, and
Windows does not promise the order within it, so a descendant can be ended
some microseconds after the child has gone and freed the fence. (The same
stress run with the fence in place: 0 late writes in 800, from a child and a
grandchild each writing every millisecond.) POSIX has the wider form of this
limit: there a descendant that did not keep the lock's descriptor is not
ended at all when the child goes.
"""

import sys

assert sys.platform == "win32"

import asyncio
import contextlib
import ctypes
import ctypes.wintypes as wt
import msvcrt
import os
import signal
import subprocess
import tempfile
from asyncio import windows_utils
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import final

from skulk.utils._runtime_files_windows import hold_children_fence

_JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x0008
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_ERROR_MORE_DATA = 234
_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_CREATE_NO_WINDOW = 0x08000000
_STARTF_USESTDHANDLES = 0x00000100
_INFINITE = 0xFFFFFFFF
_ERROR_ACCESS_DENIED = 5
# How often to look whether ended processes have finished: quickly at first,
# because it normally takes a few milliseconds, then at a slower pace.
_FINISH_POLL_SECONDS = 0.005
_FINISH_SLOW_POLL_SECONDS = 0.05
_FINISH_QUICK_POLLS = 200


class _BasicLimits(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wt.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    )


class _ExtendedLimits(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class _StartupInformation(ctypes.Structure):
    _fields_ = (
        ("cb", wt.DWORD),
        ("lpReserved", wt.LPWSTR),
        ("lpDesktop", wt.LPWSTR),
        ("lpTitle", wt.LPWSTR),
        ("dwX", wt.DWORD),
        ("dwY", wt.DWORD),
        ("dwXSize", wt.DWORD),
        ("dwYSize", wt.DWORD),
        ("dwXCountChars", wt.DWORD),
        ("dwYCountChars", wt.DWORD),
        ("dwFillAttribute", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("wShowWindow", wt.WORD),
        ("cbReserved2", wt.WORD),
        ("lpReserved2", wt.LPVOID),
        ("hStdInput", wt.HANDLE),
        ("hStdOutput", wt.HANDLE),
        ("hStdError", wt.HANDLE),
    )


class _ExtendedStartupInformation(ctypes.Structure):
    _fields_ = (
        ("StartupInfo", _StartupInformation),
        ("lpAttributeList", wt.LPVOID),
    )


# PROCESS_INFORMATION: two handles, then two 32-bit identifiers. Read as
# bytes so the values are plain integers to the type checker.
_POINTER_BYTES = ctypes.sizeof(ctypes.c_void_p)
_PROCESS_INFORMATION_BYTES = 2 * _POINTER_BYTES + 8
# JOBOBJECT_BASIC_PROCESS_ID_LIST: two 32-bit counts (processes in the job,
# identifiers that fitted in the list), then one pointer-sized identifier each.
_ID_LIST_HEADER_BYTES = 8


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateJobObjectW.argtypes = (wt.LPVOID, wt.LPCWSTR)
_kernel32.CreateJobObjectW.restype = wt.HANDLE
_kernel32.SetInformationJobObject.argtypes = (
    wt.HANDLE,
    ctypes.c_int,
    wt.LPVOID,
    wt.DWORD,
)
_kernel32.SetInformationJobObject.restype = wt.BOOL
_kernel32.QueryInformationJobObject.argtypes = (
    wt.HANDLE,
    ctypes.c_int,
    wt.LPVOID,
    wt.DWORD,
    wt.LPVOID,
)
_kernel32.QueryInformationJobObject.restype = wt.BOOL
_kernel32.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
_kernel32.OpenProcess.restype = wt.HANDLE
_kernel32.IsProcessInJob.argtypes = (wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL))
_kernel32.IsProcessInJob.restype = wt.BOOL
_kernel32.TerminateJobObject.argtypes = (wt.HANDLE, wt.UINT)
_kernel32.TerminateJobObject.restype = wt.BOOL
_kernel32.CloseHandle.argtypes = (wt.HANDLE,)
_kernel32.CloseHandle.restype = wt.BOOL
_kernel32.InitializeProcThreadAttributeList.argtypes = (
    wt.LPVOID,
    wt.DWORD,
    wt.DWORD,
    ctypes.POINTER(ctypes.c_size_t),
)
_kernel32.InitializeProcThreadAttributeList.restype = wt.BOOL
_kernel32.UpdateProcThreadAttribute.argtypes = (
    wt.LPVOID,
    wt.DWORD,
    ctypes.c_size_t,
    wt.LPVOID,
    ctypes.c_size_t,
    wt.LPVOID,
    wt.LPVOID,
)
_kernel32.UpdateProcThreadAttribute.restype = wt.BOOL
_kernel32.DeleteProcThreadAttributeList.argtypes = (wt.LPVOID,)
_kernel32.DeleteProcThreadAttributeList.restype = None
_kernel32.CreateProcessW.argtypes = (
    wt.LPCWSTR,
    wt.LPWSTR,
    wt.LPVOID,
    wt.LPVOID,
    wt.BOOL,
    wt.DWORD,
    wt.LPVOID,
    wt.LPCWSTR,
    ctypes.POINTER(_ExtendedStartupInformation),
    wt.LPVOID,
)
_kernel32.CreateProcessW.restype = wt.BOOL
_kernel32.WaitForSingleObject.argtypes = (wt.HANDLE, wt.DWORD)
_kernel32.WaitForSingleObject.restype = wt.DWORD
_kernel32.GetExitCodeProcess.argtypes = (wt.HANDLE, ctypes.POINTER(wt.DWORD))
_kernel32.GetExitCodeProcess.restype = wt.BOOL

# Typed views of the foreign functions, so their results are not ``Any``.
_create_job: Callable[[None, None], int | None] = _kernel32.CreateJobObjectW
_set_job_information: Callable[[int, int, object, int], int] = (
    _kernel32.SetInformationJobObject
)
_query_job: Callable[[int, int, object, int, None], int] = (
    _kernel32.QueryInformationJobObject
)
_open_process: Callable[[int, bool, int], int | None] = _kernel32.OpenProcess
_is_in_job: Callable[[int, int, object], int] = _kernel32.IsProcessInJob
_terminate_job: Callable[[int, int], int] = _kernel32.TerminateJobObject
_close_handle: Callable[[int], int] = _kernel32.CloseHandle
_initialize_attributes: Callable[[object, int, int, object], int] = (
    _kernel32.InitializeProcThreadAttributeList
)
_update_attribute: Callable[[object, int, int, object, int, None, None], int] = (
    _kernel32.UpdateProcThreadAttribute
)
_delete_attributes: Callable[[object], None] = _kernel32.DeleteProcThreadAttributeList
_create_process: Callable[
    [None, object, None, None, bool, int, object, str, object, object], int
] = _kernel32.CreateProcessW
_wait_for_handle: Callable[[int, int], int] = _kernel32.WaitForSingleObject
_exit_code: Callable[[int, object], int] = _kernel32.GetExitCodeProcess


def _failure(what: str) -> OSError:
    """The last Windows error as the matching ``OSError`` subclass."""
    error = ctypes.get_last_error()
    return OSError(0, f"{what}: {ctypes.FormatError(error)}", None, error)


def isolated_environment() -> dict[str, str]:
    """The least a Windows program needs: the system directory and a temp folder."""
    system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    temporary = tempfile.gettempdir()
    return {
        "SystemRoot": system_root,
        "PATH": str(Path(system_root) / "System32"),
        "TEMP": temporary,
        "TMP": temporary,
    }


def _new_job() -> int:
    """Create a job that ends its processes when its last handle closes."""
    job = _create_job(None, None)
    if not job:
        raise _failure("could not create a job")
    try:
        _set_limits(job, admit_new_members=True)
    except BaseException:
        _close_handle(job)
        raise
    return job


def _set_limits(job: int, *, admit_new_members: bool) -> None:
    """Keep *kill on close*; optionally stop the job taking any new process."""
    basic = _BasicLimits(LimitFlags=_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    if not admit_new_members:
        # With a member present, a limit of one refuses every further one.
        basic = _BasicLimits(
            LimitFlags=_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
            ActiveProcessLimit=1,
        )
    limits = _ExtendedLimits(BasicLimitInformation=basic)
    if not _set_job_information(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        raise _failure("could not configure a job")


def _create_in_job(  # noqa: PLR0913 - one CreateProcessW call, spelled out
    arguments: Sequence[str],
    cwd: Path,
    job: int,
    standard: tuple[int, ...],
    fence: int,
    *,
    console: bool,
) -> tuple[int, int]:
    """Create the process as a member of ``job``; return its handle and identifier.

    The child inherits exactly the handles named here: ``standard``, which is
    its input, its output and, when a third is given, its error output (the
    output serves for both otherwise), and ``fence``, which it only holds.
    With ``console`` the child shares this process's console; without, it
    gets none and no window.
    """
    try:
        for handle in (*standard, fence):
            os.set_handle_inheritable(handle, True)
        inherited = (wt.HANDLE * (len(standard) + 1))(*standard, fence)
        jobs = (wt.HANDLE * 1)(job)
        size = ctypes.c_size_t()
        # The first call reports the size the list needs and is expected to fail.
        _initialize_attributes(None, 2, 0, ctypes.byref(size))
        attributes = ctypes.create_string_buffer(size.value)
        if not _initialize_attributes(attributes, 2, 0, ctypes.byref(size)):
            raise _failure("could not prepare process attributes")
        try:
            if not _update_attribute(
                attributes,
                0,
                _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                inherited,
                ctypes.sizeof(inherited),
                None,
                None,
            ) or not _update_attribute(
                attributes,
                0,
                _PROC_THREAD_ATTRIBUTE_JOB_LIST,
                jobs,
                ctypes.sizeof(jobs),
                None,
                None,
            ):
                raise _failure("could not set process attributes")
            startup = _ExtendedStartupInformation(
                StartupInfo=_StartupInformation(
                    cb=ctypes.sizeof(_ExtendedStartupInformation),
                    dwFlags=_STARTF_USESTDHANDLES,
                    hStdInput=standard[0],
                    hStdOutput=standard[1],
                    hStdError=standard[-1],
                ),
                lpAttributeList=ctypes.addressof(attributes),
            )
            environment = isolated_environment()
            block = "".join(
                f"{name}={environment[name]}\0"
                for name in sorted(environment, key=str.upper)
            )
            started = ctypes.create_string_buffer(_PROCESS_INFORMATION_BYTES)
            if not _create_process(
                None,
                ctypes.create_unicode_buffer(subprocess.list2cmdline(arguments)),
                None,
                None,
                True,
                _EXTENDED_STARTUPINFO_PRESENT
                | _CREATE_UNICODE_ENVIRONMENT
                | (0 if console else _CREATE_NO_WINDOW),
                # Every entry ends with a terminator and the buffer adds the
                # one that ends the block. A character outside the Basic
                # Multilingual Plane takes two units here, which this handles.
                ctypes.create_unicode_buffer(block),
                str(cwd),
                ctypes.byref(startup),
                started,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _delete_attributes(attributes)
    finally:
        # Nothing this process starts by other means may inherit them.
        for handle in (*standard, fence):
            with contextlib.suppress(OSError):
                os.set_handle_inheritable(handle, False)
    information = started.raw
    process = int.from_bytes(information[:_POINTER_BYTES], sys.byteorder)
    thread = int.from_bytes(
        information[_POINTER_BYTES : 2 * _POINTER_BYTES], sys.byteorder
    )
    identifier = int.from_bytes(
        information[2 * _POINTER_BYTES : 2 * _POINTER_BYTES + 4], sys.byteorder
    )
    _close_handle(thread)
    return process, identifier


def _member_identifiers(job: int) -> list[int]:
    """The identifiers of the processes in the job now."""
    capacity = 256
    while True:
        size = _ID_LIST_HEADER_BYTES + _POINTER_BYTES * capacity
        listing = ctypes.create_string_buffer(size)
        if (
            not _query_job(job, _JOB_OBJECT_BASIC_PROCESS_ID_LIST, listing, size, None)
            and ctypes.get_last_error() != _ERROR_MORE_DATA
        ):
            raise _failure("could not list a job's processes")
        raw = listing.raw
        in_job = int.from_bytes(raw[0:4], sys.byteorder)
        listed = int.from_bytes(raw[4:8], sys.byteorder)
        if in_job > listed:
            capacity = in_job + 64
            continue
        return [
            int.from_bytes(
                raw[
                    _ID_LIST_HEADER_BYTES + index * _POINTER_BYTES : (
                        _ID_LIST_HEADER_BYTES + (index + 1) * _POINTER_BYTES
                    )
                ],
                sys.byteorder,
            )
            for index in range(listed)
        ]


def _end_all(job: int) -> list[int]:
    """End every process in the job; return a handle on each to wait for.

    Ending a process is not instant. Windows reports success, and the job's
    own count of members drops, while the process is still closing what it
    held; only when the process itself is finished are its files and
    directories free again. Its handle is what says so, and a handle can be
    had only while the process is still listed, which is before it is ended.

    So, in this order: stop the job taking new members, which makes the list
    complete; take a handle on every member; then end them.
    """
    _set_limits(job, admit_new_members=False)
    members: list[int] = []
    try:
        for identifier in _member_identifiers(job):
            handle = _open_process(
                _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, identifier
            )
            if not handle:
                # It ended between being listed and being opened.
                continue
            # The identifier may have gone to an unrelated process meanwhile.
            is_member = wt.BOOL()
            if _is_in_job(handle, job, ctypes.byref(is_member)) and is_member.value:
                members.append(handle)
            else:
                _close_handle(handle)
        if (
            not _terminate_job(job, 1)
            and ctypes.get_last_error() != _ERROR_ACCESS_DENIED
        ):
            raise _failure("could not end a job's processes")
    except BaseException:
        for handle in members:
            _close_handle(handle)
        raise
    return members


async def _until_finished(members: list[int]) -> None:
    """Wait until every ended process has finished, closing each handle as it does.

    There is no time limit, on purpose. Returning early would let the caller
    go on, and release its directory lock, while a process of the old tree
    still had files open there. POSIX waits for a killed child the same way.
    """
    polls = 0
    while True:
        unfinished: list[int] = []
        for handle in members:
            if _wait_for_handle(handle, 0) == _WAIT_OBJECT_0:
                _close_handle(handle)
            else:
                unfinished.append(handle)
        members = unfinished
        if not members:
            return
        polls += 1
        await asyncio.sleep(
            _FINISH_POLL_SECONDS
            if polls < _FINISH_QUICK_POLLS
            else _FINISH_SLOW_POLL_SECONDS
        )


def _await_exit(process: int) -> int:
    """Block until the process ends; return its exit code."""
    _wait_for_handle(process, _INFINITE)
    code = wt.DWORD()
    if not _exit_code(process, ctypes.byref(code)):
        raise _failure("could not read a process's exit code")
    return code.value


@final
class JobChild:
    """A child inside its own job, with its output stream and exit status."""

    def __init__(
        self,
        pid: int,
        stdout: asyncio.StreamReader,
        transport: asyncio.ReadTransport,
        exit_code: asyncio.Future[int],
        job: int,
    ) -> None:
        """Bind what :func:`start` created; use that function, not this one."""
        self.pid = pid
        self.stdout = stdout
        self._transport = transport
        self._exit_code = exit_code
        self._job: int | None = job
        self._ended: list[int] = []

    @property
    def returncode(self) -> int | None:
        """The child's exit code, or ``None`` while it is still running."""
        done = self._exit_code
        if not done.done() or done.cancelled() or done.exception() is not None:
            return None
        return done.result()

    async def wait(self) -> int:
        """Wait for the child to end; return its exit code."""
        return await asyncio.shield(self._exit_code)

    def kill(self) -> None:
        """End every process in the job; quiet when none is left."""
        if self._job is not None:
            self._ended.extend(_end_all(self._job))

    async def close(self) -> None:
        """End what is left of the tree, wait until it is gone, then release it.

        Returns only when every process of the tree has finished, however
        long that takes, so nothing of theirs is still open in the directory
        the caller goes on to use.
        """
        self._transport.close()
        if self._job is None:
            return
        self._ended.extend(_end_all(self._job))
        members, self._ended = self._ended, []
        await _until_finished(members)
        _close_handle(self._job)
        self._job = None


async def start(arguments: Sequence[str], *, cwd: Path, lock: Path) -> JobChild:
    """Start a program as a member of a new job, with its output as a stream.

    Args:
        arguments: The program and its arguments.
        cwd: The directory the program starts in.
        lock: The path of the lock that fences ``cwd``. The child holds that
            lock's children fence open, alone, for as long as it runs.

    Raises:
        ValueError: The lock's children fence is not an owner-only file.
        OSError: The program could not be started, or an earlier child still
            holds the fence.
    """
    loop = asyncio.get_running_loop()
    job = _new_job()
    try:
        # The reading end is overlapped, as the event loop requires; the
        # writing end is the plain handle a child expects for its output.
        read_end, write_end = windows_utils.pipe(overlapped=(True, False))
        try:
            fence = hold_children_fence(lock)
            empty_input = os.open(os.devnull, os.O_RDONLY)
            try:
                process, pid = _create_in_job(
                    arguments,
                    cwd,
                    job,
                    (msvcrt.get_osfhandle(empty_input), write_end),
                    msvcrt.get_osfhandle(fence),
                    console=False,
                )
            finally:
                os.close(empty_input)
                # From here the child's copy is the only one.
                os.close(fence)
        except BaseException:
            _close_handle(read_end)
            raise
        finally:
            # The child has its own copy; ours would keep the stream open.
            _close_handle(write_end)
        exit_code = loop.run_in_executor(None, _await_exit, process)
        # The waiting thread uses the handle until the process has ended.
        exit_code.add_done_callback(lambda _: _close_handle(process))
        try:
            reader = asyncio.StreamReader(loop=loop)
            transport, _ = await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader, loop=loop),
                windows_utils.PipeHandle(read_end),
            )
        except BaseException:
            # The child is already running: do not leave it, or anything of
            # its still open, behind the error.
            await asyncio.shield(_until_finished(_end_all(job)))
            raise
    except BaseException:
        _close_handle(job)
        raise
    return JobChild(pid, reader, transport, exit_code, job)


def _own_standard_stream(descriptor: int, flags: int) -> int:
    """A private copy of one of this process's standard streams, to pass on.

    A process started without one, or with one already closed, gets the null
    device in its place, as a POSIX program does when its parent closed the
    stream before starting it.
    """
    try:
        return os.dup(descriptor)
    except OSError:
        return os.open(os.devnull, flags)


def _wait_until_finished(members: list[int]) -> None:
    """Block until every ended process has finished; close each handle."""
    for handle in members:
        _wait_for_handle(handle, _INFINITE)
        _close_handle(handle)


def run_in_place(arguments: Sequence[str], *, lock: Path) -> int:
    """Run a program in this process's place and return its exit code.

    The program gets this process's own standard input, output and error, with
    nothing in between, and its console and working directory. It is a member
    of a new job and holds the lock's children fence, like any owned child, so
    it cannot outlive this process or the lock this process goes on holding.

    While the program runs, this process ignores Ctrl-C and Ctrl-Break: both
    reach every program on the console, and what they mean is the running
    program's to decide. When the program ends, whatever it left running in
    its job is ended and waited for before this returns.

    Args:
        arguments: The program and its arguments.
        lock: The path of the lock this process holds.

    Returns:
        The program's exit code, as Windows reports it.

    Raises:
        ValueError: The lock's children fence is not an owner-only file.
        OSError: The program could not be started.
    """
    job = _new_job()
    try:
        copies = [
            _own_standard_stream(0, os.O_RDONLY),
            _own_standard_stream(1, os.O_WRONLY),
            _own_standard_stream(2, os.O_WRONLY),
        ]
        ignored: dict[int, object] = {}
        try:
            fence = hold_children_fence(lock)
            try:
                # From before the program exists, so no interrupt can end
                # this process and leave the program without its owner.
                for number in (signal.SIGINT, signal.SIGBREAK):
                    with contextlib.suppress(ValueError):
                        ignored[number] = signal.signal(number, signal.SIG_IGN)
                process, _ = _create_in_job(
                    arguments,
                    Path.cwd(),
                    job,
                    tuple(msvcrt.get_osfhandle(copy) for copy in copies),
                    msvcrt.get_osfhandle(fence),
                    console=True,
                )
            finally:
                os.close(fence)
                for copy in copies:
                    os.close(copy)
            try:
                code = _await_exit(process)
            finally:
                _close_handle(process)
                _wait_until_finished(_end_all(job))
        finally:
            for number, previous in ignored.items():
                signal.signal(number, previous)  # pyright: ignore[reportArgumentType]
    finally:
        _close_handle(job)
    return code
