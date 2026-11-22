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

A file lock on Windows belongs to one process and is not shared with a child,
so the directory's lock is not passed down as it is on POSIX. The job gives
the same result by a different route: when this process dies, its lock is
released and its tree is ended in the same teardown. Those two steps are not
one atomic step. Another process that takes the lock in the instant between
them could start while the old tree is being ended; the old tree runs no more
code from that point, so it cannot write behind the new holder.
"""

import sys

assert sys.platform == "win32"

import asyncio
import ctypes
import ctypes.wintypes as wt
import msvcrt
import os
import subprocess
import tempfile
from asyncio import windows_utils
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import final

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_CREATE_NO_WINDOW = 0x08000000
_STARTF_USESTDHANDLES = 0x00000100
_INFINITE = 0xFFFFFFFF
_ERROR_ACCESS_DENIED = 5


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
    limits = _ExtendedLimits(
        BasicLimitInformation=_BasicLimits(
            LimitFlags=_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
    )
    if not _set_job_information(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        failure = _failure("could not configure a job")
        _close_handle(job)
        raise failure
    return job


def _create_in_job(
    arguments: Sequence[str], cwd: Path, job: int, output: int
) -> tuple[int, int]:
    """Create the process as a member of ``job``; return its handle and identifier.

    The child inherits exactly two handles: an empty input and ``output`` for
    both of its output streams.
    """
    descriptor = os.open(os.devnull, os.O_RDONLY)
    try:
        empty_input = msvcrt.get_osfhandle(descriptor)
        os.set_handle_inheritable(empty_input, True)
        os.set_handle_inheritable(output, True)
        inherited = (wt.HANDLE * 2)(empty_input, output)
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
                    hStdInput=empty_input,
                    hStdOutput=output,
                    hStdError=output,
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
                | _CREATE_NO_WINDOW,
                # Built element by element: the block has a terminator after
                # every entry, and one more to end it.
                (ctypes.c_wchar * (len(block) + 1))(*block),
                str(cwd),
                ctypes.byref(startup),
                started,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _delete_attributes(attributes)
    finally:
        os.close(descriptor)
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
        if (
            self._job is not None
            and not _terminate_job(self._job, 1)
            and ctypes.get_last_error() != _ERROR_ACCESS_DENIED
        ):
            raise _failure("could not end a job's processes")

    def close(self) -> None:
        """Release the job and the output stream, ending anything still in the job."""
        if self._job is not None:
            _close_handle(self._job)
            self._job = None
        self._transport.close()


async def start(arguments: Sequence[str], *, cwd: Path) -> JobChild:
    """Start a program as a member of a new job, with its output as a stream."""
    loop = asyncio.get_running_loop()
    job = _new_job()
    try:
        # The reading end is overlapped, as the event loop requires; the
        # writing end is the plain handle a child expects for its output.
        read_end, write_end = windows_utils.pipe(overlapped=(True, False))
        try:
            process, pid = _create_in_job(arguments, cwd, job, write_end)
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
            _terminate_job(job, 1)
            raise
    except BaseException:
        _close_handle(job)
        raise
    return JobChild(pid, reader, transport, exit_code, job)
