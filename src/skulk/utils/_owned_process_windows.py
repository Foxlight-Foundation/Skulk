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

The child is created suspended and resumed only after it is in the job, so
nothing it starts can be outside the job. Jobs nest, so this works when this
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
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import psutil

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_CREATE_SUSPENDED = 0x00000004
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
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
_kernel32.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
_kernel32.OpenProcess.restype = wt.HANDLE
_kernel32.AssignProcessToJobObject.argtypes = (wt.HANDLE, wt.HANDLE)
_kernel32.AssignProcessToJobObject.restype = wt.BOOL
_kernel32.TerminateJobObject.argtypes = (wt.HANDLE, wt.UINT)
_kernel32.TerminateJobObject.restype = wt.BOOL
_kernel32.CloseHandle.argtypes = (wt.HANDLE,)
_kernel32.CloseHandle.restype = wt.BOOL

# Typed views of the foreign functions, so their results are not ``Any``.
_create_job: Callable[[None, None], int | None] = _kernel32.CreateJobObjectW
_set_job_information: Callable[[int, int, object, int], int] = (
    _kernel32.SetInformationJobObject
)
_open_process: Callable[[int, bool, int], int | None] = _kernel32.OpenProcess
_assign_to_job: Callable[[int, int], int] = _kernel32.AssignProcessToJobObject
_terminate_job: Callable[[int, int], int] = _kernel32.TerminateJobObject
_close_handle: Callable[[int], int] = _kernel32.CloseHandle


def _failure(what: str) -> OSError:
    error = ctypes.get_last_error()
    return OSError(0, f"{what}: {ctypes.FormatError(error)}", None, error)


def isolated_environment() -> dict[str, str]:
    """The least a Windows program needs: the system directory and a temp folder."""
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
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


async def start(
    arguments: Sequence[str], *, cwd: Path
) -> tuple[asyncio.subprocess.Process, int]:
    """Start a program inside its own job; return the process and the job."""
    job = _new_job()
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=isolated_environment(),
            # Suspended, so it cannot start anything before it is in the job.
            creationflags=_CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW,
        )
        try:
            handle = _open_process(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, process.pid
            )
            if not handle:
                raise _failure("could not open the started process")
            try:
                if not _assign_to_job(job, handle):
                    raise _failure("could not place the started process in its job")
            finally:
                _close_handle(handle)
            psutil.Process(process.pid).resume()
        except BaseException:
            # Never leave a suspended child behind: it would hold its pipes
            # open and never exit.
            process.kill()
            await process.wait()
            raise
    except BaseException:
        _close_handle(job)
        raise
    return process, job


def terminate(job: int) -> None:
    """End every process in the job; quiet when none is left."""
    if not _terminate_job(job, 1) and ctypes.get_last_error() != _ERROR_ACCESS_DENIED:
        raise _failure("could not end a job's processes")


def close(job: int) -> None:
    """Close this process's handle to the job, ending anything still in it."""
    _close_handle(job)
