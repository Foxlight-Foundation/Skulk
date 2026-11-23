"""Run a command on Windows as an ordinary user and exit with its exit code.

A GitHub-hosted Windows runner runs every step as an administrator. Skulk's
plugin installer refuses an administrator, as it refuses root, so a test that
installs a plugin is skipped in such a step. This script starts a command with
the same account's rights reduced to an ordinary user's: the token that
``runas /trustlevel:0x20000`` makes, in which the Administrators group can only
deny access and the administrator privileges are gone. The command gets this
process's console, standard streams, environment and working directory, and
this process exits with the command's exit code.

It does not change accounts. Files the command creates belong to the same
account as before, and what that account may read it may still read. What the
account can reach only as a member of Administrators, the command cannot.

Usage::

    python scripts/run_as_ordinary_windows_user.py COMMAND [ARGUMENT ...]

A COMMAND of ``python`` means the interpreter running this script. Windows
would otherwise look the name up beside the running program first, which in a
virtual environment is the base interpreter, without the environment's
packages.
"""

import ctypes
import subprocess
import sys
from ctypes import wintypes

# The rights of an ordinary user of the same account (winsafer.h).
_SAFER_SCOPEID_USER = 2
_SAFER_LEVELID_NORMALUSER = 0x20000
_SAFER_LEVEL_OPEN = 1

_STARTF_USESTDHANDLES = 0x100
_STD_HANDLES = (-10, -11, -12)  # input, output, error
_INFINITE = 0xFFFFFFFF


class _StartupInfo(ctypes.Structure):
    _fields_ = (
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    )


class _ProcessInformation(ctypes.Structure):
    process: int
    thread: int
    _fields_ = (
        ("process", wintypes.HANDLE),
        ("thread", wintypes.HANDLE),
        ("process_id", wintypes.DWORD),
        ("thread_id", wintypes.DWORD),
    )


def run(arguments: list[str]) -> int:
    """Run the command with an ordinary user's rights; return its exit code.

    Args:
        arguments: The program and its arguments.

    Returns:
        The command's exit code, as Windows reports it.

    Raises:
        OSError: The reduced token could not be made or the command could not
            be started.
    """
    if sys.platform != "win32":
        raise SystemExit("this script runs on Windows only")
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.restype = wintypes.HANDLE

    level = wintypes.HANDLE()
    if not advapi32.SaferCreateLevel(
        _SAFER_SCOPEID_USER,
        _SAFER_LEVELID_NORMALUSER,
        _SAFER_LEVEL_OPEN,
        ctypes.byref(level),
        None,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    token = wintypes.HANDLE()
    try:
        # With no token given, the result is a reduced copy of this process's
        # own, which is the one kind of token an unprivileged caller may start
        # a process with.
        if not advapi32.SaferComputeTokenFromLevel(
            level, None, ctypes.byref(token), 0, None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        advapi32.SaferCloseLevel(level)

    startup = _StartupInfo()
    startup.cb = ctypes.sizeof(startup)
    startup.dwFlags = _STARTF_USESTDHANDLES
    startup.hStdInput, startup.hStdOutput, startup.hStdError = (
        kernel32.GetStdHandle(wintypes.DWORD(number & 0xFFFFFFFF))
        for number in _STD_HANDLES
    )
    started = _ProcessInformation()
    command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(arguments))
    try:
        if not advapi32.CreateProcessAsUserW(
            token,
            None,
            command_line,
            None,
            None,
            True,
            0,
            None,
            None,
            ctypes.byref(startup),
            ctypes.byref(started),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(token)
    try:
        kernel32.WaitForSingleObject(started.process, _INFINITE)
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(started.process, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(started.thread)
        kernel32.CloseHandle(started.process)
    return code.value


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    command = sys.argv[1:]
    if command[0] == "python":
        command[0] = sys.executable
    sys.stdout.flush()
    result = run(command)
    # Windows exit codes are unsigned 32-bit values; Python exits with a
    # signed integer of the same bits.
    raise SystemExit(result - (1 << 32) if result >= 1 << 31 else result)
