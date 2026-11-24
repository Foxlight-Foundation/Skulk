"""Which account a Windows process runs as, and whether it has root's authority."""

import ctypes
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from skulk.utils.runtime_files import has_root_authority

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows accounts and tokens"
)

_SAFER_SCOPE_USER = 2
_SAFER_LEVEL_NORMAL_USER = 0x20000
_SAFER_LEVEL_OPEN = 1
_CREATE_NO_WINDOW = 0x08000000
_WAIT_FINISHED = 0

_REPORT_AUTHORITY = (
    "import sys\n"
    "from pathlib import Path\n"
    "from skulk.utils import _windows_account as account\n"
    "from skulk.utils.runtime_files import has_root_authority\n"
    "Path(sys.argv[1]).write_text(\n"
    "    f'{account.this_process_is_administrator()} {has_root_authority()}'\n"
    ")\n"
)


@contextmanager
def _ordinary_user_token() -> Iterator[int]:
    """The token an ordinary user would hold, derived from this process's own.

    It is this account's token with its administrator rights removed, which is
    what Windows gives a program started without elevation.
    """
    assert sys.platform == "win32"
    import ctypes.wintypes as wt

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.SaferCreateLevel.argtypes = (
        wt.DWORD,
        wt.DWORD,
        wt.DWORD,
        ctypes.POINTER(wt.HANDLE),
        wt.LPVOID,
    )
    advapi32.SaferComputeTokenFromLevel.argtypes = (
        wt.HANDLE,
        wt.HANDLE,
        ctypes.POINTER(wt.HANDLE),
        wt.DWORD,
        wt.LPVOID,
    )
    advapi32.SaferCloseLevel.argtypes = (wt.HANDLE,)
    kernel32.CloseHandle.argtypes = (wt.HANDLE,)

    level = wt.HANDLE()
    assert advapi32.SaferCreateLevel(
        _SAFER_SCOPE_USER,
        _SAFER_LEVEL_NORMAL_USER,
        _SAFER_LEVEL_OPEN,
        ctypes.byref(level),
        None,
    ), ctypes.get_last_error()
    token = wt.HANDLE()
    try:
        assert advapi32.SaferComputeTokenFromLevel(
            level, None, ctypes.byref(token), 0, None
        ), ctypes.get_last_error()
    finally:
        advapi32.SaferCloseLevel(level)
    assert token.value is not None
    try:
        yield token.value
    finally:
        kernel32.CloseHandle(token)


@contextmanager
def _thread_wearing(token: int) -> Iterator[None]:
    """Have the calling thread act under ``token`` until the block ends."""
    assert sys.platform == "win32"
    import ctypes.wintypes as wt

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.ImpersonateLoggedOnUser.argtypes = (wt.HANDLE,)
    assert advapi32.ImpersonateLoggedOnUser(token), ctypes.get_last_error()
    try:
        yield
    finally:
        assert advapi32.RevertToSelf()


def _run_under(token: int, arguments: list[str]) -> int:
    """Run a program as a new process holding ``token``; return its exit code."""
    assert sys.platform == "win32"
    import ctypes.wintypes as wt

    class StartupInformation(ctypes.Structure):
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

    class ProcessInformation(ctypes.Structure):
        _fields_ = (
            ("hProcess", wt.HANDLE),
            ("hThread", wt.HANDLE),
            ("dwProcessId", wt.DWORD),
            ("dwThreadId", wt.DWORD),
        )

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.CreateProcessAsUserW.argtypes = (
        wt.HANDLE,
        wt.LPCWSTR,
        wt.LPWSTR,
        wt.LPVOID,
        wt.LPVOID,
        wt.BOOL,
        wt.DWORD,
        wt.LPVOID,
        wt.LPCWSTR,
        ctypes.POINTER(StartupInformation),
        ctypes.POINTER(ProcessInformation),
    )
    kernel32.WaitForSingleObject.argtypes = (wt.HANDLE, wt.DWORD)
    kernel32.WaitForSingleObject.restype = wt.DWORD
    kernel32.GetExitCodeProcess.argtypes = (wt.HANDLE, ctypes.POINTER(wt.DWORD))
    kernel32.CloseHandle.argtypes = (wt.HANDLE,)

    startup = StartupInformation()
    startup.cb = ctypes.sizeof(startup)
    started = ProcessInformation()
    command = ctypes.create_unicode_buffer(subprocess.list2cmdline(arguments))
    assert advapi32.CreateProcessAsUserW(
        token,
        None,
        command,
        None,
        None,
        False,
        _CREATE_NO_WINDOW,
        None,
        None,
        ctypes.byref(startup),
        ctypes.byref(started),
    ), ctypes.get_last_error()
    try:
        assert kernel32.WaitForSingleObject(started.hProcess, 60_000) == _WAIT_FINISHED
        code = wt.DWORD()
        assert kernel32.GetExitCodeProcess(started.hProcess, ctypes.byref(code))
        return code.value
    finally:
        kernel32.CloseHandle(started.hThread)
        kernel32.CloseHandle(started.hProcess)


def test_the_profile_folder_is_the_accounts_own_whatever_the_variable_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    home = Path(os.environ["USERPROFILE"])
    assert Path(account.profile_directory()) == home
    monkeypatch.setenv("USERPROFILE", r"C:\somewhere\else")
    assert Path(account.profile_directory()) == home
    monkeypatch.delenv("USERPROFILE")
    assert Path(account.profile_directory()) == home


def test_the_account_is_a_security_identifier_and_does_not_change() -> None:
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    identifier = account.this_account()
    assert identifier.startswith("S-1-")
    assert account.this_account() == identifier
    assert account.process_account(account._get_current_process()) == identifier


def test_an_administrator_has_root_authority_and_an_ordinary_user_does_not(
    tmp_path: Path,
) -> None:
    """Checked in two real processes: this one, and one without administrator rights."""
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    if not account.this_process_is_administrator():
        # Already an ordinary user: there is no administrator side to show.
        assert account.this_account() != account.SYSTEM_ACCOUNT
        assert not has_root_authority()
        return

    assert has_root_authority()
    report = tmp_path / "authority.txt"
    with _ordinary_user_token() as token:
        assert (
            _run_under(token, [sys.executable, "-c", _REPORT_AUTHORITY, str(report)])
            == 0
        )
    assert report.read_text() == "False False"


def test_a_thread_wearing_a_lesser_identity_does_not_hide_the_process(
    tmp_path: Path,
) -> None:
    """The guard is about what the process can do, whatever one thread is wearing.

    An administrator process that had a thread act as an ordinary user could
    otherwise pass the guard and then take the identity off again.
    """
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    del tmp_path
    if not account.this_process_is_administrator():
        pytest.skip("needs an administrator process to have something to hide")

    with _ordinary_user_token() as token, _thread_wearing(token):
        assert account.this_process_is_administrator()
        assert has_root_authority()
    assert has_root_authority()


def test_the_system_account_has_root_authority_whatever_its_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    monkeypatch.setattr(account, "_account", account.SYSTEM_ACCOUNT)
    monkeypatch.setattr(account, "this_process_is_administrator", lambda: False)
    assert has_root_authority()
