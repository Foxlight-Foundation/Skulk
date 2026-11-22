"""Which account a Windows process runs as, and whether it has root's authority."""

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from skulk.utils.runtime_files import has_root_authority

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows accounts and tokens"
)

_SAFER_SCOPE_USER = 2
_SAFER_LEVEL_NORMAL_USER = 0x20000
_SAFER_LEVEL_OPEN = 1


@contextmanager
def _as_an_ordinary_user() -> Iterator[None]:
    """Run the calling thread with this account's rights minus administrator.

    Windows derives the token an ordinary user would hold from the current
    one; the thread wears it until the block ends.
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
    advapi32.ImpersonateLoggedOnUser.argtypes = (wt.HANDLE,)
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
    try:
        assert advapi32.ImpersonateLoggedOnUser(token), ctypes.get_last_error()
        try:
            yield
        finally:
            assert advapi32.RevertToSelf()
    finally:
        kernel32.CloseHandle(token)


def test_the_account_is_a_security_identifier_and_does_not_change() -> None:
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    identifier = account.this_account()
    assert identifier.startswith("S-1-")
    assert account.this_account() == identifier
    assert account.process_account(account._get_current_process()) == identifier


def test_an_administrator_has_root_authority_and_an_ordinary_user_does_not() -> None:
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    if not account.this_process_is_administrator():
        # Already an ordinary user: nothing to remove.
        assert account.this_account() != account.SYSTEM_ACCOUNT
        assert not has_root_authority()
        return

    assert has_root_authority()
    with _as_an_ordinary_user():
        assert not account.this_process_is_administrator()
        assert not has_root_authority()
    # Back to the process's own rights.
    assert has_root_authority()


def test_the_system_account_has_root_authority_whatever_its_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sys.platform == "win32"
    from skulk.utils import _windows_account as account

    monkeypatch.setattr(account, "_account", account.SYSTEM_ACCOUNT)
    monkeypatch.setattr(account, "this_process_is_administrator", lambda: False)
    assert has_root_authority()
