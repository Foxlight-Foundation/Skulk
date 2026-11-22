# ruff: noqa: E402
# The platform assertion has to come before the Windows-only imports: it is
# what tells the type checker to skip this module on other platforms.
"""Which Windows account a process runs as, and with what authority.

POSIX code asks for a user ID and compares it with zero. Windows has neither:
an account is a security identifier, and authority over the machine comes
either from being the SYSTEM account or from running as administrator.
"""

import sys

assert sys.platform == "win32"

import ctypes
import ctypes.wintypes as wt
from collections.abc import Callable

SYSTEM_ACCOUNT = "S-1-5-18"

_TOKEN_QUERY = 0x0008
_TOKEN_DUPLICATE = 0x0002
_SECURITY_IDENTIFICATION = 1
_TOKEN_USER = 1
_BUILTIN_ADMINISTRATORS = 26  # WinBuiltinAdministratorsSid
_MAXIMUM_SID_BYTES = 68

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

_kernel32.GetCurrentProcess.restype = wt.HANDLE
_kernel32.GetCurrentProcess.argtypes = ()
_kernel32.CloseHandle.restype = wt.BOOL
_kernel32.CloseHandle.argtypes = (wt.HANDLE,)
_kernel32.LocalFree.restype = wt.LPVOID
_kernel32.LocalFree.argtypes = (wt.LPVOID,)
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
_advapi32.CreateWellKnownSid.restype = wt.BOOL
_advapi32.CreateWellKnownSid.argtypes = (
    ctypes.c_int,
    wt.LPVOID,
    wt.LPVOID,
    ctypes.POINTER(wt.DWORD),
)
_advapi32.DuplicateToken.restype = wt.BOOL
_advapi32.DuplicateToken.argtypes = (wt.HANDLE, ctypes.c_int, ctypes.POINTER(wt.HANDLE))
_advapi32.CheckTokenMembership.restype = wt.BOOL
_advapi32.CheckTokenMembership.argtypes = (
    wt.HANDLE,
    wt.LPVOID,
    ctypes.POINTER(wt.BOOL),
)
_advapi32.ConvertSidToStringSidW.restype = wt.BOOL
_advapi32.ConvertSidToStringSidW.argtypes = (wt.LPVOID, ctypes.POINTER(wt.LPWSTR))

# ctypes calls return untyped values. These views give each foreign function
# the result type its ``restype`` above produces.
_get_current_process: Callable[..., int] = _kernel32.GetCurrentProcess
_close_handle: Callable[..., int] = _kernel32.CloseHandle
_local_free: Callable[..., object] = _kernel32.LocalFree
_open_process_token: Callable[..., int] = _advapi32.OpenProcessToken
_get_token_information: Callable[..., int] = _advapi32.GetTokenInformation
_sid_to_text: Callable[..., int] = _advapi32.ConvertSidToStringSidW
_create_well_known_sid: Callable[..., int] = _advapi32.CreateWellKnownSid
_check_token_membership: Callable[..., int] = _advapi32.CheckTokenMembership
_duplicate_token: Callable[..., int] = _advapi32.DuplicateToken


def _last_error(action: str) -> OSError:
    return ctypes.WinError(ctypes.get_last_error(), action)


def process_account(process: int) -> str:
    """The security identifier of the account a process runs as, as text.

    Args:
        process: A handle to the process, open for querying.

    Returns:
        The identifier in its ``S-1-...`` form.
    """
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


def this_account() -> str:
    """This process's account identifier; it cannot change while it runs."""
    global _account
    if _account is None:
        _account = process_account(_get_current_process())
    return _account


def this_process_is_administrator() -> bool:
    """Whether this process holds an active membership of Administrators.

    This is the test Windows itself uses for "running as administrator". A
    standard token, or one restricted to an ordinary user's rights, carries
    the group for denials only and is not a member; an elevated token and the
    SYSTEM account are. The token's elevation flag is not used: it stays set
    on a token that has had its administrator rights removed.

    The process's own token is examined, never the calling thread's. A thread
    can wear a lesser identity for a while and then drop it, which says
    nothing about what the process can do.
    """
    size = wt.DWORD(_MAXIMUM_SID_BYTES)
    administrators = ctypes.create_string_buffer(_MAXIMUM_SID_BYTES)
    if not _create_well_known_sid(
        _BUILTIN_ADMINISTRATORS, None, administrators, ctypes.byref(size)
    ):
        raise _last_error("CreateWellKnownSid")
    process_token = wt.HANDLE()
    if not _open_process_token(
        _get_current_process(),
        _TOKEN_QUERY | _TOKEN_DUPLICATE,
        ctypes.byref(process_token),
    ):
        raise _last_error("OpenProcessToken")
    try:
        # Membership can only be asked of an impersonation token, so the
        # process's token is copied as one; nothing impersonates with it.
        copy = wt.HANDLE()
        if not _duplicate_token(
            process_token, _SECURITY_IDENTIFICATION, ctypes.byref(copy)
        ):
            raise _last_error("DuplicateToken")
        try:
            member = wt.BOOL()
            if not _check_token_membership(copy, administrators, ctypes.byref(member)):
                raise _last_error("CheckTokenMembership")
            return bool(member.value)
        finally:
            _close_handle(copy)
    finally:
        _close_handle(process_token)


def this_process_has_machine_authority() -> bool:
    """Whether this process could change the machine for every account.

    True for the SYSTEM account and for a process running as administrator:
    the two ways a Windows process holds what root holds on POSIX.
    """
    return this_account() == SYSTEM_ACCOUNT or this_process_is_administrator()
