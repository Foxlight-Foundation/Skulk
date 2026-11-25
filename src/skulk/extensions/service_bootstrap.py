"""Standard-library-only verification before starting the fixed plugin manager.

Local setup copies this file beside protected service state. The OS service runs
it with isolated base Python and site initialization disabled, so a changed .pth
or package cannot execute before the service-runtime seal is checked.

Because it may import nothing but the standard library, this file asks the
operating system who owns a file itself, and restates the rules that the rest
of Skulk takes from ``skulk.utils.runtime_files``. On Windows those rules are:
*protected* state is owned by the account, with an access list that inherits
nothing and names the account and SYSTEM only; a *runtime member* is owned by
the account and grants access to nobody but the account and SYSTEM, inherited
or not; and nothing is ever opened through a reparse point.
"""

import argparse
import functools
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import cast

_MAXIMUM_FILES = 200000
_MAXIMUM_BYTES = 17179869184
_MAXIMUM_MANIFEST = 67108864
_MAXIMUM_CONFIGURATION = 8192
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_GENERATION = re.compile(r"^[a-f0-9]{32}$")

LIFETIME_LOCK = "bootstrap.lock"
"""The lock this file's process holds on Windows for as long as it runs.

launchd and systemd know whether the service they started is still there.
Task Scheduler's answer comes as text in the system's language, so on Windows
local setup asks this lock: while it is held, a service is starting, serving
or stopping. Setup creates the file; this file only ever opens it.
"""

STOP_REQUEST = "service-stop"
"""The file whose presence in the service root asks a Windows service to stop.

Windows has no signal one process can send another to say "stop": the nearest
thing, Ctrl-Break, reaches only programs on the sender's own console. Setup
creates this file, and the process this file runs passes the request on to
the manager as Ctrl-Break and waits for it. The file then stays until the
service is installed again, and while it is there nothing starts: a stop
holds even where the task that starts the service could not be switched off.
"""

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wt
    import msvcrt
    from collections.abc import Callable

    _SYSTEM_ACCOUNT = "S-1-5-18"
    _FILE_GENERIC_READ = 0x00120089
    _FILE_SHARE_ALL = 0x00000007
    _OPEN_EXISTING = 3
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _SE_FILE_OBJECT = 1
    _OWNER_AND_ACCESS_LIST = 0x00000001 | 0x00000004
    _SE_DACL_PROTECTED = 0x1000
    _ACL_SIZE_INFORMATION = 2
    _ACCESS_ALLOWED_ACE = 0
    _ACCESS_DENIED_ACE = 1
    _TOKEN_QUERY = 0x0008
    _TOKEN_DUPLICATE = 0x0002
    _TOKEN_USER = 1
    _SECURITY_IDENTIFICATION = 1
    _BUILTIN_ADMINISTRATORS = 26
    _MAXIMUM_SID_BYTES = 68
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _LOCKFILE_EXCLUSIVE_NOW = 0x00000002 | 0x00000001
    _CTRL_BREAK_EVENT = 1
    _CREATE_NEW_PROCESS_GROUP = 0x00000200
    _CREATE_NO_WINDOW = 0x08000000

    class _Region(ctypes.Structure):
        """Where in a file a lock lies (the system's ``OVERLAPPED``)."""

        _fields_ = (
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wt.DWORD),
            ("OffsetHigh", wt.DWORD),
            ("hEvent", wt.HANDLE),
        )

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
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _kernel32.CreateFileW.argtypes = (
        wt.LPCWSTR,
        wt.DWORD,
        wt.DWORD,
        wt.LPVOID,
        wt.DWORD,
        wt.DWORD,
        wt.HANDLE,
    )
    _kernel32.CreateFileW.restype = wt.HANDLE
    _kernel32.CloseHandle.argtypes = (wt.HANDLE,)
    _kernel32.CloseHandle.restype = wt.BOOL
    _kernel32.LocalFree.argtypes = (wt.LPVOID,)
    _kernel32.LocalFree.restype = wt.LPVOID
    _kernel32.GetCurrentProcess.argtypes = ()
    _kernel32.GetCurrentProcess.restype = wt.HANDLE
    _kernel32.CreateJobObjectW.argtypes = (wt.LPVOID, wt.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wt.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wt.HANDLE,
        ctypes.c_int,
        wt.LPVOID,
        wt.DWORD,
    )
    _kernel32.SetInformationJobObject.restype = wt.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = (wt.HANDLE, wt.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wt.BOOL
    _advapi32.GetSecurityInfo.argtypes = (
        wt.HANDLE,
        ctypes.c_int,
        wt.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )
    _advapi32.GetSecurityInfo.restype = wt.DWORD
    _advapi32.GetSecurityDescriptorControl.argtypes = (
        wt.LPVOID,
        ctypes.POINTER(wt.WORD),
        ctypes.POINTER(wt.DWORD),
    )
    _advapi32.GetSecurityDescriptorControl.restype = wt.BOOL
    _advapi32.GetAclInformation.argtypes = (
        wt.LPVOID,
        wt.LPVOID,
        wt.DWORD,
        ctypes.c_int,
    )
    _advapi32.GetAclInformation.restype = wt.BOOL
    _advapi32.GetAce.argtypes = (wt.LPVOID, wt.DWORD, ctypes.POINTER(ctypes.c_void_p))
    _advapi32.GetAce.restype = wt.BOOL
    _advapi32.ConvertSidToStringSidW.argtypes = (wt.LPVOID, ctypes.POINTER(wt.LPWSTR))
    _advapi32.ConvertSidToStringSidW.restype = wt.BOOL
    _advapi32.OpenProcessToken.argtypes = (
        wt.HANDLE,
        wt.DWORD,
        ctypes.POINTER(wt.HANDLE),
    )
    _advapi32.OpenProcessToken.restype = wt.BOOL
    _advapi32.GetTokenInformation.argtypes = (
        wt.HANDLE,
        ctypes.c_int,
        wt.LPVOID,
        wt.DWORD,
        ctypes.POINTER(wt.DWORD),
    )
    _advapi32.GetTokenInformation.restype = wt.BOOL
    _advapi32.CreateWellKnownSid.argtypes = (
        ctypes.c_int,
        wt.LPVOID,
        wt.LPVOID,
        ctypes.POINTER(wt.DWORD),
    )
    _advapi32.CreateWellKnownSid.restype = wt.BOOL
    _advapi32.DuplicateToken.argtypes = (
        wt.HANDLE,
        ctypes.c_int,
        ctypes.POINTER(wt.HANDLE),
    )
    _advapi32.DuplicateToken.restype = wt.BOOL
    _advapi32.CheckTokenMembership.argtypes = (
        wt.HANDLE,
        wt.LPVOID,
        ctypes.POINTER(wt.BOOL),
    )
    _advapi32.CheckTokenMembership.restype = wt.BOOL
    _kernel32.LockFileEx.argtypes = (
        wt.HANDLE,
        wt.DWORD,
        wt.DWORD,
        wt.DWORD,
        wt.DWORD,
        ctypes.POINTER(_Region),
    )
    _kernel32.LockFileEx.restype = wt.BOOL
    _kernel32.GetConsoleCP.argtypes = ()
    _kernel32.GetConsoleCP.restype = wt.UINT
    _kernel32.AttachConsole.argtypes = (wt.DWORD,)
    _kernel32.AttachConsole.restype = wt.BOOL
    _kernel32.GenerateConsoleCtrlEvent.argtypes = (wt.DWORD, wt.DWORD)
    _kernel32.GenerateConsoleCtrlEvent.restype = wt.BOOL
    _ControlHandler = ctypes.WINFUNCTYPE(wt.BOOL, wt.DWORD)
    _kernel32.SetConsoleCtrlHandler.argtypes = (_ControlHandler, wt.BOOL)
    _kernel32.SetConsoleCtrlHandler.restype = wt.BOOL
    _userenv = ctypes.WinDLL("userenv", use_last_error=True)
    _userenv.GetUserProfileDirectoryW.argtypes = (
        wt.HANDLE,
        wt.LPWSTR,
        ctypes.POINTER(wt.DWORD),
    )
    _userenv.GetUserProfileDirectoryW.restype = wt.BOOL

    # Typed views of the foreign functions, so their results are not ``Any``.
    _create_file: Callable[[str, int, int, None, int, int, None], int | None] = (
        _kernel32.CreateFileW
    )
    _close_handle: Callable[[object], int] = _kernel32.CloseHandle
    _local_free: Callable[[object], object] = _kernel32.LocalFree
    _current_process: Callable[[], int] = _kernel32.GetCurrentProcess
    _create_job: Callable[[None, None], int | None] = _kernel32.CreateJobObjectW
    _set_job_information: Callable[[int, int, object, int], int] = (
        _kernel32.SetInformationJobObject
    )
    _assign_to_job: Callable[[int, int], int] = _kernel32.AssignProcessToJobObject
    _get_security_info: Callable[..., int] = _advapi32.GetSecurityInfo
    _get_control: Callable[[object, object, object], int] = (
        _advapi32.GetSecurityDescriptorControl
    )
    _get_acl_information: Callable[[object, object, int, int], int] = (
        _advapi32.GetAclInformation
    )
    _get_ace: Callable[[object, int, object], int] = _advapi32.GetAce
    _sid_to_text: Callable[[object, object], int] = _advapi32.ConvertSidToStringSidW
    _open_process_token: Callable[[int, int, object], int] = _advapi32.OpenProcessToken
    _token_information: Callable[[object, int, object, object, object], int] = (
        _advapi32.GetTokenInformation
    )
    _create_well_known_sid: Callable[[int, None, object, object], int] = (
        _advapi32.CreateWellKnownSid
    )
    _duplicate_token: Callable[[object, int, object], int] = _advapi32.DuplicateToken
    _check_membership: Callable[[object, object, object], int] = (
        _advapi32.CheckTokenMembership
    )
    _get_profile_directory: Callable[[object, object, object], int] = (
        _userenv.GetUserProfileDirectoryW
    )
    _lock_file: Callable[[int, int, int, int, int, object], int] = _kernel32.LockFileEx
    _console_code_page: Callable[[], int] = _kernel32.GetConsoleCP
    _attach_console: Callable[[int], int] = _kernel32.AttachConsole
    _send_console_event: Callable[[int, int], int] = _kernel32.GenerateConsoleCtrlEvent
    _set_control_handler: Callable[[object, int], int] = _kernel32.SetConsoleCtrlHandler
    _INVALID_HANDLE = wt.HANDLE(-1).value

    def _failure() -> OSError:
        return ctypes.WinError(ctypes.get_last_error())

    def _identifier(address: int | None) -> str:
        """A security identifier in its ``S-1-...`` form."""
        text = wt.LPWSTR()
        if not address or not _sid_to_text(address, ctypes.byref(text)):
            raise _failure()
        try:
            value = text.value
        finally:
            _local_free(text)
        if not value:
            raise OSError("account identifier is empty")
        return value

    @functools.cache
    def _account() -> str:
        """The account this process runs as; it cannot change while it runs."""
        token = wt.HANDLE()
        if not _open_process_token(
            _current_process(), _TOKEN_QUERY, ctypes.byref(token)
        ):
            raise _failure()
        try:
            size = wt.DWORD()
            _token_information(token, _TOKEN_USER, None, 0, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            if not _token_information(
                token, _TOKEN_USER, buffer, size, ctypes.byref(size)
            ):
                raise _failure()
            # TOKEN_USER begins with a pointer to the account's identifier.
            return _identifier(ctypes.c_void_p.from_buffer(buffer).value)
        finally:
            _close_handle(token)

    def _profile_directory() -> str:
        """The folder Windows keeps for this account, asked of Windows itself."""
        token = wt.HANDLE()
        if not _open_process_token(
            _current_process(), _TOKEN_QUERY, ctypes.byref(token)
        ):
            raise _failure()
        try:
            size = wt.DWORD()
            _get_profile_directory(token, None, ctypes.byref(size))
            buffer = ctypes.create_unicode_buffer(size.value)
            if not size.value or not _get_profile_directory(
                token, buffer, ctypes.byref(size)
            ):
                raise _failure()
            return ctypes.wstring_at(buffer)
        finally:
            _close_handle(token)

    def _is_administrator() -> bool:
        """Whether this process holds an active membership of Administrators.

        The process's own token is asked, and its elevation flag is not used:
        that stays set on a token whose administrator rights were removed.
        """
        size = wt.DWORD(_MAXIMUM_SID_BYTES)
        administrators = ctypes.create_string_buffer(_MAXIMUM_SID_BYTES)
        if not _create_well_known_sid(
            _BUILTIN_ADMINISTRATORS, None, administrators, ctypes.byref(size)
        ):
            raise _failure()
        token = wt.HANDLE()
        if not _open_process_token(
            _current_process(), _TOKEN_QUERY | _TOKEN_DUPLICATE, ctypes.byref(token)
        ):
            raise _failure()
        try:
            # Membership can only be asked of an impersonation token, so the
            # process's token is copied as one; nothing impersonates with it.
            copy = wt.HANDLE()
            if not _duplicate_token(
                token, _SECURITY_IDENTIFICATION, ctypes.byref(copy)
            ):
                raise _failure()
            try:
                member = wt.BOOL()
                if not _check_membership(copy, administrators, ctypes.byref(member)):
                    raise _failure()
                return bool(member.value)
            finally:
                _close_handle(copy)
        finally:
            _close_handle(token)

    def _open_entry(path: Path) -> int:
        """Open a file or folder for reading as itself; the caller closes it.

        A reparse point is opened, not followed, so the caller can refuse it.
        """
        handle = _create_file(
            str(path),
            _FILE_GENERIC_READ,
            _FILE_SHARE_ALL,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_OPEN_REPARSE_POINT | _FILE_FLAG_BACKUP_SEMANTICS,
            None,
        )
        if not handle or handle == _INVALID_HANDLE:
            raise _failure()
        try:
            return msvcrt.open_osfhandle(handle, os.O_RDONLY)
        except BaseException:
            _close_handle(handle)
            raise

    def _is_reparse_point(info: os.stat_result) -> bool:
        # A junction reports itself as a folder; the attribute is what says
        # that the entry stands for another.
        return bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)

    def _access(descriptor: int) -> tuple[bool, bool, bool]:
        """Who can reach an open entry: ``(owned, protected, closed)``.

        ``owned``: the account owns it. ``protected``: its access list takes
        nothing from the folder above. ``closed``: the list grants access to
        nobody but the account and SYSTEM. A volume without access lists
        (FAT, exFAT) answers False to all three.
        """
        owner = ctypes.c_void_p()
        access_list = ctypes.c_void_p()
        security = ctypes.c_void_p()
        if _get_security_info(
            msvcrt.get_osfhandle(descriptor),
            _SE_FILE_OBJECT,
            _OWNER_AND_ACCESS_LIST,
            ctypes.byref(owner),
            None,
            ctypes.byref(access_list),
            None,
            ctypes.byref(security),
        ):
            return False, False, False
        try:
            owned = owner.value is not None and _identifier(owner.value) == _account()
            control = wt.WORD()
            revision = wt.DWORD()
            protected = bool(
                _get_control(security, ctypes.byref(control), ctypes.byref(revision))
                and control.value & _SE_DACL_PROTECTED
            )
            return owned, protected, _grants_no_other(access_list.value)
        finally:
            _local_free(security)

    def _grants_no_other(access_list: int | None) -> bool:
        if access_list is None:
            # No access list at all grants everything to everyone.
            return False
        # ACL_SIZE_INFORMATION is three 32-bit counts: entries, then the bytes
        # used and free.
        size = ctypes.create_string_buffer(12)
        if not _get_acl_information(access_list, size, 12, _ACL_SIZE_INFORMATION):
            return False
        allowed = {_account(), _SYSTEM_ACCOUNT}
        for index in range(int.from_bytes(size.raw[:4], sys.byteorder)):
            entry = ctypes.c_void_p()
            if not _get_ace(access_list, index, ctypes.byref(entry)) or not entry.value:
                return False
            kind = ctypes.c_ubyte.from_address(entry.value).value
            if kind == _ACCESS_DENIED_ACE:
                continue
            if kind != _ACCESS_ALLOWED_ACE:
                # Any other kind of grant is someone this cannot vouch for.
                return False
            # A grant is a four-byte header, a four-byte mask, then the
            # identifier of the account it names.
            if _identifier(entry.value + 8) not in allowed:
                return False
        return True

    def _owner_only(descriptor: int) -> bool:
        return all(_access(descriptor))


def private_read(path: Path, maximum: int) -> bytes:
    """Read a bounded owner-only regular file without following a symlink."""
    if sys.platform == "win32":
        descriptor = _open_entry(path)
    else:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if sys.platform == "win32":
            protected = (
                stat.S_ISREG(info.st_mode)
                and not _is_reparse_point(info)
                and _owner_only(descriptor)
            )
        else:
            protected = (
                stat.S_ISREG(info.st_mode)
                and info.st_uid == os.getuid()
                and not info.st_mode & 0o077
            )
        if not protected:
            raise ValueError("service metadata is not protected")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            value = source.read(maximum + 1)
        if len(value) > maximum:
            raise ValueError("service metadata exceeds bound")
        return value
    finally:
        os.close(descriptor)


def private_directory(path: Path) -> None:
    """Require an existing protected directory, never create it during bootstrap."""
    if sys.platform == "win32":
        descriptor = _open_entry(path)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(info.st_mode)
                or _is_reparse_point(info)
                or not _owner_only(descriptor)
            ):
                raise ValueError("service directory is not protected")
        finally:
            os.close(descriptor)
        return
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("service directory is not protected")


def document(raw: bytes) -> dict[str, object]:
    """Decode one JSON object with string keys, rejecting duplicate properties."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate service metadata property")
            result[name] = value
        return result

    value = cast(object, json.loads(raw, object_pairs_hook=unique))
    if not isinstance(value, dict):
        raise ValueError("service metadata must be an object")
    return cast(dict[str, object], value)


def digest_file(path: Path) -> str:
    """Hash a regular file without allowing its last component to become a link."""
    if sys.platform == "win32":
        descriptor = _open_entry(path)
    else:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or (
            sys.platform == "win32" and _is_reparse_point(info)
        ):
            raise ValueError("service member is not a file")
        return hashlib.file_digest(source, "sha256").hexdigest()


def _desktop_metadata(path: Path, info: os.stat_result) -> bool:
    """A regular file a desktop file browser leaves behind, never imported or run.

    The same rule as ``skulk.extensions.runtime_files.is_desktop_metadata``,
    restated here because this file may import only the standard library.
    Finder writes ``.DS_Store`` into every folder it shows; sealing it would
    let an operator browsing the runtime stop the manager from starting.
    """
    named = path.name == ".DS_Store" or path.name.startswith("._")
    return named and stat.S_ISREG(info.st_mode)


def current_base() -> Path:
    """The interpreter a service copy is built on and its seal must name.

    A Windows process started from an environment has that environment's
    launcher as its executable, which is a copy and not a link to anything,
    so the base is taken from where Python itself records it. Run directly on
    the base, as the OS service runs this file, the two are the same, with
    one difference: the service runs ``pythonw.exe``, the same interpreter
    without a console window, and the base is the ``python.exe`` beside it.
    """
    if sys.platform == "win32":
        base = Path(getattr(sys, "_base_executable", sys.executable)).resolve(
            strict=True
        )
        if base.name.lower() == "pythonw.exe":
            base = base.with_name("python.exe").resolve(strict=True)
        return base
    return Path(sys.executable).resolve(strict=True)


def base_identity(base: Path) -> dict[str, str]:
    """What a manifest records about the base interpreter, as it is now.

    Windows adds the interpreter's library: ``python.exe`` there is a small
    program that loads ``python3XX.dll`` from its own folder, and the
    environment holds no link to either.

    Raises:
        OSError: The interpreter, or on Windows its library, cannot be read.
            A base missing either is refused, never recorded.
    """
    identity = {"base_python": str(base), "base_sha256": digest_file(base)}
    if sys.platform == "win32":
        library = base.with_name(
            f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        )
        identity["base_library_sha256"] = digest_file(library)
    return identity


def names_base(metadata: dict[str, object], base: Path) -> bool:
    """Whether a manifest was sealed to ``base`` with the bytes it has now."""
    return all(
        metadata.get(name) == value for name, value in base_identity(base).items()
    )


def manager_interpreter(runtime: Path) -> Path:
    """The interpreter ``venv`` gives a service copy on this platform."""
    if sys.platform == "win32":
        return runtime / "Scripts" / "python.exe"
    return runtime / "bin" / "python"


def _configured_home(configuration: bytes) -> Path:
    """The folder ``pyvenv.cfg`` says the base interpreter is in."""
    for line in configuration.decode("utf-8").splitlines():
        key, separator, value = line.partition(" = ")
        if separator and key == "home":
            return Path(value)
    raise ValueError("service runtime configuration is not recognized")


if sys.platform == "win32":

    def _windows_tree(root: Path, base_python: Path) -> dict[str, list[str | int]]:
        """The Windows form of :func:`runtime_tree`.

        ``venv`` makes no link there: the environment's interpreter is a
        launcher that finds the base through ``home`` in ``pyvenv.cfg``. So
        every reparse point is refused (a junction needs no privilege), each
        member must be the account's and grant access to nobody but the
        account and SYSTEM, and ``home`` must be the base's folder. Each
        member is opened once and examined and hashed through that handle.
        """
        entries: dict[str, list[str | int]] = {}
        size = 0
        for path in (root, *sorted(root.rglob("*"))):
            if len(entries) >= _MAXIMUM_FILES:
                raise ValueError("service runtime file count exceeds bound")
            relative = "." if path == root else path.relative_to(root).as_posix()
            descriptor = _open_entry(path)
            try:
                info = os.fstat(descriptor)
                link = _is_reparse_point(info)
                if relative != "." and not link and _desktop_metadata(path, info):
                    continue
                owned, _, closed = _access(descriptor)
                if not owned:
                    raise ValueError("service runtime ownership differs")
                if link:
                    raise ValueError("service runtime contains an unsupported link")
                if not closed:
                    raise ValueError("service runtime is writable by another user")
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISDIR(info.st_mode):
                    entries[relative] = ["directory", mode]
                elif stat.S_ISREG(info.st_mode):
                    size += info.st_size
                    if size > _MAXIMUM_BYTES:
                        raise ValueError("service runtime size exceeds bound")
                    with os.fdopen(descriptor, "rb", closefd=False) as source:
                        if relative == "pyvenv.cfg":
                            content = source.read(_MAXIMUM_CONFIGURATION + 1)
                            if len(content) > _MAXIMUM_CONFIGURATION:
                                raise ValueError(
                                    "service runtime configuration exceeds bound"
                                )
                            if _configured_home(content) != base_python.parent:
                                # The launcher starts whatever this names.
                                raise ValueError("service runtime interpreter differs")
                            digest = hashlib.sha256(content).hexdigest()
                        else:
                            digest = hashlib.file_digest(source, "sha256").hexdigest()
                    entries[relative] = ["file", mode, digest]
                else:
                    raise ValueError("service runtime contains an unsupported member")
            finally:
                os.close(descriptor)
        return entries


def runtime_tree(root: Path, base_python: Path) -> dict[str, list[str | int]]:
    """Seal the complete bounded runtime tree and its fixed interpreter aliases.

    File bytes, directory membership and permissions are included. Only venv's
    interpreter aliases and conventional lib64 link may point outside a file;
    arbitrary links back into development environments are refused. Windows
    has no such links and accepts none.
    """
    private_directory(root)
    if sys.platform == "win32":
        return _windows_tree(root, base_python)
    entries: dict[str, list[str | int]] = {
        ".": ["directory", stat.S_IMODE(root.stat().st_mode)]
    }
    size = 0
    for path in sorted(root.rglob("*")):
        if len(entries) >= _MAXIMUM_FILES:
            raise ValueError("service runtime file count exceeds bound")
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if _desktop_metadata(path, info):
            continue
        mode = stat.S_IMODE(info.st_mode)
        if info.st_uid != os.getuid():
            raise ValueError("service runtime ownership differs")
        if stat.S_ISLNK(info.st_mode):
            if relative == "lib64" and os.readlink(path) == "lib":
                entries[relative] = ["link", mode, "lib"]
            elif (
                relative
                in {
                    "bin/python",
                    "bin/python3",
                    f"bin/python{sys.version_info.major}.{sys.version_info.minor}",
                }
                and path.resolve(strict=True) == base_python
            ):
                entries[relative] = [
                    "python",
                    mode,
                    str(base_python),
                    digest_file(base_python),
                ]
            else:
                raise ValueError("service runtime contains an unsupported link")
        elif mode & 0o022:
            raise ValueError("service runtime is writable by another user")
        elif stat.S_ISDIR(info.st_mode):
            entries[relative] = ["directory", mode]
        elif stat.S_ISREG(info.st_mode):
            size += info.st_size
            if size > _MAXIMUM_BYTES:
                raise ValueError("service runtime size exceeds bound")
            entries[relative] = ["file", mode, digest_file(path)]
        else:
            raise ValueError("service runtime contains an unsupported member")
    return entries


def has_root_authority() -> bool:
    """Whether this process could change the machine for every account.

    The effective root user on POSIX; on Windows the SYSTEM account or a
    process running as administrator.
    """
    if sys.platform == "win32":
        return _account() == _SYSTEM_ACCOUNT or _is_administrator()
    return os.geteuid() == 0


def verified_runtime(root: Path) -> Path:
    """Verify the selected service copy before allowing Python site initialization."""
    if has_root_authority():
        raise ValueError("plugin manager must run without root")
    private_directory(root)
    pointer = document(private_read(root / "core-runtime.json", 4096))
    if set(pointer) != {"generation", "manifest_sha256"}:
        raise ValueError("invalid service selection")
    name = pointer["generation"]
    digest = pointer["manifest_sha256"]
    if (
        not isinstance(name, str)
        or not _GENERATION.fullmatch(name)
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
    ):
        raise ValueError("invalid service selection")
    private_directory(root / "core-runtimes")
    generation = root / "core-runtimes" / name
    private_directory(generation)
    raw = private_read(generation / "snapshot.json", _MAXIMUM_MANIFEST)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("service manifest integrity differs")
    metadata = document(raw)
    if metadata.get("protocol") != 1 or metadata.get("generation") != name:
        raise ValueError("service generation differs")
    base = current_base()
    if not names_base(metadata, base):
        raise ValueError("service base Python differs")
    if metadata.get("bootstrap_sha256") != digest_file(root / "service-bootstrap.py"):
        raise ValueError("service bootstrap differs")
    runtime = generation / "runtime"
    if metadata.get("entries") != runtime_tree(runtime, base):
        raise ValueError("service runtime integrity differs")
    return runtime


def selected_base_matches(root: Path, base: Path) -> bool:
    """Whether the selected runtime was sealed to ``base`` as it is now.

    A cheap part of :func:`verified_runtime`: it reads the selection and its
    manifest and hashes only the interpreter, which is what an app update
    replaces in place (same path, new bytes), leaving a registered service
    that can no longer start.

    Args:
        root: The service root holding ``core-runtime.json``.
        base: The interpreter the current Skulk runs on.

    Returns:
        False when the selection is missing, unreadable, or sealed to another
        interpreter path or other interpreter bytes.
    """
    try:
        pointer = document(private_read(root / "core-runtime.json", 4096))
        name = pointer.get("generation")
        digest = pointer.get("manifest_sha256")
        if (
            not isinstance(name, str)
            or not _GENERATION.fullmatch(name)
            or not isinstance(digest, str)
            or not _DIGEST.fullmatch(digest)
        ):
            return False
        raw = private_read(
            root / "core-runtimes" / name / "snapshot.json", _MAXIMUM_MANIFEST
        )
        if hashlib.sha256(raw).hexdigest() != digest:
            return False
        return names_base(document(raw), base)
    except (OSError, ValueError):
        return False


def manager_command(root: Path, runtime: Path) -> list[str]:
    """The one command this file ever starts: the manager, from ``runtime``."""
    return [
        str(manager_interpreter(runtime)),
        "-I",
        "-B",
        "-m",
        "skulk.extensions.runtime_manager",
        "serve",
        "--root",
        str(root),
    ]


def manager_environment(root: Path) -> dict[str, str]:
    """The fixed environment the manager starts with.

    Nothing of the service's own environment reaches it except, on Windows,
    where the system and a temporary folder are: a program cannot start
    there without them. Windows also names the account's home, which a POSIX
    program finds in the account database and a Windows program only through
    this variable; it is the folder Windows records for the account, not the
    service's own variable.
    """
    if sys.platform == "win32":
        import tempfile

        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        temporary = tempfile.gettempdir()
        return {
            "SystemRoot": system_root,
            "PATH": str(Path(system_root) / "System32"),
            "TEMP": temporary,
            "TMP": temporary,
            "USERPROFILE": _profile_directory(),
            "SKULK_HOME": str(root),
        }
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "SKULK_HOME": str(root)}


if sys.platform == "win32":

    def _open_stream(descriptor: int) -> int:
        """A standard stream to hand on, or the null device when it is closed."""
        import subprocess

        try:
            os.fstat(descriptor)
        except OSError:
            return subprocess.DEVNULL
        return descriptor

    def _hold_lifetime(root: Path) -> int:
        """Take this process's lifetime lock; it is released when the process ends.

        The descriptor is returned so that it stays open, and is never closed.

        Raises:
            OSError: The lock file is missing, or another process holds the
                lock: a service is already running from this root.
            ValueError: The lock file is not protected.
        """
        descriptor = _open_entry(root / LIFETIME_LOCK)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or _is_reparse_point(info)
                or not _owner_only(descriptor)
            ):
                raise ValueError("service lock is not protected")
            # The byte every Skulk lock on Windows is: the last one a file
            # can have, far from anything a file holds.
            region = _Region(Offset=0xFFFFFFFF, OffsetHigh=0x7FFFFFFF)
            if not _lock_file(
                msvcrt.get_osfhandle(descriptor),
                _LOCKFILE_EXCLUSIVE_NOW,
                0,
                1,
                0,
                ctypes.byref(region),
            ):
                raise _failure()
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @_ControlHandler
    def _stay(kind: int) -> bool:
        """Tell the system this process handled a console event, so it lives."""
        return True

    def _ask_to_stop(manager: int, shares_console: bool) -> None:
        """Send Ctrl-Break to the manager's process group, and to nothing else.

        The event can only be sent to programs on the sender's own console.
        Started by Task Scheduler this process has none, so it joins the
        manager's first. Joining a console resets a process's handlers to the
        system's, which end the process, so one that does nothing is
        registered straight after: the event is addressed to the manager's
        group, and this guards against it reaching this process anyway.
        """
        if not shares_console:
            if not _attach_console(manager):
                raise _failure()
            if not _set_control_handler(_stay, 1):
                raise _failure()
        if not _send_console_event(_CTRL_BREAK_EVENT, manager):
            raise _failure()

    def _run_manager(root: Path, runtime: Path) -> int:
        """Run the manager as a child of this process; return its exit code.

        Windows cannot replace a running process with another program, so
        this process stays as the one the OS service started and watches. It
        first joins a job that ends its processes when this process's handle
        closes: however this process ends, the manager and everything the
        manager started end with it, and nothing is left serving that the
        service does not know about.

        The manager is asked to stop in one way, Ctrl-Break to its own process
        group, whoever asks: Ctrl-C or Ctrl-Break at this process's console,
        or local setup through :data:`STOP_REQUEST`. This process then waits
        for the manager, which shuts down in an orderly way, and never ends it
        itself.
        """
        import signal
        import subprocess

        job = _create_job(None, None)
        if not job:
            raise _failure()
        limits = _ExtendedLimits(
            BasicLimitInformation=_BasicLimits(
                LimitFlags=_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
        )
        # The handle is never closed here: the system closes it when this
        # process ends, which is the moment the job is for.
        if not _set_job_information(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ) or not _assign_to_job(job, _current_process()):
            raise _failure()
        signalled: list[int] = []
        for number in (signal.SIGINT, signal.SIGBREAK):
            signal.signal(number, lambda received, _frame: signalled.append(received))
        request = root / STOP_REQUEST
        if os.path.lexists(request):
            return 0
        # Without a console of its own (Task Scheduler starts this file with
        # pythonw.exe) the manager gets one that has no window.
        shares_console = bool(_console_code_page())
        manager = subprocess.Popen(
            manager_command(root, runtime),
            cwd=root,
            env=manager_environment(root),
            stdin=subprocess.DEVNULL,
            stdout=_open_stream(1),
            stderr=_open_stream(2),
            creationflags=_CREATE_NEW_PROCESS_GROUP
            | (0 if shares_console else _CREATE_NO_WINDOW),
        )
        asked = False
        while True:
            try:
                return manager.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            if not asked and (signalled or os.path.lexists(request)):
                asked = True
                try:
                    _ask_to_stop(manager.pid, shares_console)
                except OSError:
                    # A manager that ended in this very moment has no console
                    # left to ask through, and needs no asking.
                    if manager.poll() is None:
                        raise


def main() -> None:
    """Verify, then run the fixed manager command in this process's place, without a shell.

    POSIX replaces this process with the manager. Windows cannot, so there
    this process runs the manager as a child tied to its own life and exits
    with the manager's exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        root = cast(Path, arguments.root).absolute()
        if sys.platform == "win32":
            private_directory(root)
            # Held, never closed: the system releases it when this process ends.
            lifetime = _hold_lifetime(root)
            if os.path.lexists(root / STOP_REQUEST):
                # Asked before the copy is verified, which takes time: a
                # stopped service lets go of its lock at once.
                raise SystemExit(0)
        runtime = verified_runtime(root)
        if sys.platform == "win32":
            code = _run_manager(root, runtime)
            os.close(lifetime)
        else:
            command = manager_command(root, runtime)
            os.execve(command[0], command, manager_environment(root))
    except (OSError, ValueError):
        print(
            "plugin service runtime unavailable; rerun local setup or inspect protected evidence",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    # Windows reports an exit code as an unsigned 32-bit number; Python takes
    # a signed one.
    raise SystemExit(code if code < 2**31 else code - 2**32)


if __name__ == "__main__":
    main()
