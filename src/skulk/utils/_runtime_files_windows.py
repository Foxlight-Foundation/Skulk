# ruff: noqa: E402
# The platform assertion has to come before the Windows-only imports: it is
# what tells the type checker to skip this module on other platforms.
"""Windows side of :mod:`skulk.utils.runtime_files`.

The POSIX functions rest on owner and mode bits, ``O_NOFOLLOW`` and ``flock``.
Windows has none of those, so the same guarantees come from the ``safefs``
seam of the model-preservation package, which Skulk already depends on:

- **Owner-only** means the account owns the file or folder and its access list
  is protected (nothing inherited) and names the account and SYSTEM only.
- **Never through a link** means the entry is opened as itself and refused if
  it is a reparse point.
- **The lock** is one the operating system drops when the process dies.

A folder or file created here is private from the moment it exists. One
created by other means inside a private folder inherits access for the account
and SYSTEM only, but its access list is not marked protected, so it does not
count as owner-only: runtime state has to be created through these functions,
as it is on POSIX where the mode bits are set at creation. A volume without
access lists (FAT, exFAT) cannot hold owner-only state at all.
"""

import sys

assert sys.platform == "win32"

import contextlib
import ctypes
import ctypes.wintypes as wt
import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from foxlight_model_preservation import safefs

_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3
_ERROR_SHARING_VIOLATION = 32
_GENERIC_READ = 0x80000000
_FILE_SHARE_ALL = 0x00000007
_OPEN_EXISTING = 3

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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

# Typed views of the foreign functions, so their results are not ``Any``.
_create_file: Callable[[str, int, int, None, int, int, None], int | None] = (
    _kernel32.CreateFileW
)
_close_handle: Callable[[int], int] = _kernel32.CloseHandle
_INVALID_HANDLE = wt.HANDLE(-1).value


def _owner_only(descriptor: int) -> bool:
    """Whether only this account and SYSTEM can reach an open file or folder."""
    return safefs.owned(descriptor) and safefs.private(descriptor)


def _open_directory(path: Path) -> int:
    """Open a folder as itself; a link in its place is refused."""
    try:
        return safefs.open_dir(path)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            # POSIX reports these through the directory check, not an open.
            raise ValueError("runtime directory must be owner-only") from None
        raise


def is_private_directory(path: Path) -> bool:
    """Whether ``path`` is a folder only this account and SYSTEM can reach."""
    try:
        descriptor = safefs.open_dir(path)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            # Something other than a folder is there: a link or a file.
            return False
        raise
    try:
        return _owner_only(descriptor)
    finally:
        os.close(descriptor)


def private_directory(path: Path) -> None:
    """Create or validate an owner-only directory without accepting a link."""
    safefs.make_dir(path, parents=True, exist_ok=True)
    descriptor = _open_directory(path)
    try:
        if not _owner_only(descriptor):
            raise ValueError("runtime directory must be owner-only")
    finally:
        os.close(descriptor)


def open_private(path: Path, flags: int) -> int:
    """Open an owner-only regular file, never a link; the caller closes it."""
    descriptor = safefs.open_path(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode) or not _owner_only(
            descriptor
        ):
            raise ValueError("runtime file must be owner-only")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def read_private(path: Path, limit: int) -> bytes:
    """Read bounded bytes from an owner-only regular file, never a link."""
    descriptor = safefs.open_path(path, os.O_RDONLY)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode) or not _owner_only(
            descriptor
        ):
            raise ValueError("runtime file must be owner-only")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            content = source.read(limit + 1)
        if len(content) > limit:
            raise ValueError("runtime file exceeds bound")
        return content
    finally:
        os.close(descriptor)


def write_private(path: Path, content: bytes) -> None:
    """Atomically replace a private file and make its contents durable."""
    private_directory(path.parent)
    temporary = f".{path.name}.{uuid4().hex}"
    directory = _open_directory(path.parent)
    try:
        descriptor = safefs.open_at(
            directory, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            # Replaces the target even while another handle has it open.
            safefs.replace_at(directory, temporary, directory, path.name)
            os.fsync(directory)
        finally:
            with contextlib.suppress(FileNotFoundError):
                safefs.unlink_at(directory, temporary)
    finally:
        os.close(directory)


def remove_private(path: Path) -> None:
    """Remove a private file, if present, and make the removal durable."""
    directory = _open_directory(path.parent)
    try:
        with contextlib.suppress(FileNotFoundError):
            safefs.unlink_at(directory, path.name)
        os.fsync(directory)
    finally:
        os.close(directory)


def sync_directory(path: Path) -> None:
    """Make a folder's own changes durable."""
    descriptor = safefs.open_dir(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def children_fence(lock: Path) -> Path:
    """The file a lock holder's child keeps open for as long as it runs.

    A Windows file lock belongs to one process: when the holder dies its lock
    is free at once, a moment before the operating system has ended the
    programs the holder started. Such a program is therefore started holding
    this file open with no sharing (:mod:`skulk.utils._owned_process_windows`),
    and :func:`open_lock` refuses the lock while anyone holds it. That is the
    part a POSIX child plays by inheriting the lock's own descriptor.
    """
    return lock.with_name(lock.name + ".children")


def _refuse_while_a_child_remains(lock: Path) -> None:
    """Raise ``BlockingIOError`` while a child of the lock's last holder runs."""
    handle = _create_file(
        str(children_fence(lock)),
        _GENERIC_READ,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        0,
        None,
    )
    if handle and handle != _INVALID_HANDLE:
        # Nobody holds it: the last child, if there was one, has ended.
        _close_handle(handle)
        return
    error = ctypes.get_last_error()
    if error in (_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND):
        # No holder of this lock has started a child yet.
        return
    if error == _ERROR_SHARING_VIOLATION:
        raise BlockingIOError(
            errno.EAGAIN, "a child of the lock's last holder is still running"
        )
    raise ctypes.WinError(error)


def open_lock(root: Path, name: str) -> int:
    """Open and take the named lock in ``root``; the caller closes it to release.

    Raises:
        BlockingIOError: Another holder has the lock, or a program the last
            holder started is still running.
        ValueError: The lock file is not owner-only.
    """
    private_directory(root)
    descriptor = safefs.open_path(root / name, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode) or not _owner_only(
            descriptor
        ):
            raise ValueError("runtime lock must be owner-only")
        safefs.lock(descriptor)
        _refuse_while_a_child_remains(root / name)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor
