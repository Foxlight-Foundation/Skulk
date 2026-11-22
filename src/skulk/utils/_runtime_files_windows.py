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
import errno
import os
import stat
from pathlib import Path
from uuid import uuid4

from foxlight_model_preservation import safefs


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


def open_lock(root: Path, name: str) -> int:
    """Open and take the named lock in ``root``; the caller closes it to release.

    Raises:
        BlockingIOError: Another holder has the lock.
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
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor
