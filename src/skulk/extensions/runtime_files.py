"""Protected local files and exclusive ownership for managed plugin runtimes."""

import asyncio
import fcntl
import functools
import grp
import os
import pwd
import stat
import time
from pathlib import Path
from typing import final
from uuid import uuid4


@functools.cache
def _private_group(group_id: int, user_id: int) -> bool:
    """Whether ``group_id`` is the account's own group with no other members."""
    try:
        group = grp.getgrgid(group_id)
        account = pwd.getpwuid(user_id)
    except KeyError:
        return False
    return (
        account.pw_gid == group_id
        and group.gr_name == account.pw_name
        and not group.gr_mem
    )


def writable_only_by(info: os.stat_result, user_id: int) -> bool:
    """Whether no account other than ``user_id``, or root, can change a file.

    Group write counts as private when the file belongs to the account and its
    group is the account's own private group with no other members: that is
    what Ubuntu's default 0002 umask gives every file a user creates, so a
    Python environment installed there is still the owner's alone.

    Args:
        info: The file's ``lstat`` result.
        user_id: The account the file must be private to.

    Returns:
        False when others can write it; True otherwise.
    """
    if info.st_mode & 0o002:
        return False
    if not info.st_mode & 0o020:
        return True
    return info.st_uid == user_id and _private_group(info.st_gid, user_id)


def is_desktop_metadata(path: Path) -> bool:
    """Whether ``path`` is a regular file a desktop file browser leaves behind.

    Finder writes ``.DS_Store`` into every folder it shows, and AppleDouble
    ``._`` companions onto volumes without extended attributes. Neither is
    ever imported or executed, so an operator browsing the plugin tree must
    not stop the manager by it. Only a regular file qualifies: a link or a
    directory under one of these names is still whatever the caller's rules
    make of it.
    """
    if path.name != ".DS_Store" and not path.name.startswith("._"):
        return False
    return stat.S_ISREG(path.lstat().st_mode)


def private_directory(path: Path) -> None:
    """Create or validate an owner-only directory without accepting a symlink."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("runtime directory must be owner-only")


def read_private(path: Path, limit: int = 131072) -> bytes:
    """Read bounded bytes from an owner-only regular file, never a symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
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
    """Atomically replace a private file and sync its contents and directory."""
    private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def remove_private(path: Path) -> None:
    """Remove a private file, if present, and sync its directory.

    The removal is durable before this returns: a crash afterwards cannot
    bring the file back with the directory's earlier contents.
    """
    path.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@final
class RuntimeLock:
    """Hold a nonblocking file lock until all owned runtime work has finished."""

    def __init__(self, root: Path, name: str = "installer.lock") -> None:
        """Fence this service-owned directory using a fixed local lock name."""
        if name not in {
            "installer.lock",
            "supervisor.lock",
            "service.lock",
            "manager.lock",
            "attachment.lock",
            "setup.lock",
            "catalog.lock",
        }:
            raise ValueError("unknown runtime lock")
        private_directory(root)
        self.descriptor = os.open(
            root / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            info = os.fstat(self.descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise ValueError("runtime lock must be owner-only")
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.descriptor)
            raise

    def close(self) -> None:
        """Release exclusive ownership after child and disk work has completed."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


async def acquire_runtime_lock(
    root: Path,
    *,
    wait_seconds: float,
    name: str = "installer.lock",
    poll_seconds: float = 0.05,
) -> RuntimeLock:
    """Take a runtime fence, waiting a bounded time for a brief holder to finish.

    A running owner holds the installer fence for moments at a time: while its
    start verifies the installed runtime right after an activation, and again
    at every periodic re-verification. Work that has written nothing yet waits
    that out rather than refusing; a fence still held when the wait runs out
    means real work, and the caller refuses as before.

    Args:
        root: The service-owned directory the fence protects.
        wait_seconds: How long to keep trying before giving up.
        name: The fixed lock name, as for :class:`RuntimeLock`.
        poll_seconds: The pause between attempts.

    Returns:
        The held fence; the caller closes it.

    Raises:
        BlockingIOError: Another holder still had the fence when the wait ran
            out.
    """
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return RuntimeLock(root, name)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(poll_seconds)
