"""Protected local files and exclusive ownership for independently managed runtimes."""

import asyncio
import functools
import os
import stat
import sys
import time
from pathlib import Path
from typing import Literal, NamedTuple, final
from uuid import uuid4

if sys.platform != "win32":
    # POSIX-only; the guard keeps this module loadable on Windows.
    import fcntl
    import grp
    import pwd
else:
    # Owner-only directories, files and the lock have their own Windows
    # implementation; each function below hands over to it first.
    from skulk.utils import _runtime_files_windows as _windows
    from skulk.utils import _windows_account


@functools.cache
def private_group(group_id: int, user_id: int) -> bool:
    """Cache whether an OS group is private to its owning account.

    Inspect local group/account metadata without changing permissions or members.
    Return False for missing or shared groups; callers still check file ownership.
    """
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


def has_root_authority() -> bool:
    """Whether this process could change the machine for every account.

    Code that runs installed plugins refuses to start with that authority, so
    a plugin never inherits it. On POSIX it is the effective root user. On
    Windows it is the SYSTEM account or a process running as administrator.
    """
    if sys.platform == "win32":
        return _windows_account.this_process_has_machine_authority()
    return os.geteuid() == 0


def has_borrowed_identity() -> bool:
    """Whether this process acts as one account on behalf of another.

    On POSIX that is a set-user-ID program: its effective user differs from
    the user who started it. Code that runs installed plugins refuses to start
    that way, so a plugin is never run with an identity its user was only lent.
    A Windows process has one identity for its whole life, so there it is
    never the case.
    """
    if sys.platform == "win32":
        return False
    return os.geteuid() != os.getuid()


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
    return info.st_uid == user_id and private_group(info.st_gid, user_id)


def no_other_account_can_change(path: Path) -> bool:
    """Whether only this account, or the machine's own authorities, can change a file.

    The rule for a file this account did not necessarily create, such as one
    in an installed Python environment that is about to be copied. On POSIX
    it is :func:`writable_only_by` for this user. On Windows the file's owner,
    and every account its access list lets change it, must be this account,
    SYSTEM or the Administrators group. What other accounts may read is not
    examined on either.

    Args:
        path: The file, examined as itself: a link is never followed.

    Returns:
        False when another account can change the file.

    Raises:
        OSError: When the file cannot be examined.
    """
    if sys.platform == "win32":
        return _windows.no_other_account_can_change(path)
    return writable_only_by(path.lstat(), os.getuid())


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


def is_private_directory(path: Path) -> bool:
    """Whether ``path`` is a directory only this account can reach.

    The path itself is examined, never a link's target, and nothing is
    created or changed.

    Args:
        path: The directory to examine.

    Returns:
        True for a directory owned by this account that no other account can
        reach; False for anything else found there, a link included.

    Raises:
        OSError: When the path cannot be examined, for example because it does
            not exist.
    """
    if sys.platform == "win32":
        return _windows.is_private_directory(path)
    info = path.lstat()
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and not info.st_mode & 0o077
    )


def private_directory(path: Path) -> None:
    """Create or validate an owner-only directory without accepting a symlink."""
    if sys.platform == "win32":
        _windows.private_directory(path)
        return
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("runtime directory must be owner-only")


def open_private(path: Path, flags: int) -> int:
    """Open an owner-only regular file, never a link.

    A file this call creates is created owner-only. The check is made on the
    opened file, so what is checked is what the caller then reads or writes.

    Args:
        path: The file to open.
        flags: The ``os.O_*`` access and creation flags, as for :func:`os.open`.

    Returns:
        An open descriptor; the caller closes it.

    Raises:
        ValueError: When the file is not a regular file private to this
            account.
        OSError: When it cannot be opened, as :func:`os.open` reports it; a
            link in its place is one such case.
    """
    if sys.platform == "win32":
        return _windows.open_private(path, flags)
    descriptor = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("runtime file must be owner-only")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


class Custody(NamedTuple):
    """One entry as itself: what it is, and who can change it."""

    kind: Literal["file", "directory", "link", "other"]
    """A link is any entry that stands for another: a symbolic link, and on
    Windows a junction or any other reparse point."""
    info: os.stat_result
    """The entry's own status, never a link target's."""
    owned: bool
    """Whether it belongs to this account."""
    exclusive: bool
    """Whether no other account can change it. On POSIX that is no group or
    other write permission. On Windows it is an access list that grants
    nothing to anyone but this account and SYSTEM, inherited or not."""


def examine(entry: Path | int) -> Custody:
    """Report what an entry is and who can change it, without following a link.

    Args:
        entry: A path, examined as itself, or an open descriptor.

    Returns:
        The entry's custody. Whether a link is ``exclusive`` is not examined:
        what matters about a link is where it points.

    Raises:
        OSError: When the entry cannot be examined, for example because it
            does not exist.
    """
    if sys.platform == "win32":
        return Custody(*_windows.examine(entry))
    info = os.fstat(entry) if isinstance(entry, int) else entry.lstat()
    kind: Literal["file", "directory", "link", "other"]
    if stat.S_ISLNK(info.st_mode):
        kind = "link"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    elif stat.S_ISREG(info.st_mode):
        kind = "file"
    else:
        kind = "other"
    return Custody(
        kind=kind,
        info=info,
        owned=info.st_uid == os.getuid(),
        exclusive=not info.st_mode & 0o022,
    )


def open_unfollowed(path: Path) -> int:
    """Open an existing file for reading as itself, never through a link.

    Nothing is checked about who owns it: pass the descriptor to
    :func:`examine` for that.

    Args:
        path: The file to open.

    Returns:
        An open descriptor; the caller closes it.

    Raises:
        OSError: When it cannot be opened; a link in its place is one such
            case.
    """
    if sys.platform == "win32":
        return _windows.open_unfollowed(path)
    return os.open(path, os.O_RDONLY | os.O_NOFOLLOW)


def create_file(path: Path, mode: int) -> int:
    """Create a new file for writing with exactly these permission bits.

    Whatever is already at the path, a link included, is a refusal. The
    process's umask does not decide the result. Windows has no permission
    bits: a file created there takes its access from the folder it is in,
    unless ``mode`` gives nobody else access, which makes it owner-only.

    Args:
        path: The file to create.
        mode: Its permission bits.

    Returns:
        An open descriptor; the caller closes it.

    Raises:
        OSError: When it cannot be created, for example because something is
            already there.
    """
    if sys.platform == "win32":
        return _windows.create_file(path, mode)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def read_private(path: Path, limit: int = 131072) -> bytes:
    """Read bounded bytes from an owner-only regular file, never a symlink."""
    if sys.platform == "win32":
        return _windows.read_private(path, limit)
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
    if sys.platform == "win32":
        _windows.write_private(path, content)
        return
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
    if sys.platform == "win32":
        _windows.remove_private(path)
        return
    path.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sync_directory(path: Path) -> None:
    """Make a directory's own changes durable.

    Creating, renaming or removing an entry changes the directory that holds
    it. Until that directory is synced, a crash can bring back its earlier
    contents.

    Args:
        path: The directory whose entries changed.
    """
    if sys.platform == "win32":
        _windows.sync_directory(path)
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
            "bootstrap.lock",
            "attachment.lock",
            "setup.lock",
            "catalog.lock",
            "native-policy.lock",
            "native-host-key.lock",
        }:
            raise ValueError("unknown runtime lock")
        self.path = root / name
        if sys.platform == "win32":
            self.descriptor = _windows.open_lock(root, name)
            return
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
