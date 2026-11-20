"""Journal native music outputs before inference and reclaim only proven files.

The caller serializes creation with its native lifetime lock. Reclamation is
permitted only after the native writer and addressed result readers have drained;
the Worker source-custody fence supplies that authority after a process crash.
This module never infers writer exit from a task's replicated terminal status.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

from pydantic import Field

from skulk.shared.types.common import CommandId
from skulk.shared.types.native_music import (
    NativeMusicContract,
    NativeMusicDirectoryIdentity,
)
from skulk.utils.runtime_files import remove_private, write_private

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}")
_RESULT_FILES = {
    "conditioning.wav": 64 * 1024**2,
    "result.wav": 64 * 1024**2,
    "result.json": 16 * 1024,
}


@final
class _ResultOwnership(NativeMusicContract):
    """Private pre-write directory identity, outside the native result directory."""

    version: Literal[1] = Field(default=1, description="Ownership receipt schema.")
    directory: str = Field(
        min_length=1,
        max_length=160,
        description="Exact command-correlated native request directory name.",
    )
    identity: NativeMusicDirectoryIdentity = Field(
        description="Core-observed result inode before any native writer admission."
    )


def music_result_directory(root: Path, command_id: CommandId) -> Path:
    """Derive the private native output path without accepting a caller pathname.

    Native completion uses this same opaque command identity. Validate it as one
    filename rather than hashing it into a different native request identity.
    """
    identifier = str(command_id)
    if _REQUEST_ID.fullmatch(identifier) is None:
        raise ValueError("invalid native music command identity")
    return root / "workspace" / "requests" / identifier


def _receipt(directory: Path) -> Path:
    if (
        directory.parent.name != "requests"
        or directory.parent.parent.name != "workspace"
        or _REQUEST_ID.fullmatch(directory.name) is None
    ):
        raise ValueError("native music custody requires its fixed workspace path")
    return directory.with_name("." + directory.name + ".owner.json")


def _private_directory(info: os.stat_result) -> NativeMusicDirectoryIdentity:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("native music result directory must be owner-only")
    return NativeMusicDirectoryIdentity(device=info.st_dev, inode=info.st_ino)


def _read_receipt(directory: Path) -> _ResultOwnership | None:
    path = _receipt(directory)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
            or info.st_size > 2048
        ):
            raise ValueError("native music ownership receipt is not private metadata")
        record = _ResultOwnership.model_validate_json(os.read(descriptor, 2049))
        if record.directory != directory.name:
            raise ValueError("native music receipt belongs to another request")
        return record
    finally:
        os.close(descriptor)


@final
@dataclass(frozen=True)
class PendingMusicResult:
    """Retained pre-submit cleanup identity after an uncertain filesystem effect.

    A writer must never have been admitted, or must already be drained. A known
    inode must still match; without
    an observed inode only an empty private directory without a receipt can be
    reclaimed under the original parent. Existing completed results are never
    transferred into this ownership by a failed exclusive mkdir.
    """

    directory: Path
    parent_identity: NativeMusicDirectoryIdentity
    identity: NativeMusicDirectoryIdentity | None


class MusicResultPreparationError(RuntimeError):
    """Failed pre-write preparation whose exact cleanup must remain retryable."""

    def __init__(self, pending: PendingMusicResult) -> None:
        """Retain private ownership facts without exposing paths in public errors."""
        super().__init__("native music result preparation cleanup remains pending")
        self.pending = pending


def prepare_result_directory(
    directory: Path, parent_identity: NativeMusicDirectoryIdentity
) -> NativeMusicDirectoryIdentity:
    """Create and durably journal one empty result before native submission.

    The requests parent must match the already owned lifetime. Existing request
    directories and unresolved receipts are refused, including empty leftovers;
    cleanup/recovery must settle them explicitly before an identity can be reused.
    If any filesystem operation fails, no native request may be sent.
    """
    receipt = _receipt(directory)
    parent = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if _private_directory(os.fstat(parent)) != parent_identity:
            raise ValueError("native music requests parent changed its identity")
        try:
            receipt.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("native music result has an unresolved ownership receipt")
        identity: NativeMusicDirectoryIdentity | None = None
        created = False
        try:
            os.mkdir(directory.name, mode=0o700, dir_fd=parent)
            created = True
            identity = _private_directory(
                os.stat(directory.name, dir_fd=parent, follow_symlinks=False)
            )
            os.fsync(parent)
            record = _ResultOwnership(directory=directory.name, identity=identity)
            write_private(receipt, record.model_dump_json().encode())
            return identity
        except BaseException as error:
            if isinstance(error, FileExistsError) and not created:
                # Exclusive creation did not own the existing directory. In
                # particular, never roll back a retained completed result.
                raise
            pending = PendingMusicResult(directory, parent_identity, identity)
            if reclaim_unsubmitted_result(pending):
                raise
            raise MusicResultPreparationError(pending) from error
    finally:
        os.close(parent)


def verify_prepared_result_directory(
    directory: Path,
    identity: NativeMusicDirectoryIdentity,
    parent_identity: NativeMusicDirectoryIdentity,
) -> None:
    """Require the exact empty pre-write result and durable receipt before send.

    Core's model adapter creates the result before constructing a private native
    request. The resident supervisor independently checks that preparation under
    its serial actor; a caller-supplied inode alone cannot manufacture ownership.
    """
    record = _read_receipt(directory)
    if record is None or record.identity != identity:
        raise ValueError("native music result lacks its pre-write ownership receipt")
    parent = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if _private_directory(os.fstat(parent)) != parent_identity:
            raise ValueError("native music requests parent changed its identity")
        descriptor = os.open(
            directory.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
        )
        try:
            if _private_directory(os.fstat(descriptor)) != identity or os.listdir(
                descriptor
            ):
                raise ValueError("native music result is not its empty prepared inode")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _reclaim_result_directory(
    directory: Path,
    *,
    expected_identity: NativeMusicDirectoryIdentity | None = None,
    expected_parent: NativeMusicDirectoryIdentity | None = None,
    unsubmitted: bool = False,
) -> bool:
    record = _read_receipt(directory)
    if unsubmitted and (
        expected_identity is None
        and record is not None
        or expected_identity is not None
        and record is not None
        and record.identity != expected_identity
    ):
        return False
    try:
        parent = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return record is None and expected_parent is None
    try:
        if (
            expected_parent is not None
            and _private_directory(os.fstat(parent)) != expected_parent
        ):
            return False
        _private_directory(os.fstat(parent))
        try:
            descriptor = os.open(
                directory.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except FileNotFoundError:
            if record is not None:
                remove_private(_receipt(directory))
            return True
        try:
            identity = _private_directory(os.fstat(descriptor))
            if expected_identity is not None and identity != expected_identity:
                return False
            if record is not None and record.identity != identity:
                return False
            owned: list[tuple[str, os.stat_result]] = []
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    if unsubmitted:
                        return False
                    info = entry.stat(follow_symlinks=False)
                    if (
                        record is None
                        or entry.name not in _RESULT_FILES
                        or not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.getuid()
                        or info.st_nlink != 1
                        or info.st_mode & 0o077
                        or info.st_size > _RESULT_FILES[entry.name]
                    ):
                        return False
                    owned.append((entry.name, info))
            # Validate every entry before removing any; unknown files preserve
            # the complete directory and its admission reservation. Descriptor
            # operations keep a renamed parent from redirecting file deletion.
            for name, observed in owned:
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if current != observed:
                    return False
                os.unlink(name, dir_fd=descriptor)
            os.fsync(descriptor)
            current = _private_directory(
                os.stat(directory.name, dir_fd=parent, follow_symlinks=False)
            )
            if current != identity:
                return False
            os.rmdir(directory.name, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(descriptor)
        if record is not None:
            remove_private(_receipt(directory))
        return True
    finally:
        os.close(parent)


def reclaim_result_directory(directory: Path) -> bool:
    """Reclaim known output only after the caller proves native/reader exit.

    Preserve replaced inodes, foreign names, links and malformed receipts.
    Without a receipt, only an empty private directory is reclaimable: native
    execution cannot have started before the pre-write receipt was synced.
    Return False on incomplete cleanup so the owner retains capacity and retries.
    """
    try:
        return _reclaim_result_directory(directory)
    except (OSError, ValueError):
        return False


def reclaim_unsubmitted_result(pending: PendingMusicResult) -> bool:
    """Reclaim only this pre-submit empty inode/receipt, preserving failed custody.

    The owning model serializes creation and proves no native writer can run:
    either submission never began or the resident has actually drained. Retain
    this fallback until successful completion because the resident may refuse
    before recording its own writer custody. It can reclaim only empty output.
    Expected parent and observed child identities survive a receipt-sync failure;
    any replacement, new content, foreign receipt or incomplete removal refuses
    cleanup. A False result must keep the model fenced and owner retained.
    """
    try:
        return _reclaim_result_directory(
            pending.directory,
            expected_identity=pending.identity,
            expected_parent=pending.parent_identity,
            unsubmitted=True,
        )
    except (OSError, ValueError):
        return False


def reclaim_music_result(root: Path, command_id: CommandId) -> bool:
    """Adapt native music cleanup to the Worker's drained source-custody callback."""
    try:
        directory = music_result_directory(root, command_id)
    except ValueError:
        return False
    return reclaim_result_directory(directory)
