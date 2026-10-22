"""Descriptor-pinned review of explicit operator-selected local/mounted folders."""

from __future__ import annotations

import os
import secrets
import shutil
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from skulk.store.preservation_contracts import FolderReview, FolderReviewRequest
from skulk.store.preservation_journal import PreservationHostError


class FolderObservation(BaseModel):
    """Private observation binds consent to a parent and absent/existing leaf."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    view: FolderReview
    path: Path
    parent_device: int
    parent_inode: int
    device: int | None
    inode: int | None
    enrolled_id: str | None = None


def private_folder(descriptor: int) -> os.stat_result:
    """Require a dedicated owner-only directory; never chmod shared storage."""
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise PreservationHostError("storage_private_folder_required")
    return info


def observe_folder(
    request: FolderReviewRequest,
    *,
    forbidden: tuple[Path, ...],
    enrolled_id: str | None = None,
) -> FolderObservation:
    """Inspect a dedicated empty leaf or one missing leaf, without creating it.

    Only trusted operator setup calls this. Existing parents must resolve now;
    neither this review nor confirmation creates mount paths recursively.
    """
    if any(
        ord(character) < 32 or ord(character) == 127
        for character in request.path + request.label
    ):
        raise PreservationHostError("invalid_storage_selection")
    selected = Path(request.path)
    if (
        not selected.is_absolute()
        or selected.name in {"", ".", ".."}
        or ".." in selected.parts
    ):
        raise PreservationHostError("invalid_storage_selection")
    path = selected.parent.resolve(strict=True) / selected.name
    for protected in forbidden:
        protected = protected.resolve()
        if path.is_relative_to(protected) or protected.is_relative_to(path):
            raise PreservationHostError("storage_overlaps_core")
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parent_info = os.fstat(parent)
        try:
            child = os.open(
                path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
        except FileNotFoundError:
            info = None
            free = shutil.disk_usage(parent).free
        else:
            try:
                info = private_folder(child)
                with os.scandir(child) as entries:
                    if enrolled_id is None and next(entries, None) is not None:
                        raise PreservationHostError("storage_folder_not_empty")
                free = shutil.disk_usage(child).free
            finally:
                os.close(child)
        return FolderObservation(
            view=FolderReview(
                review_id=secrets.token_hex(32),
                label=request.label,
                creates_folder=info is None,
                free_bytes=free,
                expires_at=int(time.time()) + 600,
            ),
            path=path,
            parent_device=parent_info.st_dev,
            parent_inode=parent_info.st_ino,
            device=None if info is None else info.st_dev,
            inode=None if info is None else info.st_ino,
            enrolled_id=enrolled_id,
        )
    finally:
        os.close(parent)


@contextmanager
def confirmed_folder(observation: FolderObservation) -> Iterator[int]:
    """Recheck the reviewed identity and pin/create only its private leaf.

    A crash-created empty leaf requires fresh review. No preexisting entry is
    replaced or removed, and no missing mount/parent is recreated.
    """
    path = observation.path
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(parent)
        if (info.st_dev, info.st_ino) != (
            observation.parent_device,
            observation.parent_inode,
        ):
            raise PreservationHostError("storage_review_changed")
        if observation.device is None:
            os.mkdir(path.name, mode=0o700, dir_fd=parent)
            os.fsync(parent)
        child = os.open(
            path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
        )
        try:
            info = private_folder(child)
            if observation.device is not None and (info.st_dev, info.st_ino) != (
                observation.device,
                observation.inode,
            ):
                raise PreservationHostError("storage_review_changed")
            with os.scandir(child) as entries:
                if observation.enrolled_id is None and next(entries, None) is not None:
                    raise PreservationHostError("storage_folder_not_empty")
            current = path.lstat()
            parent_current = path.parent.stat()
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino) or (
                parent_current.st_dev,
                parent_current.st_ino,
            ) != (observation.parent_device, observation.parent_inode):
                raise PreservationHostError("storage_review_changed")
            yield child
        finally:
            os.close(child)
    finally:
        os.close(parent)
