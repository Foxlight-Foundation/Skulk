"""Bounded, resumable local-kit byte ingress outside control envelopes."""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    secure_directory,
)

KIT_CHUNK_BYTES = 4 * 1024 * 1024
MAX_KIT_BYTES = 8 * 1024**4


class KitUploads:
    """One host-owned ingress directory; offsets reflect flushed file bytes.

    Callers serialize mutations. Completion is only a transfer observation;
    trusted keys, complete hashing and runtime admission happen at recovery.
    """

    def __init__(self, root: Path, journal: PreservationJournal) -> None:
        self.root = root
        self.journal = journal
        secure_directory(root)

    def path(self, upload_id: str) -> Path:
        """Resolve only an existing opaque host-created regular upload."""
        if len(upload_id) != 32 or any(c not in "0123456789abcdef" for c in upload_id):
            raise PreservationHostError("invalid_upload_identity")
        path = self.root / upload_id
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise PreservationHostError("unsafe_kit_upload")
        return path

    def status(self, upload_id: str) -> dict[str, JsonValue]:
        """Observe a durable offset without starting verification or admission."""
        path = self.path(upload_id)
        declared = self.journal.setting("upload:" + upload_id)
        if not isinstance(declared, int):
            raise PreservationHostError("upload_not_found")
        offset = path.stat().st_size
        if offset > declared:
            raise PreservationHostError("upload_length_changed")
        return {
            "upload_id": upload_id,
            "size_bytes": declared,
            "offset": offset,
            "complete": offset == declared,
        }

    def create(self, size_bytes: int) -> dict[str, JsonValue]:
        """Reserve a bounded kit transfer after observing actual free capacity."""
        if not 1 <= size_bytes <= MAX_KIT_BYTES:
            raise PreservationHostError("kit_size_out_of_range")
        entries = tuple(self.root.iterdir())
        if len(entries) >= 8:
            raise PreservationHostError("remove_unused_kit_uploads")
        reservations = sum(
            max(
                0,
                cast(int, self.journal.setting("upload:" + item.name) or 0)
                - item.stat().st_size,
            )
            for item in entries
        )
        if (
            shutil.disk_usage(self.root).free
            < size_bytes + reservations + 16 * 1024 * 1024
        ):
            raise PreservationHostError("insufficient_space")
        identity = uuid.uuid4().hex
        self.journal.set_setting("upload:" + identity, size_bytes)
        descriptor = os.open(
            self.root / identity,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        os.fsync(descriptor)
        os.close(descriptor)
        return self.status(identity)

    def append(self, upload_id: str, offset: int, data: bytes) -> dict[str, JsonValue]:
        """Flush one bounded exact-offset chunk; identical response-loss replay works."""
        status = self.status(upload_id)
        current, size = cast(int, status["offset"]), cast(int, status["size_bytes"])
        if (
            not data
            or len(data) > KIT_CHUNK_BYTES
            or offset < 0
            or offset + len(data) > size
        ):
            raise PreservationHostError("invalid_kit_chunk")
        path = self.path(upload_id)
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r+b") as stream:
            if offset < current and offset + len(data) <= current:
                stream.seek(offset)
                if stream.read(len(data)) != data:
                    raise PreservationHostError("kit_chunk_replay_changed")
            elif offset == current:
                stream.seek(offset)
                _ = stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            else:
                raise PreservationHostError("kit_chunk_offset_changed")
        return self.status(upload_id)

    def remove(self, upload_id: str) -> None:
        """Release an unused ingress file; the host fences active job references."""
        self.path(upload_id).unlink()
