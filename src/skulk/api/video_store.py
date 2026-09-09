"""Owning-API storage for finished video containers and their thumbnails.

The worker that rendered a clip streams it here over the ``OUTPUT_MEDIA``
plane in bounded raw chunks. The store assembles each artifact into a staging
file, verifies size and digest against what the producer declared, and only
then commits it under the job's directory. Committed artifacts expire after a
bounded lifetime so an API node's disk is never an unbounded archive.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal, Protocol

from pydantic import BaseModel

from skulk.shared.types.common import CommandId

VideoArtifactPurpose = Literal["video", "thumbnail"]

_ARTIFACT_FILENAMES: dict[VideoArtifactPurpose, str] = {
    "video": "output.mp4",
    "thumbnail": "thumbnail.jpg",
}
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
"""Absolute ceiling on one artifact, well above any 15 s 768p container."""


class StoredVideoArtifact(BaseModel, frozen=True):
    """One committed artifact on the owning API's disk."""

    command_id: CommandId
    purpose: VideoArtifactPurpose
    file_path: Path
    content_type: str
    size_bytes: int
    sha256: str
    expires_at: float


class _Digest(Protocol):
    """The slice of a hashlib object the assembler relies on."""

    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(eq=False)
class _Assembly:
    """One artifact being received chunk by chunk."""

    purpose: VideoArtifactPurpose
    content_type: str
    total_bytes: int
    total_chunks: int
    staging_path: Path
    handle: IO[bytes]
    digest: _Digest = field(default_factory=hashlib.sha256)
    received_bytes: int = 0
    next_sequence: int = 1


class VideoStore:
    """Disk-backed, expiring store for generated video artifacts."""

    def __init__(self, storage_dir: Path, default_expiry_seconds: int = 24 * 3600) -> None:
        self._storage_dir = storage_dir
        self._default_expiry_seconds = default_expiry_seconds
        self._artifacts: dict[tuple[CommandId, VideoArtifactPurpose], StoredVideoArtifact] = {}
        self._assemblies: dict[tuple[CommandId, VideoArtifactPurpose], _Assembly] = {}
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_dir(self) -> Path:
        """Root directory holding one subdirectory per job."""
        return self._storage_dir

    def open_assembly(
        self,
        command_id: CommandId,
        purpose: VideoArtifactPurpose,
        *,
        content_type: str,
        total_bytes: int,
        total_chunks: int,
    ) -> None:
        """Begin receiving one artifact; replaces any half-received attempt."""

        if total_bytes <= 0 or total_bytes > _MAX_ARTIFACT_BYTES:
            raise ValueError("artifact size is outside the accepted range")
        self.abort_assembly(command_id, purpose)
        directory = self._storage_dir / str(command_id)
        directory.mkdir(parents=True, exist_ok=True)
        staging_path = directory / f"{_ARTIFACT_FILENAMES[purpose]}.part"
        self._assemblies[(command_id, purpose)] = _Assembly(
            purpose=purpose,
            content_type=content_type,
            total_bytes=total_bytes,
            total_chunks=total_chunks,
            staging_path=staging_path,
            handle=staging_path.open("wb"),
        )

    def append(
        self,
        command_id: CommandId,
        purpose: VideoArtifactPurpose,
        sequence: int,
        data: bytes,
    ) -> None:
        """Append one in-order chunk to an open assembly."""

        assembly = self._assemblies.get((command_id, purpose))
        if assembly is None:
            raise ValueError("no open assembly for this artifact")
        if sequence != assembly.next_sequence:
            raise ValueError(
                f"out-of-order artifact chunk {sequence}, expected {assembly.next_sequence}"
            )
        if assembly.received_bytes + len(data) > assembly.total_bytes:
            raise ValueError("artifact chunks exceed the declared size")
        assembly.handle.write(data)
        assembly.digest.update(data)
        assembly.received_bytes += len(data)
        assembly.next_sequence += 1

    def commit(
        self,
        command_id: CommandId,
        purpose: VideoArtifactPurpose,
        *,
        sha256: str,
        total_chunks: int,
    ) -> StoredVideoArtifact:
        """Verify and publish an assembled artifact."""

        assembly = self._assemblies.pop((command_id, purpose), None)
        if assembly is None:
            raise ValueError("no open assembly for this artifact")
        assembly.handle.close()
        try:
            if assembly.next_sequence - 1 != total_chunks or total_chunks != assembly.total_chunks:
                raise ValueError("artifact chunk count does not match its declaration")
            if assembly.received_bytes != assembly.total_bytes:
                raise ValueError("artifact byte count does not match its declaration")
            if assembly.digest.hexdigest() != sha256:
                raise ValueError("artifact failed SHA-256 verification")
        except ValueError:
            assembly.staging_path.unlink(missing_ok=True)
            raise
        final_path = assembly.staging_path.with_suffix("")
        assembly.staging_path.replace(final_path)
        stored = StoredVideoArtifact(
            command_id=command_id,
            purpose=purpose,
            file_path=final_path,
            content_type=assembly.content_type,
            size_bytes=assembly.received_bytes,
            sha256=sha256,
            expires_at=time.time() + self._default_expiry_seconds,
        )
        self._artifacts[(command_id, purpose)] = stored
        return stored

    def abort_assembly(self, command_id: CommandId, purpose: VideoArtifactPurpose) -> None:
        """Discard a partial artifact."""

        assembly = self._assemblies.pop((command_id, purpose), None)
        if assembly is None:
            return
        assembly.handle.close()
        assembly.staging_path.unlink(missing_ok=True)

    def has_open_assembly(self, command_id: CommandId, purpose: VideoArtifactPurpose) -> bool:
        """Whether an artifact is currently being received."""
        return (command_id, purpose) in self._assemblies

    def get(
        self, command_id: CommandId, purpose: VideoArtifactPurpose = "video"
    ) -> StoredVideoArtifact | None:
        """Return a committed artifact unless it has expired."""

        stored = self._artifacts.get((command_id, purpose))
        if stored is None:
            return None
        if time.time() > stored.expires_at:
            self.delete(command_id)
            return None
        return stored

    def delete(self, command_id: CommandId) -> None:
        """Remove every artifact and partial assembly for one job."""

        for purpose in _ARTIFACT_FILENAMES:
            self.abort_assembly(command_id, purpose)
            self._artifacts.pop((command_id, purpose), None)
        shutil.rmtree(self._storage_dir / str(command_id), ignore_errors=True)

    def cleanup_expired(self) -> int:
        """Delete expired jobs; returns how many were removed."""

        now = time.time()
        expired = {
            command_id
            for (command_id, _purpose), stored in self._artifacts.items()
            if now > stored.expires_at
        }
        for command_id in expired:
            self.delete(command_id)
        return len(expired)
