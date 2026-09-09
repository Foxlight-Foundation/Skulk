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

from pydantic import BaseModel, ConfigDict

from skulk.shared.types.common import CommandId

VideoArtifactPurpose = Literal["video", "thumbnail"]

_ARTIFACT_FILENAMES: dict[VideoArtifactPurpose, str] = {
    "video": "output.mp4",
    "thumbnail": "thumbnail.jpg",
}
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
"""Absolute ceiling on one artifact, well above any 15 s 768p container."""
_MAX_STASHED_CHUNKS = 256
"""Out-of-order chunks held per assembly on a reordering transport."""
_MAX_STASHED_BYTES = 64 * 1024 * 1024
"""Bytes of early chunks held per assembly before the transfer is refused."""


class StoredVideoArtifact(BaseModel):
    """One committed artifact on the owning API's disk."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    command_id: CommandId
    purpose: VideoArtifactPurpose
    file_path: Path
    content_type: str
    size_bytes: int
    sha256: str
    expires_at: float
    verified: bool = True
    """Whether the bytes on disk were hashed by this process. Adopted
    artifacts start unverified and are hashed on first access."""


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
    stash: dict[int, bytes] = field(default_factory=dict)
    """Chunks that arrived ahead of ``next_sequence`` on a reordering transport."""
    stashed_bytes: int = 0


def _file_sha256(path: Path) -> str | None:
    """Hash one file; ``None`` when it cannot be read."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


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
        if sequence < assembly.next_sequence or sequence in assembly.stash:
            return  # duplicate re-delivery
        if sequence > assembly.total_chunks:
            raise ValueError(f"artifact chunk {sequence} exceeds the declared count")
        if sequence != assembly.next_sequence:
            # Zenoh delivers a stream in order; the gossipsub fallback may not.
            # Hold a bounded window of early chunks and drain them in order.
            if (
                len(assembly.stash) >= _MAX_STASHED_CHUNKS
                or assembly.stashed_bytes + len(data) > _MAX_STASHED_BYTES
            ):
                raise ValueError(
                    f"artifact reorder window exceeded waiting for chunk "
                    f"{assembly.next_sequence}"
                )
            assembly.stash[sequence] = data
            assembly.stashed_bytes += len(data)
            return
        self._write_chunk(assembly, data)
        while assembly.next_sequence in assembly.stash:
            early = assembly.stash.pop(assembly.next_sequence)
            assembly.stashed_bytes -= len(early)
            self._write_chunk(assembly, early)

    @staticmethod
    def _write_chunk(assembly: _Assembly, data: bytes) -> None:
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
            if assembly.stash:
                raise ValueError("artifact completed with chunks still missing")
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

    def adopt(
        self,
        command_id: CommandId,
        purpose: VideoArtifactPurpose,
        *,
        content_type: str,
        size_bytes: int,
        sha256: str,
        expires_at: float,
    ) -> bool:
        """Re-attach an artifact committed by a previous process.

        The job registry is the durable record of what was verified; the file
        is registered when it exists with the recorded size and is hashed
        against the recorded digest on first access before it is served.
        Returns whether the artifact is now registered.
        """

        path = self._storage_dir / str(command_id) / _ARTIFACT_FILENAMES[purpose]
        try:
            if not path.is_file() or path.stat().st_size != size_bytes:
                return False
        except OSError:
            return False
        if time.time() > expires_at:
            return False
        self._artifacts[(command_id, purpose)] = StoredVideoArtifact(
            command_id=command_id,
            purpose=purpose,
            file_path=path,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            expires_at=expires_at,
            verified=False,
        )
        return True

    def purge_unknown(self, keep: set[CommandId]) -> int:
        """Delete job directories that no served or in-flight job owns."""

        removed = 0
        try:
            entries = list(self._storage_dir.iterdir())
        except OSError:
            return 0
        for entry in entries:
            if not entry.is_dir():
                continue
            command_id = CommandId(entry.name)
            if command_id in keep or any(
                key[0] == command_id for key in (*self._artifacts, *self._assemblies)
            ):
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        return removed

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
        if not stored.verified:
            # A previous process verified these bytes; anything could have
            # touched the file since. Hash once before serving it as verified.
            if _file_sha256(stored.file_path) != stored.sha256:
                self.delete(command_id)
                return None
            stored = stored.model_copy(update={"verified": True})
            self._artifacts[(command_id, purpose)] = stored
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
