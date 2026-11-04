"""Single verified edit WAV custody over the existing generated-media assembler.

Native completion is provenance, not a content receipt. This owner independently
checks encoded bytes and requires a durable completed job before exposing them.
All I/O belongs off the API loop and under that service's admission/actor lock.
"""

from __future__ import annotations

import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import final

from skulk.api.music_edit_jobs import (
    MAX_ACTIVE_MUSIC_EDITS,
    MAX_RETAINED_MUSIC_EDITS,
    MusicEditJob,
)
from skulk.api.video_store import VideoStore
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_SOURCE_BYTES,
    pcm24_from_wav,
)
from skulk.shared.types.common import CommandId
from skulk.shared.types.native_music import NativeMusicWav
from skulk.utils.runtime_files import RuntimeLock, private_directory


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@final
@dataclass(frozen=True)
class StoredMusicEdit:
    """Private closed-file receipt; no path or WAV bytes enter replicated state."""

    facts: NativeMusicWav
    file_path: Path
    expires_at: float


@final
class MusicEditMediaStore:
    """Own result copies using the shared 64 MiB WAV assembler and 24-hour expiry.

    Keep every known retained or assembling job protected from implicit eviction.
    The typed job owner explicitly removes expired/deleted content. A lifetime
    OS fence and mutex cover adoption, verification, reads and partial cleanup.
    """

    def __init__(self, root: Path, *, max_total_bytes: int = 2 * 1024**3) -> None:
        """Open protected result storage; no existing file is implicitly adopted."""
        if (
            not 44
            <= max_total_bytes
            <= MAX_RETAINED_MUSIC_EDITS * MAX_SEPARATION_SOURCE_BYTES
        ):
            raise ValueError("Music edit media budget is outside its bounded policy")
        missing: list[Path] = []
        ancestor = root
        while not ancestor.exists():
            missing.append(ancestor)
            ancestor = ancestor.parent
        self.root = root
        self.max_total_bytes = max_total_bytes
        self._mutex = threading.RLock()
        self._ownership: RuntimeLock | None = RuntimeLock(root, name="service.lock")
        self._opened: dict[CommandId, tuple[int, int]] = {}
        self._completed: dict[CommandId, StoredMusicEdit] = {}
        self._known: set[CommandId] = set()
        try:
            private_directory(root / "wav")
            self._store = VideoStore(root / "wav", max_total_bytes=max_total_bytes)
            _sync_directory(root / "wav")
            _sync_directory(root)
            for directory in missing:
                _sync_directory(directory.parent)
        except BaseException:
            self._ownership.close()
            self._ownership = None
            raise

    def _check(self, identifier: CommandId) -> None:
        if self._ownership is None:
            raise RuntimeError("Music edit media owner is closed")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(identifier)) is None:
            raise ValueError("Invalid music edit media identity")

    def reserved_bytes(self) -> int:
        """Return committed plus admitted assembly bytes; content is never truncated."""
        with self._mutex:
            if self._ownership is None:
                raise RuntimeError("Music edit media owner is closed")
            # Failed cleanup must not release its disk promise just because the
            # shared assembler has already dropped its transient file handle.
            return sum(size for size, _ in self._opened.values()) + sum(
                receipt.facts.size_bytes for receipt in self._completed.values()
            )

    def owned_identifiers(self) -> frozenset[CommandId]:
        """Return bounded known ownership, including content awaiting failed cleanup."""
        with self._mutex:
            if self._ownership is None:
                raise RuntimeError("Music edit media owner is closed")
            return frozenset(self._known)

    def open_assembly(
        self, identifier: CommandId, *, total_bytes: int, total_chunks: int
    ) -> None:
        """Reserve one exact WAV using the common ordered/reordered bounded assembler."""
        with self._mutex:
            self._check(identifier)
            if (
                not 44 <= total_bytes <= MAX_SEPARATION_SOURCE_BYTES
                or not 1 <= total_chunks <= min(total_bytes, 4096)
            ):
                raise ValueError("Music edit output exceeds its encoded transfer bound")
            if identifier in self._completed:
                raise ValueError("Music edit output is already retained")
            existing = self._opened.get(identifier)
            if existing is not None:
                if existing != (total_bytes, total_chunks):
                    raise ValueError("Music edit open changed its declared transfer")
                return
            if len(self._opened) >= MAX_ACTIVE_MUSIC_EDITS:
                raise ValueError("Music edit active assembly capacity is full")
            if (
                identifier not in self._known
                and len(self._known) >= MAX_RETAINED_MUSIC_EDITS
            ):
                raise ValueError("Music edit retained content capacity is full")
            if self.reserved_bytes() + total_bytes > self.max_total_bytes:
                raise ValueError("Music edit media byte capacity is full")
            directory = self.root / "wav" / str(identifier)
            # A leftover is not a blank destination. Startup must reconcile it
            # against durable completed metadata before a new write can use it.
            directory.mkdir(mode=0o700)
            self._known.add(identifier)
            self._opened[identifier] = (total_bytes, total_chunks)
            try:
                evicted = self._store.open_assembly(
                    identifier,
                    "music",
                    content_type="audio/wav",
                    total_bytes=total_bytes,
                    total_chunks=total_chunks,
                    keep=self._known | {identifier},
                )
                if evicted:
                    raise RuntimeError(
                        "Protected music edit content was unexpectedly evicted"
                    )
                (directory / "output.wav.part").chmod(0o600)
            except BaseException:
                self.delete(identifier)
                raise

    def append(self, identifier: CommandId, sequence: int, data: bytes) -> None:
        """Feed raw admitted bytes; duplicates, ordering and chunk bounds stay shared."""
        with self._mutex:
            self._check(identifier)
            if identifier not in self._opened:
                raise ValueError("Music edit result was not opened")
            self._store.append(identifier, "music", sequence, data)

    def assembly_complete(self, identifier: CommandId) -> bool:
        """Observe complete chunk arrival without claiming custody or public content."""
        with self._mutex:
            self._check(identifier)
            return self._store.assembly_complete(identifier, "music")

    @staticmethod
    def _measure(path: Path, expected: NativeMusicWav) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
                or info.st_size > MAX_SEPARATION_SOURCE_BYTES
            ):
                raise ValueError("Music edit result is not a bounded private WAV")
            facts, peak = pcm24_from_wav(source.read(MAX_SEPARATION_SOURCE_BYTES + 1))
            if (
                facts.sha256,
                facts.size_bytes,
                facts.frames,
                facts.sample_rate,
                facts.channels,
                facts.bit_depth,
                facts.encoding,
                peak,
            ) != (
                expected.sha256,
                expected.size_bytes,
                expected.frames,
                expected.sample_rate,
                expected.channels,
                expected.bit_depth,
                "pcm",
                expected.peak,
            ):
                raise ValueError(
                    "Music edit WAV differs from its complete native manifest"
                )
            # The encoded facts are measured independently. Gain is native
            # provenance; it cannot be inferred from the finished WAV alone.
            os.fsync(source.fileno())
        _sync_directory(path.parent)
        _sync_directory(path.parent.parent)

    def commit(
        self, identifier: CommandId, expected: NativeMusicWav, *, total_chunks: int
    ) -> StoredMusicEdit:
        """Sync and independently verify one complete file before returning custody."""
        with self._mutex:
            self._check(identifier)
            if identifier not in self._opened:
                raise ValueError("Music edit result was not opened")
            try:
                stored = self._store.commit(
                    identifier,
                    "music",
                    sha256=expected.sha256,
                    total_chunks=total_chunks,
                )
                self._measure(stored.file_path, expected)
                if stored.size_bytes != expected.size_bytes:
                    raise ValueError("Music edit WAV differs from its transfer receipt")
                receipt = StoredMusicEdit(expected, stored.file_path, stored.expires_at)
                self._completed[identifier] = receipt
                del self._opened[identifier]
                return receipt
            except BaseException:
                self.delete(identifier)
                raise

    def adopt_complete(self, job: MusicEditJob) -> bool:
        """Recheck exact completed bytes at startup before any content is available."""
        with self._mutex:
            self._check(job.id)
            output = job.output
            if (
                output is None
                or job.expires_at is None
                or time.time() >= job.expires_at
            ):
                return False
            if (
                job.id not in self._known
                and len(self._known) >= MAX_RETAINED_MUSIC_EDITS
            ):
                raise ValueError("Music edit retained content capacity is full")
            expected = output.result.output
            try:
                if not self._store.adopt(
                    job.id,
                    "music",
                    content_type="audio/wav",
                    size_bytes=expected.size_bytes,
                    sha256=expected.sha256,
                    expires_at=float(job.expires_at),
                ):
                    self.delete(job.id)
                    return False
                stored = self._store.get(job.id, "music")
                if stored is None:
                    self.delete(job.id)
                    return False
                self._measure(stored.file_path, expected)
                self._known.add(job.id)
                self._completed[job.id] = StoredMusicEdit(
                    expected, stored.file_path, stored.expires_at
                )
                return True
            except (OSError, ValueError):
                self.delete(job.id)
                return False

    def content(self, job: MusicEditJob) -> StoredMusicEdit | None:
        """Expose only an unexpired matching completed job; caller leases its descriptor."""
        with self._mutex:
            self._check(job.id)
            output = job.output
            actual = self._completed.get(job.id)
            if (
                output is None
                or actual is None
                or actual.facts != output.result.output
                or job.expires_at is None
                or time.time() >= min(actual.expires_at, job.expires_at)
            ):
                return None
            try:
                info = actual.file_path.lstat()
            except FileNotFoundError:
                return None
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_size != actual.facts.size_bytes
                or info.st_nlink != 1
            ):
                return None
            return actual

    def delete(self, identifier: CommandId) -> None:
        """Remove owned result copies/assemblies; original context belongs elsewhere."""
        with self._mutex:
            self._check(identifier)
            self._store.delete(identifier)
            directory = self.root / "wav" / str(identifier)
            if directory.exists() or directory.is_symlink():
                raise OSError("Music edit result cleanup is incomplete")
            self._opened.pop(identifier, None)
            self._completed.pop(identifier, None)
            self._known.discard(identifier)

    def purge_unknown(self, keep: frozenset[CommandId]) -> int:
        """Remove orphan API copies only after durable completed-job reconciliation."""
        with self._mutex:
            if self._ownership is None:
                raise RuntimeError("Music edit media owner is closed")
            protected = set(keep) | self._known
            removed = self._store.purge_unknown(protected)
            # The shared video cache deliberately uses best-effort removal.
            # A new music owner must not become ready with leftover bytes that
            # its fresh in-memory reservation table does not account for.
            if any(
                (path.is_dir() or path.is_symlink())
                and CommandId(path.name) not in protected
                for path in (self.root / "wav").iterdir()
            ):
                raise OSError("Music edit orphan cleanup is incomplete")
            return removed

    def close(self) -> None:
        """Abort partial assemblies, retain complete files and release the OS fence."""
        with self._mutex:
            if self._ownership is None:
                return
            for identifier in tuple(self._opened):
                self.delete(identifier)
            self._ownership.close()
            self._ownership = None
