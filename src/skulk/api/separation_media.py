"""Private four-stem custody over the existing bounded generated-media assembler.

All methods doing file I/O belong off the API event loop. A complete stored file
is an internal receipt, not public content: reads additionally require a durable
completed job with the exact complete four-file manifest. This class owns only
API result copies, never uploaded/Studio originals or native runner files.
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

from skulk.api.separation_jobs import (
    MAX_ACTIVE_SEPARATION_JOBS,
    MAX_RETAINED_SEPARATION_JOBS,
    SeparationJob,
)
from skulk.api.video_store import VideoStore
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_SET_BYTES,
    MAX_SEPARATION_STEM_BYTES,
    SEPARATION_STEMS,
    SeparatedStem,
    SeparationStemName,
    stem_from_wav,
)
from skulk.shared.types.common import CommandId
from skulk.utils.runtime_files import RuntimeLock, private_directory


def _sync_directory(path: Path) -> None:
    """Persist directory entries before any corresponding durable job receipt."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@final
@dataclass(frozen=True)
class StoredSeparationStem:
    """Private closed-file receipt; never a public or replicated response type."""

    facts: SeparatedStem
    file_path: Path
    expires_at: float


@final
class SeparationMediaStore:
    """Own four closed output roles without changing video or music store contracts.

    Each role uses the existing WAV assembler in a distinct private directory.
    Aggregate byte/job/assembly limits apply across all four. Known complete
    results cannot be evicted to admit new work; the owning job index explicitly
    deletes expired or reviewed results. A lifetime fence excludes other owners.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_total_bytes: int = MAX_RETAINED_SEPARATION_JOBS * MAX_SEPARATION_SET_BYTES,
    ) -> None:
        """Create protected storage; existing results need complete verified adoption."""
        if (
            not 176
            <= max_total_bytes
            <= MAX_RETAINED_SEPARATION_JOBS * MAX_SEPARATION_SET_BYTES
        ):
            raise ValueError("separation media budget is outside its bounded policy")
        # mkdir(parents=True) can introduce more than one missing ancestor.
        # Persist their links bottom-up too, without syncing unrelated ancestors
        # above the nearest already-existing directory.
        created_roots: list[Path] = []
        ancestor = root
        while not ancestor.exists():
            created_roots.append(ancestor)
            ancestor = ancestor.parent
        self.root = root
        self.max_total_bytes = max_total_bytes
        self._mutex = threading.RLock()
        self._ownership: RuntimeLock | None = RuntimeLock(root, name="service.lock")
        self._stores: dict[SeparationStemName, VideoStore] = {}
        self._opened: dict[tuple[CommandId, SeparationStemName], tuple[int, int]] = {}
        self._completed: dict[
            tuple[CommandId, SeparationStemName], StoredSeparationStem
        ] = {}
        self._known: set[CommandId] = set()
        try:
            for name in SEPARATION_STEMS:
                directory = root / name
                private_directory(directory)
                self._stores[name] = VideoStore(
                    directory, max_total_bytes=max_total_bytes
                )
                _sync_directory(directory)
            _sync_directory(root)
            for created in created_roots:
                _sync_directory(created.parent)
        except BaseException:
            self._ownership.close()
            self._ownership = None
            raise

    def _check(self, command_id: CommandId) -> None:
        if self._ownership is None:
            raise RuntimeError("separation media owner is closed")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(command_id)) is None:
            raise ValueError("invalid separation artifact identity")

    def reserved_bytes(self) -> int:
        """Return declared in-flight and complete bytes across the complete store."""
        with self._mutex:
            return sum(store.reserved_bytes() for store in self._stores.values())

    def open_assembly(
        self,
        command_id: CommandId,
        name: SeparationStemName,
        *,
        total_bytes: int,
        total_chunks: int,
    ) -> None:
        """Reserve one bounded WAV without discarding progress on an identical open."""
        with self._mutex:
            self._check(command_id)
            if (
                not 44 <= total_bytes <= MAX_SEPARATION_STEM_BYTES
                or not 1 <= total_chunks <= min(total_bytes, 4096)
            ):
                raise ValueError("separation output exceeds its transfer bounds")
            key = (command_id, name)
            if key in self._completed:
                raise ValueError("separation output is already retained")
            existing = self._opened.get(key)
            if existing is not None:
                if existing != (total_bytes, total_chunks):
                    raise ValueError("separation open changed its declared transfer")
                return
            if len(self._opened) >= MAX_ACTIVE_SEPARATION_JOBS * 4:
                raise ValueError("separation assembly capacity is full")
            active = {identifier for identifier, _ in self._opened}
            if command_id not in active and len(active) >= MAX_ACTIVE_SEPARATION_JOBS:
                raise ValueError("separation active-job assembly capacity is full")
            if (
                command_id not in self._known
                and len(self._known) >= MAX_RETAINED_SEPARATION_JOBS
            ):
                raise ValueError("separation retained-result capacity is full")
            if self.reserved_bytes() + total_bytes > self.max_total_bytes:
                raise ValueError("separation media byte capacity is full")
            headroom = self._stores[name].disk_headroom_bytes()
            promised = sum(store.unwritten_bytes() for store in self._stores.values())
            # All four role stores use this owner's filesystem. Their child
            # checks cannot independently promise the same free bytes; count
            # every outstanding write before opening another role or job.
            if headroom is not None and promised + total_bytes > headroom:
                raise ValueError("separation filesystem headroom is insufficient")
            evicted = self._stores[name].open_assembly(
                command_id,
                "music",
                content_type="audio/wav",
                total_bytes=total_bytes,
                total_chunks=total_chunks,
                keep=self._known | {command_id},
            )
            if evicted:
                raise RuntimeError(
                    "protected separation content was unexpectedly evicted"
                )
            self._known.add(command_id)
            self._opened[key] = (total_bytes, total_chunks)

    def append(
        self,
        command_id: CommandId,
        name: SeparationStemName,
        sequence: int,
        data: bytes,
    ) -> None:
        """Feed an admitted stream through existing bounded ordered/stashed writes."""
        with self._mutex:
            self._check(command_id)
            if (command_id, name) not in self._opened:
                raise ValueError("separation stream was not opened")
            self._stores[name].append(command_id, "music", sequence, data)

    def assembly_complete(
        self, command_id: CommandId, name: SeparationStemName
    ) -> bool:
        """Return whether every chunk has arrived; this is not a content receipt."""
        with self._mutex:
            self._check(command_id)
            return self._stores[name].assembly_complete(command_id, "music")

    @staticmethod
    def _measure(name: SeparationStemName, path: Path) -> SeparatedStem:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > MAX_SEPARATION_STEM_BYTES
            ):
                raise ValueError("separation result is not a bounded regular WAV")
            facts = stem_from_wav(name, source.read(MAX_SEPARATION_STEM_BYTES + 1))
            if facts.sample_rate != 44100:
                raise ValueError(
                    "HTDemucs output differs from its native sampling rate"
                )
            # The durable job index must never outlive an unsynced result copy.
            os.fsync(source.fileno())
        _sync_directory(path.parent)
        # The WAV entry and the job directory's link are separate durability
        # boundaries. Syncing only the job directory can lose its parent entry.
        _sync_directory(path.parent.parent)
        return facts

    def commit(
        self,
        command_id: CommandId,
        name: SeparationStemName,
        *,
        sha256: str,
        total_chunks: int,
    ) -> StoredSeparationStem:
        """Close, independently verify and sync one file; partial sets stay private."""
        with self._mutex:
            self._check(command_id)
            if (command_id, name) not in self._opened:
                raise ValueError("separation stream was not opened")
            try:
                stored = self._stores[name].commit(
                    command_id, "music", sha256=sha256, total_chunks=total_chunks
                )
                facts = self._measure(name, stored.file_path)
                if (facts.sha256, facts.size_bytes) != (
                    stored.sha256,
                    stored.size_bytes,
                ):
                    raise ValueError(
                        "separation bytes differ from their transport receipt"
                    )
                receipt = StoredSeparationStem(
                    facts, stored.file_path, stored.expires_at
                )
                self._completed[(command_id, name)] = receipt
                del self._opened[(command_id, name)]
                return receipt
            except BaseException:
                self.delete(command_id)
                raise

    def adopt_complete(self, job: SeparationJob) -> bool:
        """Verify all four durable completed files after restart before any read."""
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
                and len(self._known) >= MAX_RETAINED_SEPARATION_JOBS
            ):
                raise ValueError("separation retained-result capacity is full")
            try:
                for stem in output.stems:
                    store = self._stores[stem.name]
                    if not store.adopt(
                        job.id,
                        "music",
                        content_type="audio/wav",
                        size_bytes=stem.size_bytes,
                        sha256=stem.sha256,
                        expires_at=float(job.expires_at),
                    ):
                        self.delete(job.id)
                        return False
                    stored = store.get(job.id, "music")
                    if (
                        stored is None
                        or self._measure(stem.name, stored.file_path) != stem
                    ):
                        self.delete(job.id)
                        return False
                    self._completed[(job.id, stem.name)] = StoredSeparationStem(
                        stem, stored.file_path, stored.expires_at
                    )
                if self.reserved_bytes() > self.max_total_bytes:
                    raise ValueError(
                        "separation adoption exceeds its total byte budget"
                    )
                self._known.add(job.id)
                return True
            except (ValueError, OSError):
                self.delete(job.id)
                return False

    def content(
        self, job: SeparationJob, name: SeparationStemName
    ) -> StoredSeparationStem | None:
        """Return a path only for an unexpired, durably completed matching set."""
        with self._mutex:
            self._check(job.id)
            output = job.output
            if (
                output is None
                or job.expires_at is None
                or time.time() >= job.expires_at
            ):
                return None
            for expected in output.stems:
                actual = self._completed.get((job.id, expected.name))
                if (
                    actual is None
                    or actual.facts != expected
                    or time.time() >= actual.expires_at
                ):
                    return None
                try:
                    info = actual.file_path.lstat()
                except FileNotFoundError:
                    return None
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size != expected.size_bytes
                ):
                    return None
            return self._completed[(job.id, name)]

    def delete(self, command_id: CommandId) -> None:
        """Remove all owned result copies and assemblies, never input/original audio."""
        with self._mutex:
            self._check(command_id)
            for store in self._stores.values():
                store.delete(command_id)
            for name in SEPARATION_STEMS:
                self._opened.pop((command_id, name), None)
                self._completed.pop((command_id, name), None)
            self._known.discard(command_id)

    def purge_unknown(self, keep: frozenset[CommandId]) -> int:
        """Remove orphan API-result directories after durable startup reconciliation."""
        with self._mutex:
            if self._ownership is None:
                raise RuntimeError("separation media owner is closed")
            protected = set(keep) | self._known
            return sum(
                store.purge_unknown(protected) for store in self._stores.values()
            )

    def close(self) -> None:
        """Close partial assemblies before releasing ownership; retain complete files."""
        with self._mutex:
            if self._ownership is None:
                return
            for command_id, name in tuple(self._opened):
                self._stores[name].abort_assembly(command_id, "music")
                del self._opened[(command_id, name)]
            self._ownership.close()
            self._ownership = None
