"""Durable node-local content store for memory traces (fabric-memory Phase 2).

The store is the source of truth; fields are rebuildable indexes over it
(implementation-plan invariant #2). Append-only JSONL segments hold
:class:`SpanRecord` rows with fp16 vectors encoded as base64; exact forgetting
tombstones a span and subtracts its stored binding; ``rebuild`` replays live
records into a fresh :class:`~skulk.memory.index.MemoryIndex`, which is both
the crash-recovery path and the estimator-bench primitive (the same stored
history can be replayed into candidate index implementations offline).

Durability posture follows the ``DiskEventLog`` patterns: line-oriented
appends with an fsync cadence, torn-tail tolerance on read (a partial final
line is detected and dropped, never fatal), segment rotation, and an ENOSPC
degraded mode in which the store stops accepting writes while reads keep
working; memory degrades to off, inference never degrades.
"""

from __future__ import annotations

import base64
import errno
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from skulk.memory.hrr import DTYPE, Vector
from skulk.memory.index import MemoryIndex
from skulk.memory.separation import Whitener

SEGMENT_ROTATION_BYTES = 64 * 1024 * 1024
"""Rotate to a new append segment past this size."""

FSYNC_EVERY_APPENDS = 8
"""Appends between fsyncs: bounded loss window, bounded write amplification."""

FREE_SPACE_FLOOR_BYTES = 256 * 1024 * 1024
"""Refuse writes below this much free disk, before ENOSPC can even occur.

Running a volume to zero bytes destabilizes the whole node (the DiskEventLog
lesson); an optional memory subsystem must never be what fills the last byte.
"""

_MANIFEST_NAME = "MANIFEST.json"
_TOMBSTONES_NAME = "tombstones.jsonl"
_STORE_SCHEMA_VERSION = 1

_LOGGER = logging.getLogger(__name__)

SpanRole = Literal["user", "assistant"]
"""Speaker roles the capture pipeline persists.

A closed set on purpose: provenance is quoted back at surfacing time
("you told me..." versus "I said..."), so an unconstrained string here
would let a caller typo become a permanently misattributed memory.
"""


def encode_vector(vector: Vector) -> str:
    """Encode a vector as base64 fp16 for compact JSONL storage (capture path)."""
    return base64.b64encode(np.asarray(vector, dtype=np.float16).tobytes()).decode(
        "ascii"
    )


def _decode_vector(data: str) -> Vector:
    """Decode a base64 fp16 vector back to the library dtype."""
    raw = np.frombuffer(base64.b64decode(data.encode("ascii")), dtype=np.float16)
    return raw.astype(DTYPE)


class SpanRecord(BaseModel):
    """One captured span: the durable truth a field trace points at.

    Key and value vectors are persisted so exact forgetting and exact
    rewrites can subtract the binding actually written (Phase 1 review:
    stored pairs are authoritative), and so fields are rebuildable.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    span_id: str = Field(description="Unique id; the field's trace id.")
    text: str = Field(description="Verbatim span text quoted at surfacing.")
    role: SpanRole = Field(description="Speaker role of the span.")
    session_id: str = Field(description="Conversation/session provenance.")
    node_id: str = Field(description="Capturing node's id at write time.")
    created_at: float = Field(description="Unix seconds at capture.")
    key_b64: str = Field(description="Whitened cue key, base64 fp16.")
    value_b64: str = Field(description="Bound value vector, base64 fp16.")
    embedding_model_id: str = Field(description="Cue-space anchor identity.")
    whitener_version: str = Field(description="Separation-stage version.")

    def key_vector(self) -> Vector:
        """Decode the stored key vector."""
        return _decode_vector(self.key_b64)

    def value_vector(self) -> Vector:
        """Decode the stored value vector."""
        return _decode_vector(self.value_b64)


class StoreFullError(RuntimeError):
    """Raised on writes while the store is in ENOSPC degraded mode."""


@dataclass
class ContentStore:
    """Append-only span store under one directory (``SKULK_DATA_HOME/memory``).

    Not thread-safe by design: the single MemoryManager owns it (plan 4.1).
    """

    root: Path
    rotation_bytes: int = SEGMENT_ROTATION_BYTES
    fsync_every: int = FSYNC_EVERY_APPENDS
    free_space_floor_bytes: int = FREE_SPACE_FLOOR_BYTES
    _appends_since_sync: int = field(default=0, init=False)
    _degraded: bool = field(default=False, init=False)
    _tails_repaired: set[str] = field(default_factory=set, init=False)
    _corrupt_lines_dropped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self._manifest_path().exists():
                self._write_manifest(self._default_manifest())
        except OSError as error:
            # First use on an already-full volume must still honor the
            # contract: the store exists in degraded mode (reads work and
            # report nothing stored) instead of construction raising a raw
            # OSError into whatever component owns it.
            if error.errno != errno.ENOSPC:
                raise
            self._degraded = True
            _LOGGER.warning(
                "memory store initialization hit ENOSPC at %s; starting "
                "degraded (capture disabled, reads continue)",
                self.root,
            )

    # --- manifest -----------------------------------------------------------

    @staticmethod
    def _default_manifest() -> dict[str, object]:
        return {
            "schema": _STORE_SCHEMA_VERSION,
            "embedding_model_id": None,
            "whitener_version": None,
            "segments": [],
            "compacted_through": 0,
        }

    def _manifest_path(self) -> Path:
        return self.root / _MANIFEST_NAME

    def _read_manifest(self) -> dict[str, object]:
        path = self._manifest_path()
        if not path.exists():
            # A store that degraded before its first manifest write reads as
            # empty rather than failing.
            return self._default_manifest()
        data = json.loads(path.read_text())  # pyright: ignore[reportAny] - json.loads stub gap
        if not isinstance(data, dict):
            raise ValueError("memory store manifest is not an object")
        return {str(key): value for key, value in data.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        # Atomic replace so a crash mid-write cannot lose the manifest. The
        # scratch is fsynced before the rename and the directory after it, so
        # a power loss cannot persist later operations (segment unlinks) while
        # dropping the manifest swap they depend on.
        scratch = self._manifest_path().with_suffix(".tmp")
        with scratch.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, self._manifest_path())
        self._fsync_directory()

    def _fsync_directory(self) -> None:
        """Make renames and unlinks in the store directory durable.

        POSIX requires a directory fsync for metadata durability. Only a
        platform's refusal to fsync a directory descriptor (EINVAL/ENOTSUP)
        is tolerated, which is no worse than the pre-fsync behavior there;
        real storage failures (EIO, ENOSPC) propagate, because acknowledging
        a write whose directory entry may not survive would be a lie.
        """
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
        finally:
            os.close(descriptor)

    # --- segments -----------------------------------------------------------

    def _segments(self) -> list[Path]:
        manifest = self._read_manifest()
        names = manifest.get("segments")
        if not isinstance(names, list):
            return []
        return [self.root / str(name) for name in names]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    def _next_sequence(self) -> int:
        """Next segment sequence number, parsed from listed names.

        Parsing (rather than counting) keeps sequences monotonic across
        compaction, so a rotation can never reuse a name an older manifest
        generation referenced.
        """
        highest = 0
        for segment in self._segments():
            digits = segment.stem.rsplit("-", 1)[-1]
            if digits.isdigit():
                highest = max(highest, int(digits))
        return highest + 1

    def _active_segment(self) -> Path:
        segments = self._segments()
        if segments and segments[-1].exists() and (
            segments[-1].stat().st_size < self.rotation_bytes
        ):
            return segments[-1]
        # Rotating away: fsync the outgoing segment's unsynced tail first, so
        # the documented at-most-``fsync_every`` loss window holds across
        # segment boundaries instead of quietly resetting on rotation.
        if segments and segments[-1].exists() and self._appends_since_sync > 0:
            with segments[-1].open("rb+") as handle:
                os.fsync(handle.fileno())
            self._appends_since_sync = 0
        segment = self.root / f"spans-{self._next_sequence():06d}.jsonl"
        # A crashed compaction can leave a stray file at a name no manifest
        # references; never let rotation silently adopt its contents.
        segment.unlink(missing_ok=True)
        manifest = self._read_manifest()
        listed = manifest.get("segments")
        names: list[object] = listed if isinstance(listed, list) else []  # pyright: ignore[reportUnknownVariableType]
        names.append(segment.name)
        manifest["segments"] = names
        self._write_manifest(manifest)
        return segment

    # --- write path ----------------------------------------------------------

    @property
    def degraded(self) -> bool:
        """Whether the store refuses writes after an out-of-space failure."""
        return self._degraded

    def _refuse_writes_if_degraded(self, pending_bytes: int = 0) -> None:
        """Refuse while degraded, or when the pending write would breach the floor.

        Span text and whitener matrices are not size-bounded by this layer,
        so the admission check accounts for the write actually being asked
        for rather than admitting anything while one byte above the floor.
        """
        if self._degraded:
            raise StoreFullError("memory store is in degraded (out of space) mode")
        free = shutil.disk_usage(self.root).free
        if free < self.free_space_floor_bytes + pending_bytes:
            self._degraded = True
            raise StoreFullError(
                f"memory store free-space floor reached ({free} bytes free, "
                f"{pending_bytes} pending); capture disabled, reads continue"
            )

    def _translate_out_of_space(self, error: OSError) -> None:
        """Flip degraded mode and re-raise as :class:`StoreFullError` on ENOSPC."""
        if error.errno == errno.ENOSPC:
            self._degraded = True
            raise StoreFullError(
                "memory store hit ENOSPC; capture disabled, reads continue"
            ) from error

    def _repair_torn_tail(self, segment: Path) -> None:
        """Truncate a crash-torn final line before resuming appends.

        ``scan()`` tolerates a torn tail, but appending after one would
        concatenate the new record onto the unterminated fragment and lose
        both. Repair runs once per segment per process, before the first
        append that touches it.
        """
        if segment.name in self._tails_repaired:
            return
        self._tails_repaired.add(segment.name)
        if not segment.exists() or segment.stat().st_size == 0:
            return
        with segment.open("rb+") as handle:
            _ = handle.seek(0, os.SEEK_END)
            size = handle.tell()
            _ = handle.seek(max(0, size - 1))
            if handle.read(1) == b"\n":
                return
            # Walk back to the last newline; everything after it is the torn
            # fragment. Segments are line-oriented so a bounded backward scan
            # (one read per 4 KiB block) finds it quickly.
            position = size
            keep = 0
            while position > 0:
                block_start = max(0, position - 4096)
                _ = handle.seek(block_start)
                block = handle.read(position - block_start)
                newline_at = block.rfind(b"\n")
                if newline_at != -1:
                    keep = block_start + newline_at + 1
                    break
                position = block_start
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, record: SpanRecord) -> None:
        """Durably append one span record.

        Raises :class:`StoreFullError` while degraded; the caller's contract
        is that capture stops and serving continues (memory degrades to off).
        """
        line = record.model_dump_json() + "\n"
        self._refuse_writes_if_degraded(pending_bytes=len(line.encode("utf-8")))
        try:
            segment = self._active_segment()
            self._repair_torn_tail(segment)
            fresh = not segment.exists()
            with segment.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                self._appends_since_sync += 1
                if self._appends_since_sync >= self.fsync_every:
                    os.fsync(handle.fileno())
                    self._appends_since_sync = 0
            if fresh:
                # A new file's directory entry is not durable until the
                # directory itself is fsynced; without this, a power loss can
                # unlink a segment whose appends were already acknowledged.
                self._fsync_directory()
        except OSError as error:
            self._translate_out_of_space(error)
            raise

    def tombstone(self, span_id: str, *, deleted_at: float) -> None:
        """Record exact forgetting of one span (drop at next compaction)."""
        entry = json.dumps({"span_id": span_id, "deleted_at": deleted_at}) + "\n"
        self._refuse_writes_if_degraded(pending_bytes=len(entry.encode("utf-8")))
        path = self.root / _TOMBSTONES_NAME
        try:
            self._repair_torn_tail(path)
            fresh = not path.exists()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(entry)
                handle.flush()
                os.fsync(handle.fileno())
            if fresh:
                # Forgetting is a promise; the first tombstone's directory
                # entry must survive power loss like every later one.
                self._fsync_directory()
        except OSError as error:
            self._translate_out_of_space(error)
            raise

    # --- read path -----------------------------------------------------------

    def _tombstoned(self) -> set[str]:
        path = self.root / _TOMBSTONES_NAME
        if not path.exists():
            return set()
        ids: set[str] = set()
        # Binary mode: a crash can tear a line mid-multibyte-character, and
        # text-mode iteration would raise UnicodeDecodeError before any
        # per-line tolerance could run. json.loads accepts bytes directly.
        # Like scan(), only the final line is the expected torn-tail artifact;
        # corruption elsewhere would silently re-expose a forgotten span, so
        # it is skipped loudly.
        with path.open("rb") as handle:
            pending: tuple[int, bytes] | None = None
            for number, raw in enumerate(handle, start=1):
                if pending is not None:
                    self._collect_tombstone(ids, path, pending, is_tail=False)
                pending = (number, raw)
            if pending is not None:
                self._collect_tombstone(ids, path, pending, is_tail=True)
        return ids

    def _collect_tombstone(
        self,
        ids: set[str],
        path: Path,
        numbered: tuple[int, bytes],
        *,
        is_tail: bool,
    ) -> None:
        number, raw = numbered
        line = raw.strip()
        if not line:
            return
        try:
            entry = json.loads(line)  # pyright: ignore[reportAny] - json.loads stub gap
        except ValueError:
            if not is_tail:
                self._corrupt_lines_dropped += 1
                _LOGGER.warning(
                    "memory store dropped unparseable tombstone at %s:%d "
                    "(a forgotten span may be re-exposed until it is "
                    "tombstoned again)",
                    path.name,
                    number,
                )
            return
        if isinstance(entry, dict):
            span_id = entry.get("span_id")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(span_id, str):
                ids.add(span_id)

    @property
    def corrupt_lines_dropped(self) -> int:
        """Mid-segment lines dropped as unparseable across this store's reads.

        The expected torn tail (last line of the last segment) does not
        count; anything else here means bytes rotted underneath us and was
        logged when skipped.
        """
        return self._corrupt_lines_dropped

    def scan(self, *, include_tombstoned: bool = False) -> Iterator[SpanRecord]:
        """Iterate records in append order; never fatal on damaged lines.

        A partial final line in the final segment is the expected crash
        artifact and is dropped silently. Corruption anywhere else is not
        expected (segments are append-only and never mutated in place), so
        it is still skipped (reads must keep working) but logged loudly and
        counted in :attr:`corrupt_lines_dropped`: silence here would let
        compaction quietly persist the loss.
        """
        dead: set[str] = set() if include_tombstoned else self._tombstoned()
        segments = [segment for segment in self._segments() if segment.exists()]
        for segment in segments:
            final_segment = segment == segments[-1]
            # Stream line by line so a full 64 MiB segment never has to sit
            # in memory at once (the DiskEventLog read posture). Binary mode:
            # a tear inside a multibyte character would raise
            # UnicodeDecodeError during text-mode iteration, before any
            # per-line tolerance could run; pydantic validates bytes directly.
            with segment.open("rb") as handle:
                pending: tuple[int, bytes] | None = None
                for number, raw in enumerate(handle, start=1):
                    if pending is not None:
                        record = self._parse_line(segment, pending, is_tail=False)
                        if record is not None and record.span_id not in dead:
                            yield record
                    pending = (number, raw)
                if pending is not None:
                    record = self._parse_line(
                        segment, pending, is_tail=final_segment
                    )
                    if record is not None and record.span_id not in dead:
                        yield record

    def _parse_line(
        self, segment: Path, numbered: tuple[int, bytes], *, is_tail: bool
    ) -> SpanRecord | None:
        number, raw = numbered
        line = raw.strip()
        if not line:
            return None
        try:
            return SpanRecord.model_validate_json(line)
        except ValueError:
            if not is_tail:
                self._corrupt_lines_dropped += 1
                _LOGGER.warning(
                    "memory store dropped unparseable record at %s:%d "
                    "(mid-segment corruption; reads continue, but this line "
                    "is lost and compaction will not carry it)",
                    segment.name,
                    number,
                )
            return None

    def get(self, span_id: str) -> SpanRecord | None:
        """Fetch one live record by id (linear; the index owns fast lookup)."""
        for record in self.scan():
            if record.span_id == span_id:
                return record
        return None

    # --- maintenance -----------------------------------------------------------

    def compact(self) -> int:
        """Rewrite segments without tombstoned spans; returns dropped count.

        The compacted data is written under a sequence-fresh name that no
        prior manifest generation references, then the manifest swap is the
        single atomic commit point, and only after it do the old segments
        retire. A crash anywhere leaves either the complete old view or the
        complete new view, never records visible twice.
        """
        self._refuse_writes_if_degraded()
        dead = self._tombstoned()
        old_segments = self._segments()
        # Reserve before rewriting: the live segment bytes are an upper bound
        # on the survivor stream, so if that plus the floor does not fit, the
        # rewrite would only churn the disk toward ENOSPC before failing.
        # Refusing here does not flip degraded: ordinary appends may still
        # fit even when a full duplicate temporarily cannot.
        live_bytes = sum(
            segment.stat().st_size for segment in old_segments if segment.exists()
        )
        free = shutil.disk_usage(self.root).free
        if free < live_bytes + self.free_space_floor_bytes:
            raise StoreFullError(
                f"compaction needs up to {live_bytes} bytes plus the "
                f"{self.free_space_floor_bytes}-byte floor but only {free} "
                "bytes are free; retry after space is reclaimed"
            )
        final = self.root / f"spans-{self._next_sequence():06d}.jsonl"
        scratch = self.root / (final.name + ".tmp")
        survivors = 0
        dropped = 0
        try:
            with scratch.open("w", encoding="utf-8") as handle:
                for record in self.scan(include_tombstoned=True):
                    if record.span_id in dead:
                        dropped += 1
                        continue
                    handle.write(record.model_dump_json() + "\n")
                    survivors += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(scratch, final)
            self._fsync_directory()
            manifest = self._read_manifest()
            manifest["segments"] = [final.name]
            manifest["compacted_through"] = survivors
            self._write_manifest(manifest)
        except OSError as error:
            # A failed rewrite must not squat on the disk's last bytes: free
            # the partial scratch before entering degraded mode.
            scratch.unlink(missing_ok=True)
            self._translate_out_of_space(error)
            raise
        # Past the commit point: cleanup failures cost disk, never records.
        for stale in old_segments:
            if stale != final:
                stale.unlink(missing_ok=True)
        (self.root / _TOMBSTONES_NAME).unlink(missing_ok=True)
        return dropped

    def rebuild(self, index_factory: Callable[[], MemoryIndex]) -> MemoryIndex:
        """Replay live records into a fresh index (crash recovery and bench).

        The rebuilt index observes the same write order the store did, so a
        field rebuilt after a node death probes identically to the original
        up to decay elapsed since the last capture (invariant #2: losing a
        field costs recency salience, never memories).
        """
        index = index_factory()
        for record in self.scan():
            index.write(record.span_id, record.key_vector(), record.value_vector())
        return index

    # --- whitener persistence ----------------------------------------------------

    def save_whitener(self, whitener: Whitener) -> Path:
        """Persist the frozen separation stage for exact cue-space rebuilds."""
        pending = whitener.mu.nbytes + whitener.matrix.nbytes + 4096
        self._refuse_writes_if_degraded(pending_bytes=pending)
        path = self.root / f"whitener-{whitener.version}.npz"
        if path.exists():
            # A version string names one immutable cue space; stored records
            # identify their keys by it. The one legitimate re-persist is
            # crash recovery: a power loss between the archive rename and the
            # manifest update leaves a complete identical orphan, and
            # retrying then only needs the manifest half.
            if self._matches_persisted_whitener(whitener):
                try:
                    manifest = self._read_manifest()
                    manifest["whitener_version"] = whitener.version
                    manifest["embedding_model_id"] = whitener.embedding_model_id
                    self._write_manifest(manifest)
                except OSError as error:
                    self._translate_out_of_space(error)
                    raise
                return path
            raise ValueError(
                f"whitener version {whitener.version!r} is already persisted "
                "with different contents; whitener versions are immutable, "
                "fit under a new version"
            )
        scratch = self.root / (path.name + ".tmp")
        try:
            # Write to a scratch handle and fsync before the rename, then
            # fsync the directory: the manifest update below is durable, so
            # the file it names must be durable first, or a power loss leaves
            # the manifest pointing at a missing or truncated archive.
            with scratch.open("wb") as handle:
                np.savez(
                    handle,
                    mu=whitener.mu,
                    matrix=whitener.matrix,
                    alpha=np.float64(whitener.alpha),
                    shrinkage=np.float64(whitener.shrinkage),
                    embedding_model_id=np.str_(whitener.embedding_model_id),
                    version=np.str_(whitener.version),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(scratch, path)
            self._fsync_directory()
            manifest = self._read_manifest()
            manifest["whitener_version"] = whitener.version
            manifest["embedding_model_id"] = whitener.embedding_model_id
            self._write_manifest(manifest)
        except OSError as error:
            scratch.unlink(missing_ok=True)
            self._translate_out_of_space(error)
            raise
        return path

    def _matches_persisted_whitener(self, whitener: Whitener) -> bool:
        """Whether the persisted archive for this version is exactly ``whitener``.

        Exact array equality is the right bar: save and load round-trip
        float32 bit-identically, so a crash-recovery retry matches exactly
        and a refit under a reused version does not.
        """
        try:
            existing = self.load_whitener(whitener.version)
        except (OSError, ValueError, KeyError):
            return False
        return (
            bool(np.array_equal(existing.mu, whitener.mu))
            and bool(np.array_equal(existing.matrix, whitener.matrix))
            and existing.alpha == whitener.alpha
            and existing.shrinkage == whitener.shrinkage
            and existing.embedding_model_id == whitener.embedding_model_id
        )

    def load_whitener(self, version: str | None = None) -> Whitener:
        """Load a persisted whitener (the manifest's active one by default)."""
        if version is None:
            manifest = self._read_manifest()
            active = manifest.get("whitener_version")
            if not isinstance(active, str):
                raise FileNotFoundError("memory store has no active whitener")
            version = active
        path = self.root / f"whitener-{version}.npz"
        with np.load(path) as data:  # pyright: ignore[reportAny] - npz mapping stub gap
            return Whitener(
                mu=data["mu"].astype(DTYPE),  # pyright: ignore[reportAny] - npz mapping stub gap
                matrix=data["matrix"].astype(DTYPE),  # pyright: ignore[reportAny] - npz mapping stub gap
                alpha=float(data["alpha"]),  # pyright: ignore[reportAny] - npz mapping stub gap
                shrinkage=float(data["shrinkage"]),  # pyright: ignore[reportAny] - npz mapping stub gap
                embedding_model_id=str(data["embedding_model_id"]),  # pyright: ignore[reportAny] - npz mapping stub gap
                version=str(data["version"]),  # pyright: ignore[reportAny] - npz mapping stub gap
            )
