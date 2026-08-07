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
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from skulk.memory.hrr import DTYPE, Vector
from skulk.memory.index import MemoryIndex
from skulk.memory.separation import Whitener

SEGMENT_ROTATION_BYTES = 64 * 1024 * 1024
"""Rotate to a new append segment past this size."""

FSYNC_EVERY_APPENDS = 8
"""Appends between fsyncs: bounded loss window, bounded write amplification."""

_MANIFEST_NAME = "MANIFEST.json"
_TOMBSTONES_NAME = "tombstones.jsonl"
_STORE_SCHEMA_VERSION = 1


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
    role: str = Field(description="Speaker role of the span (user/assistant).")
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
    _appends_since_sync: int = field(default=0, init=False)
    _degraded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path().exists():
            self._write_manifest({
                "schema": _STORE_SCHEMA_VERSION,
                "embedding_model_id": None,
                "whitener_version": None,
                "segments": [],
                "compacted_through": 0,
            })

    # --- manifest -----------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.root / _MANIFEST_NAME

    def _read_manifest(self) -> dict[str, object]:
        data = json.loads(self._manifest_path().read_text())  # pyright: ignore[reportAny] - json.loads stub gap
        if not isinstance(data, dict):
            raise ValueError("memory store manifest is not an object")
        return {str(key): value for key, value in data.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        # Atomic replace so a crash mid-write cannot lose the manifest.
        scratch = self._manifest_path().with_suffix(".tmp")
        scratch.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(scratch, self._manifest_path())

    # --- segments -----------------------------------------------------------

    def _segments(self) -> list[Path]:
        manifest = self._read_manifest()
        names = manifest.get("segments")
        if not isinstance(names, list):
            return []
        return [self.root / str(name) for name in names]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    def _active_segment(self) -> Path:
        segments = self._segments()
        if segments and segments[-1].exists() and (
            segments[-1].stat().st_size < self.rotation_bytes
        ):
            return segments[-1]
        sequence = len(segments) + 1
        segment = self.root / f"spans-{sequence:06d}.jsonl"
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

    def append(self, record: SpanRecord) -> None:
        """Durably append one span record.

        Raises :class:`StoreFullError` while degraded; the caller's contract
        is that capture stops and serving continues (memory degrades to off).
        """
        if self._degraded:
            raise StoreFullError("memory store is in degraded (out of space) mode")
        line = record.model_dump_json() + "\n"
        segment = self._active_segment()
        try:
            with segment.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                self._appends_since_sync += 1
                if self._appends_since_sync >= self.fsync_every:
                    os.fsync(handle.fileno())
                    self._appends_since_sync = 0
        except OSError as error:
            if error.errno == 28:  # ENOSPC
                self._degraded = True
                raise StoreFullError(
                    "memory store hit ENOSPC; capture disabled, reads continue"
                ) from error
            raise

    def tombstone(self, span_id: str, *, deleted_at: float) -> None:
        """Record exact forgetting of one span (drop at next compaction)."""
        if self._degraded:
            raise StoreFullError("memory store is in degraded (out of space) mode")
        entry = json.dumps({"span_id": span_id, "deleted_at": deleted_at}) + "\n"
        with (self.root / _TOMBSTONES_NAME).open("a", encoding="utf-8") as handle:
            handle.write(entry)
            handle.flush()
            os.fsync(handle.fileno())

    # --- read path -----------------------------------------------------------

    def _tombstoned(self) -> set[str]:
        path = self.root / _TOMBSTONES_NAME
        if not path.exists():
            return set()
        ids: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)  # pyright: ignore[reportAny] - json.loads stub gap
            except json.JSONDecodeError:
                continue  # torn tail on the tombstone file: skip, never fatal
            if isinstance(entry, dict):
                span_id = entry.get("span_id")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if isinstance(span_id, str):
                    ids.add(span_id)
        return ids

    def scan(self, *, include_tombstoned: bool = False) -> Iterator[SpanRecord]:
        """Iterate records in append order, dropping torn tails silently.

        A partial final line (crash mid-append) fails JSON parsing and is
        skipped; every complete line before it is preserved. This is the
        crash-safety contract the gate tests pin.
        """
        dead: set[str] = set() if include_tombstoned else self._tombstoned()
        for segment in self._segments():
            if not segment.exists():
                continue
            for line in segment.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = SpanRecord.model_validate_json(line)
                except ValueError:
                    continue  # torn or corrupt line: drop, never fatal
                if record.span_id in dead:
                    continue
                yield record

    def get(self, span_id: str) -> SpanRecord | None:
        """Fetch one live record by id (linear; the index owns fast lookup)."""
        for record in self.scan():
            if record.span_id == span_id:
                return record
        return None

    # --- maintenance -----------------------------------------------------------

    def compact(self) -> int:
        """Rewrite segments without tombstoned spans; returns dropped count.

        Compaction writes a fresh single segment then atomically swaps the
        manifest, so a crash at any point leaves either the old or the new
        view, never a mix.
        """
        dead = self._tombstoned()
        survivors = list(self.scan())
        dropped = 0
        for record in self.scan(include_tombstoned=True):
            if record.span_id in dead:
                dropped += 1
        fresh = self.root / "spans-compact.jsonl.tmp"
        with fresh.open("w", encoding="utf-8") as handle:
            for record in survivors:
                handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        final = self.root / "spans-000001.jsonl"
        old_segments = [p for p in self._segments() if p.exists()]
        os.replace(fresh, final)
        manifest = self._read_manifest()
        manifest["segments"] = [final.name]
        manifest["compacted_through"] = len(survivors)
        self._write_manifest(manifest)
        for stale in old_segments:
            if stale != final and stale.exists():
                stale.unlink()
        tombstones = self.root / _TOMBSTONES_NAME
        if tombstones.exists():
            tombstones.unlink()
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
        path = self.root / f"whitener-{whitener.version}.npz"
        np.savez(
            path,
            mu=whitener.mu,
            matrix=whitener.matrix,
            alpha=np.float64(whitener.alpha),
            shrinkage=np.float64(whitener.shrinkage),
            embedding_model_id=np.str_(whitener.embedding_model_id),
            version=np.str_(whitener.version),
        )
        manifest = self._read_manifest()
        manifest["whitener_version"] = whitener.version
        manifest["embedding_model_id"] = whitener.embedding_model_id
        self._write_manifest(manifest)
        return path

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
