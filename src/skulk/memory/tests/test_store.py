# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Gate tests for the Phase 2 content store (crash safety, rebuild, forgetting)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skulk.memory.hrr import Vector, random_vectors
from skulk.memory.index import HolographicField, MemoryIndex
from skulk.memory.separation import Whitener
from skulk.memory.store import (
    ContentStore,
    SpanRecord,
    StoreFullError,
    encode_vector,
)

DIM = 2048


def _record(
    span_id: str, key: Vector, value: Vector, text: str = "a remembered thing"
) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        text=text,
        role="user",
        session_id="session-1",
        node_id="node-a",
        created_at=1_700_000_000.0,
        key_b64=encode_vector(key),
        value_b64=encode_vector(value),
        embedding_model_id="bge-small-en-v1.5",
        whitener_version="v1",
    )


def _seeded_store(tmp_path: Path, count: int) -> tuple[ContentStore, list[SpanRecord]]:
    store = ContentStore(root=tmp_path / "memory")
    keys = random_vectors(count, DIM, seed=1)
    values = random_vectors(count, DIM, seed=2)
    records = [_record(f"s{i}", keys[i], values[i]) for i in range(count)]
    for record in records:
        store.append(record)
    return store, records


def test_round_trip_preserves_records_and_vectors(tmp_path: Path) -> None:
    store, records = _seeded_store(tmp_path, 5)
    read_back = list(store.scan())
    assert [r.span_id for r in read_back] == [r.span_id for r in records]
    # fp16 storage: vectors survive within half-precision tolerance.
    original = records[3].key_vector()
    restored = read_back[3].key_vector()
    assert np.allclose(original, restored, atol=2e-3)


def test_torn_tail_is_dropped_not_fatal(tmp_path: Path) -> None:
    """A crash mid-append leaves a readable store missing only the torn row."""
    store, records = _seeded_store(tmp_path, 4)
    segment = next(p for p in (tmp_path / "memory").glob("spans-*.jsonl"))
    with segment.open("a", encoding="utf-8") as handle:
        handle.write('{"span_id": "torn", "text": "cut off mid-w')
    survivors = [r.span_id for r in store.scan()]
    assert survivors == [r.span_id for r in records]


def test_rebuild_probes_identically_to_original(tmp_path: Path) -> None:
    store, records = _seeded_store(tmp_path, 6)
    original = HolographicField(dim=DIM)
    for record in records:
        original.write(record.span_id, record.key_vector(), record.value_vector())

    def factory() -> MemoryIndex:
        return HolographicField(dim=DIM)

    rebuilt = store.rebuild(factory)
    assert isinstance(rebuilt, HolographicField)
    assert len(rebuilt) == len(original)
    assert abs(rebuilt.energy() - original.energy()) < 1e-2
    for record in records:
        a = original.probe(record.key_vector())
        b = rebuilt.probe(record.key_vector())
        assert a.trace_id == b.trace_id == record.span_id
        assert abs(a.confidence - b.confidence) < 5e-3


def test_tombstone_compact_rebuild_forgets_exactly(tmp_path: Path) -> None:
    store, records = _seeded_store(tmp_path, 5)
    store.tombstone("s2", deleted_at=1_700_000_100.0)
    assert [r.span_id for r in store.scan()] == ["s0", "s1", "s3", "s4"]
    dropped = store.compact()
    assert dropped == 1
    assert not (tmp_path / "memory" / "tombstones.jsonl").exists()
    survivors = [r.span_id for r in store.scan()]
    assert survivors == ["s0", "s1", "s3", "s4"]

    def factory() -> MemoryIndex:
        return HolographicField(dim=DIM)

    rebuilt = store.rebuild(factory)
    forgotten = rebuilt.probe(records[2].key_vector())
    assert forgotten.trace_id != "s2"


def test_enospc_degrades_capture_and_keeps_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, records = _seeded_store(tmp_path, 3)
    keys = random_vectors(1, DIM, seed=9)
    values = random_vectors(1, DIM, seed=10)

    real_open = Path.open

    def failing_open(self: Path, *args: object, **kwargs: object) -> object:
        if self.name.startswith("spans-") and "a" in str(args) + str(kwargs):
            raise OSError(28, "No space left on device")
        return real_open(self, *args, **kwargs)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType] - passthrough shim

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(StoreFullError):
        store.append(_record("s-full", keys[0], values[0]))
    monkeypatch.setattr(Path, "open", real_open)
    assert store.degraded
    with pytest.raises(StoreFullError):
        store.append(_record("s-after", keys[0], values[0]))
    # Reads keep working in degraded mode.
    assert [r.span_id for r in store.scan()] == [r.span_id for r in records]


def test_whitener_round_trip_preserves_cue_space(tmp_path: Path) -> None:
    store = ContentStore(root=tmp_path / "memory")
    corpus = np.random.default_rng(3).normal(size=(32, 16)).astype(np.float32)
    whitener = Whitener.fit(corpus, embedding_model_id="bge-small", version="v7")
    store.save_whitener(whitener)
    loaded = store.load_whitener()
    assert loaded.version == "v7"
    assert loaded.embedding_model_id == "bge-small"
    probe = corpus[0]
    assert np.allclose(whitener.transform(probe), loaded.transform(probe), atol=1e-5)


def test_manifest_survives_corrupt_tombstone_lines(tmp_path: Path) -> None:
    store, _records = _seeded_store(tmp_path, 2)
    (tmp_path / "memory" / "tombstones.jsonl").write_text(
        '{"span_id": "s0", "deleted_at": 1}\nnot json at all\n'
    )
    assert [r.span_id for r in store.scan()] == ["s1"]


def test_segment_rotation(tmp_path: Path) -> None:
    store = ContentStore(root=tmp_path / "memory", rotation_bytes=1024)
    keys = random_vectors(8, DIM, seed=4)
    values = random_vectors(8, DIM, seed=5)
    for i in range(8):
        store.append(_record(f"r{i}", keys[i], values[i]))
    segments = sorted((tmp_path / "memory").glob("spans-*.jsonl"))
    assert len(segments) > 1
    assert [r.span_id for r in store.scan()] == [f"r{i}" for i in range(8)]
    manifest = json.loads((tmp_path / "memory" / "MANIFEST.json").read_text())
    assert manifest["segments"] == [s.name for s in segments]
