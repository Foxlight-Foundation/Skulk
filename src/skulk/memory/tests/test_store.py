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


def test_append_after_torn_tail_repairs_the_segment(tmp_path: Path) -> None:
    """The first append after crash recovery must not fuse with the torn line."""
    _store, records = _seeded_store(tmp_path, 4)
    segment = next(p for p in (tmp_path / "memory").glob("spans-*.jsonl"))
    with segment.open("a", encoding="utf-8") as handle:
        handle.write('{"span_id": "torn", "text": "cut off mid-w')

    reopened = ContentStore(root=tmp_path / "memory")
    keys = random_vectors(1, DIM, seed=11)
    values = random_vectors(1, DIM, seed=12)
    reopened.append(_record("s-recovered", keys[0], values[0]))
    survivors = [r.span_id for r in reopened.scan()]
    assert survivors == [r.span_id for r in records] + ["s-recovered"]


def test_tombstone_enospc_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, records = _seeded_store(tmp_path, 2)
    real_open = Path.open

    def failing_open(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "tombstones.jsonl":
            raise OSError(28, "No space left on device")
        return real_open(self, *args, **kwargs)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType] - passthrough shim

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(StoreFullError):
        store.tombstone("s0", deleted_at=1_700_000_100.0)
    monkeypatch.setattr(Path, "open", real_open)
    assert store.degraded
    assert [r.span_id for r in store.scan()] == [r.span_id for r in records]


def test_enospc_during_rotation_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest writes during segment rotation honor the degraded-mode contract."""
    store = ContentStore(root=tmp_path / "memory")
    keys = random_vectors(1, DIM, seed=13)
    values = random_vectors(1, DIM, seed=14)
    real_open = Path.open

    def failing_open(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "MANIFEST.tmp":
            raise OSError(28, "No space left on device")
        return real_open(self, *args, **kwargs)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType] - passthrough shim

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(StoreFullError):
        store.append(_record("s-rotate", keys[0], values[0]))
    assert store.degraded


def test_rotation_preserves_fsync_cadence(tmp_path: Path) -> None:
    """The at-most-fsync_every loss window holds across segment boundaries."""
    store = ContentStore(root=tmp_path / "memory", rotation_bytes=1024, fsync_every=100)
    keys = random_vectors(3, DIM, seed=19)
    values = random_vectors(3, DIM, seed=20)
    store.append(_record("f0", keys[0], values[0]))
    assert store._appends_since_sync == 1  # pyright: ignore[reportPrivateUsage]
    # Records exceed rotation_bytes, so this append rotates; the outgoing
    # segment's unsynced tail must be fsynced and the counter reset, leaving
    # only the new segment's single append pending.
    store.append(_record("f1", keys[1], values[1]))
    assert store._appends_since_sync == 1  # pyright: ignore[reportPrivateUsage]


def test_compact_enospc_cleans_scratch_and_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os as os_module

    store, records = _seeded_store(tmp_path, 3)
    store.tombstone("s1", deleted_at=1_700_000_100.0)
    real_replace = os_module.replace

    def failing_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os_module, "replace", failing_replace)
    with pytest.raises(StoreFullError):
        store.compact()
    monkeypatch.setattr(os_module, "replace", real_replace)
    assert store.degraded
    assert not list((tmp_path / "memory").glob("*.tmp"))
    # The old view is fully intact: reads still work and hide the tombstone.
    assert [r.span_id for r in store.scan()] == ["s0", "s2"]
    assert records[1].span_id == "s1"


def test_compact_refuses_while_degraded(tmp_path: Path) -> None:
    store, _records = _seeded_store(tmp_path, 2)
    keys = random_vectors(1, DIM, seed=21)
    values = random_vectors(1, DIM, seed=22)
    store.free_space_floor_bytes = 1 << 60
    with pytest.raises(StoreFullError):
        store.append(_record("s-floor", keys[0], values[0]))
    assert store.degraded
    with pytest.raises(StoreFullError):
        store.compact()


def test_free_space_floor_degrades_before_enospc(tmp_path: Path) -> None:
    store = ContentStore(root=tmp_path / "memory", free_space_floor_bytes=1 << 60)
    keys = random_vectors(1, DIM, seed=15)
    values = random_vectors(1, DIM, seed=16)
    with pytest.raises(StoreFullError):
        store.append(_record("s-floor", keys[0], values[0]))
    assert store.degraded


def test_compact_never_reuses_an_old_segment_name(tmp_path: Path) -> None:
    """The manifest swap is the commit point; old names are never overwritten."""
    store = ContentStore(root=tmp_path / "memory", rotation_bytes=1024)
    keys = random_vectors(6, DIM, seed=17)
    values = random_vectors(6, DIM, seed=18)
    for i in range(6):
        store.append(_record(f"c{i}", keys[i], values[i]))
    old_names = {p.name for p in (tmp_path / "memory").glob("spans-*.jsonl")}
    assert len(old_names) > 1
    store.tombstone("c1", deleted_at=1_700_000_100.0)
    dropped = store.compact()
    assert dropped == 1
    manifest = json.loads((tmp_path / "memory" / "MANIFEST.json").read_text())
    assert len(manifest["segments"]) == 1
    assert manifest["segments"][0] not in old_names
    for name in old_names:
        assert not (tmp_path / "memory" / name).exists()
    assert [r.span_id for r in store.scan()] == ["c0", "c2", "c3", "c4", "c5"]

    # A stray file from a compaction that crashed before its manifest swap
    # must never be silently adopted by a later rotation.
    compacted_sequence = int(manifest["segments"][0].split("-")[1].split(".")[0])
    next_name = f"spans-{compacted_sequence + 1:06d}.jsonl"
    (tmp_path / "memory" / next_name).write_text('{"span_id": "ghost"}\n')
    store2 = ContentStore(root=tmp_path / "memory", rotation_bytes=1024)
    store2.append(_record("c-post", keys[0], values[0]))
    assert "ghost" not in [r.span_id for r in store2.scan()]
    assert [r.span_id for r in store2.scan()][-1] == "c-post"


def test_torn_multibyte_tail_is_dropped_not_fatal(tmp_path: Path) -> None:
    """A tear inside a multibyte UTF-8 character must not abort the scan."""
    store, records = _seeded_store(tmp_path, 3)
    segment = next(p for p in (tmp_path / "memory").glob("spans-*.jsonl"))
    with segment.open("ab") as handle:
        handle.write(b'{"span_id": "torn", "text": "caf\xc3')
    assert [r.span_id for r in store.scan()] == [r.span_id for r in records]

    # And the repair path handles it too: the next append truncates cleanly.
    reopened = ContentStore(root=tmp_path / "memory")
    keys = random_vectors(1, DIM, seed=23)
    values = random_vectors(1, DIM, seed=24)
    reopened.append(_record("s-after-tear", keys[0], values[0]))
    assert [r.span_id for r in reopened.scan()][-1] == "s-after-tear"


def test_mid_segment_corruption_is_loud_but_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store, records = _seeded_store(tmp_path, 4)
    segment = next(p for p in (tmp_path / "memory").glob("spans-*.jsonl"))
    lines = segment.read_bytes().splitlines(keepends=True)
    corrupted = lines[:2] + [b"garbage not json\n"] + lines[2:]
    segment.write_bytes(b"".join(corrupted))
    with caplog.at_level("WARNING", logger="skulk.memory.store"):
        survivors = [r.span_id for r in store.scan()]
    assert survivors == [r.span_id for r in records]
    assert store.corrupt_lines_dropped == 1
    assert any("corruption" in message for message in caplog.messages)


def test_compact_reserves_space_before_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compaction refuses up front when the survivor copy cannot fit."""
    import shutil as shutil_module
    from typing import NamedTuple

    class Usage(NamedTuple):
        total: int
        used: int
        free: int

    store, _records = _seeded_store(tmp_path, 3)
    store.free_space_floor_bytes = 0
    store.tombstone("s0", deleted_at=1_700_000_100.0)

    def tiny_disk(_root: object) -> Usage:
        return Usage(total=100, used=99, free=1)

    monkeypatch.setattr(shutil_module, "disk_usage", tiny_disk)
    with pytest.raises(StoreFullError):
        store.compact()
    # Refusal is not degradation: appends may still fit when a full
    # duplicate temporarily cannot.
    assert not store.degraded
    monkeypatch.undo()
    assert store.compact() == 1
    assert [r.span_id for r in store.scan()] == ["s1", "s2"]


def test_init_on_full_volume_starts_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First use on a full disk yields a degraded store, not a raw OSError."""
    real_open = Path.open

    def failing_open(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "MANIFEST.tmp":
            raise OSError(28, "No space left on device")
        return real_open(self, *args, **kwargs)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType] - passthrough shim

    monkeypatch.setattr(Path, "open", failing_open)
    store = ContentStore(root=tmp_path / "memory")
    monkeypatch.undo()
    assert store.degraded
    assert list(store.scan()) == []
    keys = random_vectors(1, DIM, seed=25)
    values = random_vectors(1, DIM, seed=26)
    with pytest.raises(StoreFullError):
        store.append(_record("s-init-full", keys[0], values[0]))


def test_mid_file_tombstone_corruption_is_loud(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store, _records = _seeded_store(tmp_path, 3)
    (tmp_path / "memory" / "tombstones.jsonl").write_text(
        'corrupt mid-file line\n{"span_id": "s1", "deleted_at": 1}\n'
    )
    with caplog.at_level("WARNING", logger="skulk.memory.store"):
        survivors = [r.span_id for r in store.scan()]
    # The valid tombstone still applies; the corrupt one is loud, not fatal.
    assert survivors == ["s0", "s2"]
    assert store.corrupt_lines_dropped == 1
    assert any("tombstone" in message for message in caplog.messages)


def test_whitener_versions_are_immutable(tmp_path: Path) -> None:
    store = ContentStore(root=tmp_path / "memory")
    corpus = np.random.default_rng(6).normal(size=(32, 16)).astype(np.float32)
    whitener = Whitener.fit(corpus, embedding_model_id="bge-small", version="v9")
    store.save_whitener(whitener)
    refit = Whitener.fit(corpus * 2, embedding_model_id="bge-small", version="v9")
    with pytest.raises(ValueError, match="immutable"):
        store.save_whitener(refit)


def test_torn_multibyte_tombstone_tail_is_dropped_not_fatal(tmp_path: Path) -> None:
    """UnicodeDecodeError is a ValueError: the tolerance already covers it."""
    store, _records = _seeded_store(tmp_path, 3)
    store.tombstone("s0", deleted_at=1_700_000_100.0)
    with (tmp_path / "memory" / "tombstones.jsonl").open("ab") as handle:
        handle.write(b'{"span_id": "caf\xc3')
    assert [r.span_id for r in store.scan()] == ["s1", "s2"]
    # A newline-less final line is the expected torn tail: silent, uncounted.
    assert store.corrupt_lines_dropped == 0


def test_floor_admission_accounts_for_pending_write_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free space slightly above the floor must not admit an oversized write."""
    import shutil as shutil_module
    from typing import NamedTuple

    class Usage(NamedTuple):
        total: int
        used: int
        free: int

    store, _records = _seeded_store(tmp_path, 1)
    store.free_space_floor_bytes = 1000
    keys = random_vectors(1, DIM, seed=27)
    values = random_vectors(1, DIM, seed=28)

    def slightly_above_floor(_root: object) -> Usage:
        return Usage(total=10_000, used=8_900, free=1_100)

    monkeypatch.setattr(shutil_module, "disk_usage", slightly_above_floor)
    # The record line is tens of KiB; only 100 bytes sit above the floor.
    with pytest.raises(StoreFullError):
        store.append(_record("s-oversize", keys[0], values[0]))
    assert store.degraded


def test_whitener_crash_recovery_retry_is_idempotent(tmp_path: Path) -> None:
    """Re-persisting the identical whitener completes a torn manifest update."""
    store = ContentStore(root=tmp_path / "memory")
    corpus = np.random.default_rng(7).normal(size=(32, 16)).astype(np.float32)
    whitener = Whitener.fit(corpus, embedding_model_id="bge-small", version="v11")
    store.save_whitener(whitener)
    # Simulate the crash window: archive durable, manifest update lost.
    manifest_path = tmp_path / "memory" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["whitener_version"] = None
    manifest["embedding_model_id"] = None
    manifest_path.write_text(json.dumps(manifest))

    store.save_whitener(whitener)
    loaded = store.load_whitener()
    assert loaded.version == "v11"
    # A genuinely different refit under the same version is still refused.
    refit = Whitener.fit(corpus * 3, embedding_model_id="bge-small", version="v11")
    with pytest.raises(ValueError, match="immutable"):
        store.save_whitener(refit)


def test_save_whitener_refuses_to_replace_a_populated_cue_space(
    tmp_path: Path,
) -> None:
    """A store holding spans cannot be repointed at a different cue space."""
    store, _records = _seeded_store(tmp_path, 2)
    corpus = np.random.default_rng(9).normal(size=(32, 16)).astype(np.float32)
    foreign = Whitener.fit(corpus, embedding_model_id="bge-small", version="v7")
    with pytest.raises(ValueError, match="cue space"):
        store.save_whitener(foreign)

    # The matching identity is accepted, and an empty store may switch.
    matching = Whitener.fit(
        corpus, embedding_model_id="bge-small-en-v1.5", version="v1"
    )
    store.save_whitener(matching)
    empty = ContentStore(root=tmp_path / "empty-memory")
    empty.save_whitener(foreign)
    replacement = Whitener.fit(
        corpus * 2, embedding_model_id="bge-small", version="v8"
    )
    empty.save_whitener(replacement)
    assert empty.load_whitener().version == "v8"


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
    # Newline-terminated means complete: this is corruption, not a torn
    # tail, and must be counted rather than silently tolerated.
    assert store.corrupt_lines_dropped == 1


def test_compact_failure_after_rename_removes_uncommitted_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest failure after the rename must not leave the survivor copy."""
    store, _records = _seeded_store(tmp_path, 3)
    store.tombstone("s0", deleted_at=1_700_000_100.0)
    old_names = {p.name for p in (tmp_path / "memory").glob("spans-*.jsonl")}
    real_open = Path.open

    def failing_open(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "MANIFEST.tmp":
            raise OSError(28, "No space left on device")
        return real_open(self, *args, **kwargs)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType] - passthrough shim

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(StoreFullError):
        store.compact()
    monkeypatch.undo()
    assert store.degraded
    # No scratch and no uncommitted sequence-fresh file squat on the disk.
    assert not list((tmp_path / "memory").glob("*.tmp"))
    assert {p.name for p in (tmp_path / "memory").glob("spans-*.jsonl")} == old_names
    assert [r.span_id for r in store.scan()] == ["s1", "s2"]


def test_append_rejects_foreign_cue_space(tmp_path: Path) -> None:
    store = ContentStore(root=tmp_path / "memory")
    corpus = np.random.default_rng(8).normal(size=(32, 16)).astype(np.float32)
    whitener = Whitener.fit(corpus, embedding_model_id="bge-small", version="v7")
    store.save_whitener(whitener)
    keys = random_vectors(1, DIM, seed=29)
    values = random_vectors(1, DIM, seed=30)
    # _record carries embedding_model_id "bge-small-en-v1.5" / whitener "v1",
    # which is not the store's active cue space.
    with pytest.raises(ValueError, match="cue space"):
        store.append(_record("s-foreign", keys[0], values[0]))


def test_rebuild_rejects_mixed_cue_spaces(tmp_path: Path) -> None:
    store, _records = _seeded_store(tmp_path, 2)
    keys = random_vectors(1, DIM, seed=31)
    values = random_vectors(1, DIM, seed=32)
    drifted = _record("s-drift", keys[0], values[0]).model_copy(
        update={"whitener_version": "v2"}
    )
    # No whitener is registered, so append cannot arbitrate; rebuild must.
    store.append(drifted)

    def factory() -> MemoryIndex:
        return HolographicField(dim=DIM)

    with pytest.raises(ValueError, match="cue spaces"):
        store.rebuild(factory)


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
