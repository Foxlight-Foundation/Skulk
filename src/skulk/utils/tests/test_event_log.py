# pyright: reportPrivateUsage=false
from pathlib import Path

import pytest

from skulk.shared.types.events import TestEvent
from skulk.utils.disk_event_log import DiskEventLog


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "event_log"


def test_append_and_read_back(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(5)]
    for e in events:
        log.append(e)

    assert len(log) == 5

    result = list(log.read_all())
    assert len(result) == 5
    for original, restored in zip(events, result, strict=True):
        assert original.event_id == restored.event_id

    log.close()


def test_read_range(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(10)]
    for e in events:
        log.append(e)

    result = list(log.read_range(3, 7))
    assert len(result) == 4
    for i, restored in enumerate(result):
        assert events[3 + i].event_id == restored.event_id

    log.close()


def test_read_range_bounds(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(3)]
    for e in events:
        log.append(e)

    # Start beyond count
    assert list(log.read_range(5, 10)) == []
    # Negative start
    assert list(log.read_range(-1, 2)) == []
    # End beyond count is clamped
    result = list(log.read_range(1, 100))
    assert len(result) == 2

    log.close()


def test_empty_log(log_dir: Path):
    log = DiskEventLog(log_dir)
    assert len(log) == 0
    assert list(log.read_all()) == []
    assert list(log.read_range(0, 10)) == []
    log.close()


def _archives(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("events.*.bin.zst"))


def test_rotation_on_close(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()

    active = log_dir / "events.bin"
    assert not active.exists()

    archives = _archives(log_dir)
    assert len(archives) == 1
    assert archives[0].stat().st_size > 0


def test_rotation_on_construction_with_stale_file(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "events.bin").write_bytes(b"stale data")

    log = DiskEventLog(log_dir)
    archives = _archives(log_dir)
    assert len(archives) == 1
    assert archives[0].exists()
    assert len(log) == 0

    log.close()


def test_empty_log_no_archive(log_dir: Path):
    """Closing an empty log should not leave an archive."""
    log = DiskEventLog(log_dir)
    log.close()

    active = log_dir / "events.bin"

    assert not active.exists()
    assert _archives(log_dir) == []


def test_close_is_idempotent(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()
    archive = _archives(log_dir)
    log.close()  # should not raise

    assert _archives(log_dir) == archive


def test_successive_sessions(log_dir: Path):
    """Simulate two master sessions: both archives should be kept."""
    log1 = DiskEventLog(log_dir)
    log1.append(TestEvent())
    log1.close()

    first_archive = _archives(log_dir)[-1]

    log2 = DiskEventLog(log_dir)
    log2.append(TestEvent())
    log2.append(TestEvent())
    log2.close()

    # Session 1 archive shifted to slot 2, session 2 in slot 1
    second_archive = _archives(log_dir)[-1]
    should_be_first_archive = _archives(log_dir)[-2]

    assert first_archive.exists()
    assert second_archive.exists()
    assert first_archive != second_archive
    assert should_be_first_archive == first_archive


def test_rotation_keeps_at_most_5_archives(log_dir: Path):
    """After 7 sessions, only the 5 most recent archives should remain."""
    all_archives: list[Path] = []
    for _ in range(7):
        log = DiskEventLog(log_dir)
        log.append(TestEvent())
        log.close()
        all_archives.append(_archives(log_dir)[-1])

    for old in all_archives[:2]:
        assert not old.exists()
    for recent in all_archives[2:]:
        assert recent.exists()


def test_compact_keeps_tail_and_absolute_indices(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(6)]
    for event in events:
        log.append(event)

    log.compact(4)

    assert log.start_idx == 4
    assert len(log) == 6
    assert list(log.read_range(0, 6)) == events[4:]
    assert list(log.read_range(4, 6)) == events[4:]
    assert list(log.read_range(5, 6)) == events[5:]

    log.close()


def test_read_range_does_not_cache_stale_offsets_after_compaction(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(6)]
    for event in events:
        log.append(event)

    in_flight = log.read_range(1, 5)
    first = next(in_flight)
    assert first.event_id == events[1].event_id

    log.compact(4)
    assert [event.event_id for event in in_flight] == [events[2].event_id, events[3].event_id, events[4].event_id]

    reread = list(log.read_range(5, 6))
    assert len(reread) == 1
    assert reread[0].event_id == events[5].event_id

    log.close()


def test_compact_aborts_when_retained_tail_is_incomplete(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(6)]
    for event in events:
        log.append(event)

    def fake_read_range(start: int, end: int):
        assert start == 4
        assert end == 6
        return iter(events[4:5])

    monkeypatch.setattr(log, "read_range", fake_read_range)

    log.compact(4)

    assert log.start_idx == 0
    assert len(log) == 6
    assert [event.event_id for event in log.read_all()] == [
        event.event_id for event in events
    ]

    log.close()


def _record_offset(path: Path, idx: int) -> int:
    offset = 0
    with open(path, "rb") as f:
        for _ in range(idx):
            f.seek(offset)
            header = f.read(4)
            assert len(header) == 4
            length = int.from_bytes(header, byteorder="big")
            offset += 4 + length
    return offset


def test_read_range_stops_on_corrupt_record(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(6)]
    for event in events:
        log.append(event)

    assert len(list(log.read_all())) == len(events)
    active_path = log_dir / "events.bin"
    corrupt_offset = _record_offset(active_path, 3)
    with open(active_path, "r+b") as f:
        f.seek(corrupt_offset + 4)
        f.write(b"e")

    result = list(log.read_range(1, 5))

    assert [event.event_id for event in result] == [
        events[1].event_id,
        events[2].event_id,
    ]

    log.close()


def test_compact_keeps_active_log_when_retained_tail_is_corrupt(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(6)]
    for event in events:
        log.append(event)

    assert len(list(log.read_all())) == len(events)
    active_path = log_dir / "events.bin"
    corrupt_offset = _record_offset(active_path, 4)
    with open(active_path, "r+b") as f:
        f.seek(corrupt_offset + 4)
        f.write(b"e")
    corrupt_bytes = active_path.read_bytes()

    log.compact(4)

    assert log.start_idx == 0
    assert len(log) == 6
    assert active_path.read_bytes() == corrupt_bytes

    log.close()


def test_persistence_failure_degrades_without_losing_indices(log_dir: Path):
    """ENOSPC (or any OSError) on append must not propagate — the canonical
    failure killed a node during the 2026-06-05 launch E2E smoke. Indices
    derive from len(log), so the count must keep advancing in the degraded
    counting-only mode or follower replay indices would collide."""
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    assert len(log) == 1

    assert log._file is not None
    log._file.close()  # next write raises ValueError... use a stub instead

    class _FullDisk:
        closed = False

        def write(self, _data: bytes) -> int:
            raise OSError(28, "No space left on device")

        def close(self) -> None:
            self.closed = True

        def flush(self) -> None:
            pass

    log._file = _FullDisk()  # type: ignore[assignment]

    log.append(TestEvent())  # must not raise
    assert len(log) == 2
    assert log._persistence_failed

    # Subsequent appends count without touching the dead file.
    log.append(TestEvent())
    assert len(log) == 3

    # Compaction is a no-op in counting-only mode (no coherent disk tail).
    log.compact(2)
    assert len(log) == 3
    assert log.start_idx == 0

    # Reads return empty without touching the dead file — the master's
    # replay caller only catches ValueError, so a flush-time OSError here
    # would kill the node through the side door.
    assert list(log.read_range(0, len(log))) == []
    assert list(log.read_all()) == []

    log.close()  # must not raise nor archive the dirty tail


def test_metadata_failure_counts_exactly_once(log_dir: Path):
    """ENOSPC at the _write_metadata site (the observed crash) arrives with
    the count already incremented — the handler must not increment again
    (PR #209 review: double-count would corrupt follower replay indices)."""
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    assert len(log) == 1

    def _fail_metadata() -> None:
        raise OSError(28, "No space left on device")

    log._write_metadata = _fail_metadata

    # Metadata now writes on a coarse cadence (#278/#279): drive the count to
    # the cadence boundary so the injected failure actually fires.
    from skulk.utils.disk_event_log import _METADATA_WRITE_INTERVAL_APPENDS

    while len(log) % _METADATA_WRITE_INTERVAL_APPENDS != _METADATA_WRITE_INTERVAL_APPENDS - 1:
        log.append(TestEvent())
    before = len(log)
    log.append(TestEvent())  # record write succeeds, metadata fails
    assert len(log) == before + 1  # exactly once, not twice
    assert log._persistence_failed
    log.append(TestEvent())
    assert len(log) == before + 2
    log.close()


def test_init_on_failing_disk_degrades_instead_of_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A full disk at construction time (the unguarded ENOSPC site that
    killed a node in the 2026-06-06 smoke via the API event log) must yield
    a degraded counting-only log, not an exception."""
    import builtins

    target = tmp_path / "event_log"
    real_open = builtins.open

    def _failing_open(file: object, *args: object, **kwargs: object) -> object:
        if str(file).endswith("events.bin"):
            raise OSError(28, "No space left on device")
        return real_open(file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", _failing_open)
    log = DiskEventLog(target)

    # Counting-only from birth: indices stay coherent, reads are empty.
    log.append(TestEvent())
    log.append(TestEvent())
    assert len(log) == 2
    assert list(log.read_all()) == []
    log.close()


def test_compact_failure_degrades_instead_of_raising(log_dir: Path):
    """ENOSPC during compaction's rewrite must degrade, not escape into the
    master's snapshot path."""
    import builtins

    log = DiskEventLog(log_dir)
    for _ in range(10):
        log.append(TestEvent())

    real_open = builtins.open

    def _failing_open(file: object, *args: object, **kwargs: object) -> object:
        if str(file).endswith("events.bin.tmp"):
            raise OSError(28, "No space left on device")
        return real_open(file, *args, **kwargs)  # type: ignore[arg-type]

    import unittest.mock

    with unittest.mock.patch("builtins.open", _failing_open):
        log.compact(5)

    # Logical range preserved; log keeps counting in degraded mode.
    assert len(log) == 10
    log.append(TestEvent())
    assert len(log) == 11
    assert list(log.read_all()) == []
    log.close()


def test_free_space_floor_triggers_degraded_mode(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Dropping below the free-space floor proactively degrades the log
    instead of running the disk to zero."""
    import shutil as _shutil

    import skulk.utils.disk_event_log as del_module

    log = DiskEventLog(log_dir)
    monkeypatch.setattr(del_module, "_DISK_CHECK_INTERVAL_APPENDS", 2)

    fake_usage = _shutil.disk_usage(log_dir)._replace(free=1)

    def _fake_disk_usage(_path: Path) -> object:
        return fake_usage

    monkeypatch.setattr(del_module.shutil, "disk_usage", _fake_disk_usage)

    log.append(TestEvent())
    log.append(TestEvent())  # triggers the check

    assert log._persistence_failed
    # Indices keep advancing in degraded mode.
    log.append(TestEvent())
    assert len(log) == 3
    log.close()


def test_archive_total_bytes_budget(log_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Archives are pruned to the byte budget, not just the count cap."""
    import skulk.utils.disk_event_log as del_module

    monkeypatch.setattr(del_module, "_MAX_ARCHIVE_TOTAL_BYTES", 1)

    for _session in range(3):
        log = DiskEventLog(log_dir)
        for _ in range(5):
            log.append(TestEvent())
        log.close()

    archives = sorted(log_dir.glob("events.*.bin.zst"))
    # With a 1-byte budget every archive beyond the newest is pruned; the
    # loop never deletes the final remaining archive mid-iteration, so at
    # most one survives.
    assert len(archives) <= 1


def test_active_size_bytes_tracks_appends(log_dir: Path):
    log = DiskEventLog(log_dir)
    assert log.active_size_bytes == 0
    log.append(TestEvent())
    assert log.active_size_bytes > 0
    log.close()


def test_append_after_compact_does_not_overwrite_tail(log_dir: Path):
    """Regression: compact() reopened the active file with the cursor at 0,
    so the next append overwrote the retained tail record by record. The
    master's post-snapshot compaction always had this; the API log's ring
    retention makes it fire constantly."""
    log = DiskEventLog(log_dir)
    for _ in range(10):
        log.append(TestEvent())

    log.compact(6)  # retain absolute indices [6, 10)
    log.append(TestEvent())  # must append, not overwrite index 6

    events = list(log.read_range(6, 11))
    assert len(events) == 5
    assert len(log) == 11
    assert log.active_size_bytes > 0
    log.close()


def test_metadata_not_rewritten_per_append(log_dir: Path):
    """Per-append metadata rewrites were the dominant physical-write term of
    every indexed event (#278/#279); the file is diagnostic-only and is now
    refreshed on a coarse cadence plus rotation/compaction/close."""
    import json

    log = DiskEventLog(log_dir)
    meta_path = log_dir / "events.meta.json"
    assert json.loads(meta_path.read_text())["count"] == 0

    for _ in range(10):
        log.append(TestEvent())
    # Ten appends: metadata still reflects the init-time write.
    assert json.loads(meta_path.read_text())["count"] == 0

    from skulk.utils.disk_event_log import _METADATA_WRITE_INTERVAL_APPENDS

    for _ in range(_METADATA_WRITE_INTERVAL_APPENDS - 10):
        log.append(TestEvent())
    assert (
        json.loads(meta_path.read_text())["count"]
        == _METADATA_WRITE_INTERVAL_APPENDS
    )
    log.close()
