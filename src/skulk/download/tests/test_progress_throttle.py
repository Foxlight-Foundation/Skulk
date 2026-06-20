# pyright: reportPrivateUsage=false
"""Download-progress throttle bounds the in_progress event stream (#364).

A large download fires a progress callback per 8MB chunk across parallel files,
and each emitted ``NodeDownloadProgress`` is a full ordered event (master indexes
it, appends the event log, persists a snapshot, and rebroadcasts to every node).
So the in_progress stream must be bounded by total count, not by rate -- a pure
time gate still scales event volume with download duration, which saturated the
gossip send queue and dropped the terminal ``DownloadCompleted`` so placements
wedged in RunnerLoading. The fraction-delta gate caps a download to roughly
``1 / _PROGRESS_STEP`` in_progress events regardless of size or duration.
"""

from typing import cast

import pytest

from skulk.download import coordinator as coordinator_mod
from skulk.download.coordinator import DownloadCoordinator
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import NodeId
from skulk.utils.channels import channel


class _FakeDownloader:
    def on_progress(self, callback: object) -> None:
        pass


def _make_coordinator() -> DownloadCoordinator:
    _, cmd_recv = channel[object]()
    event_send, _ = channel[object]()
    return DownloadCoordinator(
        node_id=NodeId("n1"),
        shard_downloader=cast("object", _FakeDownloader()),  # pyright: ignore[reportArgumentType]
        download_command_receiver=cast("object", cmd_recv),  # pyright: ignore[reportArgumentType]
        event_sender=cast("object", event_send),  # pyright: ignore[reportArgumentType]
    )


def test_in_progress_throttle_gates_by_fraction_rate_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    co = _make_coordinator()
    mid = ModelId("org/model")
    clock = {"now": 1000.0}

    def fake_now() -> float:
        return clock["now"]

    monkeypatch.setattr(coordinator_mod, "current_time", fake_now)

    # First update always emits (establishes the baseline).
    assert co._should_emit_in_progress(mid, 0.0) is True

    # A >=step advance but within the rate floor (<1s) is suppressed.
    clock["now"] = 1000.5
    assert co._should_emit_in_progress(mid, 0.20) is False

    # A >=step advance after the rate floor emits.
    clock["now"] = 1002.0
    assert co._should_emit_in_progress(mid, 0.20) is True

    # A sub-step advance (<5%) is suppressed even after the rate floor.
    clock["now"] = 1004.0
    assert co._should_emit_in_progress(mid, 0.23) is False

    # No meaningful advance, but the heartbeat interval elapsed -> emit.
    clock["now"] = 1002.0 + co._HEARTBEAT_SECS + 1
    assert co._should_emit_in_progress(mid, 0.24) is True


def test_in_progress_throttle_bounds_event_count_for_a_long_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the raw downloader's per-8MB-chunk firing for a large model:
    # thousands of callbacks with tiny fraction deltas and sub-second spacing.
    # The gate must admit O(1/_PROGRESS_STEP) of them, not O(callbacks).
    co = _make_coordinator()
    mid = ModelId("org/big")
    clock = {"now": 0.0}

    def fake_now() -> float:
        return clock["now"]

    monkeypatch.setattr(coordinator_mod, "current_time", fake_now)

    emitted = 0
    chunks = 4000  # ~32 GB at 8MB chunks
    for i in range(1, chunks + 1):
        clock["now"] = i * 0.1  # 100ms per chunk -> 400s total, well past heartbeats
        if co._should_emit_in_progress(mid, i / chunks):
            emitted += 1

    # Without the fix this is ~ (400s / 1s) = 400 events; with the fraction gate
    # plus the heartbeat it is sharply bounded. Assert it is a small fraction of
    # the callbacks and near the step/heartbeat budget.
    assert emitted <= 30, f"expected a small bounded count, got {emitted}"
    assert emitted >= 1
