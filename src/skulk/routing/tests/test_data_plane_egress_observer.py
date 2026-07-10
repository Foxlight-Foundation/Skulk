"""Bounded router DATA-plane observer coverage."""

from skulk.routing.data_plane import (
    _MAX_TRACKED_OWNERS,  # pyright: ignore[reportPrivateUsage]
    _OVERFLOW_OWNER_KEY,  # pyright: ignore[reportPrivateUsage]
    DataPlaneEgressObserver,
)


def test_owner_diagnostics_remain_bounded_during_owner_churn() -> None:
    observer = DataPlaneEgressObserver()

    for owner_index in range(_MAX_TRACKED_OWNERS + 10):
        observer.record_stream_opened(f"owner-{owner_index}")

    snapshot = observer.snapshot()
    assert len(snapshot.owners) == _MAX_TRACKED_OWNERS + 1
    assert snapshot.owners[_OVERFLOW_OWNER_KEY].active_streams == 10
    assert sum(owner.active_streams for owner in snapshot.owners.values()) == (
        _MAX_TRACKED_OWNERS + 10
    )

    for owner_index in range(_MAX_TRACKED_OWNERS + 10):
        observer.record_stream_closed(f"owner-{owner_index}")

    assert observer.snapshot().active_stream_queues == 0
