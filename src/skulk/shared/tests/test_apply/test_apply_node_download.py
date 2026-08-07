from skulk.shared.apply import apply, apply_node_download_progress
from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    IndexedEvent,
    NodeDownloadProgress,
    StateSnapshotHydrated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgressData,
)
from skulk.worker.tests.constants import MODEL_A_ID, MODEL_B_ID


def test_apply_node_download_progress():
    state = State()
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    event = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event]}


def test_apply_two_node_download_progress():
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0, world_size=2)
    event1 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )
    event2 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard2,
        total=Memory(),
    )
    state = State(downloads={NodeId("node-1"): [event1]})

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event2), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event1, event2]}


def test_apply_ongoing_download_is_replay_compatible_noop() -> None:
    """Legacy progress events decode but never repopulate durable State."""

    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    ongoing = DownloadOngoing(
        node_id=NodeId("node-1"),
        shard_metadata=shard,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(10),
            downloaded=Memory.from_mb(1),
            downloaded_this_session=Memory.from_mb(1),
            completed_files=0,
            total_files=1,
            speed=1.0,
            eta_ms=1,
            files={},
        ),
    )

    state = State()
    assert (
        apply_node_download_progress(
            NodeDownloadProgress(download_progress=ongoing), state
        )
        == state
    )


def test_apply_pending_download_clears_prior_terminal_outcome() -> None:
    """A durable attempt reset removes history without storing live progress."""

    node = NodeId("node-1")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    completed = DownloadCompleted(
        node_id=node,
        shard_metadata=shard,
        total=Memory.from_mb(10),
    )
    pending = DownloadPending(node_id=node, shard_metadata=shard)

    updated = apply_node_download_progress(
        NodeDownloadProgress(download_progress=pending),
        State(downloads={node: [completed]}),
    )

    assert updated.downloads == {}


def test_snapshot_hydration_keeps_only_durable_download_outcomes() -> None:
    """Legacy transient snapshot entries cannot repopulate durable State."""

    node = NodeId("node-1")
    shard_a = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    shard_b = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0, world_size=1)
    completed = DownloadCompleted(
        node_id=node,
        shard_metadata=shard_a,
        total=Memory.from_mb(10),
    )
    failed = DownloadFailed(
        node_id=node,
        shard_metadata=shard_b,
        error_message="terminal failure",
    )
    pending = DownloadPending(node_id=node, shard_metadata=shard_a)
    ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard_b,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(10),
            downloaded=Memory.from_mb(1),
            downloaded_this_session=Memory.from_mb(1),
            completed_files=0,
            total_files=1,
            speed=1.0,
            eta_ms=1,
            files={},
        ),
    )
    snapshot = State(
        downloads={node: [completed, pending, ongoing, failed]},
        last_event_applied_idx=0,
    )

    hydrated = apply(
        State(),
        IndexedEvent(idx=0, event=StateSnapshotHydrated(state=snapshot)),
    )

    assert hydrated.downloads == {node: [completed, failed]}
