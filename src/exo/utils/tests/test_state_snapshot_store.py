from pathlib import Path

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state import State
from exo.shared.types.state_sync import StateSnapshot
from exo.utils.state_snapshot_store import StateSnapshotStore


def test_latest_for_session_round_trip(tmp_path: Path) -> None:
    session_id = SessionId(master_node_id=NodeId("node-a"), election_clock=2)
    store = StateSnapshotStore(tmp_path)
    snapshot = StateSnapshot(
        session_id=session_id,
        last_event_applied_idx=7,
        state=State(last_event_applied_idx=7),
    )

    store.write(snapshot)

    restored = store.latest_for_session(session_id)
    assert restored is not None
    assert restored.session_id == snapshot.session_id
    assert restored.last_event_applied_idx == snapshot.last_event_applied_idx
    assert restored.state.last_event_applied_idx == snapshot.state.last_event_applied_idx


def test_retention_keeps_recent_snapshots(tmp_path: Path) -> None:
    session_id = SessionId(master_node_id=NodeId("node-a"), election_clock=4)
    store = StateSnapshotStore(tmp_path, max_snapshots=2)

    for idx in range(3):
        store.write(
            StateSnapshot(
                session_id=session_id,
                last_event_applied_idx=idx,
                state=State(last_event_applied_idx=idx),
            )
        )

    snapshots = sorted(tmp_path.glob("snapshot.*.json.zst"))
    assert len(snapshots) == 2
    assert snapshots[0].name.endswith("1.json.zst")
    assert snapshots[1].name.endswith("2.json.zst")
