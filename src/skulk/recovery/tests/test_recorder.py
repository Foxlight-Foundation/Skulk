"""The node's recorder writes only safe, current views of the cluster."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from skulk.recovery.record import read_record
from skulk.recovery.recorder import (
    RECORD_REFRESH_AFTER,
    ClusterIntentRecorder,
    record_is_writable,
)
from skulk.recovery.tests.fixtures import sample_state
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.profiling import NodeIdentity
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView

SESSION = SessionId(master_node_id=NodeId("node-a"), election_clock=1)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def _telemetry() -> TelemetryView:
    view = TelemetryView()
    view.node_identities[NodeId("node-a")] = NodeIdentity(
        friendly_name="kite1",
        node_install_id=UUID("11111111-1111-4111-8111-111111111111"),
    )
    return view


def _recorder(
    path: Path, state: list[State | None], clock: _Clock
) -> ClusterIntentRecorder:
    return ClusterIntentRecorder(
        state_provider=lambda: state[0],
        session_provider=lambda: SESSION,
        telemetry_view=_telemetry(),
        path=path,
        clock=clock,
    )


def test_unsynced_and_recovering_views_are_never_written() -> None:
    assert not record_is_writable(None)
    assert not record_is_writable(State())
    assert not record_is_writable(
        State(last_event_applied_idx=5, restart_recovery_pending=True)
    )
    assert record_is_writable(State(last_event_applied_idx=0))


async def test_recovering_cluster_leaves_the_previous_record_in_place(tmp_path: Path) -> None:
    """A cold start's nearly empty view must not replace what restore needs."""
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    current: list[State | None] = [sample_state()]
    assert await _recorder(path, current, clock).record_once()
    before = read_record(path)

    current[0] = State(last_event_applied_idx=0, restart_recovery_pending=True)
    clock.now += RECORD_REFRESH_AFTER * 2
    assert not await _recorder(path, current, clock).record_once()

    assert read_record(path) == before


async def test_unchanged_content_is_restamped_only_after_the_refresh_interval(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    current: list[State | None] = [sample_state()]
    recorder = _recorder(path, current, clock)

    assert await recorder.record_once()
    clock.now += timedelta(seconds=30)
    assert not await recorder.record_once()
    clock.now += RECORD_REFRESH_AFTER
    assert await recorder.record_once()
    record = read_record(path)
    assert record is not None and record.stamp.written_at == clock.now


async def test_changed_content_is_written_immediately(tmp_path: Path) -> None:
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    current: list[State | None] = [sample_state()]
    recorder = _recorder(path, current, clock)
    assert await recorder.record_once()

    state = current[0]
    assert state is not None
    current[0] = state.model_copy(update={"instances": {}})
    clock.now += timedelta(seconds=5)

    assert await recorder.record_once()
    record = read_record(path)
    assert record is not None and record.placements == ()


async def test_a_new_recorder_resumes_from_the_record_on_disk(tmp_path: Path) -> None:
    """After a restart the recorder does not rewrite identical content at once."""
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    current: list[State | None] = [sample_state()]
    assert await _recorder(path, current, clock).record_once()

    clock.now += timedelta(seconds=10)
    assert not await _recorder(path, current, clock).record_once()


async def test_unwritable_disk_is_reported_not_raised(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    clock = _Clock()
    current: list[State | None] = [sample_state()]

    assert not await _recorder(blocked / "cluster-intent.json", current, clock).record_once()


async def test_an_offline_excluded_machine_stays_excluded(tmp_path: Path) -> None:
    """A machine missing from telemetry keeps its install ID while referenced."""
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    telemetry = _telemetry()
    telemetry.node_identities[NodeId("node-b")] = NodeIdentity(
        friendly_name="kite2",
        node_install_id=UUID("22222222-2222-4222-8222-222222222222"),
    )
    current: list[State | None] = [sample_state()]
    recorder = ClusterIntentRecorder(
        state_provider=lambda: current[0],
        session_provider=lambda: SESSION,
        telemetry_view=telemetry,
        path=path,
        clock=clock,
    )
    assert await recorder.record_once()

    # node-a, which the second placement excludes, drops out of telemetry.
    del telemetry.node_identities[NodeId("node-a")]
    clock.now += RECORD_REFRESH_AFTER * 2
    assert await recorder.record_once()

    record = read_record(path)
    assert record is not None
    assert record.placements[1].excluded_install_ids == (
        UUID("11111111-1111-4111-8111-111111111111"),
    )
