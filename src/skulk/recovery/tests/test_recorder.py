"""The node's recorder writes only safe, current views of the cluster."""

from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path
from uuid import UUID

import anyio

from skulk.recovery.record import DEPARTURE_GRACE, read_record
from skulk.recovery.recorder import (
    RECORD_REFRESH_AFTER,
    ClusterIntentRecorder,
    own_addresses,
    record_is_writable,
)
from skulk.recovery.tests.fixtures import connected, sample_instance, sample_state
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.profiling import NodeIdentity
from skulk.shared.types.restart_recovery import (
    ADVERT_ADDRESS_LIMIT,
    RestartRecoveryAdvert,
)
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


async def test_the_advert_describes_the_record_and_goes_out_at_once(tmp_path: Path) -> None:
    """A cold-started master needs every node's advert before it writes anything."""
    path = tmp_path / "cluster-intent.json"
    sent: list[RestartRecoveryAdvert] = []

    async def send(advert: RestartRecoveryAdvert) -> None:
        sent.append(advert)

    clock = _Clock()
    current: list[State | None] = [State(last_event_applied_idx=0, restart_recovery_pending=True)]
    recorder = ClusterIntentRecorder(
        state_provider=lambda: current[0],
        session_provider=lambda: SESSION,
        telemetry_view=_telemetry(),
        path=path,
        clock=clock,
        advert_sender=send,
        api_port=52915,
        address_provider=lambda: ("192.168.1.20",),
    )
    assert recorder.advert() == RestartRecoveryAdvert(
        written_at=None, placements=0, api_port=52915, addresses=("192.168.1.20",)
    )

    with anyio.move_on_after(0.2):
        await recorder.run()
    # Recovery is pending, so nothing was written, but the advert still went out.
    assert sent == [
        RestartRecoveryAdvert(
            written_at=None, placements=0, api_port=52915, addresses=("192.168.1.20",)
        )
    ]

    current[0] = sample_state()
    assert await recorder.record_once()
    assert recorder.advert() == RestartRecoveryAdvert(
        written_at=clock.now, placements=2, api_port=52915, addresses=("192.168.1.20",)
    )


def _two_machine_telemetry() -> TelemetryView:
    telemetry = _telemetry()
    telemetry.node_identities[NodeId("node-b")] = NodeIdentity(
        friendly_name="kite2",
        node_install_id=UUID("22222222-2222-4222-8222-222222222222"),
    )
    return telemetry


def _one_model_per_machine() -> State:
    on_a = sample_instance("on-a", "org/on-a", ("node-a",))
    on_b = sample_instance("on-b", "org/on-b", ("node-b",))
    return State(
        instances={on_a.instance_id: on_a, on_b.instance_id: on_b},
        topology=connected("node-a", "node-b"),
        last_event_applied_idx=42,
    )


def _recorded_models(path: Path) -> list[str]:
    record = read_record(path)
    assert record is not None
    return [placement.model_id for placement in record.placements]


def _two_machine_recorder(
    path: Path, current: list[State | None], clock: _Clock, telemetry: TelemetryView
) -> ClusterIntentRecorder:
    return ClusterIntentRecorder(
        state_provider=lambda: current[0],
        session_provider=lambda: SESSION,
        telemetry_view=telemetry,
        path=path,
        clock=clock,
    )


def _after_node_b_left(state: State) -> State:
    """The master's view once node-b timed out: its node and placement gone."""
    return state.model_copy(
        update={
            "instances": {
                instance_id: instance
                for instance_id, instance in state.instances.items()
                if NodeId("node-b") not in instance.shard_assignments.node_to_runner
            },
            "topology": connected("node-a"),
        }
    )


async def test_a_departed_machines_placement_outlives_it_for_the_grace(
    tmp_path: Path,
) -> None:
    """Machines in a shutdown go down minutes apart; the last one standing
    must still record the models of the ones already gone."""
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    current: list[State | None] = [_one_model_per_machine()]
    recorder = _two_machine_recorder(path, current, clock, _two_machine_telemetry())
    assert await recorder.record_once()

    state = current[0]
    assert state is not None
    current[0] = _after_node_b_left(state)
    clock.now += timedelta(seconds=40)
    # The content is unchanged, so nothing new is written.
    assert not await recorder.record_once()
    assert _recorded_models(path) == ["org/on-a", "org/on-b"]

    clock.now += RECORD_REFRESH_AFTER
    assert await recorder.record_once()
    assert _recorded_models(path) == ["org/on-a", "org/on-b"]

    clock.now += DEPARTURE_GRACE
    assert await recorder.record_once()
    assert _recorded_models(path) == ["org/on-a"]


async def test_a_stopped_placement_leaves_the_record_at_once(tmp_path: Path) -> None:
    """Its machine is still connected, so the operator (or the steward) stopped it."""
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    current: list[State | None] = [_one_model_per_machine()]
    recorder = _two_machine_recorder(path, current, clock, _two_machine_telemetry())
    assert await recorder.record_once()

    state = current[0]
    assert state is not None
    current[0] = state.model_copy(
        update={
            "instances": {
                instance_id: instance
                for instance_id, instance in state.instances.items()
                if instance_id != "on-b"
            }
        }
    )
    clock.now += timedelta(seconds=5)
    assert await recorder.record_once()
    assert _recorded_models(path) == ["org/on-a"]


async def test_a_departed_placement_repaired_elsewhere_is_recorded_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    current: list[State | None] = [_one_model_per_machine()]
    recorder = _two_machine_recorder(path, current, clock, _two_machine_telemetry())
    assert await recorder.record_once()

    state = current[0]
    assert state is not None
    departed = _after_node_b_left(state)
    current[0] = departed
    clock.now += timedelta(seconds=40)
    assert not await recorder.record_once()

    moved = sample_instance("moved", "org/on-b", ("node-a",))
    current[0] = departed.model_copy(
        update={"instances": {**departed.instances, moved.instance_id: moved}}
    )
    clock.now += timedelta(seconds=5)
    assert await recorder.record_once()
    record = read_record(path)
    assert record is not None
    assert [
        (placement.model_id, placement.host_install_ids) for placement in record.placements
    ] == [
        ("org/on-a", (UUID("11111111-1111-4111-8111-111111111111"),)),
        ("org/on-b", (UUID("11111111-1111-4111-8111-111111111111"),)),
    ]


async def test_a_returning_machine_without_its_model_ends_the_grace(
    tmp_path: Path,
) -> None:
    """Back in the cluster and not running the model: the record follows suit."""
    path = tmp_path / "cluster-intent.json"
    clock = _Clock()
    telemetry = _two_machine_telemetry()
    current: list[State | None] = [_one_model_per_machine()]
    recorder = _two_machine_recorder(path, current, clock, telemetry)
    assert await recorder.record_once()

    state = current[0]
    assert state is not None
    departed = _after_node_b_left(state)
    current[0] = departed
    clock.now += timedelta(seconds=40)
    assert not await recorder.record_once()

    # node-b restarts under a new runtime node ID with the same install ID.
    telemetry.node_identities[NodeId("node-b-again")] = NodeIdentity(
        friendly_name="kite2",
        node_install_id=UUID("22222222-2222-4222-8222-222222222222"),
    )
    current[0] = departed.model_copy(update={"topology": connected("node-a", "node-b-again")})
    clock.now += timedelta(seconds=30)
    assert await recorder.record_once()
    assert _recorded_models(path) == ["org/on-a"]


def test_a_node_without_an_api_advertises_no_addresses(tmp_path: Path) -> None:
    """Nothing could fetch its record, so it names nowhere to fetch it from."""
    current: list[State | None] = [sample_state()]
    recorder = ClusterIntentRecorder(
        state_provider=lambda: current[0],
        session_provider=lambda: SESSION,
        telemetry_view=_telemetry(),
        path=tmp_path / "cluster-intent.json",
        api_port=None,
        address_provider=lambda: ("192.168.1.20",),
    )
    assert recorder.advert().addresses == ()


def test_own_addresses_leave_out_what_no_peer_can_use() -> None:
    addresses = own_addresses()
    assert len(addresses) <= ADVERT_ADDRESS_LIMIT
    assert all(not ip_address(address).is_loopback for address in addresses)
    assert all(
        not (ip_address(address).version == 6 and ip_address(address).is_link_local)
        for address in addresses
    )
