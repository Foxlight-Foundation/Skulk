"""A cold-started master carries steward history back from the node's record."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from uuid import UUID

import anyio
import pytest

from skulk.master.main import Master
from skulk.master.placement import PlacementError, PlacementInfoPendingError
from skulk.recovery.record import (
    RESTARTED_BEFORE_DECISION,
    ClusterIntentRecord,
    PlacementIntent,
    RecoveryStamp,
)
from skulk.recovery.restore import (
    HOST_WAIT_AFTER_SETTLE_SECONDS,
    INFO_WAIT_AFTER_SETTLE_SECONDS,
)
from skulk.recovery.tests.fixtures import KITE1, KITE2, sample_instance, sample_proposal
from skulk.shared.apply import apply
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    InstanceFailureRecorded,
    LocalForwarderEvent,
    RestartRecoveryCompleted,
    StateSnapshotHydrated,
    StewardActionProposalChanged,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeIdentity, NodeNetworkInfo
from skulk.shared.types.restart_recovery import RestartRecoveryAdvert
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.steward_actions import StewardActionProposal
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.shards import Sharding
from skulk.utils.channels import Receiver, channel

LOCAL_NODE = NodeId("local-node")


def _master(
    initial_state: State | None,
    *,
    record: ClusterIntentRecord | None,
    enabled: bool = True,
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    global_send, global_recv = channel[GlobalForwarderEvent]()
    master = Master(
        LOCAL_NODE,
        SessionId(master_node_id=NodeId(), election_clock=1),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_send,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=initial_state,
        restart_recovery_record=record,
        restart_recovery_enabled=enabled,
    )
    return master, global_recv


def _advance(
    master: Master,
    *,
    settled: bool = True,
    nodes: set[NodeId] | None = None,
    now: datetime | None = None,
) -> list[Event]:
    """Run one restart recovery step of the master's planning pass."""
    return master._restart_recovery_events(  # pyright: ignore[reportPrivateUsage]
        now or datetime.now(tz=timezone.utc),
        topology_settled=settled,
        connected_node_ids=nodes if nodes is not None else {LOCAL_NODE},
    )


def _record() -> ClusterIntentRecord:
    return ClusterIntentRecord(
        stamp=RecoveryStamp(
            session_master_node_id=NodeId("before-restart"),
            session_election_clock=4,
            last_event_applied_idx=120,
            written_at=datetime.now(tz=timezone.utc) - timedelta(minutes=1),
        ),
        steward_proposals=(sample_proposal("rejected", 30), sample_proposal("pending", 2)),
    )


async def _index_seed(master: Master, global_recv: Receiver[GlobalForwarderEvent]) -> list[GlobalForwarderEvent]:
    received: list[GlobalForwarderEvent] = []

    async def consume() -> None:
        with global_recv as events:
            async for event in events:
                received.append(event)
                return

    with anyio.fail_after(10):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(consume)
            await master._index_seed_event()  # pyright: ignore[reportPrivateUsage]
    return received


async def test_cold_start_restores_terminal_proposal_history_as_event_zero() -> None:
    record = _record()
    master, global_recv = _master(None, record=record)

    received = await _index_seed(master, global_recv)

    assert len(received) == 1 and received[0].origin_idx == 0
    seed = received[0].event
    assert isinstance(seed, StateSnapshotHydrated)
    assert seed.state.restart_recovery_pending is True
    restored = seed.state.steward_action_proposals
    assert {proposal.status for proposal in restored.values()} == {"rejected", "expired"}
    pending_before = record.steward_proposals[1]
    assert restored[pending_before.proposal_id].outcome == RESTARTED_BEFORE_DECISION
    # The master's ordered view matches what it replicated.
    assert dict(master._ordered_steward_proposals) == dict(restored)  # pyright: ignore[reportPrivateUsage]


async def test_cold_start_without_a_record_still_holds_records_until_complete() -> None:
    """Other nodes' records stay protected even when this node has none."""
    master, global_recv = _master(None, record=None)

    received = await _index_seed(master, global_recv)

    seed = received[0].event
    assert isinstance(seed, StateSnapshotHydrated)
    assert seed.state.restart_recovery_pending is True
    assert seed.state.steward_action_proposals == {}


async def test_disabled_recovery_keeps_the_historical_empty_cold_start() -> None:
    master, _ = _master(None, record=_record(), enabled=False)

    await master._index_seed_event()  # pyright: ignore[reportPrivateUsage]

    assert master.state.last_event_applied_idx == -1
    assert master.state.restart_recovery_pending is False
    assert master.state.steward_action_proposals == {}


def test_failover_keeps_live_history_and_ignores_the_disk_record() -> None:
    """A surviving view is fresher than any record; failover is unchanged."""
    live = sample_proposal("dispatched", 1)
    prior = State(
        steward_action_proposals={live.proposal_id: live}, last_event_applied_idx=9
    )
    master, _ = _master(seed_state_for_new_session(prior), record=_record())

    assert dict(master._ordered_steward_proposals) == {live.proposal_id: live}  # pyright: ignore[reportPrivateUsage]
    assert master._restart_recovery_record is None  # pyright: ignore[reportPrivateUsage]
    assert _advance(master) == []


async def test_recovery_completes_once_and_clears_the_flag() -> None:
    record = _record()
    master, global_recv = _master(None, record=record)
    await _index_seed(master, global_recv)

    completion = _advance(master)

    assert completion == [
        RestartRecoveryCompleted(
            event_id=completion[0].event_id,
            restored_placements=0,
            restored_proposals=2,
        )
    ]
    assert _advance(master) == []
    cleared = apply(master.state, IndexedEvent(idx=1, event=completion[0]))
    assert cleared.restart_recovery_pending is False


def test_master_promoted_mid_recovery_keeps_it_pending() -> None:
    """Failover carries an unfinished restore, and keeps every record untouched.

    Inheriting a restore means a second node existed, which is the multi-node
    case: even if only this node remains, it does not restore from its own
    record alone.
    """
    prior = State(restart_recovery_pending=True, last_event_applied_idx=3)
    master, _ = _master(seed_state_for_new_session(prior), record=_record())

    assert master._restart_recovery_record is not None  # pyright: ignore[reportPrivateUsage]
    master.state = master.state.model_copy(update={"restart_recovery_pending": True})
    assert _advance(master) == []
    assert master.state.restart_recovery_pending is True


def _card(model: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model),
        storage_size=Memory.from_gb(4),
        n_layers=4,
        hidden_size=8,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _intent(model: str) -> PlacementIntent:
    return PlacementIntent(
        model_id=ModelId(model),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        requested_context_tokens=8192,
        host_install_ids=(KITE1,),
    )


class _Planner:
    """Stand-in for the placement path: records commands, places one instance each."""

    def __init__(self, *, failing: set[str] | None = None, pending: bool = False) -> None:
        self.commands: list[PlaceInstance] = []
        self._failing = failing or set()
        self._pending = pending

    def __call__(
        self, command: PlaceInstance, current: Mapping[InstanceId, Instance]
    ) -> dict[InstanceId, Instance]:
        self.commands.append(command)
        model = str(command.model_card.model_id)
        if self._pending:
            raise PlacementInfoPendingError("node memory has not been reported yet")
        if model in self._failing:
            raise PlacementError("not enough memory on any node")
        placed = sample_instance(f"restored-{len(self.commands)}", model, (str(LOCAL_NODE),))
        return {**current, placed.instance_id: placed}


async def _cold_master_with(
    monkeypatch: pytest.MonkeyPatch, *models: str, planner: _Planner, catalog: set[str] | None = None
) -> Master:
    record = _record().model_copy(update={"placements": tuple(_intent(m) for m in models)})
    master, global_recv = _master(None, record=record)
    await _index_seed(master, global_recv)
    known = set(models) if catalog is None else catalog

    def ordered_card(model_id: ModelId) -> ModelCard | None:
        return _card(str(model_id)) if str(model_id) in known else None

    monkeypatch.setattr(master, "_ordered_placement_model_card", ordered_card)
    monkeypatch.setattr(master, "_place_for_steward_action", planner)
    return master


async def test_single_node_cold_start_brings_placements_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _Planner()
    master = await _cold_master_with(monkeypatch, "org/first", "org/second", planner=planner)

    # Nothing happens before the settling window: node information and the
    # returning machines must be visible first.
    assert _advance(master, settled=False) == []

    events = _advance(master)

    created = [event for event in events if isinstance(event, InstanceCreated)]
    assert [str(event.instance.shard_assignments.model_id) for event in created] == [
        "org/first",
        "org/second",
    ]
    assert isinstance(events[-1], RestartRecoveryCompleted)
    assert events[-1].restored_placements == 2
    assert [command.requested_context_tokens for command in planner.commands] == [8192, 8192]
    # The second placement planned against the first, before either echoed.
    assert _advance(master) == []


async def test_multi_node_cold_start_keeps_every_record_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _Planner()
    master = await _cold_master_with(monkeypatch, "org/first", planner=planner)

    assert _advance(master, nodes={LOCAL_NODE, NodeId("other-node")}) == []
    assert planner.commands == []
    assert master.state.restart_recovery_pending is True


async def test_placements_that_cannot_come_back_are_reported_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _Planner(failing={"org/huge"})
    master = await _cold_master_with(
        monkeypatch,
        "org/gone",
        "org/huge",
        "org/fits",
        planner=planner,
        catalog={"org/huge", "org/fits"},
    )

    events = _advance(master)

    failures = [event for event in events if isinstance(event, InstanceFailureRecorded)]
    assert [failure.failure.error_message for failure in failures] == [
        "Could not bring back org/gone after the restart: it is no longer in the model catalog.",
        "Could not bring back org/huge after the restart: not enough memory on any node",
    ]
    assert {failure.failure.error_code for failure in failures} == {"placement_failed"}
    assert [type(event).__name__ for event in events] == [
        "InstanceFailureRecorded",
        "InstanceFailureRecorded",
        "InstanceCreated",
        "RestartRecoveryCompleted",
    ]
    completion = events[-1]
    assert isinstance(completion, RestartRecoveryCompleted)
    assert completion.restored_placements == 1


async def test_missing_node_information_is_retried_then_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = await _cold_master_with(monkeypatch, "org/first", planner=_Planner(pending=True))
    start = datetime.now(tz=timezone.utc)

    assert _advance(master, now=start) == []
    assert _advance(master, now=start + timedelta(seconds=60)) == []

    events = _advance(master, now=start + timedelta(seconds=INFO_WAIT_AFTER_SETTLE_SECONDS + 1))
    assert isinstance(events[0], InstanceFailureRecorded)
    assert "node information did not arrive in time" in events[0].failure.error_message
    assert isinstance(events[-1], RestartRecoveryCompleted)


def test_old_machines_are_preferred_then_the_planner_chooses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, _ = _master(None, record=None)
    old_machine, other = NodeId("old-machine"), NodeId("other-machine")
    attempts: list[list[NodeId]] = []

    def planner(
        command: PlaceInstance, current: Mapping[InstanceId, Instance]
    ) -> dict[InstanceId, Instance]:
        attempts.append(list(command.excluded_nodes))
        if len(attempts) == 1:
            raise PlacementError("the old machine is full")
        return dict(current)

    monkeypatch.setattr(master, "_place_for_steward_action", planner)
    command = PlaceInstance(
        model_card=_card("org/model"),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )

    master._place_restored_intent(  # pyright: ignore[reportPrivateUsage]
        command, {}, frozenset({old_machine}), {old_machine, other}
    )

    assert attempts == [[other], []]


async def test_a_winner_without_placements_still_waits_for_the_whole_cluster() -> None:
    """An empty local record must not release peers whose records hold placements."""
    master, global_recv = _master(None, record=None)
    await _index_seed(master, global_recv)

    assert _advance(master, settled=False) == []
    assert _advance(master, nodes={LOCAL_NODE, NodeId("peer-with-a-record")}) == []
    assert master.state.restart_recovery_pending is True
    # The peer dropping out later does not turn this into a one-node restore.
    assert _advance(master) == []
    assert master.state.restart_recovery_pending is True


async def test_a_lone_winner_without_placements_completes_after_settling() -> None:
    master, global_recv = _master(None, record=None)
    await _index_seed(master, global_recv)

    assert _advance(master, settled=False) == []
    completion = _advance(master)
    assert len(completion) == 1 and isinstance(completion[0], RestartRecoveryCompleted)


async def test_a_peer_seen_only_while_settling_still_counts() -> None:
    """A peer that leaves before the window ends still makes it multi-node."""
    master, global_recv = _master(None, record=_record())
    await _index_seed(master, global_recv)

    assert _advance(master, settled=False, nodes={LOCAL_NODE, NodeId("brief-peer")}) == []
    assert _advance(master) == []
    assert master.state.restart_recovery_pending is True


PEER = NodeId("peer-node")


def _intent_on(model: str, *hosts: UUID) -> PlacementIntent:
    return PlacementIntent(
        model_id=ModelId(model),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        host_install_ids=hosts,
    )


def _record_written(minutes_ago: int, *placements: PlacementIntent, proposals: tuple[StewardActionProposal, ...] = ()) -> ClusterIntentRecord:
    return ClusterIntentRecord(
        stamp=RecoveryStamp(
            session_master_node_id=NodeId("before-restart"),
            session_election_clock=2,
            last_event_applied_idx=50,
            written_at=datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago),
        ),
        placements=placements,
        steward_proposals=proposals,
    )


class _Fetcher:
    """Stand-in for fetching a peer's record over its API."""

    def __init__(self, records: dict[NodeId, ClusterIntentRecord | None]) -> None:
        self.records = records
        self.calls: list[NodeId] = []

    async def __call__(
        self,
        node_id: NodeId,
        advert: RestartRecoveryAdvert,
        network: NodeNetworkInfo | None,
    ) -> ClusterIntentRecord | None:
        self.calls.append(node_id)
        return self.records.get(node_id)


async def _two_node_master(
    monkeypatch: pytest.MonkeyPatch,
    *,
    own: ClusterIntentRecord | None,
    fetcher: _Fetcher,
    planner: _Planner | None = None,
) -> Master:
    global_send, global_recv = channel[GlobalForwarderEvent]()
    master = Master(
        LOCAL_NODE,
        SessionId(master_node_id=LOCAL_NODE, election_clock=1),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_send,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        restart_recovery_record=own,
        restart_recovery_enabled=True,
        restart_recovery_fetcher=fetcher,
    )
    await _index_seed(master, global_recv)
    view = master._telemetry_view  # pyright: ignore[reportPrivateUsage]
    view.node_identities[LOCAL_NODE] = NodeIdentity(node_install_id=KITE1)
    view.node_identities[PEER] = NodeIdentity(node_install_id=KITE2)
    monkeypatch.setattr(master, "_ordered_placement_model_card", lambda model_id: _card(str(model_id)))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    monkeypatch.setattr(master, "_place_for_steward_action", planner or _Planner())
    return master


async def _choose(master: Master, now: datetime | None = None) -> list[Event]:
    return await master._choose_restart_recovery_record(  # pyright: ignore[reportPrivateUsage]
        now or datetime.now(tz=timezone.utc),
        topology_settled=True,
        connected_node_ids={LOCAL_NODE, PEER},
    )


def _advert_for(record: ClusterIntentRecord) -> RestartRecoveryAdvert:
    return RestartRecoveryAdvert(
        written_at=record.stamp.written_at, placements=len(record.placements), api_port=52415
    )


async def test_a_fresher_peer_record_is_fetched_and_restored_across_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = _record_written(60, _intent_on("org/stale", KITE1))
    pending_then = sample_proposal("pending", 2)
    peer_record = _record_written(
        1, _intent_on("org/fresh-a", KITE1), _intent_on("org/fresh-b", KITE2), proposals=(pending_then,)
    )
    fetcher = _Fetcher({PEER: peer_record})
    planner = _Planner()
    master = await _two_node_master(monkeypatch, own=own, fetcher=fetcher, planner=planner)

    # The peer has not advertised yet: wait rather than pick from what is known.
    assert await _choose(master) == []
    assert fetcher.calls == []

    master._telemetry_view.node_restart_recovery[PEER] = _advert_for(peer_record)  # pyright: ignore[reportPrivateUsage]
    carried = await _choose(master)

    assert fetcher.calls == [PEER]
    assert [type(event).__name__ for event in carried] == ["StewardActionProposalChanged"]
    restored_proposal = carried[0]
    assert isinstance(restored_proposal, StewardActionProposalChanged)
    assert restored_proposal.proposal.status == "expired"
    events = _advance(master, nodes={LOCAL_NODE, PEER})
    created = [event for event in events if isinstance(event, InstanceCreated)]
    assert [str(event.instance.shard_assignments.model_id) for event in created] == ["org/fresh-a", "org/fresh-b"]
    assert isinstance(events[-1], RestartRecoveryCompleted)


async def test_the_masters_own_fresher_record_needs_no_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = _record_written(1, _intent_on("org/own", KITE1))
    peer_record = _record_written(30, _intent_on("org/older", KITE2))
    fetcher = _Fetcher({PEER: peer_record})
    master = await _two_node_master(monkeypatch, own=own, fetcher=fetcher)
    master._telemetry_view.node_restart_recovery[PEER] = _advert_for(peer_record)  # pyright: ignore[reportPrivateUsage]

    assert await _choose(master) == []
    assert fetcher.calls == []
    created = [e for e in _advance(master, nodes={LOCAL_NODE, PEER}) if isinstance(e, InstanceCreated)]
    assert [str(e.instance.shard_assignments.model_id) for e in created] == ["org/own"]


async def test_an_unfetchable_peer_falls_back_to_the_next_freshest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = _record_written(60, _intent_on("org/own", KITE1))
    peer_record = _record_written(1, _intent_on("org/peer", KITE2))
    fetcher = _Fetcher({PEER: None})
    master = await _two_node_master(monkeypatch, own=own, fetcher=fetcher)
    master._telemetry_view.node_restart_recovery[PEER] = _advert_for(peer_record)  # pyright: ignore[reportPrivateUsage]

    await _choose(master)

    assert fetcher.calls == [PEER]
    created = [e for e in _advance(master, nodes={LOCAL_NODE, PEER}) if isinstance(e, InstanceCreated)]
    assert [str(e.instance.shard_assignments.model_id) for e in created] == ["org/own"]


async def test_restore_waits_for_the_recorded_machines_then_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = UUID("99999999-9999-4999-8999-999999999999")
    own = _record_written(1, _intent_on("org/needs-a-third-machine", missing))
    master = await _two_node_master(monkeypatch, own=own, fetcher=_Fetcher({}))
    master._telemetry_view.node_restart_recovery[PEER] = RestartRecoveryAdvert()  # pyright: ignore[reportPrivateUsage]
    start = datetime.now(tz=timezone.utc)

    assert await _choose(master, start) == []
    assert _advance(master, nodes={LOCAL_NODE, PEER}) == []

    later = start + timedelta(seconds=HOST_WAIT_AFTER_SETTLE_SECONDS + 1)
    await _choose(master, later)
    events = _advance(master, nodes={LOCAL_NODE, PEER}, now=later)
    assert [type(event).__name__ for event in events] == ["InstanceCreated", "RestartRecoveryCompleted"]


async def test_a_promoted_master_does_not_place_what_was_already_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carried = sample_instance("already-restored", "org/first", (str(LOCAL_NODE),))
    prior = State(
        instances={carried.instance_id: carried},
        restart_recovery_pending=True,
        last_event_applied_idx=7,
    )
    own = _record_written(1, _intent_on("org/first", KITE1), _intent_on("org/second", KITE1))
    planner = _Planner()
    global_send, _ = channel[GlobalForwarderEvent]()
    master = Master(
        LOCAL_NODE,
        SessionId(master_node_id=LOCAL_NODE, election_clock=3),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_send,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=seed_state_for_new_session(prior),
        restart_recovery_record=own,
        restart_recovery_enabled=True,
        restart_recovery_fetcher=_Fetcher({}),
    )
    master.state = seed_state_for_new_session(prior).model_copy(update={"last_event_applied_idx": 0})
    # The recorded machine is this node, so restore need not wait for it.
    master._telemetry_view.node_identities[LOCAL_NODE] = NodeIdentity(node_install_id=KITE1)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(master, "_ordered_placement_model_card", lambda model_id: _card(str(model_id)))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    monkeypatch.setattr(master, "_place_for_steward_action", planner)

    await master._choose_restart_recovery_record(  # pyright: ignore[reportPrivateUsage]
        datetime.now(tz=timezone.utc), topology_settled=True, connected_node_ids={LOCAL_NODE}
    )
    events = _advance(master)

    created = [e for e in events if isinstance(e, InstanceCreated)]
    assert [str(e.instance.shard_assignments.model_id) for e in created] == ["org/second"]
    assert isinstance(events[-1], RestartRecoveryCompleted)


async def test_each_placement_gets_its_own_wait_for_node_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowFirst(_Planner):
        def __init__(self) -> None:
            super().__init__()
            self.ready = False

        def __call__(self, command: PlaceInstance, current: Mapping[InstanceId, Instance]) -> dict[InstanceId, Instance]:
            if not self.ready:
                self.commands.append(command)
                raise PlacementInfoPendingError("node memory has not been reported yet")
            return super().__call__(command, current)

    planner = _SlowFirst()
    master = await _cold_master_with(monkeypatch, "org/first", "org/second", planner=planner)
    start = datetime.now(tz=timezone.utc)
    assert _advance(master, now=start) == []
    # Most of the first placement's window passes before information arrives.
    nearly = start + timedelta(seconds=INFO_WAIT_AFTER_SETTLE_SECONDS - 10)
    planner.ready = True
    events = _advance(master, now=nearly)
    created = [e for e in events if isinstance(e, InstanceCreated)]
    assert [str(e.instance.shard_assignments.model_id) for e in created] == ["org/first", "org/second"]
