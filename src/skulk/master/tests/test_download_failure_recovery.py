"""Tests for detecting instances wedged by a rank's failed download (#381).

A multi-node instance whose ring forms but where one rank's model download
terminally fails sits at ``RunnerConnected`` forever with nothing to recover it.
``instances_wedged_by_download_failure`` is the master-side detector that finds
exactly that wedge from replicated state so the plan loop can fail and re-place
the instance.
"""

from collections.abc import Mapping

import pytest

from skulk.master.main import (
    instances_wedged_by_download_failure,
    retire_download_failure_lineages,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    LlamaRpcInstance,
    MlxRingInstance,
    RunnerId,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import (
    RunnerConnected,
    RunnerReady,
    RunnerStatus,
)
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    RpcDonorShardMetadata,
)

_MODEL = ModelId("test-model")


def _model_card(model_id: ModelId = _MODEL) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        n_layers=16,
        storage_size=Memory.from_bytes(1024),
        hidden_size=64,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _shard(card: ModelCard, rank: int, world: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        start_layer=0,
        end_layer=16,
        n_layers=16,
        model_card=card,
        device_rank=rank,
        world_size=world,
    )


def _two_node_instance(
    instance_id: InstanceId, node_a: NodeId, node_b: NodeId
) -> tuple[Instance, RunnerId, RunnerId]:
    card = _model_card()
    runner_a, runner_b = RunnerId(), RunnerId()
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=_MODEL,
            runner_to_shard={
                runner_a: _shard(card, 0, 2),
                runner_b: _shard(card, 1, 2),
            },
            node_to_runner={node_a: runner_a, node_b: runner_b},
        ),
        hosts_by_node={node_a: [], node_b: []},
        ephemeral_port=12345,
    )
    return instance, runner_a, runner_b


def _failed(node_id: NodeId, model_id: ModelId = _MODEL) -> DownloadFailed:
    return DownloadFailed(
        node_id=node_id,
        shard_metadata=_shard(_model_card(model_id), 0, 2),
        error_message="[Errno 28] No space left on device",
    )


def _completed(node_id: NodeId) -> DownloadCompleted:
    return DownloadCompleted(
        node_id=node_id,
        shard_metadata=_shard(_model_card(), 0, 2),
        total=Memory.from_bytes(1024),
    )


def _state(
    instance: Instance,
    runners: dict[RunnerId, RunnerStatus],
    downloads: dict[NodeId, list[DownloadProgress]],
) -> State:
    return State(
        instances={instance.instance_id: instance},
        runners=runners,
        downloads=downloads,
    )


def test_wedge_detected_when_a_rank_download_failed_and_not_ready() -> None:
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a)], node_b: [_completed(node_b)]},
    )

    wedged = instances_wedged_by_download_failure(state)

    assert set(wedged) == {iid}
    failed_nodes, cause = wedged[iid]
    assert failed_nodes == frozenset({node_a})
    assert "No space left on device" in cause


def test_ready_instance_never_reported_even_with_stale_failure() -> None:
    # A serving instance must never be torn down by this path, even if a stale
    # DownloadFailed lingers in state.
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerReady(), runner_b: RunnerReady()},
        {node_a: [_failed(node_a)], node_b: [_completed(node_b)]},
    )

    assert instances_wedged_by_download_failure(state) == {}


def test_not_ready_without_any_failure_is_not_wedged() -> None:
    # Still legitimately loading: connected, downloads completed, no failure.
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_completed(node_a)], node_b: [_completed(node_b)]},
    )

    assert instances_wedged_by_download_failure(state) == {}


def test_failure_for_a_different_model_is_ignored() -> None:
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a, ModelId("other-model"))], node_b: []},
    )

    assert instances_wedged_by_download_failure(state) == {}


def test_multiple_failed_ranks_all_reported() -> None:
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a)], node_b: [_failed(node_b)]},
    )

    failed_nodes, _cause = instances_wedged_by_download_failure(state)[iid]
    assert failed_nodes == frozenset({node_a, node_b})


async def test_recovery_consumes_the_terminal_failure_record() -> None:
    """Recovery must reset the failed node's download status to Pending.

    Without the reset, the stale terminal DownloadFailed lingers in session
    state and the wedge scan condemns EVERY future placement of the model
    touching that node, long after the cause is gone (observed live: an
    ENOSPC during a pooled placement kept killing fresh placements an hour
    after the disk was freed, until a whole-fleet restart).
    """
    from anyio import WouldBlock

    from skulk.master.main import Master
    from skulk.routing.router import get_node_id_keypair
    from skulk.shared.types.commands import (
        ForwarderCommand,
        ForwarderDownloadCommand,
    )
    from skulk.shared.types.common import SessionId
    from skulk.shared.types.events import (
        Event,
        GlobalForwarderEvent,
        InstanceFailureRecorded,
        LocalForwarderEvent,
        NodeDownloadProgress,
    )
    from skulk.shared.types.state_sync import StateSyncMessage
    from skulk.shared.types.worker.downloads import DownloadPending
    from skulk.utils.channels import channel

    master_node = NodeId(get_node_id_keypair().to_node_id())
    session_id = SessionId(master_node_id=master_node, election_clock=0)
    ge_sender, _ = channel[GlobalForwarderEvent]()
    _, co_receiver = channel[ForwarderCommand]()
    _, le_receiver = channel[LocalForwarderEvent]()
    ss_sender, ss_receiver = channel[StateSyncMessage]()
    fcds, _ = channel[ForwarderDownloadCommand]()
    ev_send, ev_recv = channel[Event]()
    master = Master(
        master_node,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=ss_receiver,
        state_sync_sender=ss_sender,
        download_command_sender=fcds,
    )

    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    master.state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a)], node_b: [_completed(node_b)]},
    )

    await master._recover_download_failed_instances()  # pyright: ignore[reportPrivateUsage]

    emitted: list[Event] = []
    while True:
        try:
            emitted.append(ev_recv.receive_nowait())
        except WouldBlock:
            break
    resets = [event for event in emitted if isinstance(event, NodeDownloadProgress)]
    assert any(
        isinstance(reset.download_progress, DownloadPending)
        and reset.download_progress.node_id == node_a
        and reset.download_progress.shard_metadata.model_card.model_id == _MODEL
        for reset in resets
    ), "recovery must emit a DownloadPending reset for the failed node"
    failure = next(
        event for event in emitted if isinstance(event, InstanceFailureRecorded)
    )
    assert failure.failure.error_code == "download_failed"
    assert "No space left on device" not in failure.failure.error_message
    assert "Inspect cluster logs" in failure.failure.error_message


async def test_recovery_stops_when_the_failure_follows_the_model_to_every_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each repair hop excludes every node that already failed in its lineage.

    Recovery stamps only the caller's own exclusions on a replacement (#667),
    and it resets the failed node's download status, so each hop used to see
    only the latest failure. A failure that repeats on every node (offline,
    auth, a missing revision) then re-placed forever under new instance ids:
    a fails, the replacement lands on b and c, b fails, and a is eligible
    again. With the lineage carried, the second hop finds no two nodes left
    and recovery stops at the teardown.
    """
    from anyio import WouldBlock

    from skulk.master import main as master_main
    from skulk.master.main import Master
    from skulk.master.placement import PlacementError
    from skulk.routing.router import get_node_id_keypair
    from skulk.shared.types.commands import (
        ForwarderCommand,
        ForwarderDownloadCommand,
        PlaceInstance,
    )
    from skulk.shared.types.common import SessionId
    from skulk.shared.types.events import (
        Event,
        GlobalForwarderEvent,
        InstanceCreated,
        InstanceFailureRecorded,
        LocalForwarderEvent,
    )
    from skulk.shared.types.state_sync import StateSyncMessage
    from skulk.utils.channels import channel

    node_a, node_b, node_c = NodeId("a"), NodeId("b"), NodeId("c")
    searches: list[tuple[set[NodeId], set[NodeId]]] = []
    placed: list[tuple[Instance, RunnerId, RunnerId]] = []

    def place_on_two_healthy_nodes(
        command: PlaceInstance,
        topology: object,
        current_instances: Mapping[InstanceId, Instance],
        node_memory: object,
        node_network: object,
        *,
        excluded_nodes: set[NodeId] | None = None,
        stamped_exclusions: set[NodeId] | None = None,
        **_: object,
    ) -> dict[InstanceId, Instance]:
        excluded = set(excluded_nodes or ())
        searches.append((excluded, set(stamped_exclusions or ())))
        healthy = sorted({node_a, node_b, node_c} - excluded)
        if len(healthy) < command.min_nodes:
            raise PlacementError("no two healthy nodes left")
        replacement = _two_node_instance(
            InstanceId(str(command.command_id)), healthy[0], healthy[1]
        )
        placed.append(replacement)
        return {**current_instances, replacement[0].instance_id: replacement[0]}

    monkeypatch.setattr(master_main, "place_instance", place_on_two_healthy_nodes)

    master_node = NodeId(get_node_id_keypair().to_node_id())
    session_id = SessionId(master_node_id=master_node, election_clock=0)
    ge_sender, _ = channel[GlobalForwarderEvent]()
    _, co_receiver = channel[ForwarderCommand]()
    _, le_receiver = channel[LocalForwarderEvent]()
    ss_sender, ss_receiver = channel[StateSyncMessage]()
    fcds, _ = channel[ForwarderDownloadCommand]()
    ev_send, ev_recv = channel[Event]()
    master = Master(
        master_node,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=ss_receiver,
        state_sync_sender=ss_sender,
        download_command_sender=fcds,
    )

    def drain() -> list[Event]:
        emitted: list[Event] = []
        while True:
            try:
                emitted.append(ev_recv.receive_nowait())
            except WouldBlock:
                return emitted

    original, runner_a, runner_b = _two_node_instance(InstanceId(), node_a, node_b)
    master.state = _state(
        original,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a)], node_b: [_completed(node_b)]},
    )
    await master._recover_download_failed_instances()  # pyright: ignore[reportPrivateUsage]
    first_hop = drain()

    assert searches[0] == ({node_a}, set())
    replacement, runner_b2, runner_c = placed[0]
    assert set(replacement.shard_assignments.node_to_runner) == {node_b, node_c}
    assert any(
        isinstance(event, InstanceCreated)
        and event.instance.instance_id == replacement.instance_id
        for event in first_hop
    )

    # The replacement's download now fails on b. a's failure was consumed
    # (reset to pending) by the first hop, so only the lineage remembers it.
    master.state = _state(
        replacement,
        {runner_b2: RunnerConnected(), runner_c: RunnerConnected()},
        {node_b: [_failed(node_b)], node_c: [_completed(node_c)]},
    )
    await master._recover_download_failed_instances()  # pyright: ignore[reportPrivateUsage]
    second_hop = drain()

    assert searches[1] == ({node_a, node_b}, set()), (
        "the second hop must exclude the first hop's failed node too, while "
        "stamping only the caller's original exclusions"
    )
    assert len(placed) == 1, "no third placement: recovery stops at the teardown"
    assert not any(isinstance(event, InstanceCreated) for event in second_hop)
    assert any(isinstance(event, InstanceFailureRecorded) for event in second_hop)


def test_stale_donor_download_failure_does_not_wedge_rpc_instance() -> None:
    # RPC donors never download the model (#328). A stale terminal
    # DownloadFailed for the same model on the DONOR node (from an earlier
    # placement attempt there) must not condemn a pooled instance during the
    # driver's load window; only a failure on the driver node counts.
    iid = InstanceId()
    driver_node, donor_node = NodeId("driver"), NodeId("donor")
    card = _model_card()
    driver_runner, donor_runner = RunnerId(), RunnerId()
    instance = LlamaRpcInstance(
        instance_id=iid,
        shard_assignments=ShardAssignments(
            model_id=_MODEL,
            runner_to_shard={
                driver_runner: _shard(card, 0, 2),
                donor_runner: RpcDonorShardMetadata(
                    start_layer=0,
                    end_layer=0,
                    n_layers=16,
                    model_card=card,
                    device_rank=1,
                    world_size=2,
                ),
            },
            node_to_runner={driver_node: driver_runner, donor_node: donor_runner},
        ),
        driver_node=driver_node,
        donor_endpoints={donor_node: "10.99.0.2:50052"},
    )
    # Donor Ready, driver still loading; stale failure sits on the donor node.
    state = _state(
        instance,
        {driver_runner: RunnerConnected(), donor_runner: RunnerReady()},
        {donor_node: [_failed(donor_node)], driver_node: []},
    )
    assert instances_wedged_by_download_failure(state) == {}

    # A failure on the DRIVER node is still a real wedge.
    state = _state(
        instance,
        {driver_runner: RunnerConnected(), donor_runner: RunnerReady()},
        {donor_node: [], driver_node: [_failed(driver_node)]},
    )
    failed_nodes, _cause = instances_wedged_by_download_failure(state)[iid]
    assert failed_nodes == frozenset({driver_node})


def test_a_lineage_waits_for_its_replacement_to_appear() -> None:
    """A replacement not yet in state is a pending creation, not a deletion."""
    pending = InstanceId()
    lineages = {pending: frozenset({NodeId("a")})}

    kept, present = retire_download_failure_lineages(lineages, set(), State())

    assert kept == lineages
    assert present == set()


def test_a_lineage_in_progress_is_kept_and_marked_present() -> None:
    instance, runner_a, runner_b = _two_node_instance(
        InstanceId(), NodeId("b"), NodeId("c")
    )
    lineages = {instance.instance_id: frozenset({NodeId("a")})}
    state = _state(
        instance, {runner_a: RunnerConnected(), runner_b: RunnerConnected()}, {}
    )

    kept, present = retire_download_failure_lineages(lineages, set(), state)

    assert kept == lineages
    assert present == {instance.instance_id}


def test_a_lineage_retires_once_its_replacement_is_ready() -> None:
    """Recovery worked: a later failure of the same instance starts afresh."""
    instance, runner_a, runner_b = _two_node_instance(
        InstanceId(), NodeId("b"), NodeId("c")
    )
    state = _state(instance, {runner_a: RunnerReady(), runner_b: RunnerReady()}, {})

    kept, present = retire_download_failure_lineages(
        {instance.instance_id: frozenset({NodeId("a")})},
        {instance.instance_id},
        state,
    )

    assert kept == {}
    assert present == set()


def test_a_lineage_retires_once_its_replacement_is_deleted() -> None:
    gone = InstanceId()

    kept, present = retire_download_failure_lineages(
        {gone: frozenset({NodeId("a")})}, {gone}, State()
    )

    assert kept == {}
    assert present == set()

