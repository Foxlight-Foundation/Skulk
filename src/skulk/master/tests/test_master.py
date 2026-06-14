from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import anyio
import pytest
from loguru import logger

from skulk.master.main import REPLAY_TAIL_RETENTION_EVENTS, Master
from skulk.master.placement import place_instance
from skulk.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.topology import Topology
from skulk.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    RefuseInstancePlacement,
    TextGeneration,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    LocalForwarderEvent,
    NodeGatheredInfo,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    MemoryUsage,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.tasks import TaskId, TaskStatus
from skulk.shared.types.tasks import TextGeneration as TextGenerationTask
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.topology import Connection
from skulk.shared.types.worker.instances import (
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding
from skulk.utils.channels import channel


@pytest.mark.asyncio
async def test_master():
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    fcds, _fcdr = channel[ForwarderDownloadCommand]()
    ev_send, ev_recv = channel[Event]()

    async def mock_event_router():
        idx = 0
        sid = SystemId()
        with ev_recv as master_events:
            async for event in master_events:
                await local_event_sender.send(
                    LocalForwarderEvent(
                        origin=sid,
                        origin_idx=idx,
                        session=session_id,
                        event=event,
                    )
                )
                idx += 1

    all_events: list[IndexedEvent] = []

    def _get_events() -> Sequence[IndexedEvent]:
        orig_events = global_event_receiver.collect()
        for e in orig_events:
            all_events.append(
                IndexedEvent(
                    event=e.event,
                    idx=len(all_events),  # origin=e.origin,
                )
            )
        return all_events

    master = Master(
        node_id,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=fcds,
    )
    logger.info("run the master")
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        tg.start_soon(mock_event_router)

        # inject a NodeGatheredInfo event
        logger.info("inject a NodeGatheredInfo event")
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=0,
                origin=SystemId("Worker"),
                session=session_id,
                event=(
                    NodeGatheredInfo(
                        when=str(datetime.now(tz=timezone.utc)),
                        node_id=node_id,
                        info=MemoryUsage(
                            ram_total=Memory.from_bytes(678948 * 1024),
                            ram_available=Memory.from_bytes(678948 * 1024),
                            swap_total=Memory.from_bytes(0),
                            swap_available=Memory.from_bytes(0),
                        ),
                    )
                ),
            )
        )

        # wait for initial topology event (the NodeGatheredInfo above still
        # builds topology + last_seen even though its memory now rides telemetry)
        logger.info("wait for initial topology event")
        while len(list(master.state.topology.list_nodes())) == 0:
            await anyio.sleep(0.001)
        # node_memory lives on the telemetry plane now (#279 slice 2); placement
        # reads it from the master's TelemetryView, so seed it there directly.
        master._telemetry_view.node_memory[node_id] = MemoryUsage(  # pyright: ignore[reportPrivateUsage]
            ram_total=Memory.from_bytes(678948 * 1024),
            ram_available=Memory.from_bytes(678948 * 1024),
            swap_total=Memory.from_bytes(0),
            swap_available=Memory.from_bytes(0),
        )

        logger.info("inject a CreateInstance Command")
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("API"),
                command=(
                    PlaceInstance(
                        command_id=CommandId(),
                        model_card=ModelCard(
                            model_id=ModelId("llama-3.2-1b"),
                            n_layers=16,
                            storage_size=Memory.from_bytes(678948),
                            hidden_size=7168,
                            supports_tensor=True,
                            tasks=[ModelTask.TextGeneration],
                        ),
                        sharding=Sharding.Pipeline,
                        instance_meta=InstanceMeta.MlxRing,
                        min_nodes=1,
                    )
                ),
            )
        )
        logger.info("wait for an instance")
        while len(master.state.instances.keys()) == 0:
            await anyio.sleep(0.001)
        logger.info("inject a TextGeneration Command")
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("API"),
                command=(
                    TextGeneration(
                        command_id=CommandId(),
                        task_params=TextGenerationTaskParams(
                            model=ModelId("llama-3.2-1b"),
                            input=[
                                InputMessage(role="user", content="Hello, how are you?")
                            ],
                        ),
                    )
                ),
            )
        )
        while len(_get_events()) < 3:
            await anyio.sleep(0.01)

        events = _get_events()
        assert len(events) == 3
        assert events[0].idx == 0
        assert events[1].idx == 1
        assert events[2].idx == 2
        assert isinstance(events[0].event, NodeGatheredInfo)
        assert isinstance(events[1].event, InstanceCreated)
        created_instance = events[1].event.instance
        assert isinstance(created_instance, MlxRingInstance)
        runner_id = list(created_instance.shard_assignments.runner_to_shard.keys())[0]
        # Validate the shard assignments
        expected_shard_assignments = ShardAssignments(
            model_id=ModelId("llama-3.2-1b"),
            runner_to_shard={
                (runner_id): PipelineShardMetadata(
                    start_layer=0,
                    end_layer=16,
                    n_layers=16,
                    model_card=ModelCard(
                        model_id=ModelId("llama-3.2-1b"),
                        n_layers=16,
                        storage_size=Memory.from_bytes(678948),
                        hidden_size=7168,
                        supports_tensor=True,
                        tasks=[ModelTask.TextGeneration],
                    ),
                    device_rank=0,
                    world_size=1,
                )
            },
            node_to_runner={node_id: runner_id},
        )
        assert created_instance.shard_assignments == expected_shard_assignments
        # For single-node, hosts_by_node should have one entry with self-binding
        assert len(created_instance.hosts_by_node) == 1
        assert node_id in created_instance.hosts_by_node
        assert len(created_instance.hosts_by_node[node_id]) == 1
        assert created_instance.hosts_by_node[node_id][0].ip == "0.0.0.0"
        assert created_instance.ephemeral_port > 0
        assert isinstance(events[2].event, TaskCreated)
        assert events[2].event.task.task_status == TaskStatus.Pending
        assert isinstance(events[2].event.task, TextGenerationTask)
        assert events[2].event.task.task_params == TextGenerationTaskParams(
            model=ModelId("llama-3.2-1b"),
            input=[InputMessage(role="user", content="Hello, how are you?")],
        )

        ev_send.close()
        await master.shutdown()


@pytest.mark.asyncio
async def test_state_sync_response_includes_config_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    config_path = tmp_path / "skulk.yaml"
    config_yaml = (
        "model_store:\n"
        "  enabled: true\n"
        "  store_host: kite3.local\n"
        "  store_path: /Volumes/models\n"
        "hf_token: super-secret-token\n"
    )
    config_path.write_text(config_yaml)
    monkeypatch.setattr("skulk.master.main.resolve_config_path", lambda: config_path)

    global_sender, _global_receiver = channel[GlobalForwarderEvent]()
    _command_sender, command_receiver = channel[ForwarderCommand]()
    _local_event_sender, local_event_receiver = channel[LocalForwarderEvent]()
    request_sender, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, response_receiver = channel[StateSyncMessage]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, _event_receiver = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        await request_sender.send(
            StateSyncMessage(
                kind="request",
                requester=SystemId("requester"),
                session_id=session_id,
            )
        )

        response: StateSyncMessage | None = None
        while response is None:
            candidate = await response_receiver.receive()
            if candidate.kind == "response":
                response = candidate

        assert response.config_yaml is not None
        assert "super-secret-token" not in response.config_yaml
        assert "hf_token" not in response.config_yaml
        assert "store_host: kite3.local" in response.config_yaml
        assert response.snapshot is not None
        assert response.snapshot.session_id == session_id

        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_state_sync_response_survives_invalid_config_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("model_store: [")
    monkeypatch.setattr("skulk.master.main.resolve_config_path", lambda: config_path)

    global_sender, _global_receiver = channel[GlobalForwarderEvent]()
    _command_sender, command_receiver = channel[ForwarderCommand]()
    _local_event_sender, local_event_receiver = channel[LocalForwarderEvent]()
    request_sender, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, response_receiver = channel[StateSyncMessage]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, _event_receiver = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        await request_sender.send(
            StateSyncMessage(
                kind="request",
                requester=SystemId("requester"),
                session_id=session_id,
            )
        )

        response: StateSyncMessage | None = None
        while response is None:
            candidate = await response_receiver.receive()
            if candidate.kind == "response":
                response = candidate

        assert response.snapshot is not None
        assert response.config_yaml is None

        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_persist_snapshot_keeps_bounded_replay_tail() -> None:
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    global_sender, _global_receiver = channel[GlobalForwarderEvent]()
    _command_sender, command_receiver = channel[ForwarderCommand]()
    _local_event_sender, local_event_receiver = channel[LocalForwarderEvent]()
    _request_sender, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, _response_receiver = channel[StateSyncMessage]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, _event_receiver = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
    )

    compact_calls: list[int] = []

    class _FakeEventLog:
        def compact(self, keep_from_idx: int) -> None:
            compact_calls.append(keep_from_idx)

    class _FakeSnapshotStore:
        def write(self, _snapshot: object) -> None:
            return None

    master._event_log = _FakeEventLog()  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
    master._snapshot_store = _FakeSnapshotStore()  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
    master.state = master.state.model_copy(update={"last_event_applied_idx": 25})

    await master._persist_snapshot(force=True)  # pyright: ignore[reportPrivateUsage]

    assert compact_calls == [max(26 - REPLAY_TAIL_RETENTION_EVENTS, 0)]


@pytest.mark.asyncio
async def test_task_deleted_event_clears_command_mapping() -> None:
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    global_sender, _global_receiver = channel[GlobalForwarderEvent]()
    _command_sender, command_receiver = channel[ForwarderCommand]()
    local_event_sender, local_event_receiver = channel[LocalForwarderEvent]()
    _request_sender, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, _response_receiver = channel[StateSyncMessage]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, _event_receiver = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
    )

    command_id = CommandId("cmd-a")
    task_id = TaskId("task-a")
    master.command_task_mapping[command_id] = task_id

    async with anyio.create_task_group() as tg:
        tg.start_soon(master._event_processor)  # pyright: ignore[reportPrivateUsage]
        await local_event_sender.send(
            LocalForwarderEvent(
                origin=SystemId("Worker"),
                origin_idx=0,
                session=session_id,
                event=TaskDeleted(task_id=task_id),
            )
        )
        with anyio.fail_after(2):
            while command_id in master.command_task_mapping:
                await anyio.sleep(0.01)
        tg.cancel_scope.cancel()

    assert command_id not in master.command_task_mapping


@pytest.mark.asyncio
async def test_noop_task_events_for_unknown_tasks_are_not_indexed() -> None:
    """The master must not index task events for tasks absent from state.

    A misbehaving emitter (the #278 idle-runner mint) produced unbounded
    TaskStatusUpdated(Cancelled)+TaskDeleted pairs for long-dead tasks; each
    was a state no-op yet was still indexed, persisted, and broadcast
    cluster-wide. The ingest backstop drops them before indexing.
    """
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    _command_sender, command_receiver = channel[ForwarderCommand]()
    local_event_sender, local_event_receiver = channel[LocalForwarderEvent]()
    _request_sender, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, _response_receiver = channel[StateSyncMessage]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, _event_receiver = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
    )

    known_task = TextGenerationTask(
        task_id=TaskId("task-known"),
        instance_id=InstanceId("instance-a"),
        command_id=CommandId("cmd-known"),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content="hi")],
        ),
    )
    master.state = master.state.model_copy(
        update={"tasks": {known_task.task_id: known_task}}
    )

    def _local(event: Event, idx: int) -> LocalForwarderEvent:
        return LocalForwarderEvent(
            origin=SystemId("Worker"),
            origin_idx=idx,
            session=session_id,
            event=event,
        )

    indexed: GlobalForwarderEvent | None = None
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._event_processor)  # pyright: ignore[reportPrivateUsage]

        # No-op events for a task state has never seen: must NOT be indexed.
        await local_event_sender.send(
            _local(
                TaskStatusUpdated(
                    task_id=TaskId("task-ghost"), task_status=TaskStatus.Cancelled
                ),
                0,
            )
        )
        await local_event_sender.send(
            _local(TaskDeleted(task_id=TaskId("task-ghost")), 1)
        )
        # A legitimate status for a known task: must be indexed and broadcast.
        await local_event_sender.send(
            _local(
                TaskStatusUpdated(
                    task_id=known_task.task_id, task_status=TaskStatus.Running
                ),
                2,
            )
        )

        with anyio.fail_after(2):
            indexed = await global_receiver.receive()
        tg.cancel_scope.cancel()

    # The first (and only) indexed event is the known-task status — the two
    # ghost events were dropped without consuming indices.
    assert indexed is not None
    assert indexed.origin_idx == 0
    assert isinstance(indexed.event, TaskStatusUpdated)
    assert indexed.event.task_id == known_task.task_id
    assert len(master._event_log) == 1  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_refuse_instance_placement_replaces_wider_once() -> None:
    """A memory refusal re-places one node wider, and simultaneous refusals
    from multiple ranks re-place exactly once (#290).

    The command processor generates events without applying them (self.state
    only updates when they round-trip through the event processor), so without
    a dedup guard two ranks refusing the same instance would each spawn a wider
    replacement. This drives two RefuseInstancePlacement commands and asserts a
    single surviving instance at the wider width.
    """
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    # 3-node fully-connected topology with generous memory.
    node_ids = [node_id, NodeId(), NodeId()]
    topology = Topology()
    for nid in node_ids:
        topology.add_node(nid)
    port = 1
    for source in node_ids:
        for sink in node_ids:
            if source != sink:
                topology.add_connection(
                    Connection(
                        source=source, sink=sink, edge=create_socket_connection(port)
                    )
                )
                port += 1
    node_memory = {
        nid: create_node_memory(Memory.from_gb(40).in_bytes) for nid in node_ids
    }
    node_network = {nid: create_node_network() for nid in node_ids}

    card = ModelCard(
        model_id=ModelId("refuse-replace-model"),
        storage_size=Memory.from_gb(6),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    two_node_cmd = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )
    placed = place_instance(two_node_cmd, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    instance_id = instance.instance_id
    assert len(instance.shard_assignments.node_to_runner) == 2
    refused_nodes = list(instance.shard_assignments.node_to_runner.keys())

    seed_state = State(
        topology=topology,
        instances={instance_id: instance},
        node_network=node_network,
    )
    telemetry = TelemetryView()
    for nid, mem in node_memory.items():
        telemetry.node_memory[nid] = mem

    ge_sender, global_event_receiver = channel[GlobalForwarderEvent]()
    command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    fcds, _fcdr = channel[ForwarderDownloadCommand]()
    ev_send, ev_recv = channel[Event]()

    async def mock_event_router() -> None:
        idx = 0
        sid = SystemId()
        with ev_recv as master_events:
            async for event in master_events:
                await local_event_sender.send(
                    LocalForwarderEvent(
                        origin=sid,
                        origin_idx=idx,
                        session=session_id,
                        event=event,
                    )
                )
                idx += 1

    master = Master(
        node_id,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=fcds,
        initial_state=seed_state,
        telemetry_view=telemetry,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        tg.start_soon(mock_event_router)

        # wait for the seeded 2-node instance to land in state
        with anyio.fail_after(5):
            while instance_id not in master.state.instances:
                await anyio.sleep(0.005)

        # two ranks refuse the same instance, back to back
        for refusing in refused_nodes:
            await command_sender.send(
                ForwarderCommand(
                    origin=SystemId("Worker"),
                    command=RefuseInstancePlacement(
                        instance_id=instance_id,
                        node_id=refusing,
                        reason="not enough GPU-wireable memory",
                    ),
                )
            )

        # wait for a wider replacement to appear and the original to be gone
        with anyio.fail_after(5):
            while True:
                instances = dict(master.state.instances)
                if instance_id not in instances and len(instances) == 1:
                    break
                await anyio.sleep(0.005)

        replaced = next(iter(master.state.instances.values()))
        assert len(replaced.shard_assignments.node_to_runner) == 3
        # exactly one replacement despite two refusals
        assert len(master.state.instances) == 1

        global_event_receiver.collect()
        tg.cancel_scope.cancel()
