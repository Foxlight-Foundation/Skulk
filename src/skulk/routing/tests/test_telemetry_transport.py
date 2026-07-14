"""Telemetry must stay bounded and unable to delay correctness traffic."""

from collections.abc import Sequence
from typing import cast

import anyio
from skulk_pyo3_bindings import NetworkingHandle

from skulk.routing.router import OutboundPacket, Router
from skulk.routing.topics import (
    COMMANDS,
    ELECTION_MESSAGES,
    GLOBAL_EVENTS,
    LOCAL_EVENTS,
    STATE_SYNC_MESSAGES,
    TELEMETRY,
)
from skulk.shared.apply import apply
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelId
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import (
    CommandId,
    CreateInstance,
    ForwarderCommand,
    TaskCancelled,
    TaskFinished,
)
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    StateSnapshotHydrated,
    TaskCreated,
    TaskStatusUpdated,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.tasks import LoadModel, TaskId, TaskStatus
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import ShardAssignments
from skulk.utils.channels import Receiver
from skulk.utils.info_gatherer.info_gatherer import MiscData


class _BlockingTelemetryNetwork:
    """Transport that stalls telemetry while recording all other publications."""

    def __init__(self, *, latest_marker: bytes) -> None:
        self.telemetry_started = anyio.Event()
        self.release_telemetry = anyio.Event()
        self.latest_published = anyio.Event()
        self.control_published = anyio.Event()
        self.control_topics: list[str] = []
        self.telemetry_payloads: list[bytes] = []
        self._latest_marker = latest_marker

    async def gossipsub_publish(self, topic: str, data: bytes) -> None:
        if topic == TELEMETRY.topic:
            self.telemetry_started.set()
            await self.release_telemetry.wait()
            self.telemetry_payloads.append(data)
            if self._latest_marker in data:
                self.latest_published.set()
            return
        self.control_topics.append(topic)
        if len(self.control_topics) == 5:
            self.control_published.set()


class _MeshEndpoint:
    """One endpoint in a deterministic in-process gossipsub mesh."""

    def __init__(self, node_id: NodeId, endpoints: list["_MeshEndpoint"]) -> None:
        self.node_id = node_id
        self.endpoints = endpoints
        self.router: Router | None = None
        self.telemetry_started = anyio.Event()
        self.release_telemetry = anyio.Event()

    async def gossipsub_publish(self, topic: str, data: bytes) -> None:
        """Broadcast one packet, independently stalling telemetry egress."""

        if topic == TELEMETRY.topic:
            self.telemetry_started.set()
            await self.release_telemetry.wait()
        for endpoint in self.endpoints:
            if endpoint is self:
                continue
            assert endpoint.router is not None
            await endpoint.router.topic_routers[topic].publish_bytes(
                data, str(self.node_id)
            )


def _packet(topic: str) -> OutboundPacket:
    return OutboundPacket(
        topic=topic,
        routing_key=None,
        stream_key=None,
        is_terminal=False,
        data=topic.encode(),
    )


async def test_saturated_telemetry_cannot_delay_control_egress() -> None:
    """A stalled telemetry peer cannot consume any control publish loop."""

    latest_name = "reading-9999"
    network = _BlockingTelemetryNetwork(latest_marker=latest_name.encode())
    router = Router(
        handle=cast(NetworkingHandle, cast(object, network)),
        node_id="test-node",
    )
    await router.register_topic(TELEMETRY)
    telemetry_sender = router.telemetry_sender()
    telemetry_router = router.topic_routers[TELEMETRY.topic]
    ordinary_sender = router.networking_receiver.clone_sender()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(telemetry_router.run)
        task_group.start_soon(router._networking_publish)  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(router._election_networking_publish)  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(router._telemetry_networking_publish)  # pyright: ignore[reportPrivateUsage]

        await telemetry_sender.send(
            NodeTelemetry(
                node_id=NodeId("producer"),
                info=MiscData(friendly_name="reading-0"),
            )
        )
        with anyio.fail_after(0.5):
            await network.telemetry_started.wait()

        for index in range(1, 10_000):
            await telemetry_sender.send(
                NodeTelemetry(
                    node_id=NodeId("producer"),
                    info=MiscData(friendly_name=f"reading-{index}"),
                )
            )

        for topic in (
            COMMANDS.topic,
            LOCAL_EVENTS.topic,
            GLOBAL_EVENTS.topic,
            STATE_SYNC_MESSAGES.topic,
        ):
            await ordinary_sender.send(_packet(topic))
        await router._election_out_send.send(  # pyright: ignore[reportPrivateUsage]
            _packet(ELECTION_MESSAGES.topic)
        )
        with anyio.fail_after(0.5):
            await network.control_published.wait()

        blocked = router.telemetry_plane_diagnostics()
        assert blocked.pending_readings <= 1
        assert blocked.network_queue_depth <= 1
        assert blocked.max_queue_depth <= 3
        assert blocked.readings_offered == 10_000
        assert blocked.readings_coalesced > 9_000
        assert blocked.readings_dropped == 0

        network.release_telemetry.set()
        with anyio.fail_after(1):
            await network.latest_published.wait()
        recovered = router.telemetry_plane_diagnostics()
        assert recovered.readings_published >= 2
        assert recovered.last_successful_publish_age_seconds is not None
        assert set(network.control_topics) == {
            COMMANDS.topic,
            LOCAL_EVENTS.topic,
            GLOBAL_EVENTS.topic,
            STATE_SYNC_MESSAGES.topic,
            ELECTION_MESSAGES.topic,
        }
        task_group.cancel_scope.cancel()


async def test_telemetry_admission_evicts_oldest_key_at_fixed_bound() -> None:
    """Distinct telemetry identities cannot grow admission memory without bound."""

    network = _BlockingTelemetryNetwork(latest_marker=b"unused")
    router = Router(
        handle=cast(NetworkingHandle, cast(object, network)),
        node_id="test-node",
    )
    await router.register_topic(TELEMETRY)
    telemetry_sender = router.telemetry_sender()

    for index in range(1_000):
        telemetry_sender.send_nowait(
            NodeTelemetry(
                node_id=NodeId(f"producer-{index}"),
                info=MiscData(friendly_name=f"reading-{index}"),
            )
        )

    diagnostics = router.telemetry_plane_diagnostics()
    assert diagnostics.pending_readings == diagnostics.admission_capacity == 256
    assert diagnostics.readings_offered == 1_000
    assert diagnostics.readings_dropped == 744
    assert diagnostics.readings_coalesced == 0
    assert diagnostics.network_queue_depth == 0


def _lifecycle_events() -> tuple[
    list[GlobalForwarderEvent],
    State,
    SessionId,
    MlxRingInstance,
]:
    """Build valid placement, inference, cancellation, and failover events."""

    first_session = SessionId(master_node_id=NodeId("node-0"), election_clock=1)
    second_session = SessionId(master_node_id=NodeId("node-1"), election_clock=2)
    instance = MlxRingInstance(
        instance_id=InstanceId("stress-instance"),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test/stress-model"),
            runner_to_shard={},
            node_to_runner={},
        ),
        hosts_by_node={},
        ephemeral_port=52_415,
    )
    completed_task = LoadModel(
        task_id=TaskId("completed-before-failover"),
        instance_id=instance.instance_id,
    )
    cancelled_task = LoadModel(
        task_id=TaskId("cancelled-before-failover"),
        instance_id=instance.instance_id,
    )
    after_failover_task = LoadModel(
        task_id=TaskId("completed-after-failover"),
        instance_id=instance.instance_id,
    )
    first_events = [
        InstanceCreated(instance=instance),
        TaskCreated(task_id=completed_task.task_id, task=completed_task),
        TaskStatusUpdated(
            task_id=completed_task.task_id,
            task_status=TaskStatus.Complete,
        ),
        TaskCreated(task_id=cancelled_task.task_id, task=cancelled_task),
        TaskStatusUpdated(
            task_id=cancelled_task.task_id,
            task_status=TaskStatus.Cancelled,
        ),
    ]
    first_state = State()
    for index, event in enumerate(first_events):
        first_state = apply(first_state, IndexedEvent(idx=index, event=event))

    seed = seed_state_for_new_session(first_state).model_copy(
        update={"last_event_applied_idx": 0}
    )
    second_events = [
        StateSnapshotHydrated(state=seed),
        TaskCreated(task_id=after_failover_task.task_id, task=after_failover_task),
        TaskStatusUpdated(
            task_id=after_failover_task.task_id,
            task_status=TaskStatus.Complete,
        ),
    ]
    expected = first_state
    expected = apply(expected, IndexedEvent(idx=0, event=second_events[0]))
    expected = apply(expected, IndexedEvent(idx=1, event=second_events[1]))
    expected = apply(expected, IndexedEvent(idx=2, event=second_events[2]))

    forwarded = [
        GlobalForwarderEvent(
            origin_idx=index,
            origin=first_session.master_node_id,
            session=first_session,
            event=event,
        )
        for index, event in enumerate(first_events)
    ]
    forwarded.extend(
        GlobalForwarderEvent(
            origin_idx=index,
            origin=second_session.master_node_id,
            session=second_session,
            event=event,
        )
        for index, event in enumerate(second_events)
    )
    return forwarded, expected, second_session, instance


async def _wait_for_count[T](
    receivers: Sequence[Receiver[T]], expected: int
) -> list[list[T]]:
    """Drain receiver buffers until every peer has the expected packet count."""

    collected: list[list[T]] = [[] for _ in receivers]
    with anyio.fail_after(1):
        while min(len(items) for items in collected) < expected:
            for index, receiver in enumerate(receivers):
                collected[index].extend(receiver.collect())
            await anyio.sleep(0)
    return collected


async def test_multi_peer_lifecycle_stays_converged_under_telemetry_pressure() -> None:
    """Three peers converge through placement, cancellation, and failover."""

    endpoints: list[_MeshEndpoint] = []
    endpoints.extend(
        _MeshEndpoint(NodeId(f"node-{index}"), endpoints) for index in range(3)
    )
    routers: list[Router] = []
    for endpoint in endpoints:
        router = Router(
            handle=cast(NetworkingHandle, cast(object, endpoint)),
            node_id=str(endpoint.node_id),
        )
        endpoint.router = router
        routers.append(router)
        await router.register_topic(TELEMETRY)
        await router.register_topic(COMMANDS)
        await router.register_topic(GLOBAL_EVENTS)
        await router.register_topic(STATE_SYNC_MESSAGES)
        await router.register_topic(ELECTION_MESSAGES)

    telemetry_senders = [router.telemetry_sender() for router in routers]
    telemetry_receivers = [router.receiver(TELEMETRY) for router in routers]
    command_receivers = [router.receiver(COMMANDS) for router in routers]
    global_receivers = [router.receiver(GLOBAL_EVENTS) for router in routers]
    sync_receivers = [router.receiver(STATE_SYNC_MESSAGES) for router in routers]
    election_receivers = [router.receiver(ELECTION_MESSAGES) for router in routers]
    flood_done = [anyio.Event() for _ in routers]

    lifecycle, expected_state, failover_session, instance = _lifecycle_events()
    origin = SystemId("stress-api")
    commands = [
        ForwarderCommand(origin=origin, command=CreateInstance(instance=instance)),
        ForwarderCommand(
            origin=origin,
            command=TaskFinished(finished_command_id=CommandId("completed")),
        ),
        ForwarderCommand(
            origin=origin,
            command=TaskCancelled(cancelled_command_id=CommandId("cancelled")),
        ),
    ]

    async def flood(index: int) -> None:
        try:
            for sample in range(5_000):
                await telemetry_senders[index].send(
                    NodeTelemetry(
                        node_id=endpoints[index].node_id,
                        info=MiscData(friendly_name=f"node-{index}-sample-{sample}"),
                    )
                )
                if sample % 25 == 0:
                    await anyio.sleep(0)
        finally:
            flood_done[index].set()

    async with anyio.create_task_group() as task_group:
        for router in routers:
            for topic_router in router.topic_routers.values():
                task_group.start_soon(topic_router.run)
            task_group.start_soon(router._networking_publish)  # pyright: ignore[reportPrivateUsage]
            task_group.start_soon(router._election_networking_publish)  # pyright: ignore[reportPrivateUsage]
            task_group.start_soon(router._telemetry_networking_publish)  # pyright: ignore[reportPrivateUsage]

        for index, sender in enumerate(telemetry_senders):
            await sender.send(
                NodeTelemetry(
                    node_id=endpoints[index].node_id,
                    info=MiscData(friendly_name=f"node-{index}-initial"),
                )
            )
        with anyio.fail_after(1):
            for endpoint in endpoints:
                await endpoint.telemetry_started.wait()

        for index in range(3):
            task_group.start_soon(flood, index)

        command_sender = routers[0].sender(COMMANDS)
        global_sender = routers[0].sender(GLOBAL_EVENTS)
        for command in commands:
            await command_sender.send(command)
        for event in lifecycle:
            await global_sender.send(event)
        await routers[0].sender(STATE_SYNC_MESSAGES).send(
            StateSyncMessage(
                kind="request",
                requester=origin,
                session_id=failover_session,
            )
        )
        await routers[0].sender(ELECTION_MESSAGES).send(
            ElectionMessage(
                clock=2,
                seniority=1,
                proposed_session=failover_session,
                commands_seen=len(commands),
            )
        )

        received_commands = await _wait_for_count(command_receivers, len(commands))
        received_events = await _wait_for_count(global_receivers, len(lifecycle))
        await _wait_for_count(sync_receivers, 1)
        await _wait_for_count(election_receivers, 1)
        with anyio.fail_after(1):
            for done in flood_done:
                await done.wait()

        assert all(messages == commands for messages in received_commands)
        states: list[State] = []
        for messages in received_events:
            state = State()
            for message in messages:
                state = apply(
                    state,
                    IndexedEvent(idx=message.origin_idx, event=message.event),
                )
            states.append(state)
        expected_state_json = expected_state.model_dump_json()
        assert [state.model_dump_json() for state in states] == [
            expected_state_json
        ] * len(routers)
        assert instance.instance_id in expected_state.instances
        assert expected_state.tasks[TaskId("completed-after-failover")].task_status == (
            TaskStatus.Complete
        )

        for router in routers:
            diagnostics = router.telemetry_plane_diagnostics()
            assert diagnostics.pending_readings <= 1
            assert diagnostics.network_queue_depth <= 1
            assert diagnostics.readings_offered == 5_001
            assert diagnostics.readings_coalesced > 4_500

        for endpoint in endpoints:
            endpoint.release_telemetry.set()

        views = [TelemetryView() for _ in routers]
        with anyio.fail_after(2):
            while True:
                for index, receiver in enumerate(telemetry_receivers):
                    for message in receiver.collect():
                        views[index].apply(message)
                if all(
                    all(
                        view.node_identities.get(endpoint.node_id) is not None
                        and view.node_identities[endpoint.node_id].friendly_name
                        == f"node-{endpoint_index}-sample-4999"
                        for endpoint_index, endpoint in enumerate(endpoints)
                    )
                    for view in views
                ):
                    break
                await anyio.sleep(0)
        task_group.cancel_scope.cancel()
