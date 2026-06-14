import copy
import time
from collections.abc import Set as AbstractSet
from datetime import datetime, timedelta, timezone
from typing import cast

import anyio
import yaml
from loguru import logger

from skulk.master.placement import (
    PlacementError,
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
    replacement_command_for_refused_instance,
)
from skulk.shared.apply import apply
from skulk.shared.constants import SKULK_EVENT_LOG_DIR, SKULK_TRACING_ENABLED
from skulk.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RefuseInstancePlacement,
    RequestEventLog,
    SendInputChunk,
    SetTracingEnabled,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextEmbedding,
    TextGeneration,
)
from skulk.shared.types.common import CommandId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    StateSnapshotHydrated,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
    TracingStateChanged,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSnapshot, StateSyncMessage
from skulk.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from skulk.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from skulk.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from skulk.shared.types.tasks import (
    TextEmbedding as TextEmbeddingTask,
)
from skulk.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import InstanceId
from skulk.store.config import resolve_config_path
from skulk.utils.channels import Receiver, Sender
from skulk.utils.disk_event_log import DiskEventLog
from skulk.utils.event_buffer import MultiSourceBuffer
from skulk.utils.state_snapshot_store import StateSnapshotStore
from skulk.utils.task_group import TaskGroup

EVENT_LOG_REPLAY_BATCH_SIZE = 10_000
SNAPSHOT_EVENT_CADENCE = 10_000
REPLAY_TAIL_RETENTION_EVENTS = SNAPSHOT_EVENT_CADENCE

TOPOLOGY_SETTLE_GRACE_SECONDS = 60.0
"""How long after master start the plan loop trusts topology for pruning.

A new session's topology starts empty and is rebuilt from live gossip:
worker connection probes re-emit edges on a 10s cycle, plus router/mDNS
events. A failover-seeded master (#273) carries instances from the prior
session but deliberately NOT the prior topology (a dead node's out-edges
would persist forever — only their source node ever deletes them), so for
the first moments every carried instance's nodes look "disconnected".
Pruning during that window would delete the very placements the seed
preserved. 60s comfortably covers several probe cycles; the dead master's
instances are still pruned — just one minute later, once absence reflects
real liveness rather than an unsettled view."""
JsonObject = dict[str, object]

# API-facing task types: the ones whose loss strands an open HTTP request.
# Worker lifecycle tasks (CreateRunner, LoadModel, ...) are reconciled by the
# worker's own plan loop and must not be failed from here.
_COMMAND_TASK_TYPES = (
    TextGenerationTask,
    ImageGenerationTask,
    ImageEditsTask,
    TextEmbeddingTask,
)


def instances_on_dead_nodes(
    state: State,
    connected_node_ids: AbstractSet[NodeId],
    timed_out_node_ids: AbstractSet[NodeId],
) -> set[InstanceId]:
    """Instances with at least one shard on a disconnected or timed-out node.

    Timed-out nodes matter even while still present in topology: NodeTimedOut
    removes the node's instances AND their tasks from state in one apply, so
    any TaskFailed for those tasks must be emitted before that event — a
    later plan pass would no longer see them (#224 review catch).
    """
    dying: set[InstanceId] = set()
    for instance_id, instance in state.instances.items():
        for node_id in instance.shard_assignments.node_to_runner:
            if node_id not in connected_node_ids or node_id in timed_out_node_ids:
                dying.add(instance_id)
                break
    return dying


def orphaned_task_failure_events(
    state: State,
    dying_instance_ids: AbstractSet[InstanceId],
) -> list[TaskFailed]:
    """Fail in-flight API tasks whose instance is gone or being torn down.

    Without this, a node death mid-generation leaves the task in state
    forever and the API's chunk queue never receives a terminal chunk — the
    client request hangs until its own timeout (issue #223). The master is
    the only component with the global view to declare these tasks dead.

    Pure function of the master's current state so it can be tested without
    channel plumbing; ``dying_instance_ids`` covers instances whose
    InstanceDeleted was emitted in the same plan pass (state still lists
    them until the event round-trips through indexing and apply).
    """
    events: list[TaskFailed] = []
    for task_id, task in state.tasks.items():
        if not isinstance(task, _COMMAND_TASK_TYPES):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue
        instance_gone = (
            task.instance_id not in state.instances
            or task.instance_id in dying_instance_ids
        )
        if not instance_gone:
            continue
        events.append(
            TaskFailed(
                task_id=task_id,
                error_type="instance_lost",
                error_message=(
                    "The instance executing this request was lost "
                    "(node disconnected or instance deleted)"
                ),
            )
        )
    return events


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        state_sync_receiver: Receiver[StateSyncMessage],
        state_sync_sender: Sender[StateSyncMessage],
        download_command_sender: Sender[ForwarderDownloadCommand],
        snapshot_event_cadence: int = SNAPSHOT_EVENT_CADENCE,
        initial_state: State | None = None,
        telemetry_view: TelemetryView | None = None,
    ):
        self.node_id = node_id
        self.session_id = session_id
        # Live node telemetry off the event log (#279). Node-owned so it
        # survives this master's election: a freshly promoted master keeps the
        # cluster's current node_resources instead of starting blind and
        # risking a placement on a management node. None only in tests/standalone
        # construction; the planner falls back to "no telemetry constraints".
        self._telemetry_view = (
            telemetry_view if telemetry_view is not None else TelemetryView()
        )
        # A promoted master seeds its session from the node's prior
        # replicated state (shared/session_carryover.py) so placements
        # survive failover (#273) — previously every new session started
        # empty, the empty snapshot propagated, and every worker shut down
        # its healthy runners (a full serving outage from one master
        # restart). The seed is indexed as the FIRST EVENT of the new
        # session in run() (see _index_seed_event) rather than assigned
        # here: a pre-seeded snapshot at idx -1 is indistinguishable from
        # "fresh empty state" to the event router, which deliberately skips
        # hydration for idx < 0 — making the seed an ordinary logged event
        # gives every consumer exactly one delivery path. A genuinely fresh
        # node (cold start, or a rebooted node winning election before ever
        # hydrating) passes None and starts empty exactly as before — a
        # stale-boot winner cannot resurrect a cluster view it does not
        # have.
        self._seed_state = initial_state
        self.state = State(tracing_enabled=SKULK_TRACING_ENABLED)
        self._started_monotonic = time.monotonic()
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.state_sync_receiver = state_sync_receiver
        self.state_sync_sender = state_sync_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(SKULK_EVENT_LOG_DIR / "master")
        self._snapshot_store = StateSnapshotStore(
            SKULK_EVENT_LOG_DIR / "master" / "snapshots"
        )
        self._snapshot_event_cadence = snapshot_event_cadence
        self._last_snapshot_idx = -1
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}

    def _configure_expected_trace_ranks(
        self, task_id: TaskId, instance_id: InstanceId, *, trace_enabled: bool
    ) -> None:
        """Track which device ranks must report traces for a newly traced task."""

        if not trace_enabled:
            return

        selected_instance = self.state.instances.get(instance_id)
        if selected_instance is None:
            logger.warning(
                f"Unable to configure trace ranks for task {task_id}; instance {instance_id} not found"
            )
            return

        self._expected_ranks[task_id] = {
            shard.device_rank
            for shard in selected_instance.shard_assignments.runner_to_shard.values()
        }

    async def _index_seed_event(self) -> None:
        """Index the failover seed as the first event of this session (#273).

        Making the carried state an ordinary logged ``StateSnapshotHydrated``
        event gives every consumer exactly one delivery path: followers that
        snapshot-bootstrap after this point receive it inside the snapshot;
        followers that bootstrapped against the momentarily-empty state (the
        promotion race — including this node's own worker) receive it as the
        live event at index 0 and apply it like any other event. A seeded
        snapshot at idx ``-1`` instead looked identical to "fresh empty
        state", which the event router deliberately skips hydrating — the
        first live deployment of the seed lost it to exactly that race on
        the promoted node while a later-bootstrapping follower kept it.
        """
        if self._seed_state is None:
            return
        idx = len(self._event_log)
        seed = self._seed_state.model_copy(update={"last_event_applied_idx": idx})
        # Release the pre-index reference so the seed's object graph can be
        # collected as state evolves past it.
        self._seed_state = None
        indexed = IndexedEvent(event=StateSnapshotHydrated(state=seed), idx=idx)
        self.state = apply(self.state, indexed)
        self._event_log.append(indexed.event)
        await self._send_event(indexed)
        logger.info(
            f"Indexed failover seed as event {idx}: "
            f"{len(seed.instances)} carried instance(s)"
        )

    async def run(self):
        logger.info("Starting Master")

        try:
            await self._index_seed_event()
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._state_sync_processor)
                tg.start_soon(self._plan)
        finally:
            await self._persist_snapshot(force=True)
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()
            self.state_sync_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                            self._configure_expected_trace_ranks(
                                task_id,
                                selected_instance_id,
                                trace_enabled=trace_enabled,
                            )
                        case ImageGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                            self._configure_expected_trace_ranks(
                                task_id,
                                selected_instance_id,
                                trace_enabled=trace_enabled,
                            )
                        case ImageEdits():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                            self._configure_expected_trace_ranks(
                                task_id,
                                selected_instance_id,
                                trace_enabled=trace_enabled,
                            )
                        case TextEmbedding():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextEmbeddingTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                            self._configure_expected_trace_ranks(
                                task_id,
                                selected_instance_id,
                                trace_enabled=trace_enabled,
                            )
                        case SetTracingEnabled():
                            generated_events.append(
                                TracingStateChanged(enabled=command.enabled)
                            )
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case RefuseInstancePlacement():
                            # A worker could not fit its shard at load time
                            # (#290). Delete the refused instance and re-place
                            # the model one node wider so each node holds a
                            # smaller share. If even a full-width split will not
                            # fit, place_instance raises PlacementError and we
                            # stop at the deletion — that terminal case bounds
                            # the refuse→re-place loop to the cluster size.
                            refused = self.state.instances.get(command.instance_id)
                            if refused is None:
                                # Already gone (operator delete, redelivery, or
                                # a prior refusal for the same instance) — no-op.
                                logger.info(
                                    "RefuseInstancePlacement for unknown instance "
                                    f"{command.instance_id}; ignoring"
                                )
                            else:
                                after_delete = delete_instance(
                                    DeleteInstance(instance_id=command.instance_id),
                                    self.state.instances,
                                )
                                replace_command = (
                                    replacement_command_for_refused_instance(refused)
                                )
                                try:
                                    final_placement = place_instance(
                                        replace_command,
                                        self.state.topology,
                                        after_delete,
                                        self._telemetry_view.node_memory,
                                        self.state.node_network,
                                        download_status=self.state.downloads,
                                        node_resources=self._telemetry_view.node_resources,
                                    )
                                    logger.warning(
                                        "Re-placing "
                                        f"{replace_command.model_card.model_id} at "
                                        f"min_nodes={replace_command.min_nodes} after "
                                        f"{command.node_id} refused its shard "
                                        f"({command.reason})"
                                    )
                                except PlacementError as err:
                                    final_placement = after_delete
                                    logger.error(
                                        "Cannot re-place "
                                        f"{replace_command.model_card.model_id} after "
                                        f"refusal on {command.node_id} (tried "
                                        f"min_nodes={replace_command.min_nodes}): {err}. "
                                        "Giving up on this placement."
                                    )
                                transition_events = get_transition_events(
                                    self.state.instances,
                                    final_placement,
                                    self.state.tasks,
                                )
                                for cmd in cancel_unnecessary_downloads(
                                    final_placement, self.state.downloads
                                ):
                                    await self.download_command_sender.send(
                                        ForwarderDownloadCommand(
                                            origin=self._system_id, command=cmd
                                        )
                                    )
                                generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                # node_memory now lives on the telemetry plane
                                # (#279 slice 2), not in event-sourced State.
                                self._telemetry_view.node_memory,
                                self.state.node_network,
                                download_status=self.state.downloads,
                                excluded_nodes=set(command.excluded_nodes),
                                node_resources=self._telemetry_view.node_resources,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                                # telemetry plane (#279 slice 2) — stamp the
                                # memory-derived ceiling on exact placements too
                                self._telemetry_view.node_memory,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            if (
                                task_id := self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                            ) is not None:
                                generated_events.append(TaskDeleted(task_id=task_id))
                            else:
                                logger.warning(
                                    f"Finished command {command.finished_command_id} finished"
                                )

                        case AddCustomModelCard():
                            generated_events.append(
                                CustomModelCardAdded(model_card=command.model_card)
                            )
                        case DeleteCustomModelCard():
                            generated_events.append(
                                CustomModelCardDeleted(model_id=command.model_id)
                            )
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            # Large sessions can take many minutes to replay at 1k events per request,
                            # which leaves freshly joined nodes with an incomplete topology view.
                            replay_start = max(
                                command.since_idx,
                                self._event_log.start_idx,
                            )
                            if replay_start != command.since_idx:
                                logger.warning(
                                    "Requested replay index predates retained master tail; "
                                    f"serving from {replay_start} instead of {command.since_idx}"
                                )
                            end = min(
                                replay_start + EVENT_LOG_REPLAY_BATCH_SIZE,
                                len(self._event_log),
                            )
                            for i, event in enumerate(
                                self._event_log.read_range(replay_start, end),
                                start=replay_start,
                            ):
                                await self._send_event(IndexedEvent(idx=i, event=event))
                    for event in generated_events:
                        await self.event_sender.send(event)
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            connected_node_ids = set(self.state.topology.list_nodes())
            now = datetime.now(tz=timezone.utc)
            # ALL liveness-based action is suppressed while this session's
            # topology is still settling (#273): a failover-seeded master
            # carries instances but rebuilds topology and last_seen from
            # live gossip, so for the first probe cycles every carried node
            # looks "disconnected" — acting on that view would delete
            # exactly the placements the seed preserved. The suppression
            # must cover timed_out_node_ids too, not just the instance
            # pruning: NodeTimedOut's apply removes the node's instances
            # AND their tasks outright, and the TaskFailed-before-removal
            # invariant (#223/#224) is only upheld when the corresponding
            # dying_instance_ids pass ran in the same tick — a NodeTimedOut
            # emitted during the grace would strand in-flight API requests
            # without a terminal chunk (review catch on #274). After the
            # grace, absence means absence and cleanup proceeds normally.
            topology_settled = (
                time.monotonic() - self._started_monotonic
                >= TOPOLOGY_SETTLE_GRACE_SECONDS
            )
            timed_out_node_ids: set[NodeId] = (
                {
                    node_id
                    for node_id, last_seen_at in self.state.last_seen.items()
                    if now - last_seen_at > timedelta(seconds=30)
                }
                if topology_settled
                else set()
            )
            dying_instance_ids: set[InstanceId] = (
                instances_on_dead_nodes(
                    self.state, connected_node_ids, timed_out_node_ids
                )
                if topology_settled
                else set()
            )

            # Fail in-flight API tasks stranded by a dead or dying instance
            # so open HTTP requests terminate with an error instead of
            # hanging (#223). Emitted BEFORE InstanceDeleted/NodeTimedOut so
            # TaskFailed indexes ahead of the applies that remove the task
            # from state (NodeTimedOut deletes its instances' tasks
            # outright). TaskFailed flips task_status to Failed on apply, so
            # each task is emitted at most once across passes.
            for task_failed in orphaned_task_failure_events(
                self.state, dying_instance_ids
            ):
                logger.warning(
                    f"Failing orphaned task {task_failed.task_id}: "
                    f"{task_failed.error_message}"
                )
                await self.event_sender.send(task_failed)

            # kill broken instances (suppressed during the topology-settle
            # grace, same rationale as dying_instance_ids above)
            if topology_settled:
                for instance_id, instance in self.state.instances.items():
                    for node_id in instance.shard_assignments.node_to_runner:
                        if node_id not in connected_node_ids:
                            await self.event_sender.send(
                                InstanceDeleted(instance_id=instance_id)
                            )
                            break

            # time out dead nodes
            for node_id in timed_out_node_ids:
                logger.info(f"Manually removing node {node_id} due to inactivity")
                await self.event_sender.send(NodeTimedOut(node_id=node_id))

            await anyio.sleep(10)

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    if isinstance(event, TaskDeleted):
                        for command_id, task_id in list(
                            self.command_task_mapping.items()
                        ):
                            if task_id == event.task_id:
                                self.command_task_mapping.pop(command_id, None)

                    # Refuse to index task-lifecycle events that are state
                    # no-ops (the task is already gone). Without this cap a
                    # single misbehaving emitter could mint unbounded
                    # status/delete events for dead tasks — each one indexed,
                    # persisted, and broadcast cluster-wide — drowning
                    # replicas and starving liveness into election churn
                    # (#278; observed at ~800 events/s, 12k+ events for one
                    # task). Ordering makes this safe: TaskCreated is always
                    # indexed before any follower can reference the task, so
                    # an unknown task_id here is necessarily stale. The
                    # command-mapping sweep above still runs — it is
                    # in-memory hygiene, not amplification.
                    if (
                        isinstance(event, (TaskStatusUpdated, TaskDeleted, TaskFailed))
                        and event.task_id not in self.state.tasks
                    ):
                        logger.debug(
                            f"Dropping no-op task event for unknown task: "
                            f"{type(event).__name__}({event.task_id})"
                        )
                        continue

                    logger.debug(f"Master indexing event: {str(event)[:100]}")
                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]
                    if isinstance(event, NodeGatheredInfo):
                        event.when = str(datetime.now(tz=timezone.utc))

                    self._event_log.append(event)
                    await self._send_event(indexed)
                    await self._persist_snapshot()

    # This function is re-entrant, take care!
    async def _send_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self.event_sender.send(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]

    async def _state_sync_processor(self) -> None:
        with self.state_sync_receiver as messages:
            async for message in messages:
                if message.kind != "request":
                    continue
                if message.session_id != self.session_id:
                    continue

                config_yaml = self._load_state_sync_config_yaml()
                logger.info(
                    f"Serving state snapshot to {message.requester}: "
                    f"{len(self.state.instances)} instance(s), "
                    f"last_event_applied_idx={self.state.last_event_applied_idx}"
                )
                await self.state_sync_sender.send(
                    StateSyncMessage(
                        kind="response",
                        requester=message.requester,
                        session_id=self.session_id,
                        snapshot=StateSnapshot(
                            session_id=self.session_id,
                            last_event_applied_idx=self.state.last_event_applied_idx,
                            state=self.state,
                        ),
                        config_yaml=config_yaml,
                    )
                )

    def _load_state_sync_config_yaml(self) -> str | None:
        """Return a sanitized config payload for bootstrap responses.

        State-sync responses travel over cluster pub/sub, so they must never
        include secrets such as ``hf_token``. Read/parse failures are treated
        as non-fatal so bootstrap requests cannot crash master coordination.
        """

        config_path = resolve_config_path()
        if not config_path.exists():
            return None

        try:
            decoded_config = cast(object, yaml.safe_load(config_path.read_text()))
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "Failed to read local config for state-sync response"
            )
            return None

        if decoded_config is None:
            return None

        if not isinstance(decoded_config, dict):
            logger.warning(
                "Ignoring non-object config while preparing state-sync response"
            )
            return None

        raw_config = cast(dict[object, object], decoded_config)
        sanitized_config: JsonObject = {
            str(key): copy.deepcopy(value) for key, value in raw_config.items()
        }
        sanitized_config.pop("hf_token", None)
        return yaml.safe_dump(
            sanitized_config,
            default_flow_style=False,
            sort_keys=False,
        )

    async def _persist_snapshot(self, force: bool = False) -> None:
        snapshot_idx = self.state.last_event_applied_idx
        if snapshot_idx < 0:
            return
        if not force and (
            snapshot_idx - self._last_snapshot_idx < self._snapshot_event_cadence
        ):
            return
        if snapshot_idx == self._last_snapshot_idx:
            return

        snapshot = StateSnapshot(
            session_id=self.session_id,
            last_event_applied_idx=snapshot_idx,
            state=self.state,
        )
        try:
            self._snapshot_store.write(snapshot)
        except Exception as exc:
            logger.opt(exception=exc).warning("Failed to persist state snapshot")
            return

        # Keep a bounded overlap after the latest durable snapshot so a
        # follower that bootstrapped from a recently served snapshot can still
        # replay the missing tail even if another snapshot is persisted before
        # its replay request arrives.
        keep_from_idx = max(
            snapshot.last_event_applied_idx + 1 - REPLAY_TAIL_RETENTION_EVENTS,
            0,
        )
        self._event_log.compact(keep_from_idx)
        self._last_snapshot_idx = snapshot.last_event_applied_idx
