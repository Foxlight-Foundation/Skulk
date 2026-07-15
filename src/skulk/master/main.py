import copy
import time
from collections import deque
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import cast, final

import anyio
import yaml
from loguru import logger

from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    fallback_command_for_refused_instance,
    get_transition_events,
    place_instance,
    replacement_command_for_download_failed_instance,
    replacement_command_for_refused_instance,
)
from skulk.master.placement_utils import usable_vram_by_node
from skulk.shared.apply import apply
from skulk.shared.constants import SKULK_EVENT_LOG_DIR, SKULK_TRACING_ENABLED
from skulk.shared.models.memory_estimate import (
    estimate_shard_footprint,
    shard_fraction_of_model,
)
from skulk.shared.types.chunks import AudioInputChunk, InputImageChunk
from skulk.shared.types.commands import (
    AddCustomModelCard,
    AudioTranscription,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    EvictStagedModel,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RealtimeAudioTranscription,
    RefuseInstancePlacement,
    RequestEventLog,
    SendInputChunk,
    SetTracingEnabled,
    SpeechSynthesis,
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
    NodeDownloadProgress,
    NodeGatheredInfo,
    NodeTimedOut,
    NodeTimeoutEvidence,
    StagedModelEvicted,
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
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSnapshot, StateSyncMessage
from skulk.shared.types.tasks import (
    AudioTranscription as AudioTranscriptionTask,
)
from skulk.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from skulk.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from skulk.shared.types.tasks import (
    RealtimeAudioTranscription as RealtimeAudioTranscriptionTask,
)
from skulk.shared.types.tasks import (
    SpeechSynthesis as SpeechSynthesisTask,
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
from skulk.shared.types.telemetry import TelemetryView, record_membership_from_event
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import Instance, InstanceId
from skulk.shared.types.worker.runners import RunnerReady, RunnerRunning
from skulk.shared.types.worker.shards import RpcDonorShardMetadata
from skulk.store.config import resolve_config_path
from skulk.utils.channels import Receiver, Sender
from skulk.utils.disk_event_log import DiskEventLog
from skulk.utils.event_buffer import MultiSourceBuffer
from skulk.utils.state_snapshot_store import StateSnapshotStore
from skulk.utils.task_group import TaskGroup

EVENT_LOG_REPLAY_BATCH_SIZE = 10_000
EVENT_LOG_REPLAY_CHUNK_SIZE = 32
EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS = 0.25
SNAPSHOT_EVENT_CADENCE = 10_000
REPLAY_TAIL_RETENTION_EVENTS = SNAPSHOT_EVENT_CADENCE
EVENT_LOG_GROWTH_WINDOW_SECONDS = 60.0
EVENT_LOG_IDLE_GROWTH_WARNING_EVENTS_PER_MINUTE = 60.0
EVENT_LOG_GROWTH_WARNING_COOLDOWN_SECONDS = 300.0


@final
@dataclass(slots=True)
class EventLogGrowthMonitor:
    """Detect sustained event-log growth while the master has no active work."""

    window_seconds: float = EVENT_LOG_GROWTH_WINDOW_SECONDS
    warning_rate_per_minute: float = EVENT_LOG_IDLE_GROWTH_WARNING_EVENTS_PER_MINUTE
    warning_cooldown_seconds: float = EVENT_LOG_GROWTH_WARNING_COOLDOWN_SECONDS
    _idle_event_times: deque[float] = field(default_factory=deque)
    _idle_since: float | None = None
    _last_warning_at: float | None = None

    def observe(self, *, now: float, idle: bool) -> float | None:
        """Record one indexed event and return its warning rate when elevated.

        Active placement, download, and inference work resets the idle window so
        legitimate lifecycle bursts cannot prime a later warning. The returned
        rate is limited by ``warning_cooldown_seconds``; callers may log it or
        expose it through operator diagnostics without retaining event payloads.
        """

        if not idle:
            self._idle_event_times.clear()
            self._idle_since = None
            return None

        if self._idle_since is None:
            self._idle_since = now
        self._idle_event_times.append(now)
        cutoff = now - self.window_seconds
        while self._idle_event_times and self._idle_event_times[0] < cutoff:
            self._idle_event_times.popleft()

        observed_seconds = now - self._idle_since
        if observed_seconds < self.window_seconds:
            return None
        if (
            self._last_warning_at is not None
            and now - self._last_warning_at < self.warning_cooldown_seconds
        ):
            return None

        rate = len(self._idle_event_times) * 60.0 / self.window_seconds
        if rate < self.warning_rate_per_minute:
            return None
        self._last_warning_at = now
        return rate

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

RECENTLY_FREED_MEMORY_GRACE_SECONDS = 0.0
"""How long the planner credits just-deleted instance memory during placement.

This is intentionally disabled by default. The old 30s credit bridged gossip
lag after teardown, but live MLX runs showed the inverse failure: Metal memory
can remain unwired locally after the master has deleted the instance, so
crediting the freed footprint over-admits the next placement and leaves the
worker to refuse it. Placement must prefer the observed telemetry over an
optimistic teardown credit; the worker's local guard should be a last resort,
not the normal correction path."""
JsonObject = dict[str, object]

# API-facing task types: the ones whose loss strands an open HTTP request.
# Worker lifecycle tasks (CreateRunner, LoadModel, ...) are reconciled by the
# worker's own plan loop and must not be failed from here.
_COMMAND_TASK_TYPES = (
    TextGenerationTask,
    ImageGenerationTask,
    ImageEditsTask,
    TextEmbeddingTask,
    SpeechSynthesisTask,
    AudioTranscriptionTask,
    RealtimeAudioTranscriptionTask,
)


NODE_LIVENESS_TIMEOUT = timedelta(seconds=30)
NODE_HEARTBEAT_GAP_WARNING = timedelta(seconds=10)


def _aware_timestamp(when: datetime) -> datetime:
    """Return a timestamp that is safe to compare with UTC receipt times."""
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _signal_age_seconds(*, now: datetime, seen_at: datetime) -> float:
    """Return a non-negative signal age, defensively tolerating clock drift."""
    return max(
        0.0,
        (_aware_timestamp(now) - _aware_timestamp(seen_at)).total_seconds(),
    )


def compute_node_timeout_evidence(
    last_seen: Mapping[NodeId, datetime],
    heartbeat_last_seen: Mapping[NodeId, datetime],
    telemetry_last_seen: Mapping[NodeId, datetime],
    *,
    now: datetime,
    timeout: timedelta = NODE_LIVENESS_TIMEOUT,
) -> dict[NodeId, NodeTimeoutEvidence]:
    """Build reproducible liveness evidence for every event-known node.

    Args:
        last_seen: Last indexed control-plane event per node.
        heartbeat_last_seen: Local receipt time of each dedicated heartbeat.
        telemetry_last_seen: Local receipt time of ordinary fallback telemetry.
        now: Decision wall clock, injected for deterministic tests.
        timeout: Configured node-prune timeout.

    Returns:
        Evidence keyed by every node represented in ``last_seen``.
    """
    evidence: dict[NodeId, NodeTimeoutEvidence] = {}
    for node_id, last_logged_event_at in last_seen.items():
        last_logged_event_age = _signal_age_seconds(
            now=now, seen_at=last_logged_event_at
        )
        heartbeat_at = heartbeat_last_seen.get(node_id)
        heartbeat_age = (
            _signal_age_seconds(now=now, seen_at=heartbeat_at)
            if heartbeat_at is not None
            else None
        )
        fallback_telemetry_at = telemetry_last_seen.get(node_id)
        fallback_telemetry_age = (
            _signal_age_seconds(now=now, seen_at=fallback_telemetry_at)
            if fallback_telemetry_at is not None
            else None
        )
        available_ages = [last_logged_event_age]
        if heartbeat_age is not None:
            available_ages.append(heartbeat_age)
        if fallback_telemetry_age is not None:
            available_ages.append(fallback_telemetry_age)
        evidence[node_id] = NodeTimeoutEvidence(
            last_logged_event_age_seconds=last_logged_event_age,
            heartbeat_age_seconds=heartbeat_age,
            fallback_telemetry_age_seconds=fallback_telemetry_age,
            effective_age_seconds=min(available_ages),
            timeout_seconds=timeout.total_seconds(),
        )
    return evidence


def _timed_out_nodes_from_evidence(
    evidence: Mapping[NodeId, NodeTimeoutEvidence],
) -> set[NodeId]:
    """Select nodes whose freshest liveness signal exceeds the timeout."""
    return {
        node_id
        for node_id, observation in evidence.items()
        if observation.effective_age_seconds > observation.timeout_seconds
    }


def compute_timed_out_nodes(
    last_seen: Mapping[NodeId, datetime],
    heartbeat_last_seen: Mapping[NodeId, datetime],
    telemetry_last_seen: Mapping[NodeId, datetime],
    *,
    now: datetime,
    timeout: timedelta = NODE_LIVENESS_TIMEOUT,
) -> set[NodeId]:
    """Return nodes whose event, heartbeat, and telemetry signals are stale.

    Args:
        last_seen: ``State.last_seen`` (last logged event per node).
        heartbeat_last_seen: Local receipt time of each dedicated heartbeat.
        telemetry_last_seen: Local receipt time of ordinary fallback telemetry.
        now: The wall clock (tz-aware); injected for testing.
        timeout: Staleness past which a node is considered gone.

    Returns:
        Node ids to time out.
    """
    return _timed_out_nodes_from_evidence(
        compute_node_timeout_evidence(
            last_seen,
            heartbeat_last_seen,
            telemetry_last_seen,
            now=now,
            timeout=timeout,
        )
    )


def compute_heartbeat_gap_nodes(
    last_seen: Mapping[NodeId, datetime],
    heartbeat_last_seen: Mapping[NodeId, datetime],
    *,
    now: datetime,
    warning_after: timedelta = NODE_HEARTBEAT_GAP_WARNING,
) -> set[NodeId]:
    """Return live members whose dedicated heartbeat is late or absent.

    Before a node's first heartbeat, its last indexed event anchors the warning
    window so a new member gets a full grace period without making a missing
    heartbeat invisible forever.
    """
    return {
        node_id
        for node_id, last_logged_event_at in last_seen.items()
        if _signal_age_seconds(
            now=now,
            seen_at=heartbeat_last_seen.get(node_id, last_logged_event_at),
        )
        >= warning_after.total_seconds()
    }


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


def instances_wedged_by_download_failure(
    state: State,
) -> dict[InstanceId, tuple[frozenset[NodeId], str]]:
    """Not-yet-ready instances that can never load because a rank's download failed.

    A multi-node instance forms its ring and every rank waits for all ranks to
    become load-ready. If one rank's model download terminally fails (disk full,
    transient HF or network error) that rank never advances, so the whole
    instance sits at ``RunnerConnected`` forever with nothing to fail or recover
    it (#381). This detects exactly that wedge from the master's own state: an
    instance not all-ready whose any rank node carries a terminal
    ``DownloadFailed`` for the instance's model. Returns, per wedged instance,
    the failed node id(s) and a human-readable cause. Pure for testability.

    A fully-ready instance (all runners ``RunnerReady``/``RunnerRunning``) is
    never reported even if a stale ``DownloadFailed`` lingers in state, so a
    serving instance is never torn down by this path.
    """
    wedged: dict[InstanceId, tuple[frozenset[NodeId], str]] = {}
    for instance_id, instance in state.instances.items():
        runner_ids = list(instance.shard_assignments.runner_to_shard.keys())
        if not runner_ids:
            continue
        all_ready = all(
            isinstance(state.runners.get(runner_id), (RunnerReady, RunnerRunning))
            for runner_id in runner_ids
        )
        if all_ready:
            continue
        shards = list(instance.shard_assignments.runner_to_shard.values())
        model_id = shards[0].model_card.model_id
        failed_nodes: set[NodeId] = set()
        cause = ""
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            # RPC donors never download the model, so a stale DownloadFailed
            # for this model on a donor node (from an earlier placement) must
            # not condemn a pooled instance that only needs the DRIVER's copy.
            shard = instance.shard_assignments.runner_to_shard.get(runner_id)
            if isinstance(shard, RpcDonorShardMetadata):
                continue
            for progress in state.downloads.get(node_id, []):
                if (
                    isinstance(progress, DownloadFailed)
                    and progress.shard_metadata.model_card.model_id == model_id
                ):
                    failed_nodes.add(node_id)
                    cause = progress.error_message
        if failed_nodes:
            wedged[instance_id] = (frozenset(failed_nodes), cause)
    return wedged


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
        self._realtime_instance_by_command: dict[CommandId, InstanceId] = {}
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
        self._pending_replay_start_idx: int | None = None
        self._replay_worker_running = False
        self._active_replay_next_idx: int | None = None
        self._active_replay_end_idx: int | None = None
        self._event_log_growth_monitor = EventLogGrowthMonitor()
        # Nodes with an active dedicated-heartbeat gap warning. Tracking the
        # transition makes a 10-second planning loop emit one warning and one
        # recovery message instead of repeating the same warning indefinitely.
        self._heartbeat_gap_warned_nodes: set[NodeId] = set()
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}
        # Instance ids whose memory-refusal re-placement has already been
        # initiated (#290). The command processor generates events but does not
        # apply them — self.state only updates when they round-trip through
        # _event_processor — so two ranks of the same instance refusing in the
        # same window would both still see the instance present and each spawn a
        # wider replacement. Deduping on the refused id makes re-placement
        # happen at most once per instance regardless of how many ranks refuse
        # or of redelivery. Grows only by refused ids (rare); never reused since
        # InstanceIds are unique.
        self._refusal_replaced: set[InstanceId] = set()
        # Instances minted by the anywhere-but-the-refuser FALLBACK (second
        # recovery hop). A refusal against one of these is TERMINAL: without
        # this, exclusions are lost across recovery cycles (the replacement
        # command is rebuilt from shard assignments alone) and a tight fleet
        # oscillates -- A refuses full width, fallback lands on B/C, B refuses,
        # the wider re-place returns to A at the share it already refused.
        # Two hops per original placement bounds recovery; ids are unique and
        # rare, matching _refusal_replaced's growth rationale.
        self._fallback_placed_instances: set[InstanceId] = set()
        # Instance ids whose download-failure recovery has already been initiated
        # (#381), same dedup rationale as _refusal_replaced: the plan pass emits
        # events but does not apply them, so the wedged instance stays visible
        # for several ticks until InstanceDeleted round-trips. Deduping on the
        # id makes recovery fire once per wedged instance. Grows only by wedged
        # ids (rare); never reused since InstanceIds are unique.
        self._download_failure_recovered: set[InstanceId] = set()
        # Per-node memory (bytes) freed by a just-deleted instance. The grace
        # window is zero by default, so entries are normally pruned without being
        # applied; keeping the structure preserves one place to revisit this if
        # we later have a shutdown-complete signal that proves memory recovered.
        self._recently_freed_bytes: dict[NodeId, list[tuple[int, float]]] = {}

    def _effective_downloads(self) -> dict[NodeId, list[DownloadProgress]]:
        """Return durable outcomes overlaid with live download telemetry."""

        return self._telemetry_view.effective_downloads(self.state.downloads)

    def _apply_indexed_event(self, indexed: IndexedEvent) -> None:
        """Apply one durable event and synchronize the master's telemetry view."""

        self.state = apply(self.state, indexed)
        record_membership_from_event(self._telemetry_view, indexed.event)

    def _record_freed_instance(self, instance: Instance) -> None:
        """Record a deleted instance's per-node footprint for the grace window.

        The default grace is zero, which makes this a no-credit bookkeeping path.
        A future nonzero grace should be tied to a signal that the local worker
        has actually released the model memory, not merely to InstanceDeleted.
        """
        assignments = instance.shard_assignments
        deadline = time.monotonic() + RECENTLY_FREED_MEMORY_GRACE_SECONDS
        for node_id, runner_id in assignments.node_to_runner.items():
            shard = assignments.runner_to_shard.get(runner_id)
            if shard is None:
                continue
            fraction = shard_fraction_of_model(shard)
            if fraction is None or fraction <= 0.0:
                continue
            footprint = estimate_shard_footprint(shard.model_card, fraction)
            self._recently_freed_bytes.setdefault(node_id, []).append(
                (footprint.in_bytes, deadline)
            )

    def _freed_credit_by_node(self) -> dict[NodeId, int]:
        """Current non-expired freed-memory credit (bytes) per node, pruning the
        expired entries as a side effect (#314)."""
        now = time.monotonic()
        credit: dict[NodeId, int] = {}
        for node_id in list(self._recently_freed_bytes):
            live = [(b, d) for (b, d) in self._recently_freed_bytes[node_id] if d > now]
            if live:
                self._recently_freed_bytes[node_id] = live
                credit[node_id] = sum(b for (b, _) in live)
            else:
                del self._recently_freed_bytes[node_id]
        return credit

    def _placement_memory_inputs(
        self,
    ) -> tuple[
        Mapping[NodeId, MemoryUsage],
        Mapping[NodeId, Memory],
    ]:
        """Build the node-memory and usable-GPU placement inputs.

        Expired recently-freed entries are pruned before building the inputs.
        With the default zero grace, no speculative credit is applied: live MLX
        teardown can lag InstanceDeleted, and using observed telemetry produces
        placements whose concrete shard allocations match the worker's local
        pre-load guard.

        If an operator later enables a nonzero grace, both ``ram_available`` and
        the derived usable-GPU map are built from the credited memory snapshot;
        ``ram_total`` is never credited, so context-ceiling math stays anchored
        to physical capacity.
        """
        credit = self._freed_credit_by_node()
        base_memory = self._telemetry_view.node_memory
        if not credit:
            base_vram = usable_vram_by_node(
                self._telemetry_view.node_system,
                self._telemetry_view.node_resources,
                node_memory=base_memory,
            )
            return base_memory, base_vram
        # Credit the freed bytes onto each node's ram_available, clamped to
        # ram_total so credited availability never exceeds capacity (telemetry
        # may already have partly caught up, or the footprint estimate may be
        # conservative).
        memory = {
            node_id: (
                usage.model_copy(
                    update={
                        "ram_available": Memory.from_bytes(
                            min(
                                usage.ram_total.in_bytes,
                                usage.ram_available.in_bytes + credit[node_id],
                            )
                        )
                    }
                )
                if node_id in credit
                else usage
            )
            for node_id, usage in base_memory.items()
        }
        # Derive VRAM from the credited memory rather than crediting the VRAM
        # figure directly: usable_vram_by_node applies its own working-set /
        # GTT ceiling, so the credited VRAM is naturally capped and can never
        # exceed the ceiling or total VRAM.
        vram = usable_vram_by_node(
            self._telemetry_view.node_system,
            self._telemetry_view.node_resources,
            node_memory=memory,
        )
        return memory, vram

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
        self._apply_indexed_event(indexed)
        self._append_event_log(indexed.event)
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
                                        # Carry the owning API node onto the task
                                        # so the rank-0 supervisor can address
                                        # output over the Zenoh data plane (#279
                                        # Phase 2).
                                        owner_node=command.owner_node,
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
                                        owner_node=command.owner_node,  # #279 Phase 2
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
                                        owner_node=command.owner_node,  # #279 Phase 2
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
                                        owner_node=command.owner_node,  # #279 Phase 2
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
                        case SpeechSynthesis():
                            if command.task_params.reference_audio_data is not None:
                                raise ValueError(
                                    "Reference audio bytes must not enter commands or State"
                                )
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

                            if (
                                not instance_task_counts
                                and command.target_instance_id is None
                            ):
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            task_id = TaskId()
                            target_unavailable = False
                            if command.target_instance_id is not None:
                                if command.target_instance_id not in instance_task_counts:
                                    target_unavailable = True
                                selected_instance_id = command.target_instance_id
                            else:
                                available_instance_ids = sorted(
                                    instance_task_counts.keys(),
                                    key=lambda instance_id: instance_task_counts[
                                        instance_id
                                    ],
                                )
                                selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=SpeechSynthesisTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,  # #279 Phase 2
                                        instance_id=selected_instance_id,
                                        task_status=(
                                            TaskStatus.Failed
                                            if target_unavailable
                                            else TaskStatus.Pending
                                        ),
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                            if target_unavailable:
                                generated_events.append(
                                    TaskFailed(
                                        task_id=task_id,
                                        error_type="instance_unavailable",
                                        error_message=(
                                            "Requested TTS instance is unavailable or "
                                            "does not serve the requested model"
                                        ),
                                    )
                                )
                            else:
                                self._configure_expected_trace_ranks(
                                    task_id,
                                    selected_instance_id,
                                    trace_enabled=trace_enabled,
                                )
                        case AudioTranscription():
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
                                    task=AudioTranscriptionTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,  # #279 Phase 2
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
                        case RealtimeAudioTranscription():
                            instance = self.state.instances.get(
                                command.target_instance_id
                            )
                            if instance is None:
                                raise ValueError(
                                    "No target instance found for realtime STT "
                                    f"command {command.command_id}"
                                )
                            if (
                                instance.shard_assignments.model_id
                                != command.task_params.model
                            ):
                                raise ValueError(
                                    "Realtime STT target instance model does not "
                                    f"match {command.task_params.model}"
                                )
                            if len(instance.shard_assignments.node_to_runner) != 1:
                                raise ValueError(
                                    "Realtime STT requires a single-host target "
                                    f"instance, got {command.target_instance_id}"
                                )
                            instance_busy = command.target_instance_id in (
                                self._realtime_instance_by_command.values()
                            ) or any(
                                task.instance_id == command.target_instance_id
                                and task.task_status
                                in (TaskStatus.Pending, TaskStatus.Running)
                                for task in self.state.tasks.values()
                            )

                            task_id = TaskId()
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=RealtimeAudioTranscriptionTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,
                                        instance_id=command.target_instance_id,
                                        task_status=(
                                            TaskStatus.Failed
                                            if instance_busy
                                            else TaskStatus.Pending
                                        ),
                                        task_params=command.task_params,
                                        # Realtime STT does not emit trace
                                        # sessions yet. Do not register ranks
                                        # that can never report completion.
                                        trace_enabled=False,
                                    ),
                                )
                            )
                            self.command_task_mapping[command.command_id] = task_id
                            if instance_busy:
                                generated_events.append(
                                    TaskFailed(
                                        task_id=task_id,
                                        error_type="instance_busy",
                                        error_message=(
                                            "Realtime STT target instance is already "
                                            "reserved"
                                        ),
                                    )
                                )
                            else:
                                self._realtime_instance_by_command[
                                    command.command_id
                                ] = command.target_instance_id
                        case SetTracingEnabled():
                            generated_events.append(
                                TracingStateChanged(enabled=command.enabled)
                            )
                        case DeleteInstance():
                            # Credit the freed memory back to placement admission
                            # for a short grace window so a back-to-back placement
                            # is not refused on gossip-lagged node memory (#314).
                            deleted_instance = self.state.instances.get(
                                command.instance_id
                            )
                            if deleted_instance is not None:
                                self._record_freed_instance(deleted_instance)
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self._effective_downloads()
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
                            if (
                                command.instance_id
                                in self._fallback_placed_instances
                                and refused is not None
                            ):
                                # Second recovery hop already used: tear down
                                # and stop. Re-placing again would oscillate
                                # (see _fallback_placed_instances).
                                self._refusal_replaced.add(command.instance_id)
                                logger.error(
                                    "Fallback placement "
                                    f"{command.instance_id} was itself refused "
                                    f"on {command.node_id}; giving up on this "
                                    "placement (two recovery hops used)."
                                )
                                after_delete = delete_instance(
                                    DeleteInstance(
                                        instance_id=command.instance_id
                                    ),
                                    self.state.instances,
                                )
                                # Same download hygiene as every other delete
                                # path: a rank still mid-download for the
                                # torn-down instance must be cancelled, or it
                                # wastes bandwidth/disk and can re-trigger
                                # wedged-download recovery later.
                                for cancel_cmd in cancel_unnecessary_downloads(
                                    after_delete, self._effective_downloads()
                                ):
                                    await self.download_command_sender.send(
                                        ForwarderDownloadCommand(
                                            origin=self._system_id,
                                            command=cancel_cmd,
                                        )
                                    )
                                for event in get_transition_events(
                                    self.state.instances,
                                    after_delete,
                                    self.state.tasks,
                                ):
                                    await self.event_sender.send(event)
                            elif command.instance_id in self._refusal_replaced:
                                # Another rank of the same instance already
                                # triggered re-placement (self.state lags command
                                # processing, so both ranks can still see the
                                # instance present) — re-place exactly once.
                                logger.info(
                                    "RefuseInstancePlacement for "
                                    f"{command.instance_id} already handled; ignoring"
                                )
                            elif refused is None:
                                # Already gone (operator delete or redelivery) —
                                # no-op.
                                logger.info(
                                    "RefuseInstancePlacement for unknown instance "
                                    f"{command.instance_id}; ignoring"
                                )
                            else:
                                self._refusal_replaced.add(command.instance_id)
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
                                        download_status=self._effective_downloads(),
                                        node_resources=self._telemetry_view.node_resources,
                                        node_vram=usable_vram_by_node(
                                            self._telemetry_view.node_system,
                                            self._telemetry_view.node_resources,
                                            node_memory=self._telemetry_view.node_memory,
                                        ),
                                    )
                                    logger.warning(
                                        "Re-placing "
                                        f"{replace_command.model_card.model_id} at "
                                        f"min_nodes={replace_command.min_nodes} after "
                                        f"{command.node_id} refused its shard "
                                        f"({command.reason})"
                                    )
                                except PlacementInfoPendingError as err:
                                    # Telemetry for a needed node has not arrived
                                    # yet (rare during a load-time refusal, since
                                    # the cluster is already running). Distinct
                                    # from a true shortfall: the refused instance
                                    # still can't stay on the node that rejected
                                    # it, so it is torn down, but log the
                                    # transient cause rather than "cannot fit".
                                    final_placement = after_delete
                                    logger.error(
                                        "Cannot re-place "
                                        f"{replace_command.model_card.model_id} after "
                                        f"refusal on {command.node_id}: cluster info "
                                        f"still gossiping ({err}). Torn down."
                                    )
                                except PlacementError as err:
                                    # The wider width can be unsatisfiable by
                                    # construction on a heterogeneous fleet (an
                                    # MLX model refused at the full Mac width
                                    # cannot add an AMD node). Fall back to
                                    # anywhere-but-the-refuser at min_nodes=1:
                                    # the memory fit-check, not the width,
                                    # decides. Only a second failure is
                                    # terminal.
                                    fallback = fallback_command_for_refused_instance(
                                        refused, command.node_id
                                    )
                                    try:
                                        final_placement = place_instance(
                                            fallback,
                                            self.state.topology,
                                            after_delete,
                                            self._telemetry_view.node_memory,
                                            self.state.node_network,
                                            download_status=self._effective_downloads(),
                                            excluded_nodes={command.node_id},
                                            node_resources=self._telemetry_view.node_resources,
                                            node_vram=usable_vram_by_node(
                                                self._telemetry_view.node_system,
                                                self._telemetry_view.node_resources,
                                                node_memory=self._telemetry_view.node_memory,
                                            ),
                                        )
                                        for new_id in final_placement:
                                            if new_id not in after_delete:
                                                self._fallback_placed_instances.add(
                                                    new_id
                                                )
                                        logger.warning(
                                            "Re-placing "
                                            f"{fallback.model_card.model_id} "
                                            f"excluding refusing node "
                                            f"{command.node_id} (wider width "
                                            f"min_nodes={replace_command.min_nodes} "
                                            f"was unplaceable: {err})"
                                        )
                                    except (
                                        PlacementError,
                                        PlacementInfoPendingError,
                                    ) as fallback_err:
                                        final_placement = after_delete
                                        logger.error(
                                            "Cannot re-place "
                                            f"{replace_command.model_card.model_id} after "
                                            f"refusal on {command.node_id} (tried "
                                            f"min_nodes={replace_command.min_nodes}, then "
                                            f"excluding the refuser: {fallback_err}). "
                                            "Giving up on this placement."
                                        )
                                transition_events = get_transition_events(
                                    self.state.instances,
                                    final_placement,
                                    self.state.tasks,
                                )
                                for cmd in cancel_unnecessary_downloads(
                                    final_placement, self._effective_downloads()
                                ):
                                    await self.download_command_sender.send(
                                        ForwarderDownloadCommand(
                                            origin=self._system_id, command=cmd
                                        )
                                    )
                                generated_events.extend(transition_events)
                        case PlaceInstance():
                            # node_memory/node_vram come from the telemetry plane
                            # (#279 slice 2). Recently-freed credit is pruned
                            # here and disabled by default (#314); the usable-GPU
                            # map admits discrete/UMA GPU nodes against the pool
                            # their backend can actually allocate from.
                            credited_memory, credited_vram = (
                                self._placement_memory_inputs()
                            )
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                credited_memory,
                                self.state.node_network,
                                download_status=self._effective_downloads(),
                                excluded_nodes=set(command.excluded_nodes),
                                node_resources=self._telemetry_view.node_resources,
                                node_vram=credited_vram,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            # Placement inputs come from telemetry (#279 slice 2);
                            # the recently-freed bookkeeping path is pruned here
                            # and normally contributes no speculative credit.
                            credited_memory, credited_vram = (
                                self._placement_memory_inputs()
                            )
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                                credited_memory,
                                node_vram=credited_vram,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=AudioInputChunk() as chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case SendInputChunk():
                            logger.warning(
                                "Rejected legacy image input command; vision media "
                                "must use the node-addressed data transport"
                            )
                        case TaskCancelled():
                            self._realtime_instance_by_command.pop(
                                command.cancelled_command_id, None
                            )
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
                            self._realtime_instance_by_command.pop(
                                command.finished_command_id, None
                            )
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
                        case EvictStagedModel():
                            # Broadcast a fleet-wide eviction of the store-deleted
                            # model: apply() drops its download entries; workers
                            # remove their staged copies on disk (#427).
                            generated_events.append(
                                StagedModelEvicted(model_id=command.model_id)
                            )
                        case RequestEventLog():
                            self._schedule_event_log_replay(command.since_idx)
                    for event in generated_events:
                        await self.event_sender.send(event)
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    def _report_heartbeat_gap_changes(self, *, now: datetime) -> None:
        """Log dedicated-heartbeat degradation and recovery once per transition."""
        current_nodes = set(self.state.last_seen)
        gap_nodes = compute_heartbeat_gap_nodes(
            self.state.last_seen,
            self._telemetry_view.node_last_heartbeat,
            now=now,
        )
        # Once absence has produced a warning, only an actual heartbeat receipt
        # clears it. A later control event grants initial grace to a new node but
        # must not masquerade as heartbeat recovery.
        gap_nodes |= {
            node_id
            for node_id in self._heartbeat_gap_warned_nodes & current_nodes
            if node_id not in self._telemetry_view.node_last_heartbeat
        }
        observations = compute_node_timeout_evidence(
            self.state.last_seen,
            self._telemetry_view.node_last_heartbeat,
            self._telemetry_view.node_last_telemetry,
            now=now,
        )
        for node_id in sorted(
            gap_nodes - self._heartbeat_gap_warned_nodes, key=str
        ):
            evidence = observations[node_id]
            logger.bind(
                liveness_event="heartbeat_gap",
                node_id=str(node_id),
                heartbeat_age_seconds=evidence.heartbeat_age_seconds,
                fallback_telemetry_age_seconds=(
                    evidence.fallback_telemetry_age_seconds
                ),
                last_logged_event_age_seconds=(
                    evidence.last_logged_event_age_seconds
                ),
            ).warning(
                f"Dedicated heartbeat from node {node_id} is late or absent; "
                "ordinary telemetry and logged events remain liveness fallbacks"
            )
        recovered_nodes = (
            self._heartbeat_gap_warned_nodes - gap_nodes
        ) & current_nodes
        for node_id in sorted(recovered_nodes, key=str):
            heartbeat_at = self._telemetry_view.node_last_heartbeat[node_id]
            logger.bind(
                liveness_event="heartbeat_recovered",
                node_id=str(node_id),
                heartbeat_age_seconds=_signal_age_seconds(
                    now=now, seen_at=heartbeat_at
                ),
            ).info(f"Dedicated heartbeat from node {node_id} recovered")
        self._heartbeat_gap_warned_nodes = gap_nodes

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            connected_node_ids = set(self.state.topology.list_nodes())
            now = datetime.now(tz=timezone.utc)
            self._report_heartbeat_gap_changes(now=now)
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
            timeout_evidence = (
                compute_node_timeout_evidence(
                    self.state.last_seen,
                    self._telemetry_view.node_last_heartbeat,
                    self._telemetry_view.node_last_telemetry,
                    now=now,
                )
                if topology_settled
                else {}
            )
            timed_out_node_ids = _timed_out_nodes_from_evidence(timeout_evidence)
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
                evidence = timeout_evidence[node_id]
                logger.bind(
                    liveness_event="node_timed_out",
                    node_id=str(node_id),
                    last_logged_event_age_seconds=(
                        evidence.last_logged_event_age_seconds
                    ),
                    heartbeat_age_seconds=evidence.heartbeat_age_seconds,
                    fallback_telemetry_age_seconds=(
                        evidence.fallback_telemetry_age_seconds
                    ),
                    effective_age_seconds=evidence.effective_age_seconds,
                    timeout_seconds=evidence.timeout_seconds,
                ).warning(
                    f"Removing node {node_id}: all liveness signals exceeded "
                    f"the {evidence.timeout_seconds:.0f}s timeout"
                )
                await self.event_sender.send(
                    NodeTimedOut(node_id=node_id, evidence=evidence)
                )

            # Recover instances wedged because a rank's download failed (#381).
            # Suppressed during the settle grace for the same reason as the
            # liveness passes above: a freshly seeded master sees stale download
            # state until live gossip refreshes it. A terminal DownloadFailed on
            # a not-yet-ready instance can never clear on its own, so fail the
            # instance (surfacing the cause to any waiting request) and re-place
            # it excluding the failed node(s). Deduped so it acts once per wedged
            # instance, not every tick while the events round-trip.
            if topology_settled:
                await self._recover_download_failed_instances()

            await anyio.sleep(10)

    async def _recover_download_failed_instances(self) -> None:
        """Fail and re-place instances stuck because a rank's download failed (#381).

        See :func:`instances_wedged_by_download_failure`. For each newly-detected
        wedged instance: fail any in-flight API task bound to it (so an open
        request gets a clean error instead of hanging), tear the instance down,
        and re-place the model at the same width excluding the failed node(s),
        reusing the #290 placement machinery. If no healthy node set can host the
        width, placement raises and we stop at the teardown, which bounds
        recovery to the available nodes instead of looping.
        """
        wedged = instances_wedged_by_download_failure(self.state)
        for instance_id, (failed_nodes, cause) in wedged.items():
            if instance_id in self._download_failure_recovered:
                continue
            instance = self.state.instances.get(instance_id)
            if instance is None:
                continue
            self._download_failure_recovered.add(instance_id)
            failed_list = sorted(str(node_id) for node_id in failed_nodes)
            logger.error(
                f"Instance {instance_id} wedged: a rank's download failed on "
                f"{failed_list} ({cause}). Failing it and re-placing without the "
                "failed node(s)."
            )
            # Fail any in-flight API task bound to this instance before the
            # teardown events index (the TaskFailed-before-removal invariant,
            # #223): get_transition_events below emits InstanceDeleted, whose
            # apply drops the instance's tasks.
            for task_failed in orphaned_task_failure_events(self.state, {instance_id}):
                await self.event_sender.send(task_failed)
            after_delete = delete_instance(
                DeleteInstance(instance_id=instance_id), self.state.instances
            )
            final_placement = after_delete
            try:
                replace_command = replacement_command_for_download_failed_instance(
                    instance, failed_nodes
                )
                final_placement = place_instance(
                    replace_command,
                    self.state.topology,
                    after_delete,
                    self._telemetry_view.node_memory,
                    self.state.node_network,
                    download_status=self._effective_downloads(),
                    excluded_nodes=set(failed_nodes),
                    node_resources=self._telemetry_view.node_resources,
                    node_vram=usable_vram_by_node(
                        self._telemetry_view.node_system,
                        self._telemetry_view.node_resources,
                        node_memory=self._telemetry_view.node_memory,
                    ),
                )
                logger.warning(
                    f"Re-placing {replace_command.model_card.model_id} excluding "
                    f"{failed_list} after a download failure"
                )
            except (PlacementError, PlacementInfoPendingError) as err:
                final_placement = after_delete
                logger.error(
                    f"Cannot re-place {instance_id}'s model after a download "
                    f"failure on {failed_list}: {err}. Torn down."
                )
            transition_events = get_transition_events(
                self.state.instances, final_placement, self.state.tasks
            )
            for cmd in cancel_unnecessary_downloads(
                final_placement, self._effective_downloads()
            ):
                await self.download_command_sender.send(
                    ForwarderDownloadCommand(origin=self._system_id, command=cmd)
                )
            for event in transition_events:
                await self.event_sender.send(event)
            # Recovery CONSUMES the terminal failure record: reset each failed
            # node's download status for this model back to Pending. Without
            # this, the stale DownloadFailed lingers in session state and this
            # same scan condemns EVERY future placement of the model that
            # touches the node, long after the cause (disk full, network blip)
            # is gone -- one transient failure permanently poisoned the model
            # on that node until a whole-fleet restart (observed live: an
            # ENOSPC during a pooled placement kept killing fresh placements
            # an hour after the disk was freed). This recovery pass already
            # acted on the failure (teardown + re-place excluding the node);
            # if the cause persists, the next download fails afresh and
            # recovery repeats, so nothing is lost by clearing history.
            for node_id in failed_nodes:
                runner_id = instance.shard_assignments.node_to_runner.get(node_id)
                shard = (
                    instance.shard_assignments.runner_to_shard.get(runner_id)
                    if runner_id is not None
                    else None
                )
                if shard is None:
                    continue
                await self.event_sender.send(
                    NodeDownloadProgress(
                        download_progress=DownloadPending(
                            node_id=node_id,
                            shard_metadata=shard,
                            model_directory="",
                            attempt_id=DownloadAttemptId(),
                        )
                    )
                )
                logger.info(
                    f"Reset stale DownloadFailed for "
                    f"{shard.model_card.model_id} on {node_id} (consumed by "
                    "recovery)"
                )

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                if isinstance(local_event.event, InputChunkReceived) and isinstance(
                    local_event.event.chunk, InputImageChunk
                ):
                    logger.warning(
                        "Rejected legacy image input event before ordering/indexing; "
                        "vision media must use the node-addressed data transport"
                    )
                    self._multi_buffer.skip(
                        local_event.origin_idx, local_event.origin
                    )
                else:
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
                                self._realtime_instance_by_command.pop(
                                    command_id, None
                                )

                    if isinstance(event, TaskFailed) or (
                        isinstance(event, TaskStatusUpdated)
                        and event.task_status
                        not in (TaskStatus.Pending, TaskStatus.Running)
                    ):
                        for command_id, task_id in self.command_task_mapping.items():
                            if task_id == event.task_id:
                                # Terminal task state is authoritative even if
                                # the owning API disappears before TaskFinished.
                                # Preserve command mapping for eventual deletion.
                                self._realtime_instance_by_command.pop(
                                    command_id, None
                                )

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
                    self._apply_indexed_event(indexed)

                    event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]
                    if isinstance(event, NodeGatheredInfo):
                        event.when = str(datetime.now(tz=timezone.utc))

                    self._append_event_log(event)
                    await self._send_event(indexed)
                    await self._persist_snapshot()

    def _schedule_event_log_replay(self, since_idx: int) -> None:
        """Coalesce replay requests onto one paced background worker."""

        active_next = self._active_replay_next_idx
        active_end = self._active_replay_end_idx
        if (
            active_next is not None
            and active_end is not None
            and active_next <= since_idx < active_end
        ):
            # The active replay has not sent this index yet, so its existing pass
            # will satisfy the request without another broadcast of the same tail.
            return
        if self._pending_replay_start_idx is None:
            self._pending_replay_start_idx = since_idx
        else:
            self._pending_replay_start_idx = min(
                self._pending_replay_start_idx, since_idx
            )
        if self._replay_worker_running:
            return
        self._replay_worker_running = True
        self._tg.start_soon(self._drain_event_log_replays)

    async def _drain_event_log_replays(self) -> None:
        """Serve coalesced replay requests without blocking command processing."""

        try:
            while self._pending_replay_start_idx is not None:
                requested_start = self._pending_replay_start_idx
                self._pending_replay_start_idx = None
                await self._serve_event_log_replay(requested_start)
        finally:
            self._active_replay_next_idx = None
            self._active_replay_end_idx = None
            self._replay_worker_running = False

    async def _serve_event_log_replay(self, requested_start: int) -> None:
        """Broadcast one retained replay tail in bounded, paced chunks."""

        retained_start = max(requested_start, self._event_log.start_idx)
        replay_start = min(retained_start, len(self._event_log))
        if requested_start < self._event_log.start_idx:
            logger.warning(
                "Requested replay index predates retained master tail; "
                f"serving from {replay_start} instead of {requested_start}"
            )
        elif requested_start > len(self._event_log):
            logger.debug(
                "Requested replay index is beyond the current master tail; "
                f"serving an empty range at {replay_start}"
            )
        replay_end = min(
            replay_start + EVENT_LOG_REPLAY_BATCH_SIZE,
            len(self._event_log),
        )
        self._active_replay_next_idx = replay_start
        self._active_replay_end_idx = replay_end
        replayed = 0
        for idx, event in enumerate(
            self._event_log.read_range(replay_start, replay_end),
            start=replay_start,
        ):
            await self._send_event(IndexedEvent(idx=idx, event=event))
            replayed += 1
            self._active_replay_next_idx = idx + 1
            if (
                replayed % EVENT_LOG_REPLAY_CHUNK_SIZE == 0
                and idx + 1 < replay_end
            ):
                await anyio.sleep(EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS)
        logger.info(
            "Served paced event-log replay "
            f"(start={replay_start}, end={replay_end}, events={replayed})"
        )

    def _append_event_log(self, event: Event) -> None:
        """Append one decision and warn on sustained idle-state log growth."""

        self._event_log.append(event)
        active_download = any(
            isinstance(progress, DownloadOngoing)
            for progress_values in self._effective_downloads().values()
            for progress in progress_values
        )
        idle = not self.state.tasks and not active_download
        rate = self._event_log_growth_monitor.observe(
            now=time.monotonic(),
            idle=idle,
        )
        if rate is not None:
            logger.warning(
                "Event log is growing during an idle cluster state "
                f"({rate:.1f} events/min over "
                f"{self._event_log_growth_monitor.window_seconds:.0f}s); "
                "inspect periodic control-plane event sources before replay pressure accumulates"
            )

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
