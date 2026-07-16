from datetime import datetime
from typing import Literal, final

from pydantic import Field

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.topology import Connection
from skulk.shared.types.chunks import GenerationChunk, InputChunk
from skulk.shared.types.common import (
    CommandId,
    Id,
    ModelId,
    NodeId,
    SessionId,
    SystemId,
)
from skulk.shared.types.state import State
from skulk.shared.types.tasks import Task, TaskId, TaskStatus
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import Instance, InstanceId
from skulk.shared.types.worker.runners import RunnerId, RunnerStatus
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MacThunderboltConnections,
    MacThunderboltIdentifiers,
    NodeNetworkInterfaces,
    ThunderboltBridgeInfo,
)
from skulk.utils.pydantic_ext import CamelCaseModel, FrozenModel, TaggedModel


class EventId(Id):
    """
    Newtype around `ID`
    """


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    __test__ = False


class TaskCreated(BaseEvent):
    task_id: TaskId
    task: Task


class TaskAcknowledged(BaseEvent):
    task_id: TaskId


class TaskDeleted(BaseEvent):
    task_id: TaskId


class TaskStatusUpdated(BaseEvent):
    task_id: TaskId
    task_status: TaskStatus


class TaskFailed(BaseEvent):
    task_id: TaskId
    error_type: str
    error_message: str


class InstanceCreated(BaseEvent):
    instance: Instance

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InstanceCreated):
            return self.instance == other.instance and self.event_id == other.event_id

        return False


class InstanceDeleted(BaseEvent):
    instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


class NodeTimeoutEvidence(FrozenModel):
    """Signal ages captured when the master decides to prune a node."""

    last_logged_event_age_seconds: float = Field(
        ge=0,
        description="Age of the node's last indexed control-plane event.",
    )
    heartbeat_age_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Age of the node's last dedicated telemetry heartbeat receipt.",
    )
    fallback_telemetry_age_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Age of the node's last non-heartbeat telemetry receipt.",
    )
    effective_age_seconds: float = Field(
        ge=0,
        description="Age of the freshest available liveness signal.",
    )
    timeout_seconds: float = Field(
        gt=0,
        description="Configured liveness timeout used for the prune decision.",
    )


class NodeTimedOut(BaseEvent):
    """Remove a node after every available liveness signal becomes stale."""

    node_id: NodeId = Field(description="Node removed from cluster membership.")
    evidence: NodeTimeoutEvidence | None = Field(
        default=None,
        description=(
            "Signal ages captured by the deciding master; absent only on legacy "
            "or manually constructed events."
        ),
    )


# TODO: bikeshed this name
class NodeGatheredInfo(BaseEvent):
    node_id: NodeId
    when: str  # this is a manually cast datetime overrode by the master when the event is indexed, rather than the local time on the device
    info: GatheredInfo


class NodeDownloadProgress(BaseEvent):
    """Durable download outcome or reset decision.

    Current producers emit only ``DownloadCompleted``/``DownloadFailed`` and
    rare ``DownloadPending`` reset decisions here. ``DownloadOngoing`` remains
    decodable for replay of older event logs but is ignored by current apply.
    """

    download_progress: DownloadProgress


class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: GenerationChunk


class InputChunkReceived(BaseEvent):
    command_id: CommandId
    chunk: InputChunk


class TopologyEdgeCreated(BaseEvent):
    conn: Connection


class TopologyEdgeDeleted(BaseEvent):
    conn: Connection


class CustomModelCardAdded(BaseEvent):
    model_card: ModelCard


class CustomModelCardDeleted(BaseEvent):
    model_id: ModelId


class StagedModelEvicted(BaseEvent):
    """A store-deleted model whose staged copies every node should drop (#427).

    apply() clears the model's download entries from State (so the planner
    re-stages on a future placement instead of loading deleted files); each
    worker reacts by removing the model's staged directory from disk.
    """

    model_id: ModelId


class StateSnapshotHydrated(BaseEvent):
    """Local-only bootstrap event that replaces follower state from a snapshot."""

    state: State


@final
class TraceEventData(FrozenModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str
    node_id: str | None = None
    model_id: str | None = None
    task_kind: Literal["image", "text", "embedding", "speech"] | None = None
    tags: list[str] = Field(default_factory=list)
    attrs: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict
    )


@final
class TracesCollected(BaseEvent):
    task_id: TaskId
    rank: int
    traces: list[TraceEventData]


@final
class TracesMerged(BaseEvent):
    task_id: TaskId
    traces: list[TraceEventData]


@final
class TracingStateChanged(BaseEvent):
    """Cluster-wide runtime tracing state change."""

    enabled: bool


Event = (
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | RunnerStatusUpdated
    | NodeTimedOut
    | NodeGatheredInfo
    | NodeDownloadProgress
    | ChunkGenerated
    | InputChunkReceived
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
    | TracesCollected
    | TracesMerged
    | TracingStateChanged
    | CustomModelCardAdded
    | CustomModelCardDeleted
    | StagedModelEvicted
    | StateSnapshotHydrated
)


_PERSISTED_CONTROL_EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    TestEvent,
    TaskCreated,
    TaskStatusUpdated,
    TaskFailed,
    TaskDeleted,
    TaskAcknowledged,
    InstanceCreated,
    InstanceDeleted,
    RunnerStatusUpdated,
    NodeTimedOut,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
    TracingStateChanged,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    StagedModelEvicted,
    StateSnapshotHydrated,
)

_CONTROL_TOPOLOGY_INFO_TYPES = (
    NodeNetworkInterfaces,
    MacThunderboltIdentifiers,
    MacThunderboltConnections,
    ThunderboltBridgeInfo,
)


def is_persistable_control_event(event: Event) -> bool:
    """Return whether an event is a durable control-plane decision or fact.

    Payload-bearing runner IPC events and observational telemetry remain
    decodable for compatibility, but the master must reject them before
    ordering, indexing, persistence, replay, or global broadcast.
    """

    if isinstance(event, NodeGatheredInfo):
        return isinstance(event.info, _CONTROL_TOPOLOGY_INFO_TYPES)
    if isinstance(event, NodeDownloadProgress):
        return isinstance(
            event.download_progress,
            (DownloadPending, DownloadCompleted, DownloadFailed),
        )
    return isinstance(event, _PERSISTED_CONTROL_EVENT_TYPES)


class IndexedEvent(CamelCaseModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: Event


class GlobalForwarderEvent(CamelCaseModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event


class LocalForwarderEvent(CamelCaseModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: SystemId
    session: SessionId
    event: Event
