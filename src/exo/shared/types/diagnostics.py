"""Read-only diagnostic response models for live Skulk node inspection."""

from typing import Literal

from pydantic import Field

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeIdentity,
    NodeNetworkInfo,
    SystemPerformanceProfile,
)
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.pydantic_ext import CamelCaseModel

ProcessRole = Literal["skulk", "runner", "vector", "python", "other"]
RunnerTaskCancelStatus = Literal[
    "cancel_requested",
    "already_cancelled",
    "already_completed",
]


class DiagnosticsProcess(CamelCaseModel):
    """One local operating-system process relevant to Skulk diagnostics."""

    pid: int = Field(description="Operating-system process ID.")
    parent_pid: int | None = Field(
        default=None,
        description="Operating-system parent process ID, when visible.",
    )
    role: ProcessRole = Field(
        description="Best-effort Skulk role inferred from process lineage and command line.",
    )
    command: str = Field(description="Joined process command line.")
    status: str | None = Field(
        default=None,
        description="Operating-system process status such as running or sleeping.",
    )
    cpu_percent: float | None = Field(
        default=None,
        description="Recent CPU percentage reported by psutil.",
    )
    memory_percent: float | None = Field(
        default=None,
        description="Percent of physical memory used by this process.",
    )
    rss: Memory | None = Field(
        default=None,
        description="Resident set size for this process, when available.",
    )
    elapsed_seconds: float | None = Field(
        default=None,
        description="Seconds since process creation, when available.",
    )
    is_child_of_skulk: bool = Field(
        default=False,
        description="Whether this process is in the current Skulk API process tree.",
    )


class RunnerTaskDiagnostics(CamelCaseModel):
    """Compact task information for a task known to a runner or cluster state."""

    task_id: str = Field(description="Skulk task ID.")
    task_kind: str = Field(description="Concrete task model name.")
    task_status: str = Field(description="Current event-sourced task status.")
    instance_id: str = Field(description="Instance associated with the task.")
    command_id: str | None = Field(
        default=None,
        description="External command ID for user-facing inference tasks.",
    )
    runner_id: str | None = Field(
        default=None,
        description="Runner assigned to the task, if known.",
    )
    model_id: str | None = Field(
        default=None,
        description="Model associated with the task, if known.",
    )


class RunnerLifecycleMilestone(CamelCaseModel):
    """Bounded live milestone emitted by the runner supervisor."""

    at: str = Field(description="UTC timestamp when the milestone was recorded.")
    name: str = Field(description="Short milestone name.")
    detail: str | None = Field(
        default=None,
        description="Optional compact detail for the milestone.",
    )


class RunnerSupervisorDiagnostics(CamelCaseModel):
    """Live runner-supervisor state that is not event-sourced."""

    runner_id: str = Field(description="Runner ID.")
    instance_id: str = Field(description="Instance ID.")
    node_id: str = Field(description="Node ID that owns this runner.")
    model_id: str = Field(description="Model assigned to this runner.")
    device_rank: int = Field(description="Distributed device rank.")
    world_size: int = Field(description="Distributed world size.")
    start_layer: int = Field(description="Inclusive first model layer on this shard.")
    end_layer: int = Field(description="Exclusive final model layer on this shard.")
    n_layers: int = Field(description="Total number of model layers.")
    pid: int | None = Field(
        default=None,
        description="Runner subprocess PID, when started.",
    )
    process_alive: bool = Field(description="Whether the runner subprocess is alive.")
    exit_code: int | None = Field(
        default=None,
        description="Runner subprocess exit code, when exited.",
    )
    status_kind: str = Field(description="Current runner status variant.")
    status_since: str = Field(description="UTC timestamp for the current status.")
    seconds_in_status: float = Field(
        description="Wall-clock seconds spent in the current runner status."
    )
    pending_task_ids: list[str] = Field(
        default_factory=list,
        description="Tasks sent to the supervisor but not acknowledged by the runner.",
    )
    in_progress_tasks: list[RunnerTaskDiagnostics] = Field(
        default_factory=list,
        description="Tasks currently known as in progress by the supervisor.",
    )
    completed_task_count: int = Field(
        description="Number of tasks completed by this supervisor."
    )
    cancelled_task_ids: list[str] = Field(
        default_factory=list,
        description="Task IDs cancelled through this supervisor.",
    )
    last_task_sent_at: str | None = Field(
        default=None,
        description="UTC timestamp for the last task submitted to the runner.",
    )
    last_event_received_at: str | None = Field(
        default=None,
        description="UTC timestamp for the last event received from the runner.",
    )
    last_event_type: str | None = Field(
        default=None,
        description="Class name of the last event received from the runner.",
    )
    milestones: list[RunnerLifecycleMilestone] = Field(
        default_factory=list,
        description="Recent lifecycle milestones retained by the supervisor.",
    )


class RunnerTaskCancelRequest(CamelCaseModel):
    """Payload for a direct live-runner task cancellation request."""

    task_id: TaskId = Field(description="Live task ID to request cancellation for.")


class RunnerTaskCancelResponse(CamelCaseModel):
    """Result of a direct live-runner task cancellation request."""

    node_id: NodeId = Field(description="Node that accepted the cancellation request.")
    runner_id: RunnerId = Field(
        description="Runner supervisor that handled the request."
    )
    task_id: TaskId = Field(description="Task ID the request targeted.")
    status: RunnerTaskCancelStatus = Field(
        description="Cancellation outcome as observed by the local runner supervisor."
    )
    message: str = Field(
        description="Human-readable description of what the node did."
    )


class PlacementRunnerDiagnostics(CamelCaseModel):
    """Event-sourced placement details for one runner assignment."""

    runner_id: str = Field(description="Runner ID.")
    node_id: str = Field(description="Node ID assigned to this runner.")
    friendly_name: str | None = Field(
        default=None,
        description="Friendly node name, when known.",
    )
    status_kind: str | None = Field(
        default=None,
        description="Current event-sourced runner status variant.",
    )
    device_rank: int = Field(description="Distributed device rank.")
    world_size: int = Field(description="Distributed world size.")
    start_layer: int = Field(description="Inclusive first model layer on this shard.")
    end_layer: int = Field(description="Exclusive final model layer on this shard.")
    n_layers: int = Field(description="Total number of model layers.")
    is_local: bool = Field(description="Whether this assignment is on the API node.")
    is_master: bool = Field(description="Whether this assignment is on the master node.")
    tasks: list[RunnerTaskDiagnostics] = Field(
        default_factory=list,
        description="Event-sourced tasks associated with this runner assignment.",
    )


class InstancePlacementDiagnostics(CamelCaseModel):
    """Cluster placement analysis for one model instance."""

    instance_id: str = Field(description="Instance ID.")
    model_id: str = Field(description="Placed model ID.")
    master_node_id: str | None = Field(
        default=None,
        description="Current master node ID, when known.",
    )
    master_is_placement_node: bool = Field(
        description="Whether the current master is part of this model placement."
    )
    local_node_is_placement_node: bool = Field(
        description="Whether the API node is part of this model placement."
    )
    placement_node_ids: list[str] = Field(
        default_factory=list,
        description="Node IDs participating in the placement.",
    )
    runners: list[PlacementRunnerDiagnostics] = Field(
        default_factory=list,
        description="Per-runner placement details.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Heuristic warnings that may help explain a stuck placement.",
    )


class NodeRuntimeDiagnostics(CamelCaseModel):
    """Static and slow-changing runtime information for one Skulk node."""

    node_id: str = Field(description="Local node ID.")
    hostname: str = Field(description="Local hostname.")
    friendly_name: str | None = Field(
        default=None,
        description="Friendly node name from gathered identity data.",
    )
    is_master: bool = Field(description="Whether this node is the current master.")
    master_node_id: str | None = Field(
        default=None,
        description="Current master node ID, when known.",
    )
    cwd: str = Field(description="Current working directory of the API process.")
    config_path: str = Field(description="Config path resolved by this API process.")
    config_file_exists: bool = Field(
        description="Whether the resolved config path exists from this process cwd."
    )
    skulk_version: str = Field(description="Installed Skulk package version.")
    skulk_commit: str = Field(description="Git commit reported by node identity.")
    libp2p_namespace: str | None = Field(
        default=None,
        description="Configured libp2p namespace environment value, if set.",
    )
    python_unbuffered: bool = Field(
        description="Whether PYTHONUNBUFFERED is enabled for this process."
    )
    tracing_enabled: bool = Field(
        description="Current cluster runtime tracing state as seen by this API node."
    )
    structured_logging_configured: bool = Field(
        description="Whether config enables centralized structured logging."
    )
    logging_ingest_url: str | None = Field(
        default=None,
        description="Configured centralized logging ingest URL, when present.",
    )


class NodeResourceDiagnostics(CamelCaseModel):
    """Resource readings for one node from gathered state and local psutil."""

    gathered_memory: MemoryUsage | None = Field(
        default=None,
        description="Last event-sourced memory reading for this node.",
    )
    current_memory: MemoryUsage | None = Field(
        default=None,
        description="Live memory reading from the API process.",
    )
    disk: DiskUsage | None = Field(
        default=None,
        description="Last event-sourced disk reading for this node.",
    )
    system: SystemPerformanceProfile | None = Field(
        default=None,
        description="Last event-sourced system performance reading.",
    )
    network: NodeNetworkInfo | None = Field(
        default=None,
        description="Last event-sourced network interface reading.",
    )


class NodeDiagnostics(CamelCaseModel):
    """Read-only diagnostic bundle for one Skulk node."""

    generated_at: str = Field(description="UTC timestamp when this bundle was built.")
    runtime: NodeRuntimeDiagnostics = Field(description="Runtime identity and config.")
    identity: NodeIdentity | None = Field(
        default=None,
        description="Last gathered node identity data.",
    )
    resources: NodeResourceDiagnostics = Field(description="Resource readings.")
    processes: list[DiagnosticsProcess] = Field(
        default_factory=list,
        description="Relevant local OS processes.",
    )
    supervisor_runners: list[RunnerSupervisorDiagnostics] = Field(
        default_factory=list,
        description="Live local runner-supervisor diagnostics.",
    )
    placements: list[InstancePlacementDiagnostics] = Field(
        default_factory=list,
        description="Event-sourced placement analysis for current instances.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Top-level diagnostic warnings for this node.",
    )


class ClusterNodeDiagnostics(CamelCaseModel):
    """One node result inside a cluster diagnostics response."""

    node_id: str = Field(description="Node ID for this cluster diagnostics result.")
    url: str | None = Field(
        default=None,
        description="Peer API URL used to collect diagnostics, if remote.",
    )
    ok: bool = Field(description="Whether diagnostics were collected successfully.")
    diagnostics: NodeDiagnostics | None = Field(
        default=None,
        description="Collected node diagnostics when ok is true.",
    )
    error: str | None = Field(
        default=None,
        description="Collection error when ok is false.",
    )


class ClusterDiagnostics(CamelCaseModel):
    """Read-only diagnostic bundle collected from reachable cluster nodes."""

    generated_at: str = Field(description="UTC timestamp when collection finished.")
    local_node_id: str = Field(description="Node ID of the API serving this response.")
    master_node_id: str | None = Field(
        default=None,
        description="Current master node ID, when known.",
    )
    nodes: list[ClusterNodeDiagnostics] = Field(
        default_factory=list,
        description="Local and reachable peer diagnostic results.",
    )
