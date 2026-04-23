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
RunnerPhaseName = Literal[
    "created",
    "idle",
    "connect_group",
    "load_model",
    "warmup",
    "task_submission",
    "task_agreement",
    "prompt_build",
    "vision_preprocess",
    "kv_cache_lookup",
    "prefill_barrier",
    "prefill_pipeline",
    "prefill_stream",
    "decode_barrier",
    "decode_wait_first_token",
    "decode_stream",
    "parser",
    "cancel_requested",
    "cancel_observed",
    "completion",
    "error",
    "shutdown_cleanup",
]
RunnerTaskCancelStatus = Literal[
    "cancel_requested",
    "already_cancelled",
    "already_completed",
]


class MlxMemorySnapshot(CamelCaseModel):
    """Best-effort snapshot of memory reported by MLX/Metal."""

    generated_at: str = Field(description="UTC timestamp when the snapshot was taken.")
    active: Memory | None = Field(
        default=None,
        description="Currently active MLX memory, when the runtime exposes it.",
    )
    cache: Memory | None = Field(
        default=None,
        description="MLX cache memory, when the runtime exposes it.",
    )
    peak: Memory | None = Field(
        default=None,
        description="Peak MLX memory since the last reset, when available.",
    )
    wired_limit: Memory | None = Field(
        default=None,
        description=(
            "Configured MLX wired memory limit when known. Current MLX releases "
            "do not expose a getter on all platforms, so this may be null."
        ),
    )
    source: str = Field(
        description="Runtime module that supplied the measurement, such as mlx.core."
    )


RunnerDiagnosticValue = str | int | float | bool | list[str]


class RunnerDiagnosticContext(CamelCaseModel):
    """Stable runner identity included with each flight-recorder update."""

    node_id: str = Field(description="Node ID that owns this runner.")
    runner_id: str = Field(description="Runner ID.")
    pid: int | None = Field(default=None, description="Runner subprocess PID.")
    instance_id: str = Field(description="Instance ID.")
    model_id: str = Field(description="Model assigned to this runner.")
    rank: int = Field(description="Distributed rank for this runner.")
    world_size: int = Field(description="Distributed world size.")
    start_layer: int = Field(description="Inclusive first layer on this shard.")
    end_layer: int = Field(description="Exclusive final layer on this shard.")
    n_layers: int = Field(description="Total model layers.")


class RunnerDiagnosticUpdate(CamelCaseModel):
    """One non-blocking runner-to-supervisor diagnostic update."""

    at: str = Field(description="UTC timestamp when the runner emitted the update.")
    phase: RunnerPhaseName = Field(description="Current runner phase.")
    event: str = Field(description="Short event name within the phase.")
    detail: str | None = Field(
        default=None,
        description="Compact human-readable detail for diagnostics.",
    )
    attrs: dict[str, RunnerDiagnosticValue] = Field(
        default_factory=dict,
        description="Structured low-cardinality diagnostic attributes.",
    )
    context: RunnerDiagnosticContext = Field(
        description="Stable runner identity fields for this update."
    )
    task_id: str | None = Field(
        default=None,
        description="Active task ID associated with the update, when known.",
    )
    command_id: str | None = Field(
        default=None,
        description="External command ID associated with the update, when known.",
    )
    mlx_memory: MlxMemorySnapshot | None = Field(
        default=None,
        description="MLX memory snapshot captured with this update, when requested.",
    )


class RunnerFlightRecorderEntry(CamelCaseModel):
    """One retained runner flight-recorder entry."""

    at: str = Field(description="UTC timestamp when the runner emitted the update.")
    phase: RunnerPhaseName = Field(description="Runner phase at this entry.")
    event: str = Field(description="Short event name within the phase.")
    detail: str | None = Field(
        default=None,
        description="Compact human-readable detail for diagnostics.",
    )
    attrs: dict[str, RunnerDiagnosticValue] = Field(
        default_factory=dict,
        description="Structured low-cardinality diagnostic attributes.",
    )
    context: RunnerDiagnosticContext = Field(
        description="Stable runner identity fields for this entry."
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID associated with the entry, when known.",
    )
    command_id: str | None = Field(
        default=None,
        description="Command ID associated with the entry, when known.",
    )
    mlx_memory: MlxMemorySnapshot | None = Field(
        default=None,
        description="MLX memory snapshot captured with this entry, when present.",
    )


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
    phase: RunnerPhaseName = Field(description="Last runner phase reported.")
    phase_started_at: str = Field(
        description="UTC timestamp when the current phase started."
    )
    seconds_in_phase: float = Field(
        description="Wall-clock seconds spent in the current phase."
    )
    last_progress_at: str | None = Field(
        default=None,
        description="UTC timestamp for the last flight-recorder update.",
    )
    active_task_id: str | None = Field(
        default=None,
        description="Task ID associated with the current phase, when known.",
    )
    active_command_id: str | None = Field(
        default=None,
        description="Command ID associated with the current phase, when known.",
    )
    phase_detail: str | None = Field(
        default=None,
        description="Compact human-readable detail for the current phase.",
    )
    last_mlx_memory: MlxMemorySnapshot | None = Field(
        default=None,
        description="Most recent MLX memory snapshot reported by the runner.",
    )
    flight_recorder: list[RunnerFlightRecorderEntry] = Field(
        default_factory=list,
        description="Last 128 local-only runner diagnostic events.",
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


class DiagnosticProcessSample(CamelCaseModel):
    """One macOS process-sampling command result in a diagnostic capture bundle."""

    name: str = Field(description="Sampling command name.")
    command: list[str] = Field(description="Command argv that was attempted.")
    ok: bool = Field(description="Whether the sampling command completed cleanly.")
    exit_code: int | None = Field(
        default=None,
        description="Process exit code, when a command was launched.",
    )
    duration_seconds: float = Field(
        description="Wall-clock duration spent waiting for the command."
    )
    stdout: str | None = Field(
        default=None,
        description="Truncated standard output from the sampling command.",
    )
    stderr: str | None = Field(
        default=None,
        description="Truncated standard error from the sampling command.",
    )
    error: str | None = Field(
        default=None,
        description="Structured failure reason when the command could not run.",
    )


class DiagnosticCaptureRequest(CamelCaseModel):
    """Request payload for on-demand local or proxied node diagnostics capture."""

    runner_id: RunnerId | None = Field(
        default=None,
        description="Optional runner ID to focus the capture bundle.",
    )
    task_id: TaskId | None = Field(
        default=None,
        description="Optional task ID to find the relevant runner for capture.",
    )
    include_process_samples: bool = Field(
        default=True,
        description="Whether to run heavyweight local process sampling commands.",
    )
    sample_duration_seconds: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Requested duration for macOS sample collection.",
    )


class DiagnosticCaptureResponse(CamelCaseModel):
    """On-demand diagnostics capture bundle for a local or proxied node."""

    generated_at: str = Field(description="UTC timestamp when capture completed.")
    node_id: NodeId = Field(description="Node that produced the capture bundle.")
    node_diagnostics: "NodeDiagnostics" = Field(
        description="Full live node diagnostics at capture time."
    )
    runner: RunnerSupervisorDiagnostics | None = Field(
        default=None,
        description="Matched runner diagnostics, when a runner or task filter matched.",
    )
    flight_recorder: list[RunnerFlightRecorderEntry] = Field(
        default_factory=list,
        description="Flight-recorder entries retained for the matched runner.",
    )
    mlx_memory: MlxMemorySnapshot | None = Field(
        default=None,
        description="Most recent MLX memory snapshot reported by the matched runner.",
    )
    process_samples: list[DiagnosticProcessSample] = Field(
        default_factory=list,
        description="Heavyweight process sampling results, when requested.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Capture warnings and partial-failure notes.",
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
