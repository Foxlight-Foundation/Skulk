import base64
import hashlib
import io
import sys
import time
from collections import defaultdict, deque
from collections.abc import Container, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, fail_after, to_thread
from loguru import logger
from PIL import Image

from skulk.download.download_utils import resolve_model_in_path
from skulk.routing.realtime_audio import RealtimeAudioPacket
from skulk.routing.router import TelemetrySender
from skulk.routing.speech_media import SpeechMediaPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.apply import apply
from skulk.shared.constants import SKULK_IMAGE_TRANSPORT_DEBUG
from skulk.shared.models.memory_estimate import (
    GPU_VRAM_WORKING_SET_FRACTION,
    UMA_GPU_OS_HEADROOM,
    estimate_shard_footprint,
    gpu_working_set_ceiling,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    add_to_card_cache,
    delete_custom_card,
)
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.chunks import AudioInputChunk, DataChunk, InputImageChunk
from skulk.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    RefuseInstancePlacement,
    StartDownload,
    TaskCancelled,
)
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.diagnostics import (
    RunnerSupervisorDiagnostics,
    RunnerTaskCancelResponse,
    VisionMediaIngressDiagnostics,
)
from skulk.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    IndexedEvent,
    InputChunkReceived,
    NodeDownloadProgress,
    NodeGatheredInfo,
    StagedModelEvicted,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    AudioTranscription,
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    LoadModel,
    RealtimeAudioTranscription,
    Shutdown,
    SpeechSynthesis,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.telemetry import (
    TELEMETRY_PLANE_INFO,
    NodeTelemetry,
    TelemetryView,
    record_membership_from_event,
)
from skulk.shared.types.topology import Connection, SocketConnection
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import InstanceId, LlamaRpcInstance
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerStatus
from skulk.shared.types.worker.shards import ShardMetadata, TensorShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient
from skulk.store.staging_eviction import (
    StagingEvictionReport,
    enforce_staging_budget,
    staging_directory_name,
    touch_last_used,
)
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.crash_window import CrashWindow
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
)
from skulk.utils.info_gatherer.net_profile import check_reachable
from skulk.utils.keyed_backoff import KeyedBackoff
from skulk.utils.task_group import TaskGroup
from skulk.worker.plan import plan
from skulk.worker.runner.bootstrap import WEDGE_FAILURE_MARKER
from skulk.worker.runner.runner_supervisor import RunnerSupervisor

_STALE_RESET_MAX_WAIT_TICKS = 300
"""How many ~100ms planning ticks to hold planning while stale download
resets round-trip through the master (~30s). The deadline exists so a
masterless interval cannot freeze the worker forever - past it, planning
resumes and the download coordinator's missing-directory self-heal covers
the residual risk."""


_RUNNER_CRASH_THRESHOLD = 3
"""Runner failures (crashes + local fit-refusals) for one instance within
``_RUNNER_CRASH_WINDOW_SECONDS`` before the worker gives up and deletes it."""

_RUNNER_CRASH_WINDOW_SECONDS = 60.0
"""Rolling window for ``_RUNNER_CRASH_THRESHOLD`` (see CrashWindow)."""


_LOAD_FIT_TOLERANCE = 0.10
"""Fraction by which a shard's estimated footprint may exceed live usable memory
before the pre-load guard refuses (#383).

The footprint from ``estimate_shard_footprint`` already bakes in the engine
overhead factor (1.30 for MLX), a full ``KV_CONTEXT_BUDGET_TOKENS`` KV
reservation, and a flat floor, so it overstates the real resident need. The
master admits on the same padded estimate but against a *gossiped* usable
figure, while this guard re-measures *live*; on a borderline multi-node split
the live reading can sit a few hundred MB below the gossiped one (staging churn,
telemetry lag), so requiring ``usable >= footprint`` exactly flips a placement
the master already admitted into a refusal on a sub-GB jitter, observed as a
0.2GB / 2% miss refusing Devstral-24B at the LoadModel re-check across the
3-kite ring (#383, Phase 0). Allowing the footprint to exceed usable by this
margin absorbs that jitter (the 30% pad means actual resident still fits) while
a *gross* shortfall (a node that genuinely loaded another model since admission)
still refuses, preserving the leak-on-OOM guard (#290/#239)."""

_REALTIME_AUDIO_PENDING_FRAMES = 256
_REALTIME_AUDIO_PENDING_BYTES = 16 * 1024 * 1024
_REALTIME_AUDIO_PENDING_TTL_SECONDS = 30.0
_REALTIME_FINISHED_COMMANDS = 1024
_SPEECH_MEDIA_PENDING_FRAMES = 32
_SPEECH_MEDIA_PENDING_BYTES = 25 * 1024 * 1024
_SPEECH_MEDIA_PENDING_TTL_SECONDS = 30.0
_VISION_MEDIA_PENDING_STREAMS = 64
_VISION_MEDIA_PENDING_FRAMES = 64
_VISION_MEDIA_PENDING_BYTES = 32 * 1024 * 1024
_VISION_MEDIA_PENDING_TOTAL_BYTES = 512 * 1024 * 1024
_VISION_MEDIA_PENDING_TTL_SECONDS = 5 * 60.0
_VISION_MEDIA_ACK_SEND_TIMEOUT_SECONDS = 2.0


_RUNNER_FIRST_REPORT_DEADLINE_SECONDS = 120.0
"""How long a spawned runner has to report its first status before the worker
gives up on its instance (#272). A runner frozen between spawn and its first
report (SIGSTOP, a hang in early import, a stuck device init) otherwise stalls
pre-init coordination forever: ``_init_distributed_backend`` only plans
``ConnectToGroup`` once every rank has reported, and the crash breaker never
trips because the process is alive. Generous enough to cover slow Python imports
and weight mmaps, far under an indefinite hang."""


def _wedged_live_instances(
    runners: Mapping[RunnerId, "RunnerSupervisor"],
    live_instances: Container[InstanceId],
) -> list[tuple[InstanceId, ModelId]]:
    """Local supervisors whose runner died wedge-marked while the instance lives.

    These never get a planner ``Shutdown`` (``plan._kill_runner`` only fires
    when the instance is gone or a PEER failed), so the worker must give the
    instance up directly on observation - otherwise a single-node wedge
    strands a dead runner behind a live instance forever.
    """
    return [
        (
            supervisor.bound_instance.instance.instance_id,
            supervisor.shard_metadata.model_card.model_id,
        )
        for supervisor in runners.values()
        if supervisor.bound_instance.instance.instance_id in live_instances
        and _runner_failed_wedged(supervisor.status)
    ]


def runners_never_reported(
    runners: Mapping[RunnerId, "RunnerSupervisor"],
    live_instances: Container[InstanceId],
    now_monotonic: float,
    deadline_seconds: float,
) -> list[tuple[InstanceId, ModelId]]:
    """Live instances whose spawned runner never reported a status in time (#272).

    A runner frozen between spawn and its first status report stalls pre-init
    coordination forever: the group-init gate only plans ``ConnectToGroup`` once
    every rank has reported, and the crash breaker never trips because the
    process is alive. Returns ``(instance_id, model_id)`` for each live-instance
    supervisor older than ``deadline_seconds`` that has not yet reported. Pure for
    testability.
    """
    stalled: list[tuple[InstanceId, ModelId]] = []
    for supervisor in runners.values():
        instance_id = supervisor.bound_instance.instance.instance_id
        if instance_id not in live_instances:
            continue
        if supervisor.has_reported_status:
            continue
        if supervisor.seconds_since_created(now_monotonic) >= deadline_seconds:
            stalled.append(
                (instance_id, supervisor.shard_metadata.model_card.model_id)
            )
    return stalled


def footprint_exceeds_usable(
    footprint: Memory, usable: Memory, tolerance: float
) -> bool:
    """Whether a shard footprint exceeds usable memory beyond the fit tolerance.

    Pure for testability (#383). The footprint is already padded (engine overhead
    factor + KV reservation + floor), so a marginal excess is within that pad and
    within live-vs-gossip jitter; only a shortfall beyond ``tolerance`` (a
    fraction of usable) is treated as a real refusal.
    """
    return footprint.in_bytes > usable.in_bytes * (1 + tolerance)


def _runner_failed_wedged(status: RunnerStatus | None) -> bool:
    """Whether a dead runner's last gossiped status marks a GPU wedge.

    The supervisor embeds ``WEDGE_FAILURE_MARKER`` in the failure message
    when the runner exited with the deadline watchdog's ``WEDGE_EXIT_CODE``.
    Wedge deaths must never be retried: each wedge-exit leaks wired GPU
    memory that only a reboot reclaims (measured 2026-06-09, ~5GB/attempt).
    """
    return (
        isinstance(status, RunnerFailed)
        and status.error_message is not None
        and WEDGE_FAILURE_MARKER in status.error_message
    )


def _local_usable_vram() -> Memory | None:
    """Usable discrete-GPU VRAM on THIS node, or ``None`` if it has no discrete GPU.

    The worker counterpart of the master's ``usable_vram_by_node``: a GPU-offload
    node (AMD/Linux amdgpu) allocates a shard from its VRAM pool, not system RAM,
    so the local pre-spawn fit guard must size against VRAM or it would falsely
    refuse a placement the VRAM-aware master correctly admitted. Reads the local
    amdgpu sysfs (passive, the same source as the telemetry collector). It mirrors
    the master's two architectures so the two checks agree:

    * Discrete GPU: ``min(vram_total - vram_used, GPU_VRAM_WORKING_SET_FRACTION x
      vram_total)``.
    * Unified-memory APU (GTT spans the whole system: ``gtt_total > vram_total``
      AND ``gtt_total >= ram_total``, e.g. Strix Halo):
      ``min(vram_avail, GPU_VRAM_WORKING_SET_FRACTION x vram_total) +
      min(max(0, local_ram_available - UMA_GPU_OS_HEADROOM), gtt_total)`` so the
      GPU's GTT-mapped system RAM counts toward the pool. The live GPU-wireable
      snapshot matches the ceiling the master derives from gossiped telemetry.

    NVIDIA nodes (no amdgpu sysfs, NVML present) take the same discrete-GPU
    path from NVML readings; NVIDIA exposes no UMA/GTT signature here.
    Returns ``None`` on Apple unified-memory nodes (no amdgpu device), which keep
    the system-RAM path. The master is the backend authority that decides a shard
    belongs on this GPU node in the first place.
    """
    # Both collectors are Linux-only; bail before any import attempt so
    # macOS shard-fit checks never pay repeated pynvml import probes.
    if sys.platform != "linux":
        return None

    from skulk.utils.info_gatherer.linux_gpu import (
        find_amd_gpu_device,
        read_accelerator_metrics,
    )
    from skulk.utils.info_gatherer.nvidia_gpu import (
        has_nvidia_gpu,
        load_nvml,
        prefer_nvidia_telemetry,
    )
    from skulk.utils.info_gatherer.nvidia_gpu import (
        read_accelerator_metrics as read_nvidia_accelerator_metrics,
    )

    device = None if prefer_nvidia_telemetry() else find_amd_gpu_device()
    if device is None:
        # NVIDIA fallthrough (rented CUDA nodes): same discrete-VRAM sizing so
        # the worker's last-minute guard agrees with the master's admission.
        # NVIDIA reports no UMA/GTT signature, so the discrete path applies.
        # Gated on the node actually ADVERTISING a CUDA llama.cpp backend:
        # with pynvml present but only llama_cpp-cpu advertised (env unset,
        # or the GPU wheel clobbered so the probe dropped the tag), the
        # master admits against system RAM, and sizing the local guard
        # against VRAM would falsely refuse CPU placements that fit.
        from skulk.shared.backends import probe_node_backends

        if "llama_cpp-cuda" not in probe_node_backends():
            return None
        nvml = load_nvml()
        if nvml is None or not has_nvidia_gpu(nvml):
            return None
        nvidia = read_nvidia_accelerator_metrics(nvml)
        if not nvidia.vram_total_bytes or nvidia.vram_total_bytes <= 0:
            return None
        nvidia_available = max(
            0, nvidia.vram_total_bytes - (nvidia.vram_used_bytes or 0)
        )
        nvidia_ceiling = int(nvidia.vram_total_bytes * GPU_VRAM_WORKING_SET_FRACTION)
        return Memory.from_bytes(min(nvidia_available, nvidia_ceiling))
    accelerator = read_accelerator_metrics(device)
    total = accelerator.vram_total_bytes
    if not total or total <= 0:
        return None
    used = accelerator.vram_used_bytes or 0
    available = max(0, total - used)
    ceiling = int(total * GPU_VRAM_WORKING_SET_FRACTION)
    vram_usable = min(available, ceiling)
    gtt_total = accelerator.gtt_total_bytes
    local_memory = MemoryUsage.from_local_gpu_wireable()
    # UMA signature: GTT spans the whole system (exceeds the VRAM carve-out AND
    # covers all of system RAM). A discrete AMD card also reports a GTT total
    # (often ~= VRAM), so requiring it to cover system RAM keeps a discrete GPU
    # on the VRAM-only path. Matches the master's gate in ``usable_vram_by_node``.
    if (
        gtt_total is not None
        and gtt_total > total
        and gtt_total >= local_memory.ram_total.in_bytes
    ):
        # UMA APU: the GPU maps host RAM via GTT beyond the BIOS VRAM carve-out,
        # so count current free system RAM (minus OS headroom, capped by GTT).
        # The VRAM portion keeps its working-set headroom.
        sys_for_gpu = min(
            max(
                0, local_memory.ram_available.in_bytes - UMA_GPU_OS_HEADROOM.in_bytes
            ),
            gtt_total,
        )
        return Memory.from_bytes(vram_usable + sys_for_gpu)
    return Memory.from_bytes(vram_usable)


def _summarize_worker_task(task: Task) -> str:
    """Return a compact task summary for worker lifecycle logs."""
    if isinstance(task, CreateRunner):
        shard = task.bound_instance.bound_shard
        return (
            "CreateRunner("
            f"instance_id={task.instance_id!r}, "
            f"runner_id={task.bound_instance.bound_runner_id!r}, "
            f"node_id={task.bound_instance.bound_node_id!r}, "
            f"device_rank={shard.device_rank}, "
            f"world_size={shard.world_size}, "
            f"layers={shard.start_layer}:{shard.end_layer})"
        )
    if isinstance(task, DownloadModel):
        shard = task.shard_metadata
        return (
            "DownloadModel("
            f"instance_id={task.instance_id!r}, "
            f"model={shard.model_card.model_id!r}, "
            f"device_rank={shard.device_rank}, "
            f"world_size={shard.world_size}, "
            f"layers={shard.start_layer}:{shard.end_layer})"
        )
    if isinstance(task, Shutdown):
        return (
            f"Shutdown(instance_id={task.instance_id!r}, runner_id={task.runner_id!r})"
        )
    if isinstance(task, LoadModel):
        return f"LoadModel(instance_id={task.instance_id!r})"
    if isinstance(task, StartWarmup):
        return f"StartWarmup(instance_id={task.instance_id!r})"
    if isinstance(task, CancelTask):
        return (
            "CancelTask("
            f"instance_id={task.instance_id!r}, "
            f"runner_id={task.runner_id!r}, "
            f"cancelled_task_id={task.cancelled_task_id!r})"
        )
    if isinstance(task, TextGeneration):
        params = task.task_params
        return (
            "TextGeneration("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"input_messages={len(params.input)}, "
            f"chat_template_messages={len(params.chat_template_messages or [])}, "
            f"images={len(params.images)}, "
            f"cached_image_indices={sorted(params.image_hashes.keys())}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"image_count={params.image_count}, "
            f"stream={params.stream}, "
            f"reasoning_effort={params.reasoning_effort!r}, "
            f"enable_thinking={params.enable_thinking!r})"
        )
    if isinstance(task, ImageEdits):
        params = task.task_params
        return (
            "ImageEdits("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"instance_id={task.instance_id!r}, "
            f"model={params.model!r}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"has_inline_image_data={bool(params.image_data)}, "
            f"n={params.n!r}, "
            f"size={params.size!r}, "
            f"stream={params.stream!r})"
        )
    return task.__class__.__name__


def _inject_assembled_image_edit(task: ImageEdits, assembled_image: str) -> ImageEdits:
    """Return the ImageEdits task with the reassembled image injected.

    Uses ``model_copy`` so every other field is preserved - in particular
    ``owner_node``, which the Zenoh data plane needs to address generation output
    back to the owning API node (#279 Phase 2). Rebuilding the task field-by-field
    here previously dropped ``owner_node`` (and would silently drop any future
    field), causing the supervisor to stamp ``owner_node=None`` and the Router to
    publish output to a Zenoh key no node subscribes to (#310 review).
    """
    return task.model_copy(
        update={
            "task_params": task.task_params.model_copy(
                update={"image_data": assembled_image}
            )
        }
    )


def _audio_input_cleanup_command_id(
    event: Event,
    previous_tasks: Mapping[TaskId, Task],
    current_tasks: Mapping[TaskId, Task],
) -> CommandId | None:
    """Return the STT command whose uploaded chunks can be released."""
    task: Task | None = None
    if isinstance(event, TaskDeleted):
        task = previous_tasks.get(event.task_id)
    elif isinstance(event, TaskStatusUpdated) and event.task_status in {
        TaskStatus.Cancelled,
        TaskStatus.Complete,
        TaskStatus.Failed,
        TaskStatus.TimedOut,
    }:
        task = current_tasks.get(event.task_id) or previous_tasks.get(event.task_id)
    if isinstance(task, AudioTranscription):
        return task.command_id
    return None


def _realtime_input_cleanup_command_id(
    event: Event,
    previous_tasks: Mapping[TaskId, Task],
    current_tasks: Mapping[TaskId, Task],
) -> CommandId | None:
    """Return the realtime STT command whose live input route can close."""

    task: Task | None = None
    if isinstance(event, TaskDeleted):
        task = previous_tasks.get(event.task_id)
    elif isinstance(event, TaskStatusUpdated) and event.task_status in {
        TaskStatus.Cancelled,
        TaskStatus.Complete,
        TaskStatus.Failed,
        TaskStatus.TimedOut,
    }:
        task = current_tasks.get(event.task_id) or previous_tasks.get(event.task_id)
    if isinstance(task, RealtimeAudioTranscription):
        return task.command_id
    return None


def _speech_media_cleanup_command_id(
    event: Event,
    previous_tasks: Mapping[TaskId, Task],
    current_tasks: Mapping[TaskId, Task],
) -> CommandId | None:
    """Return the TTS command whose ephemeral reference media can be released."""

    task: Task | None = None
    if isinstance(event, TaskDeleted):
        task = previous_tasks.get(event.task_id)
    elif isinstance(event, TaskStatusUpdated) and event.task_status in {
        TaskStatus.Cancelled,
        TaskStatus.Complete,
        TaskStatus.Failed,
        TaskStatus.TimedOut,
    }:
        task = current_tasks.get(event.task_id) or previous_tasks.get(event.task_id)
    if isinstance(task, SpeechSynthesis):
        return task.command_id
    return None


def _vision_media_cleanup_command_id(
    event: Event,
    previous_tasks: Mapping[TaskId, Task],
    current_tasks: Mapping[TaskId, Task],
) -> CommandId | None:
    """Return the vision command whose ephemeral input can be released."""

    task: Task | None = None
    if isinstance(event, TaskDeleted):
        task = previous_tasks.get(event.task_id)
    elif isinstance(event, TaskStatusUpdated) and event.task_status in {
        TaskStatus.Cancelled,
        TaskStatus.Complete,
        TaskStatus.Failed,
        TaskStatus.TimedOut,
    }:
        task = current_tasks.get(event.task_id) or previous_tasks.get(event.task_id)
    if isinstance(task, (TextGeneration, ImageEdits)):
        return task.command_id
    return None


def _log_image_transport(message: str) -> None:
    """Emit image transport logs only at INFO when explicitly requested.

    Multimodal traffic can be large and frequent, so the default path keeps
    these diagnostics at DEBUG. Operators can opt in with
    ``SKULK_IMAGE_TRANSPORT_DEBUG=true`` (or the legacy ``SKULK_*`` alias) when
    they need end-to-end payload fingerprints.
    """
    if SKULK_IMAGE_TRANSPORT_DEBUG:
        logger.info(message)
    else:
        logger.debug(message)


def _log_image_payload_debug(
    source: str,
    image_index: int,
    image_b64: str,
    *,
    expected_b64_sha256: str | None = None,
) -> None:
    """Log stable fingerprints for an image payload crossing worker boundaries."""
    b64_sha256 = hashlib.sha256(image_b64.encode("ascii")).hexdigest()
    message = (
        f"{source} image {image_index}: b64_chars={len(image_b64)} "
        f"b64_sha256={b64_sha256[:12]}..."
    )
    if expected_b64_sha256 is not None:
        message += (
            f" expected_b64_sha256={expected_b64_sha256[:12]}..."
            f" matches_expected={b64_sha256 == expected_b64_sha256}"
        )

    if SKULK_IMAGE_TRANSPORT_DEBUG:
        try:
            raw_bytes = base64.b64decode(image_b64)
            raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            with Image.open(io.BytesIO(raw_bytes)) as pil_image:
                message += (
                    f" raw_bytes={len(raw_bytes)} raw_sha256={raw_sha256[:12]}..."
                    f" decoded={pil_image.width}x{pil_image.height}"
                    f" mode={pil_image.mode}"
                )
        except Exception as exc:
            message += f" decode_failed={type(exc).__name__}: {exc}"

    _log_image_transport(message)


def resolve_cached_vlm_images(
    image_cache: dict[str, str],
    image_hashes: dict[int, str],
) -> tuple[dict[int, str], list[tuple[int, str]]]:
    """Resolve cached VLM image hashes without treating cache misses as fatal."""
    by_index: dict[int, str] = {}
    missing: list[tuple[int, str]] = []
    for image_index, image_hash in image_hashes.items():
        cached_image = image_cache.get(image_hash)
        if cached_image is None:
            missing.append((image_index, image_hash))
            continue
        by_index[image_index] = cached_image
        _log_image_payload_debug(
            "Worker resolved cached VLM",
            image_index,
            cached_image,
            expected_b64_sha256=image_hash,
        )
    return by_index, missing


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: Receiver[IndexedEvent],
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        telemetry_sender: TelemetrySender | Sender[NodeTelemetry] | None = None,
        telemetry_view: TelemetryView | None = None,
        data_sender: Sender[DataChunk] | None = None,
        realtime_audio_receiver: Receiver[RealtimeAudioInputFrame] | None = None,
        realtime_audio_packet_receiver: Receiver[RealtimeAudioPacket] | None = None,
        speech_media_packet_receiver: Receiver[SpeechMediaPacket] | None = None,
        vision_media_packet_sender: Sender[VisionMediaPacket] | None = None,
        vision_media_packet_receiver: Receiver[VisionMediaPacket] | None = None,
        store_client: ModelStoreClient | None = None,
        staging_config: StagingNodeConfig | None = None,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self._telemetry_sender = telemetry_sender
        # Data plane (#279 Phase 2): per-token output chunks stream direct to the
        # owning API node via this sender (DATA topic), bypassing the master's
        # event log. Threaded into each RunnerSupervisor. None falls back to the
        # event path (no DATA topic wired / tests).
        self._data_sender = data_sender
        self._realtime_audio_receiver = realtime_audio_receiver
        self._realtime_audio_packet_receiver = realtime_audio_packet_receiver
        self._speech_media_packet_receiver = speech_media_packet_receiver
        self._vision_media_packet_sender = vision_media_packet_sender
        self._vision_media_packet_receiver = vision_media_packet_receiver
        # Shared, Node-owned telemetry view (#279). The worker prunes a node's
        # telemetry here when it sees NodeTimedOut, because the worker runs on
        # EVERY node regardless of role - so a --no-api node (or a --no-api
        # master placing off this view) still drops dead nodes, where an
        # API-only prune hook would leak unbounded across churn.
        self._telemetry_view = telemetry_view
        self._store_client = store_client
        self._staging_config = staging_config

        self.state: State = State()
        # Event-log-bound readings (connectivity) confirmed indexed by the
        # master: keyed by reading type, holding the payload we saw echoed
        # back as an indexed event. _forward_info gates re-sends on THIS (not
        # on "last sent"), so a change dropped after bounded delivery retries
        # keeps being re-sent each poll until it truly lands in the log.
        self._confirmed_forwarded_info: dict[type[GatheredInfo], GatheredInfo] = {}
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        # Staging DIRECTORY NAMES (forward-sanitized) of models evicted by
        # startup reconciliation whose DownloadCompleted entries may still
        # be replicating in; countered lazily by
        # plan_step once state carries them. Names stay in the pending set
        # until the reset has APPLIED (state stops advertising the stale
        # entry) - discarding on send would reopen the plan gate while the
        # event is still round-tripping through the master.
        self._stale_downloads_pending_reset: set[str] = set()
        self._stale_resets_sent: set[str] = set()
        self._stale_reset_wait_ticks: int = 0
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.input_audio_chunk_buffer: dict[CommandId, dict[int, AudioInputChunk]] = {}
        self.input_audio_chunk_counts: dict[CommandId, int] = {}
        self._speech_media_chunks: dict[
            CommandId, dict[int, SpeechMediaPacket]
        ] = {}
        self._speech_media_completed: dict[CommandId, SpeechMediaPacket] = {}
        self._speech_media_pending_bytes: dict[CommandId, int] = {}
        self._speech_media_pending_since: dict[CommandId, float] = {}
        self._vision_media_chunks: dict[
            CommandId, dict[int, VisionMediaPacket]
        ] = {}
        self._vision_media_opened: dict[CommandId, VisionMediaPacket] = {}
        self._vision_media_completed: dict[CommandId, VisionMediaPacket] = {}
        self._vision_media_verified: dict[CommandId, VisionMediaPacket] = {}
        self._vision_media_verified_chunks: dict[
            CommandId, dict[int, InputImageChunk]
        ] = {}
        self._vision_media_accepted: set[CommandId] = set()
        self._vision_media_ack_inflight: set[CommandId] = set()
        self._vision_media_failures: dict[CommandId, str] = {}
        self._vision_media_failure_since: dict[CommandId, float] = {}
        self._vision_media_pending_bytes: dict[CommandId, int] = {}
        self._vision_media_pending_total_bytes = 0
        self._vision_media_pending_since: dict[CommandId, float] = {}
        self._vision_media_completed_streams = 0
        self._vision_media_rejected_streams = 0
        self._vision_media_expired_streams = 0
        self._realtime_audio_pending: dict[
            CommandId, list[RealtimeAudioInputFrame]
        ] = {}
        self._realtime_audio_pending_bytes: dict[CommandId, int] = {}
        self._realtime_audio_pending_since: dict[CommandId, float] = {}
        self._realtime_runner_by_command: dict[CommandId, RunnerSupervisor] = {}
        self._realtime_finished_order: deque[CommandId] = deque()
        self._realtime_finished_commands: set[CommandId] = set()
        self.image_cache: dict[str, str] = {}

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        # Crash circuit breaker: stop relaunching an instance whose runner keeps
        # failing (e.g. OOM on load). Each abnormal Metal termination can leak
        # wired GPU memory reclaimable only by reboot, so an unbounded relaunch
        # loop compounds the damage (the GLM-4.7-Flash incident, 2026-06-08).
        self._crash_breaker: CrashWindow[InstanceId] = CrashWindow(
            _RUNNER_CRASH_THRESHOLD, _RUNNER_CRASH_WINDOW_SECONDS
        )
        self._stopped: anyio.Event = anyio.Event()

    def _effective_downloads(self) -> dict[NodeId, list[DownloadProgress]]:
        """Return durable outcomes overlaid with live download telemetry."""

        if self._telemetry_view is None:
            return {
                node_id: list(progresses)
                for node_id, progresses in self.state.downloads.items()
            }
        return self._telemetry_view.effective_downloads(self.state.downloads)

    def _shard_memory_fraction(self, shard: ShardMetadata) -> float:
        """Fraction of a model's memory this node's shard holds.

        Weights and KV both scale by this single fraction (see
        ``estimate_shard_footprint``). Tensor parallelism splits every weight
        matrix and KV head by ``world_size``; pipeline/CFG hold a contiguous
        layer range.
        """
        if isinstance(shard, TensorShardMetadata):
            return 1.0 / shard.world_size if shard.world_size > 0 else 1.0
        if shard.n_layers > 0:
            return (shard.end_layer - shard.start_layer) / shard.n_layers
        return 1.0 / shard.world_size if shard.world_size > 0 else 1.0

    def _local_shard_fit_error(self, shard: ShardMetadata) -> str | None:
        """Reason this node cannot hold ``shard``, or ``None`` if it fits.

        Last-resort guard using *local, current* memory, not the master's
        gossiped view. Availability is the same GPU-wireable figure the master
        admits on (``total − wired − anonymous − compressor`` from a vm_stat
        snapshot, capped at the Metal GPU working-set ceiling) - psutil's
        ``available`` counts reclaimable file cache as used, so right after a
        model download it would veto the very placement the master just
        correctly admitted. Falls back to psutil when vm_stat fails. Refusing
        here fails the placement cleanly instead of letting the runner
        OOM-abort, which on an abnormal Metal termination leaks wired GPU
        memory reclaimable only by reboot (the GLM-4.7-Flash class,
        2026-06-08).
        """
        footprint = estimate_shard_footprint(
            shard.model_card, self._shard_memory_fraction(shard)
        )
        # On a discrete-GPU node the engine allocates from VRAM, not system RAM,
        # so size the guard against local usable VRAM or it would falsely refuse
        # the very placement the (VRAM-aware) master just admitted. None on
        # unified-memory (Apple) nodes, which keep the system-RAM path.
        vram = _local_usable_vram()
        if vram is not None:
            usable = vram
            pool = f"{vram.in_gb:.1f}GB usable GPU VRAM"
        else:
            local = MemoryUsage.from_local_gpu_wireable()
            usable = min(local.ram_available, gpu_working_set_ceiling(local.ram_total))
            pool = (
                f"{local.ram_available.in_gb:.1f}GB GPU-wireable, capped at "
                "the GPU working-set ceiling"
            )
        # Refuse only on a shortfall beyond the tolerance (#383): the footprint
        # is already padded (overhead factor + KV reservation + floor), so a
        # marginal miss is within that pad and within live-vs-gossip jitter on a
        # placement the master already admitted. A gross shortfall still trips.
        if footprint_exceeds_usable(footprint, usable, _LOAD_FIT_TOLERANCE):
            return (
                f"Refusing to load a shard of {shard.model_card.model_id} on "
                f"{self.node_id}: ~{footprint.in_gb:.1f}GB needed but only "
                f"~{usable.in_gb:.1f}GB usable locally ({pool}), beyond the "
                f"{_LOAD_FIT_TOLERANCE:.0%} fit tolerance. Refusing before load "
                "to avoid an OOM abort that leaks GPU memory."
            )
        return None

    async def _give_up_on_instance(self, instance_id: InstanceId, reason: str) -> None:
        """Tear down a repeatedly-failing instance instead of relaunching it.

        Sends ``DeleteInstance`` to the master so the doomed instance stops
        being reconciled into fresh runners - each relaunch risks another
        leak-on-abort. The crash window is deliberately NOT cleared: the trip is
        edge-triggered, so leaving the failure history in place keeps the latch
        set and suppresses re-tripping (and duplicate ``DeleteInstance``) while
        the instance lingers in replicated state before the deletion lands.
        ``InstanceId``s are unique, so the stale entry can never collide with a
        future instance.
        """
        logger.error(f"Worker: giving up on instance {instance_id}: {reason}")
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=DeleteInstance(instance_id=instance_id),
            )
        )

    async def _refuse_instance_placement(
        self, instance_id: InstanceId, reason: str
    ) -> None:
        """Ask the master to re-place this instance on a wider split (#290).

        Used instead of :meth:`_give_up_on_instance` when the give-up is driven
        by the *memory fit guard* rather than a runner crash or GPU wedge. The
        master admits placements on the gossiped (telemetry-plane) availability,
        which can sit just above the live GPU-wireable figure this node measures
        at load time; on a borderline split that gap makes the master admit a
        cycle this worker then refuses. Rather than letting the instance vanish,
        the master re-places the model one node wider (smaller per-node share),
        and only gives up for good once even a full-width split won't fit. The
        crash window is left set, exactly as in :meth:`_give_up_on_instance`, so
        the edge-triggered breaker does not re-send while the deletion that the
        re-placement performs propagates through replicated state.
        """
        logger.error(
            f"Worker: refusing instance {instance_id} for placement "
            f"(asking master to re-place wider): {reason}"
        )
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=RefuseInstancePlacement(
                    instance_id=instance_id,
                    node_id=self.node_id,
                    reason=reason,
                ),
            )
        )

    def _local_capabilities_provider(self) -> frozenset[str]:
        """Snapshot the capabilities extensions have advertised on this node.

        The outbound set lives on the shared :class:`TelemetryView` so the
        API-side extension surface can mutate it (``advertise_capability``) and
        this worker's info gatherer can read it, without threading a new channel
        through :class:`Node` construction. Returns an immutable frozenset so the
        gatherer cannot mutate the live set, and an empty set when no view is
        wired (``--no-worker`` builds have no gatherer; a bare worker with no
        view simply advertises nothing).

        Returns:
            The capability tags currently advertised by this node.
        """
        if self._telemetry_view is None:
            return frozenset()
        return frozenset(self._telemetry_view.local_advertised_capabilities)

    async def run(self):
        logger.info("Starting Worker")
        self._reconcile_staging_on_startup()

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(
            info_send,
            # Fabric-citizenship: the gatherer publishes whatever the API-side
            # extension surface has advertised on the shared TelemetryView. The
            # provider snapshots the outbound set each poll (empty -> nothing
            # published), so a plugin advertising a capability rides the same
            # telemetry emit path as native node readings.
            capabilities_provider=self._local_capabilities_provider,
        )

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
                if self._realtime_audio_receiver is not None:
                    tg.start_soon(self._realtime_audio_ingress)
                if self._realtime_audio_packet_receiver is not None:
                    tg.start_soon(self._realtime_audio_packet_ingress)
                if self._speech_media_packet_receiver is not None:
                    tg.start_soon(self._speech_media_packet_ingress)
                    tg.start_soon(self._speech_media_janitor)
                if self._vision_media_packet_receiver is not None:
                    tg.start_soon(self._vision_media_packet_ingress)
                    tg.start_soon(self._vision_media_janitor)
                if (
                    self._realtime_audio_receiver is not None
                    or self._realtime_audio_packet_receiver is not None
                ):
                    tg.start_soon(self._realtime_audio_janitor)
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()
            self._stopped.set()

    async def _realtime_audio_ingress(self) -> None:
        """Route bounded non-event PCM frames to the selected local runner."""

        assert self._realtime_audio_receiver is not None
        with self._realtime_audio_receiver as frames:
            async for frame in frames:
                await self._route_realtime_audio_frame(frame)

    async def _realtime_audio_packet_ingress(self) -> None:
        """Consume node-addressed remote PCM packets without event persistence."""

        assert self._realtime_audio_packet_receiver is not None
        with self._realtime_audio_packet_receiver as packets:
            async for packet in packets:
                if packet.target_node != self.node_id or packet.kind == "transport_failed":
                    continue
                await self._route_realtime_audio_frame(packet.to_input_frame())

    async def _speech_media_packet_ingress(self) -> None:
        """Collect bounded node-addressed reference audio outside State."""

        assert self._speech_media_packet_receiver is not None
        with self._speech_media_packet_receiver as packets:
            async for packet in packets:
                if packet.target_node != self.node_id or packet.kind == "transport_failed":
                    continue
                command_id = packet.command_id
                if packet.kind in ("cancelled",):
                    self._clear_speech_media(command_id)
                    continue
                self._speech_media_pending_since.setdefault(
                    command_id, time.monotonic()
                )
                if packet.kind == "completed":
                    self._speech_media_completed[command_id] = packet
                    continue
                chunks = self._speech_media_chunks.setdefault(command_id, {})
                pending_bytes = self._speech_media_pending_bytes.get(command_id, 0)
                if packet.sequence in chunks:
                    continue
                if (
                    len(chunks) >= _SPEECH_MEDIA_PENDING_FRAMES
                    or pending_bytes + len(packet.data) > _SPEECH_MEDIA_PENDING_BYTES
                ):
                    self._clear_speech_media(command_id)
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=TaskCancelled(cancelled_command_id=command_id),
                        )
                    )
                    continue
                chunks[packet.sequence] = packet
                self._speech_media_pending_bytes[command_id] = (
                    pending_bytes + len(packet.data)
                )

    async def _vision_media_packet_ingress(self) -> None:
        """Collect ordered node-addressed image input outside replicated State."""

        assert self._vision_media_packet_receiver is not None
        with self._vision_media_packet_receiver as packets:
            async for packet in packets:
                if packet.target_node != self.node_id or packet.kind in (
                    "accepted",
                    "transport_failed",
                ):
                    continue
                command_id = packet.command_id
                if packet.kind == "cancelled":
                    source_packet = (
                        self._vision_media_opened.get(command_id)
                        or self._vision_media_verified.get(command_id)
                        or self._vision_media_completed.get(command_id)
                        or next(
                            iter(
                                self._vision_media_chunks.get(
                                    command_id, {}
                                ).values()
                            ),
                            None,
                        )
                    )
                    task = next(
                        (
                            candidate
                            for candidate in self.state.tasks.values()
                            if isinstance(candidate, (TextGeneration, ImageEdits))
                            and candidate.command_id == command_id
                        ),
                        None,
                    )
                    expected_source = (
                        source_packet.source_node
                        if source_packet is not None
                        else task.owner_node
                        if task is not None
                        else None
                    )
                    if (
                        expected_source is not None
                        and expected_source != packet.source_node
                    ):
                        logger.warning(
                            "Ignoring vision media cancellation from a source that "
                            f"does not own command {command_id}"
                        )
                        continue
                    self._clear_vision_media(command_id)
                    continue
                if command_id in self._vision_media_verified:
                    # The complete verified payload is immutable; late duplicate
                    # network frames cannot reopen or replace it.
                    continue
                if packet.kind == "opened":
                    existing_open = self._vision_media_opened.get(command_id)
                    if existing_open is not None:
                        if existing_open != packet:
                            await self._reject_vision_media(
                                packet,
                                "Vision media open metadata changed in flight",
                            )
                        continue
                    if (
                        packet.total_chunks is None
                        or packet.total_chunks > _VISION_MEDIA_PENDING_FRAMES
                        or packet.image_count is None
                        or packet.image_count > packet.total_chunks
                    ):
                        await self._reject_vision_media(
                            packet, "Vision media open exceeds declared bounds"
                        )
                        continue
                    buffered_completion = self._vision_media_completed.get(command_id)
                    buffered_chunks = self._vision_media_chunks.get(command_id, {})
                    buffered_packets = [
                        *buffered_chunks.values(),
                        *(
                            [buffered_completion]
                            if buffered_completion is not None
                            else []
                        ),
                    ]
                    if any(
                        buffered.source_node != packet.source_node
                        or buffered.model != packet.model
                        or buffered.total_chunks != packet.total_chunks
                        for buffered in buffered_packets
                    ) or (
                        buffered_completion is not None
                        and buffered_completion.image_count != packet.image_count
                    ):
                        await self._reject_vision_media(
                            packet,
                            "Vision media open does not match buffered stream metadata",
                        )
                        continue
                    if command_id not in self._vision_media_pending_since:
                        if len(self._vision_media_pending_since) >= (
                            _VISION_MEDIA_PENDING_STREAMS
                        ):
                            await self._reject_vision_media(
                                packet,
                                "Vision media ingress rejected the stream because "
                                "worker capacity is exhausted",
                            )
                            continue
                        self._vision_media_pending_since[command_id] = time.monotonic()
                    self._vision_media_opened[command_id] = packet
                    await self._finalize_vision_media(command_id)
                    continue

                opened = self._vision_media_opened.get(command_id)
                if (
                    packet.total_chunks is None
                    or packet.total_chunks > _VISION_MEDIA_PENDING_FRAMES
                    or (
                        packet.kind == "completed"
                        and (
                            packet.image_count is None
                            or packet.image_count > packet.total_chunks
                        )
                    )
                ):
                    await self._reject_vision_media(
                        packet, "Vision media frame exceeds declared bounds"
                    )
                    continue
                if command_id not in self._vision_media_pending_since:
                    if len(self._vision_media_pending_since) >= (
                        _VISION_MEDIA_PENDING_STREAMS
                    ):
                        await self._reject_vision_media(
                            packet,
                            "Vision media ingress rejected the stream because worker "
                            "capacity is exhausted",
                        )
                        continue
                    self._vision_media_pending_since[command_id] = time.monotonic()
                existing_completion = self._vision_media_completed.get(command_id)
                existing_chunks = self._vision_media_chunks.get(command_id, {})
                reference = opened or existing_completion or next(
                    iter(existing_chunks.values()), None
                )
                if (
                    reference is not None
                    and (
                        packet.source_node != reference.source_node
                        or packet.model != reference.model
                        or packet.total_chunks != reference.total_chunks
                    )
                ):
                    await self._reject_vision_media(
                        packet, "Vision media frame metadata changed in flight"
                    )
                    continue
                if packet.kind == "completed":
                    if opened is not None and packet.image_count != opened.image_count:
                        await self._reject_vision_media(
                            packet, "Vision media completion does not match its open"
                        )
                        continue
                    if existing_completion is not None and existing_completion != packet:
                        await self._reject_vision_media(
                            packet, "Vision media completion metadata changed in flight"
                        )
                        continue
                    self._vision_media_completed[command_id] = packet
                    await self._finalize_vision_media(command_id)
                    continue

                chunks = self._vision_media_chunks.setdefault(
                    command_id, existing_chunks
                )
                existing = chunks.get(packet.sequence)
                if existing is not None:
                    if existing != packet:
                        await self._reject_vision_media(
                            packet, "Vision media sequence was reused with different data"
                        )
                    continue
                pending_bytes = self._vision_media_pending_bytes.get(command_id, 0)
                packet_bytes = len(packet.data)
                if (
                    len(chunks) >= _VISION_MEDIA_PENDING_FRAMES
                    or pending_bytes + packet_bytes > _VISION_MEDIA_PENDING_BYTES
                    or self._vision_media_pending_total_bytes + packet_bytes
                    > _VISION_MEDIA_PENDING_TOTAL_BYTES
                ):
                    await self._reject_vision_media(
                        packet, "Vision media ingress exceeded its bounded capacity"
                    )
                    continue
                chunks[packet.sequence] = packet
                self._vision_media_pending_bytes[command_id] = (
                    pending_bytes + packet_bytes
                )
                self._vision_media_pending_total_bytes += packet_bytes
                await self._finalize_vision_media(command_id)

    async def _finalize_vision_media(self, command_id: CommandId) -> None:
        """Expose a stream to planning only after count and hash verification."""

        completed = self._vision_media_completed.get(command_id)
        opened = self._vision_media_opened.get(command_id)
        if completed is None or opened is None or completed.total_chunks is None:
            return
        chunks = self._vision_media_chunks.get(command_id, {})
        if len(chunks) < completed.total_chunks:
            return
        if set(chunks) != set(range(1, completed.total_chunks + 1)):
            await self._reject_vision_media(
                completed, "Vision media stream has missing or out-of-range sequences"
            )
            return
        ordered = [chunks[index] for index in range(1, completed.total_chunks + 1)]
        if any(
            packet.source_node != opened.source_node
            or packet.model != opened.model
            or packet.total_chunks != opened.total_chunks
            for packet in ordered
        ):
            await self._reject_vision_media(
                completed, "Vision media completion does not match its chunks"
            )
            return
        payload = b"".join(packet.data for packet in ordered)
        if hashlib.sha256(payload).hexdigest() != completed.sha256:
            await self._reject_vision_media(
                completed, "Vision media failed SHA-256 integrity verification"
            )
            return
        try:
            verified_chunks = {
                packet.sequence - 1: InputImageChunk(
                    model=packet.model,
                    command_id=packet.command_id,
                    data=packet.data.decode("ascii"),
                    chunk_index=packet.sequence - 1,
                    total_chunks=completed.total_chunks,
                    image_index=packet.image_index or 0,
                )
                for packet in ordered
            }
        except UnicodeDecodeError:
            await self._reject_vision_media(
                completed, "Vision media input is not ASCII base64 data"
            )
            return
        self._vision_media_verified_chunks[command_id] = verified_chunks
        self._vision_media_verified[command_id] = completed
        self._vision_media_completed_streams += 1
        self._clear_pending_vision_media(
            command_id,
            release_bytes=False,
            clear_age=False,
        )
        await self._acknowledge_vision_media_if_admitted(command_id)

    async def _acknowledge_vision_media_if_admitted(
        self, command_id: CommandId
    ) -> None:
        """Acknowledge verified bytes only after matching authoritative task input."""

        if (
            command_id in self._vision_media_accepted
            or command_id in self._vision_media_ack_inflight
        ):
            return
        completed = self._vision_media_verified.get(command_id)
        if completed is None:
            return
        task = next(
            (
                candidate
                for candidate in self.state.tasks.values()
                if isinstance(candidate, (TextGeneration, ImageEdits))
                and candidate.command_id == command_id
            ),
            None,
        )
        if task is None:
            return
        chunks = self._vision_media_verified_chunks.get(command_id, {})
        image_indexes = {chunk.image_index for chunk in chunks.values()}
        if isinstance(task, ImageEdits):
            matches_task = (
                completed.model == ModelId(task.task_params.model)
                and task.owner_node is not None
                and completed.source_node == task.owner_node
                and completed.total_chunks == task.task_params.total_input_chunks
                and completed.image_count == 1
                and image_indexes == {0}
            )
        else:
            cached_image_indexes = set(task.task_params.image_hashes)
            all_image_indexes = set(
                range(task.task_params.image_count + len(cached_image_indexes))
            )
            matches_task = (
                completed.model == task.task_params.model
                and task.owner_node is not None
                and completed.source_node == task.owner_node
                and completed.total_chunks == task.task_params.total_input_chunks
                and completed.image_count is not None
                and completed.image_count == task.task_params.image_count
                and len(image_indexes) == completed.image_count
                and image_indexes.isdisjoint(cached_image_indexes)
                and image_indexes | cached_image_indexes == all_image_indexes
            )
        if not matches_task:
            await self._reject_vision_media(
                completed,
                "Verified vision input does not match the authoritative task",
            )
            return
        sender = self._vision_media_packet_sender
        if sender is None:
            await self._reject_vision_media(
                completed, "Vision media acknowledgement transport is unavailable"
            )
            return
        self._vision_media_ack_inflight.add(command_id)
        try:
            try:
                with anyio.move_on_after(
                    _VISION_MEDIA_ACK_SEND_TIMEOUT_SECONDS, shield=True
                ) as acknowledgement_scope:
                    await sender.send(completed.accepted())
            except (BrokenResourceError, ClosedResourceError):
                return
            if (
                not acknowledgement_scope.cancel_called
                and command_id in self._vision_media_verified
            ):
                verified_chunks = self._vision_media_verified_chunks.get(command_id)
                if verified_chunks is None:
                    return
                self.input_chunk_buffer[command_id] = verified_chunks
                self.input_chunk_counts[command_id] = len(verified_chunks)
                self._vision_media_accepted.add(command_id)
        finally:
            self._vision_media_ack_inflight.discard(command_id)

    async def _reject_vision_media(
        self, packet: VisionMediaPacket, message: str
    ) -> None:
        """Reject one malformed stream and report a typed source-side failure."""

        command_id = packet.command_id
        self._vision_media_rejected_streams += 1
        self._clear_vision_media(command_id)
        if (
            command_id not in self._vision_media_failures
            and len(self._vision_media_failures) >= _VISION_MEDIA_PENDING_STREAMS
        ):
            oldest_failure = min(
                self._vision_media_failure_since,
                key=self._vision_media_failure_since.__getitem__,
            )
            self._vision_media_failures.pop(oldest_failure, None)
            self._vision_media_failure_since.pop(oldest_failure, None)
        self._vision_media_failures[command_id] = message
        self._vision_media_failure_since[command_id] = time.monotonic()
        sender = self._vision_media_packet_sender
        if sender is not None:
            with suppress(BrokenResourceError, ClosedResourceError):
                with anyio.move_on_after(2, shield=True):
                    await sender.send(packet.transport_failure(message))
        task = next(
            (
                task
                for task in self.state.tasks.values()
                if isinstance(task, (TextGeneration, ImageEdits))
                and task.command_id == command_id
            ),
            None,
        )
        if task is not None:
            self._vision_media_failures.pop(command_id, None)
            self._vision_media_failure_since.pop(command_id, None)
            await self.event_sender.send(
                TaskFailed(
                    task_id=task.task_id,
                    error_type="invalid_vision_media",
                    error_message=message,
                )
            )

    def _clear_pending_vision_media(
        self,
        command_id: CommandId,
        *,
        release_bytes: bool = True,
        clear_age: bool = True,
    ) -> None:
        """Delete unverified packet storage while preserving assembled input."""

        self._vision_media_opened.pop(command_id, None)
        self._vision_media_chunks.pop(command_id, None)
        self._vision_media_completed.pop(command_id, None)
        if release_bytes:
            released = self._vision_media_pending_bytes.pop(command_id, 0)
            self._vision_media_pending_total_bytes = max(
                0, self._vision_media_pending_total_bytes - released
            )
        if clear_age:
            self._vision_media_pending_since.pop(command_id, None)

    def _clear_vision_media(self, command_id: CommandId) -> None:
        """Delete all request-scoped vision input retained for one command."""

        self._clear_pending_vision_media(command_id)
        self._vision_media_verified.pop(command_id, None)
        self._vision_media_verified_chunks.pop(command_id, None)
        self._vision_media_accepted.discard(command_id)
        self._vision_media_failures.pop(command_id, None)
        self._vision_media_failure_since.pop(command_id, None)
        self.input_chunk_buffer.pop(command_id, None)
        self.input_chunk_counts.pop(command_id, None)

    async def _vision_media_janitor(self) -> None:
        """Reject incomplete image streams after their bounded lifetime."""

        while True:
            await anyio.sleep(5)
            for command_id in tuple(self._vision_media_verified):
                if (
                    command_id not in self._vision_media_accepted
                    and command_id not in self._vision_media_ack_inflight
                ):
                    self._tg.start_soon(
                        self._acknowledge_vision_media_if_admitted,
                        command_id,
                    )
            cutoff = time.monotonic() - _VISION_MEDIA_PENDING_TTL_SECONDS
            expired = [
                command_id
                for command_id, started_at in self._vision_media_pending_since.items()
                if started_at <= cutoff
            ]
            for command_id in expired:
                packet = self._vision_media_completed.get(command_id)
                if packet is None:
                    packet = self._vision_media_verified.get(command_id)
                if packet is None:
                    packet = next(
                        iter(self._vision_media_chunks.get(command_id, {}).values()),
                        None,
                    )
                if packet is None:
                    packet = self._vision_media_opened.get(command_id)
                if packet is not None:
                    self._vision_media_expired_streams += 1
                    await self._reject_vision_media(
                        packet, "Vision media ingress timed out before completion"
                    )
                else:
                    self._clear_vision_media(command_id)
            expired_failures = [
                command_id
                for command_id, failed_at in self._vision_media_failure_since.items()
                if failed_at <= cutoff
            ]
            for command_id in expired_failures:
                self._vision_media_failures.pop(command_id, None)
                self._vision_media_failure_since.pop(command_id, None)

    def collect_vision_media_ingress_diagnostics(
        self,
    ) -> VisionMediaIngressDiagnostics:
        """Return bounded worker-side image upload occupancy and outcomes."""

        return VisionMediaIngressDiagnostics(
            pending_api_commands=0,
            pending_api_bytes=0,
            active_api_commands=0,
            active_api_bytes=0,
            pending_worker_acknowledgements=0,
            active_streams=len(self._vision_media_pending_since),
            pending_frames=sum(
                len(chunks) for chunks in self._vision_media_chunks.values()
            ),
            retained_bytes=self._vision_media_pending_total_bytes,
            verified_streams=len(self._vision_media_verified),
            pending_failures=len(self._vision_media_failures),
            completed_streams=self._vision_media_completed_streams,
            rejected_streams=self._vision_media_rejected_streams,
            expired_streams=self._vision_media_expired_streams,
        )

    def _clear_speech_media(self, command_id: CommandId) -> None:
        """Delete all request-scoped media retained for one command."""

        self._speech_media_chunks.pop(command_id, None)
        self._speech_media_completed.pop(command_id, None)
        self._speech_media_pending_bytes.pop(command_id, None)
        self._speech_media_pending_since.pop(command_id, None)

    async def _fail_speech_media_task(self, task: SpeechSynthesis, message: str) -> None:
        """Terminate a synthesis task whose request-scoped media is invalid."""

        self._clear_speech_media(task.command_id)
        await self.event_sender.send(
            TaskFailed(
                task_id=task.task_id,
                error_type="invalid_reference_audio",
                error_message=message,
            )
        )

    async def _speech_media_janitor(self) -> None:
        """Cancel reference media that never acquires a serving task."""

        while True:
            await anyio.sleep(5)
            cutoff = time.monotonic() - _SPEECH_MEDIA_PENDING_TTL_SECONDS
            expired = [
                command_id
                for command_id, started_at in self._speech_media_pending_since.items()
                if started_at <= cutoff
            ]
            for command_id in expired:
                self._clear_speech_media(command_id)
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id,
                        command=TaskCancelled(cancelled_command_id=command_id),
                    )
                )

    async def _route_realtime_audio_frame(
        self, frame: RealtimeAudioInputFrame
    ) -> None:
        """Route one local or remote frame through shared bounded worker state."""

        if frame.command_id in self._realtime_finished_commands:
            return
        runner = self._realtime_runner_by_command.get(frame.command_id)
        if runner is not None:
            await runner.send_realtime_audio(frame)
            if frame.kind != "chunk":
                self._finish_realtime_command(frame.command_id)
            return
        pending = self._realtime_audio_pending.setdefault(frame.command_id, [])
        self._realtime_audio_pending_since.setdefault(
            frame.command_id, time.monotonic()
        )
        pending_bytes = self._realtime_audio_pending_bytes.get(frame.command_id, 0)
        if (
            len(pending) >= _REALTIME_AUDIO_PENDING_FRAMES
            or pending_bytes + len(frame.data) > _REALTIME_AUDIO_PENDING_BYTES
        ):
            self._realtime_audio_pending.pop(frame.command_id, None)
            self._realtime_audio_pending_bytes.pop(frame.command_id, None)
            self._realtime_audio_pending_since.pop(frame.command_id, None)
            logger.warning(
                "Realtime STT input exceeded the pre-dispatch bound "
                f"for command {frame.command_id}; cancelling"
            )
            await self.command_sender.send(
                ForwarderCommand(
                    origin=self._system_id,
                    command=TaskCancelled(cancelled_command_id=frame.command_id),
                )
            )
            self._mark_realtime_command_finished(frame.command_id)
            return
        pending.append(frame)
        self._realtime_audio_pending_bytes[frame.command_id] = pending_bytes + len(
            frame.data
        )

    async def _realtime_audio_janitor(self) -> None:
        """Cancel and release pre-dispatch audio that never acquires a task."""

        while True:
            await anyio.sleep(5)
            cutoff = time.monotonic() - _REALTIME_AUDIO_PENDING_TTL_SECONDS
            expired = [
                command_id
                for command_id, started_at in self._realtime_audio_pending_since.items()
                if started_at <= cutoff
                and command_id not in self._realtime_runner_by_command
            ]
            for command_id in expired:
                logger.warning(
                    "Realtime STT input expired before task dispatch for "
                    f"command {command_id}; cancelling"
                )
                self._finish_realtime_command(command_id)
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id,
                        command=TaskCancelled(cancelled_command_id=command_id),
                    )
                )

    def _mark_realtime_command_finished(self, command_id: CommandId) -> None:
        """Bound the late-frame tombstones retained by the worker."""

        if command_id in self._realtime_finished_commands:
            return
        self._realtime_finished_commands.add(command_id)
        self._realtime_finished_order.append(command_id)
        while len(self._realtime_finished_order) > _REALTIME_FINISHED_COMMANDS:
            expired = self._realtime_finished_order.popleft()
            self._realtime_finished_commands.discard(expired)

    def _finish_realtime_command(self, command_id: CommandId) -> None:
        """Release one realtime input route only after task termination."""

        self._realtime_runner_by_command.pop(command_id, None)
        self._realtime_audio_pending.pop(command_id, None)
        self._realtime_audio_pending_bytes.pop(command_id, None)
        self._realtime_audio_pending_since.pop(command_id, None)
        self._mark_realtime_command_finished(command_id)

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        # Connectivity readings (network interfaces, thunderbolt) ride the ordered
        # event log. They rarely change, but were emitted every poll, so on a
        # long-lived cluster they filled the master's replay tail and stormed the
        # AMD nodes on join (the 5-node gossip storm). Forward a reading only when
        # its value differs from the last value the master CONFIRMED (echoed back
        # as an indexed event; see _event_applier, which records the echo into
        # _confirmed_forwarded_info). Gating on confirmation rather than on "last
        # sent" matters: the delivery retry (event_router ``out_for_delivery``) is
        # bounded (retry_max_attempts), so a change sent during a long masterless
        # window can be dropped permanently; gating on the echo means we simply
        # re-send it on the next poll until it actually lands in the log, and go
        # quiet once it does. No periodic keepalive is needed; node liveness is
        # carried by the telemetry plane (the master prune and node-health read
        # the explicit telemetry heartbeat), not by these events.
        with recv as info_stream:
            async for info in info_stream:
                try:
                    # Telemetry plane (#279): live readings are gossiped on the
                    # telemetry topic, off the event log.
                    if (
                        isinstance(info, TELEMETRY_PLANE_INFO)
                        and self._telemetry_sender is not None
                    ):
                        await self._telemetry_sender.send(
                            NodeTelemetry(node_id=self.node_id, info=info)
                        )
                        continue
                    # Event-log-bound reading: skip only when the log already
                    # holds this exact value (confirmed by the master's echo).
                    if self._confirmed_forwarded_info.get(type(info)) == info:
                        continue
                    await self.event_sender.send(
                        NodeGatheredInfo(
                            node_id=self.node_id,
                            when=str(datetime.now(tz=timezone.utc)),
                            info=info,
                        )
                    )
                except (ClosedResourceError, BrokenResourceError):
                    logger.debug(
                        "Worker info forwarding stopped because the event stream "
                        "was already closed"
                    )
                    return

    async def _event_applier(self):
        with self.event_receiver as events:
            async for event in events:
                # 2. for each event, apply it to the state
                previous_tasks = self.state.tasks
                self.state = apply(self.state, event=event)
                event = event.event

                # Confirm our own connectivity readings once the master echoes
                # them back indexed; _forward_info gates on this so an
                # unconfirmed change keeps re-sending (see its comment).
                if (
                    isinstance(event, NodeGatheredInfo)
                    and event.node_id == self.node_id
                ):
                    self._confirmed_forwarded_info[type(event.info)] = event.info

                # Prune telemetry for timed-out nodes from the worker applier.
                # The API applier does the same; together they cover --no-api
                # and --no-worker nodes (#279 slice 2).
                if self._telemetry_view is not None:
                    record_membership_from_event(self._telemetry_view, event)

                # Batch STT still uses the legacy event-sourced input path.
                # Vision payloads arrive on VISION_MEDIA and never enter State.
                if isinstance(event, InputChunkReceived) and isinstance(
                    event.chunk, AudioInputChunk
                ):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_audio_chunk_buffer:
                        self.input_audio_chunk_buffer[cmd_id] = {}
                        self.input_audio_chunk_counts[cmd_id] = (
                            event.chunk.total_chunks
                        )
                    self.input_audio_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk
                    )

                if isinstance(event, TaskCreated) and isinstance(
                    event.task, (TextGeneration, ImageEdits)
                ):
                    vision_failure = self._vision_media_failures.pop(
                        event.task.command_id, None
                    )
                    self._vision_media_failure_since.pop(event.task.command_id, None)
                    if vision_failure is not None:
                        await self.event_sender.send(
                            TaskFailed(
                                task_id=event.task.task_id,
                                error_type="invalid_vision_media",
                                error_message=vision_failure,
                            )
                        )
                    else:
                        self._tg.start_soon(
                            self._acknowledge_vision_media_if_admitted,
                            event.task.command_id,
                        )

                if (
                    cleanup_cmd_id := _audio_input_cleanup_command_id(
                        event, previous_tasks, self.state.tasks
                    )
                ) is not None:
                    self.input_audio_chunk_buffer.pop(cleanup_cmd_id, None)
                    self.input_audio_chunk_counts.pop(cleanup_cmd_id, None)

                if (
                    realtime_cleanup_cmd_id := _realtime_input_cleanup_command_id(
                        event, previous_tasks, self.state.tasks
                    )
                ) is not None:
                    self._finish_realtime_command(realtime_cleanup_cmd_id)

                if (
                    speech_cleanup_cmd_id := _speech_media_cleanup_command_id(
                        event, previous_tasks, self.state.tasks
                    )
                ) is not None:
                    self._clear_speech_media(speech_cleanup_cmd_id)

                if (
                    vision_cleanup_cmd_id := _vision_media_cleanup_command_id(
                        event, previous_tasks, self.state.tasks
                    )
                ) is not None:
                    self._clear_vision_media(vision_cleanup_cmd_id)

                if isinstance(event, CustomModelCardAdded):
                    try:
                        await event.model_card.save_to_custom_dir()
                        add_to_card_cache(event.model_card)
                    except Exception:
                        logger.exception(
                            f"Failed to save custom model card (model_id={event.model_card.model_id})"
                        )

                if isinstance(event, CustomModelCardDeleted):
                    try:
                        await delete_custom_card(event.model_id)
                    except Exception:
                        logger.exception(
                            f"Failed to delete custom model card (model_id={event.model_id})"
                        )

                if isinstance(event, StagedModelEvicted):
                    # A store-delete removed the canonical copy; drop our local
                    # staged copy too (#427). evict_shard never touches the
                    # store's canonical files and is a no-op when nothing is
                    # staged here, so this is safe to broadcast to every node.
                    await self._evict_staged_model(event.model_id)

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            if self._stale_downloads_pending_reset:
                await self._reset_stale_downloads_from_state()
                if self._state_still_advertises_evicted_downloads():
                    # The DownloadPending resets round-trip through the
                    # master before our replicated state drops the stale
                    # DownloadCompleted entries; planning against the
                    # stale state could dispatch a load straight at the
                    # files we just deleted. Skip planning until the
                    # resets land (bounded: see the deadline below).
                    self._stale_reset_wait_ticks += 1
                    if self._stale_reset_wait_ticks <= _STALE_RESET_MAX_WAIT_TICKS:
                        continue
                    logger.warning(
                        "Worker: stale download resets have not applied "
                        f"after {_STALE_RESET_MAX_WAIT_TICKS} planning "
                        "ticks; resuming planning anyway (the coordinator "
                        "self-heals missing files at download time)"
                    )
                    self._stale_downloads_pending_reset.clear()
                    self._stale_resets_sent.clear()
                    self._stale_reset_wait_ticks = 0
            # Bound the crash breaker's memory: drop entries for instances that
            # no longer exist. We deliberately don't clear on give-up (that would
            # let a lingering instance re-trip and re-send DeleteInstance), so
            # this is where dead-instance keys are reclaimed.
            self._crash_breaker.retain(self.state.instances)

            # Wedge-marked LOCAL runner deaths give their instance up here, on
            # observation (see _wedged_live_instances). The breaker's
            # edge-latch makes each fire exactly once per instance per loop.
            for _wedge_iid, _wedge_model in _wedged_live_instances(
                self.runners, self.state.instances
            ):
                if self._crash_breaker.record(_wedge_iid):
                    await self._give_up_on_instance(
                        _wedge_iid,
                        f"runner for {_wedge_model} died with a suspected GPU "
                        f"wedge ({WEDGE_FAILURE_MARKER}); not retrying - each "
                        "wedge attempt leaks wired GPU memory. If this node's "
                        "available memory dropped, a reboot is the only way "
                        "to reclaim it.",
                    )

            # First-report deadline (#272): a runner frozen between spawn and its
            # first status report stalls pre-init coordination forever (the
            # group-init gate waits for every rank, and the breaker never trips
            # because the process is alive). Give the instance up via the same
            # edge-latched breaker so it fires once and the master can recover or
            # surface the failure instead of the instance "loading" forever.
            for _stall_iid, _stall_model in runners_never_reported(
                self.runners,
                self.state.instances,
                time.monotonic(),
                _RUNNER_FIRST_REPORT_DEADLINE_SECONDS,
            ):
                if self._crash_breaker.record(_stall_iid):
                    await self._give_up_on_instance(
                        _stall_iid,
                        f"runner for {_stall_model} did not report any status "
                        f"within {_RUNNER_FIRST_REPORT_DEADLINE_SECONDS:.0f}s of "
                        "spawn (suspected frozen process); giving up so it does "
                        "not stall pre-init coordination forever.",
                    )

            task: Task | None = plan(
                self.node_id,
                self.runners,
                self._effective_downloads(),
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.input_chunk_buffer,
                self.input_audio_chunk_buffer,
                self._speech_media_completed,
            )
            if task is None:
                continue

            # Gate DownloadModel on backoff BEFORE emitting TaskCreated
            # to prevent flooding the event log with useless events
            if isinstance(task, DownloadModel):
                model_id = task.shard_metadata.model_card.model_id
                if not self._download_backoff.should_proceed(model_id):
                    continue

            logger.info(f"Worker plan: {_summarize_worker_task(task)}")
            assert task.task_status
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    # The local fit guard sizes a shard's footprint against this
                    # node's memory, but an RPC placement's shares are decided
                    # by llama.cpp at load (the driver's shard nominally spans
                    # ALL layers and a donor holds none), so per-shard sizing
                    # is meaningless here. Pooled admission already used the
                    # strict per-node VRAM figure; a genuine misfit fails at
                    # llama-server load and the crash cascade recovers (#328).
                    fit_error = (
                        None
                        if isinstance(task.bound_instance.instance, LlamaRpcInstance)
                        else self._local_shard_fit_error(
                            task.bound_instance.bound_shard
                        )
                    )
                    if fit_error is not None:
                        logger.error(fit_error)
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Failed
                            )
                        )
                        # Memory refusal (not a crash/wedge): ask the master to
                        # re-place wider instead of silently deleting (#290).
                        if self._crash_breaker.record(task.instance_id):
                            await self._refuse_instance_placement(
                                task.instance_id, fit_error
                            )
                    else:
                        self._create_supervisor(task)
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Complete
                            )
                        )
                case DownloadModel(shard_metadata=shard):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(model_id)

                    found_path = resolve_model_in_path(model_id)
                    if found_path is not None:
                        logger.info(
                            f"Model {model_id} found in SKULK_MODELS_PATH at {found_path}"
                        )
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=True,
                                )
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id,
                                command=StartDownload(
                                    target_node_id=self.node_id,
                                    shard_metadata=shard,
                                ),
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Running,
                            )
                        )
                case Shutdown(runner_id=runner_id):
                    runner = self.runners.pop(runner_id)
                    shard_for_eviction = runner.shard_metadata
                    # Only evict staged files if the instance has been deleted.
                    # If the instance still exists (e.g., runner crashed but will
                    # be retried), keep the files so the next runner can find them.
                    instance_deleted = task.instance_id not in self.state.instances
                    try:
                        with fail_after(3):
                            await runner.start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                    finally:
                        runner.shutdown()
                        if instance_deleted:
                            await self._maybe_evict_shard(shard_for_eviction)
                        elif _runner_failed_wedged(self.state.runners.get(runner_id)):
                            # GPU-wedge deaths are never retried: the wedge is
                            # deterministic for the model that triggered it, and
                            # each wedge-exit leaks ~a shard of wired GPU memory
                            # that only a reboot reclaims (measured 2026-06-09:
                            # two retries cost a 24GB node ~10GB). Latch the
                            # breaker too so a lingering instance can't re-trip.
                            self._crash_breaker.record(task.instance_id)
                            await self._give_up_on_instance(
                                task.instance_id,
                                f"runner for "
                                f"{shard_for_eviction.model_card.model_id} died "
                                "with a suspected GPU wedge "
                                f"({WEDGE_FAILURE_MARKER}); not retrying - each "
                                "wedge attempt leaks wired GPU memory. If this "
                                "node's available memory dropped, a reboot is "
                                "the only way to reclaim it.",
                            )
                        elif self._crash_breaker.record(task.instance_id):
                            # Runner keeps crashing (e.g. OOM on load). Give up
                            # rather than relaunching into another leak-on-abort.
                            await self._give_up_on_instance(
                                task.instance_id,
                                f"runner for "
                                f"{shard_for_eviction.model_card.model_id} crashed "
                                f"{_RUNNER_CRASH_THRESHOLD}x within "
                                f"{_RUNNER_CRASH_WINDOW_SECONDS:.0f}s "
                                "(likely insufficient memory)",
                            )
                        else:
                            # Runner crashed but instance still exists and the
                            # breaker has not tripped - reset download status so
                            # the planner re-stages the model instead of assuming
                            # it's still on disk, and retry.
                            logger.info(
                                f"Worker: resetting download status for "
                                f"{shard_for_eviction.model_card.model_id} after runner crash"
                            )
                            await self.event_sender.send(
                                NodeDownloadProgress(
                                    download_progress=DownloadPending(
                                        node_id=self.node_id,
                                        shard_metadata=shard_for_eviction,
                                        model_directory="",
                                        attempt_id=DownloadAttemptId(),
                                    )
                                )
                            )
                case CancelTask(
                    cancelled_task_id=cancelled_task_id, runner_id=runner_id
                ):
                    await self.runners[runner_id].cancel_task(cancelled_task_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case ImageEdits() if task.task_params.total_input_chunks > 0:
                    # Assemble image from chunks and inject into task
                    cmd_id = task.command_id
                    chunks = self.input_chunk_buffer.get(cmd_id, {})
                    completed = self._vision_media_verified.get(cmd_id)
                    if (
                        completed is None
                        or completed.model != ModelId(task.task_params.model)
                        or task.owner_node is None
                        or completed.source_node != task.owner_node
                        or completed.total_chunks
                        != task.task_params.total_input_chunks
                        or completed.image_count != 1
                        or {chunk.image_index for chunk in chunks.values()} != {0}
                    ):
                        self._clear_vision_media(cmd_id)
                        await self.event_sender.send(
                            TaskFailed(
                                task_id=task.task_id,
                                error_type="invalid_vision_media",
                                error_message=(
                                    "Verified image-edit input does not match the "
                                    "admitted task"
                                ),
                            )
                        )
                        continue
                    assembled = "".join(chunks[i].data for i in range(len(chunks)))
                    logger.info(
                        f"Assembled input image from {len(chunks)} chunks, "
                        f"total size: {len(assembled)} bytes"
                    )
                    _log_image_payload_debug(
                        "Worker assembled image edit", 0, assembled
                    )
                    # Inject the assembled image while preserving every other
                    # field (notably owner_node for Zenoh routing - #279 Phase 2 /
                    # #310 review).
                    modified_task = _inject_assembled_image_edit(task, assembled)
                    # Cleanup buffers
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    self._vision_media_verified.pop(cmd_id, None)
                    self._vision_media_verified_chunks.pop(cmd_id, None)
                    self._clear_pending_vision_media(cmd_id)
                    await self._start_runner_task(modified_task)

                case TextGeneration() if (
                    task.task_params.image_hashes
                    or task.task_params.total_input_chunks > 0
                ):
                    cmd_id = task.command_id
                    by_index, missing_cached_images = resolve_cached_vlm_images(
                        self.image_cache,
                        task.task_params.image_hashes,
                    )
                    if missing_cached_images:
                        missing = ", ".join(
                            f"{idx}:{image_hash[:12]}"
                            for idx, image_hash in missing_cached_images
                        )
                        logger.error(
                            "TextGeneration VLM task references missing cached "
                            f"image(s); failing task instead of crashing worker "
                            f"(task_id={task.task_id}, command_id={cmd_id}, "
                            f"model={task.task_params.model}, missing={missing})"
                        )
                        self._clear_vision_media(cmd_id)
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Failed,
                            )
                        )
                        continue

                    if task.task_params.total_input_chunks > 0:
                        chunk_buffer = self.input_chunk_buffer.get(cmd_id, {})
                        completed = self._vision_media_verified.get(cmd_id)
                        received_image_indexes = {
                            chunk.image_index for chunk in chunk_buffer.values()
                        }
                        cached_image_indexes = set(task.task_params.image_hashes)
                        all_image_indexes = set(
                            range(
                                task.task_params.image_count
                                + len(cached_image_indexes)
                            )
                        )
                        if (
                            completed is None
                            or completed.model != task.task_params.model
                            or task.owner_node is None
                            or completed.source_node != task.owner_node
                            or completed.total_chunks
                            != task.task_params.total_input_chunks
                            or completed.image_count != task.task_params.image_count
                            or len(received_image_indexes) != completed.image_count
                            or not received_image_indexes.isdisjoint(
                                cached_image_indexes
                            )
                            or received_image_indexes | cached_image_indexes
                            != all_image_indexes
                        ):
                            self._clear_vision_media(cmd_id)
                            await self.event_sender.send(
                                TaskFailed(
                                    task_id=task.task_id,
                                    error_type="invalid_vision_media",
                                    error_message=(
                                        "Verified VLM image input does not match "
                                        "the admitted task"
                                    ),
                                )
                            )
                            continue
                        per_image: defaultdict[int, list[InputImageChunk]] = (
                            defaultdict(list)
                        )
                        for chunk in chunk_buffer.values():
                            per_image[chunk.image_index].append(chunk)
                        for img_idx in sorted(per_image):
                            sorted_chunks = sorted(
                                per_image[img_idx], key=lambda c: c.chunk_index
                            )
                            img = "".join(c.data for c in sorted_chunks)
                            b64_sha256 = hashlib.sha256(img.encode("ascii")).hexdigest()
                            self.image_cache[b64_sha256] = img
                            by_index[img_idx] = img
                            _log_image_payload_debug(
                                "Worker assembled VLM",
                                img_idx,
                                img,
                                expected_b64_sha256=b64_sha256,
                            )
                        logger.info(
                            f"Assembled {len(per_image)} VLM image(s) "
                            f"from {len(chunk_buffer)} chunks"
                        )

                    resolved_images = [by_index[i] for i in sorted(by_index)]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    self._vision_media_verified.pop(cmd_id, None)
                    self._vision_media_verified_chunks.pop(cmd_id, None)
                    self._clear_pending_vision_media(cmd_id)
                    await self._start_runner_task(modified_task)
                case SpeechSynthesis() if task.task_params.reference_audio_present:
                    command_id = task.command_id
                    chunks = self._speech_media_chunks.get(command_id, {})
                    completed = self._speech_media_completed.get(command_id)
                    if completed is None:
                        continue
                    try:
                        reference_audio = b"".join(
                            chunks[index].data for index in range(completed.sequence)
                        )
                    except KeyError as exc:
                        logger.error(
                            "Reference audio reached dispatch with missing chunk "
                            f"{exc.args[0]} for command {command_id}"
                        )
                        await self._fail_speech_media_task(
                            task,
                            f"Reference audio is missing chunk {exc.args[0]}",
                        )
                        continue
                    if hashlib.sha256(reference_audio).hexdigest() != completed.sha256:
                        logger.error(
                            f"Reference audio checksum mismatch for command {command_id}"
                        )
                        await self._fail_speech_media_task(
                            task,
                            "Reference audio checksum mismatch",
                        )
                        continue
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"reference_audio_data": reference_audio}
                            )
                        }
                    )
                    self._clear_speech_media(command_id)
                    await self._start_runner_task(modified_task)
                case AudioTranscription() if task.task_params.total_input_chunks > 0:
                    cmd_id = task.command_id
                    chunks = self.input_audio_chunk_buffer.get(cmd_id, {})
                    try:
                        assembled = "".join(
                            chunks[i].data
                            for i in range(task.task_params.total_input_chunks)
                        )
                    except KeyError as exc:
                        logger.error(
                            "AudioTranscription task reached dispatch with missing "
                            f"audio chunk {exc.args[0]} "
                            f"(task_id={task.task_id}, command_id={cmd_id})"
                        )
                        self.input_audio_chunk_buffer.pop(cmd_id, None)
                        self.input_audio_chunk_counts.pop(cmd_id, None)
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Failed,
                            )
                        )
                        continue
                    logger.info(
                        f"Assembled speech input from {len(chunks)} chunks, "
                        f"base64 size: {len(assembled)} bytes"
                    )
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"audio_data": assembled}
                            )
                        }
                    )
                    self.input_audio_chunk_buffer.pop(cmd_id, None)
                    self.input_audio_chunk_counts.pop(cmd_id, None)
                    await self._start_runner_task(modified_task)
                case task:
                    await self._start_runner_task(task)

    async def shutdown(self):
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is not None:
            runner_id = instance.shard_assignments.node_to_runner[self.node_id]
            shard = instance.shard(runner_id)
            if (
                isinstance(task, LoadModel)
                and shard is not None
                # Same bypass as the CreateRunner guard: an RPC driver's
                # bookkeeping shard nominally spans ALL layers, but llama.cpp
                # splits the actual allocation across the donors at load, so
                # sizing the whole model against this node refuses exactly the
                # pooled-only placements the feature exists for. Pooled
                # admission already checked the strict per-node VRAM figures;
                # a genuine misfit fails at llama-server load and the crash
                # cascade recovers (#328).
                and not isinstance(instance, LlamaRpcInstance)
            ):
                # Re-check fit at load dispatch. The CreateRunner guard runs
                # before download and before any concurrently-placed instance
                # has loaded, so this is the last accurate point - current free
                # memory now reflects those other loads - to refuse before the
                # runner allocates and risks an OOM-abort that leaks GPU memory.
                fit_error = self._local_shard_fit_error(shard)
                if fit_error is not None:
                    logger.error(fit_error)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Failed
                        )
                    )
                    # Memory refusal (not a crash/wedge): ask the master to
                    # re-place wider instead of silently deleting (#290).
                    if self._crash_breaker.record(task.instance_id):
                        await self._refuse_instance_placement(
                            task.instance_id, fit_error
                        )
                    return
            logger.info(
                "Dispatching worker task "
                f"({_summarize_worker_task(task)}, "
                f"target_runner_id={runner_id}, "
                f"device_rank={shard.device_rank if shard is not None else 'unknown'}, "
                f"world_size={shard.world_size if shard is not None else 'unknown'})"
            )
            runner = self.runners[runner_id]
            if isinstance(task, RealtimeAudioTranscription):
                command_id = task.command_id
                try:
                    await runner.start_task(task)
                    # The runner acknowledges before entering its receive loop.
                    # Flush while ingress still buffers, then publish the direct
                    # route only once no older frame remains. This preserves
                    # order and prevents a full IPC queue from blocking task
                    # dispatch before the runner can drain it.
                    while command_id not in self._realtime_finished_commands:
                        pending = self._realtime_audio_pending.pop(command_id, [])
                        self._realtime_audio_pending_bytes.pop(command_id, None)
                        self._realtime_audio_pending_since.pop(command_id, None)
                        if not pending:
                            self._realtime_runner_by_command[command_id] = runner
                            break
                        terminal_forwarded = False
                        for frame in pending:
                            await runner.send_realtime_audio(frame)
                            if frame.kind != "chunk":
                                terminal_forwarded = True
                                break
                        if terminal_forwarded:
                            self._finish_realtime_command(command_id)
                            break
                except BaseException:
                    self._finish_realtime_command(command_id)
                    raise
                if task.task_id not in runner.in_progress:
                    self._finish_realtime_command(command_id)
                return
            await runner.start_task(task)

    def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        shard = task.bound_instance.bound_shard
        logger.info(
            "Creating runner supervisor "
            f"(instance_id={task.instance_id}, "
            f"runner_id={task.bound_instance.bound_runner_id}, "
            f"node_id={task.bound_instance.bound_node_id}, "
            f"device_rank={shard.device_rank}, "
            f"world_size={shard.world_size}, "
            f"layers={shard.start_layer}:{shard.end_layer})"
        )
        # Context-admission ceiling (#145/#279 slice 2): the master computed
        # this once at placement time and stamped it into the event-sourced
        # placement decision, so every rank reads the identical value here.
        # Recomputing from node memory would now be non-deterministic - memory
        # moved to the last-write-wins telemetry plane and is no longer in the
        # ordered, replicated event log.
        context_token_limit = task.bound_instance.instance.context_token_limit
        logger.info(
            "Context admission limit for instance "
            f"{task.instance_id}: {context_token_limit} tokens"
        )
        runner = RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
            context_token_limit=context_token_limit,
            data_sender=self._data_sender.clone()
            if self._data_sender is not None
            else None,
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    def collect_runner_diagnostics(self) -> list[RunnerSupervisorDiagnostics]:
        """Return live read-only diagnostics for local runner supervisors."""

        return [runner.diagnostics() for runner in self.runners.values()]

    async def cancel_runner_task(
        self,
        runner_id: RunnerId,
        task_id: TaskId,
    ) -> RunnerTaskCancelResponse:
        """Request cooperative cancellation for one live task on one runner."""

        runner = self.runners.get(runner_id)
        if runner is None:
            raise KeyError(f"Runner not found on this node: {runner_id}")

        if task_id in runner.completed:
            return RunnerTaskCancelResponse(
                node_id=self.node_id,
                runner_id=runner_id,
                task_id=task_id,
                status="already_completed",
                message="Task already completed; no cancellation was sent.",
            )
        if task_id in runner.cancelled:
            return RunnerTaskCancelResponse(
                node_id=self.node_id,
                runner_id=runner_id,
                task_id=task_id,
                status="already_cancelled",
                message="Task was already marked cancelled on this runner.",
            )
        if task_id not in runner.in_progress and task_id not in runner.pending:
            raise KeyError(
                f"Task {task_id} is not pending or in progress on runner {runner_id}"
            )

        await runner.cancel_task(task_id)
        return RunnerTaskCancelResponse(
            node_id=self.node_id,
            runner_id=runner_id,
            task_id=task_id,
            status="cancel_requested",
            message=(
                "Cooperative cancellation requested on the live runner. "
                "Native or wedged work may continue until the runner observes it."
            ),
        )

    def _models_in_use(self) -> frozenset[str]:
        """Repo-form IDs of every model a live runner depends on.

        Includes companion repos (MTP sidecar, assistant, split vision
        weights) of active models: no instance names them directly, but
        evicting one corrupts a live runner just the same - MLX loads
        weights lazily.
        """

        def _add_card(card: ModelCard) -> None:
            in_use.add(str(card.model_id))
            if card.vision and card.vision.weights_repo:
                in_use.add(card.vision.weights_repo)
            runtime = card.runtime
            if runtime is not None:
                if runtime.mtp_sidecar_repo:
                    in_use.add(runtime.mtp_sidecar_repo)
                if runtime.assistant_model_repo:
                    in_use.add(runtime.assistant_model_repo)

        in_use: set[str] = set()
        for runner in self.runners.values():
            _add_card(runner.shard_metadata.model_card)
        # A store-backed download in progress has already created its
        # staging directory but no runner exists yet - a concurrent
        # teardown's budget pass must not delete a directory that is
        # actively being written.
        for progress in self._effective_downloads().get(self.node_id, []):
            if isinstance(progress, DownloadOngoing):
                _add_card(progress.shard_metadata.model_card)
        return frozenset(in_use)

    async def _evict_staged_model(self, model_id: ModelId) -> None:
        """Remove this node's staged copy of a store-deleted model (#427).

        Unlike ``_maybe_evict_shard`` (LRU budget pass on teardown), this is a
        direct, unconditional eviction triggered by a ``StagedModelEvicted``
        broadcast. ``evict_shard`` is a no-op when nothing is staged here and
        never touches the store's canonical files; the apply() side already drops
        the model's download entries from State, so this only frees disk.
        """
        if self._store_client is None or self._staging_config is None:
            return
        cache_path = Path(self._staging_config.node_cache_path).expanduser()
        try:
            await self._store_client.evict_shard(model_id, cache_path)
        except Exception:
            logger.exception(
                f"Failed to evict staged model on store-delete (model_id={model_id})"
            )

    async def _maybe_evict_shard(self, shard: ShardMetadata | None) -> None:
        """Hold the staging cache to its recent-use budget after teardown.

        The torn-down model becomes an eviction candidate like any other
        not-in-use staged copy; the grace budget decides what actually goes
        (so repeated place/delete cycles of the same model do not re-pay
        the staging copy every time).
        """
        if (
            shard is None
            or self._store_client is None
            or self._staging_config is None
            or not self._staging_config.cleanup_on_deactivate
        ):
            return
        # The just-deactivated model was in use until this very moment -
        # refresh its last-use marker (and its companions') BEFORE the
        # budget pass, or a long-staged but heavily-used model sorts as
        # old and gets evicted despite being the most recently used thing
        # on the node (the downloader only touches markers when it runs,
        # and reuse from an existing DownloadCompleted skips it).
        self._touch_staged_model_and_companions(shard.model_card)
        # The walk + rmtree are synchronous filesystem work on potentially
        # tens of GB - run off the event loop so teardown doesn't stall
        # other worker tasks. The in-use snapshot is taken HERE, on the
        # loop thread: the threaded pass must not iterate self.runners /
        # self.state while the loop mutates them.
        models_in_use = self._models_in_use()
        report = await to_thread.run_sync(self._enforce_staging_budget, models_in_use)
        if report is None:
            return
        await self._reset_download_state_for_evicted(report, shard)

    async def _reset_download_state_for_evicted(
        self, report: StagingEvictionReport, deactivated_shard: ShardMetadata
    ) -> None:
        """Reset download status for EVERY evicted model on this node.

        The budget pass can evict models other than the one just torn down;
        leaving them DownloadCompleted in master state would make the
        planner skip re-staging and fail later loads. Shard metadata for
        the other models comes from this node's own download-status state.
        """
        if self._staging_config is None:
            return
        cache_path = Path(self._staging_config.node_cache_path).expanduser()
        # Keyed by forward-sanitized directory name: the report's model ids
        # are best-effort inverses of directory names and can be ambiguous
        # for ids containing "--" - sanitizing both sides forward makes the
        # match exact.
        own_downloads = {
            staging_directory_name(
                str(progress.shard_metadata.model_card.model_id)
            ): progress.shard_metadata
            for progress in self._effective_downloads().get(self.node_id, [])
        }
        deactivated_directory = staging_directory_name(
            str(deactivated_shard.model_card.model_id)
        )
        for evicted_model_id in report.evicted_model_ids:
            evicted_directory = staging_directory_name(evicted_model_id)
            shard_metadata = (
                deactivated_shard
                if evicted_directory == deactivated_directory
                else own_downloads.get(evicted_directory)
            )
            if shard_metadata is None:
                # Never advertised as downloaded on this node - nothing to
                # reset (e.g. a companion repo staged without its own
                # download entry).
                continue
            pending = DownloadPending(
                shard_metadata=shard_metadata,
                node_id=self.node_id,
                model_directory=str(cache_path / evicted_model_id.replace("/", "--")),
                attempt_id=DownloadAttemptId(),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )

    def _touch_staged_model_and_companions(self, card: ModelCard) -> None:
        """Refresh .last_used for a model (and companions) just taken out of use."""
        if self._staging_config is None:
            return
        cache_path = Path(self._staging_config.node_cache_path).expanduser()
        model_ids = [str(card.model_id)]
        if card.vision and card.vision.weights_repo:
            model_ids.append(card.vision.weights_repo)
        if card.runtime is not None:
            if card.runtime.mtp_sidecar_repo:
                model_ids.append(card.runtime.mtp_sidecar_repo)
            if card.runtime.assistant_model_repo:
                model_ids.append(card.runtime.assistant_model_repo)
        for model_id in model_ids:
            staged_dir = cache_path / model_id.replace("/", "--")
            if staged_dir.is_dir():
                touch_last_used(staged_dir)

    def _enforce_staging_budget(
        self, models_in_use: frozenset[str]
    ) -> StagingEvictionReport | None:
        """Run one staging-budget enforcement pass (best-effort).

        ``models_in_use`` is snapshotted by the caller on the event-loop
        thread - this method may run in a worker thread and must not touch
        the loop's mutable structures.
        """
        if self._staging_config is None or not self._staging_config.enabled:
            return None
        staging_root = Path(self._staging_config.node_cache_path).expanduser()
        # A store host configured for direct loading points node_cache_path
        # at the CANONICAL store directory (a common override that was safe
        # under the old no-cleanup default). Eviction there would delete the
        # cluster's only copy of every model beyond the budget - refuse,
        # whatever the config says (codex review on #215).
        if self._store_client is not None:
            local_store_path = self._store_client.local_store_path
            if (
                local_store_path is not None
                and local_store_path.expanduser().resolve() == staging_root.resolve()
            ):
                logger.warning(
                    "Worker: staging eviction skipped - node_cache_path is "
                    f"the canonical store directory ({staging_root}); the "
                    "store is never evicted. Set a separate staging path if "
                    "this node should have a managed staging cache."
                )
                return None
        keep_recent_bytes = int(self._staging_config.staging_keep_recent_gb * 1024**3)
        try:
            return enforce_staging_budget(
                staging_root,
                keep_recent_bytes,
                models_in_use,
            )
        except Exception as exc:
            logger.warning(f"Worker: staging budget enforcement failed: {exc}")
            return None

    def _state_still_advertises_evicted_downloads(self) -> bool:
        """True while replicated state shows DownloadCompleted for a model
        whose staged files the startup pass deleted."""
        for progress in self.state.downloads.get(self.node_id, []):
            if (
                isinstance(progress, DownloadCompleted)
                and staging_directory_name(
                    str(progress.shard_metadata.model_card.model_id)
                )
                in self._stale_downloads_pending_reset
            ):
                return True
        return False

    async def _reset_stale_downloads_from_state(self) -> None:
        """Counter stale DownloadCompleted entries for startup-evicted models.

        Runs from plan_step while any evicted ID lacks its reset: once the
        replicated state shows a DownloadCompleted for this node naming an
        evicted model, emit DownloadPending with that entry's own shard
        metadata so the planner re-stages instead of dispatching a load
        against deleted files.
        """
        if self._staging_config is None:
            self._stale_downloads_pending_reset.clear()
            self._stale_resets_sent.clear()
            return
        cache_path = Path(self._staging_config.node_cache_path).expanduser()
        still_completed: set[str] = set()
        for progress in self.state.downloads.get(self.node_id, []):
            if not isinstance(progress, DownloadCompleted):
                continue
            model_id = str(progress.shard_metadata.model_card.model_id)
            directory_name = staging_directory_name(model_id)
            if directory_name not in self._stale_downloads_pending_reset:
                continue
            staged_dir = cache_path / directory_name
            if staged_dir.exists():
                # Re-staged since eviction - state is truthful again.
                self._stale_downloads_pending_reset.discard(directory_name)
                self._stale_resets_sent.discard(directory_name)
                continue
            still_completed.add(directory_name)
            if directory_name in self._stale_resets_sent:
                continue  # reset in flight; keep gating until it applies
            pending = DownloadPending(
                shard_metadata=progress.shard_metadata,
                node_id=self.node_id,
                model_directory=str(staged_dir),
                attempt_id=DownloadAttemptId(),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
            self._stale_resets_sent.add(directory_name)
            logger.info(
                f"Worker: reset stale DownloadCompleted for {model_id} "
                "(staged files were evicted at startup)"
            )
        # Anything we sent a reset for that state no longer advertises has
        # APPLIED - only then does it stop gating the planner.
        for directory_name in list(self._stale_resets_sent):
            if directory_name not in still_completed:
                self._stale_downloads_pending_reset.discard(directory_name)
                self._stale_resets_sent.discard(directory_name)
        if not self._stale_downloads_pending_reset:
            self._stale_reset_wait_ticks = 0

    def _reconcile_staging_on_startup(self) -> None:
        """Reconcile staging orphans left by a crashed or killed session.

        A node that dies never runs the deactivate-time eviction, so its
        staged copies survive forever without this. At startup nothing is
        in use yet, so every staged model is a candidate and the grace
        budget alone decides what survives - which is exactly the crash
        recovery behavior we want: recent models stay warm for the
        restart, the old tail goes.
        """
        if (
            self._staging_config is None
            or not self._staging_config.cleanup_on_deactivate
        ):
            return
        report = self._enforce_staging_budget(self._models_in_use())
        if report is not None and report.evicted_model_ids:
            logger.info(
                "Worker: startup staging reconciliation evicted "
                f"{len(report.evicted_model_ids)} orphaned model(s) "
                f"({report.evicted_bytes / 2**30:.1f} GiB)"
            )
            # The master may still hold DownloadCompleted entries for the
            # files we just deleted, but this runs BEFORE state replay -
            # there is no shard metadata to build the reset events from
            # yet. Remember the DIRECTORY names (forward-sanitized; the
            # report's repo-form ids are best-effort inverses and would be
            # ambiguous for ids containing "--"); plan_step counters the
            # stale entries as soon as they appear in replicated state
            # (the plan layer consults state.downloads, so leaving them
            # would skip re-staging and fail a later load).
            self._stale_downloads_pending_reset.update(
                staging_directory_name(model_id)
                for model_id in report.evicted_model_ids
            )

    async def _poll_connection_updates(self):
        while True:
            edges = set(
                conn.edge for conn in self.state.topology.out_edges(self.node_id)
            )
            conns: defaultdict[NodeId, set[str]] = defaultdict(set)
            async for ip, nid in check_reachable(
                self.state.topology,
                self.node_id,
                self.state.node_network,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)
                edge = SocketConnection(
                    # nonsense multiaddr
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415")
                    if "." in ip
                    # nonsense multiaddr
                    else Multiaddr(address=f"/ip6/{ip}/tcp/52415"),
                )
                if edge not in edges:
                    logger.debug(f"ping discovered {edge=}")
                    await self.event_sender.send(
                        TopologyEdgeCreated(
                            conn=Connection(source=self.node_id, sink=nid, edge=edge)
                        )
                    )

            for conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn.edge, SocketConnection):
                    continue
                # ignore mDNS discovered connections
                if conn.edge.sink_multiaddr.port != 52415:
                    continue
                if (
                    conn.sink not in conns
                    or conn.edge.sink_multiaddr.ip_address not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(10)
