import base64
import hashlib
import io
from collections import defaultdict
from collections.abc import Container, Mapping
from datetime import datetime, timezone
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, fail_after, to_thread
from loguru import logger
from PIL import Image

from skulk.download.download_utils import resolve_model_in_path
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
from skulk.shared.types.chunks import DataChunk, InputImageChunk
from skulk.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    RefuseInstancePlacement,
    StartDownload,
)
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.diagnostics import (
    RunnerSupervisorDiagnostics,
    RunnerTaskCancelResponse,
)
from skulk.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    IndexedEvent,
    InputChunkReceived,
    NodeDownloadProgress,
    NodeGatheredInfo,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.state import State
from skulk.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    LoadModel,
    Shutdown,
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
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
)
from skulk.shared.types.worker.instances import InstanceId
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

    Returns ``None`` on Apple unified-memory nodes (no amdgpu device), which keep
    the system-RAM path. The master is the backend authority that decides a shard
    belongs on this GPU node in the first place.
    """
    from skulk.utils.info_gatherer.linux_gpu import (
        find_amd_gpu_device,
        read_accelerator_metrics,
    )

    device = find_amd_gpu_device()
    if device is None:
        return None
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
        telemetry_sender: Sender[NodeTelemetry] | None = None,
        telemetry_view: TelemetryView | None = None,
        data_sender: Sender[DataChunk] | None = None,
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
        # Shared, Node-owned telemetry view (#279). The worker prunes a node's
        # telemetry here when it sees NodeTimedOut, because the worker runs on
        # EVERY node regardless of role - so a --no-api node (or a --no-api
        # master placing off this view) still drops dead nodes, where an
        # API-only prune hook would leak unbounded across churn.
        self._telemetry_view = telemetry_view
        self._store_client = store_client
        self._staging_config = staging_config

        self.state: State = State()
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
        if footprint > usable:
            return (
                f"Refusing to load a shard of {shard.model_card.model_id} on "
                f"{self.node_id}: ~{footprint.in_gb:.1f}GB needed but only "
                f"~{usable.in_gb:.1f}GB usable locally ({pool}). Refusing before "
                "load to avoid an OOM abort that leaks GPU memory."
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

    async def run(self):
        logger.info("Starting Worker")
        self._reconcile_staging_on_startup()

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send)

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()
            self._stopped.set()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                try:
                    # Telemetry plane (#279): live readings are gossiped on the
                    # telemetry topic, off the event log. The remaining node_*
                    # readings still travel as indexed NodeGatheredInfo events
                    # until later slices migrate them.
                    if (
                        isinstance(info, TELEMETRY_PLANE_INFO)
                        and self._telemetry_sender is not None
                    ):
                        await self._telemetry_sender.send(
                            NodeTelemetry(node_id=self.node_id, info=info)
                        )
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
                self.state = apply(self.state, event=event)
                event = event.event

                # Prune telemetry for timed-out nodes from the worker applier.
                # The API applier does the same; together they cover --no-api
                # and --no-worker nodes (#279 slice 2).
                if self._telemetry_view is not None:
                    record_membership_from_event(self._telemetry_view, event)

                # Buffer input image chunks for image editing
                if isinstance(event, InputChunkReceived):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_chunk_buffer:
                        self.input_chunk_buffer[cmd_id] = {}
                        self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

                    self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk
                    )

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

            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.input_chunk_buffer,
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
                    fit_error = self._local_shard_fit_error(
                        task.bound_instance.bound_shard
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
                        self.input_chunk_buffer.pop(cmd_id, None)
                        self.input_chunk_counts.pop(cmd_id, None)
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Failed,
                            )
                        )
                        continue

                    if task.task_params.total_input_chunks > 0:
                        chunk_buffer = self.input_chunk_buffer.get(cmd_id, {})
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
            if isinstance(task, LoadModel) and shard is not None:
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
            await self.runners[runner_id].start_task(task)

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
        for progress in self.state.downloads.get(self.node_id, []):
            if isinstance(progress, DownloadOngoing):
                _add_card(progress.shard_metadata.model_card)
        return frozenset(in_use)

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
            for progress in self.state.downloads.get(self.node_id, [])
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
