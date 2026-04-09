import base64
import hashlib
import io
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, fail_after
from loguru import logger
from PIL import Image

from exo.api.types import ImageEditsTaskParams
from exo.download.download_utils import resolve_model_in_path
from exo.shared.apply import apply
from exo.shared.constants import EXO_IMAGE_TRANSPORT_DEBUG
from exo.shared.models.model_cards import ModelId, add_to_card_cache, delete_custom_card
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.events import (
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
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.store.config import StagingNodeConfig
from exo.store.model_store_client import ModelStoreClient
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.plan import plan
from exo.worker.runner.runner_supervisor import RunnerSupervisor


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
            "Shutdown("
            f"instance_id={task.instance_id!r}, "
            f"runner_id={task.runner_id!r})"
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


def _log_image_transport(message: str) -> None:
    """Emit image transport logs only at INFO when explicitly requested.

    Multimodal traffic can be large and frequent, so the default path keeps
    these diagnostics at DEBUG. Operators can opt in with
    ``SKULK_IMAGE_TRANSPORT_DEBUG=true`` (or the legacy ``EXO_*`` alias) when
    they need end-to-end payload fingerprints.
    """
    if EXO_IMAGE_TRANSPORT_DEBUG:
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

    if EXO_IMAGE_TRANSPORT_DEBUG:
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
        store_client: ModelStoreClient | None = None,
        staging_config: StagingNodeConfig | None = None,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self._store_client = store_client
        self._staging_config = staging_config

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.image_cache: dict[str, str] = {}

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._stopped: anyio.Event = anyio.Event()

    async def run(self):
        logger.info("Starting Worker")

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
                            f"Model {model_id} found in EXO_MODELS_PATH at {found_path}"
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
                        elif shard_for_eviction is not None:
                            # Runner crashed but instance still exists — reset
                            # download status so the planner re-stages the model
                            # instead of assuming it's still on disk.
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
                    _log_image_payload_debug("Worker assembled image edit", 0, assembled)
                    # Create modified task with assembled image data
                    modified_task = ImageEdits(
                        task_id=task.task_id,
                        command_id=task.command_id,
                        instance_id=task.instance_id,
                        task_status=task.task_status,
                        task_params=ImageEditsTaskParams(
                            image_data=assembled,
                            total_input_chunks=task.task_params.total_input_chunks,
                            prompt=task.task_params.prompt,
                            model=task.task_params.model,
                            n=task.task_params.n,
                            quality=task.task_params.quality,
                            output_format=task.task_params.output_format,
                            response_format=task.task_params.response_format,
                            size=task.task_params.size,
                            image_strength=task.task_params.image_strength,
                            bench=task.task_params.bench,
                            stream=task.task_params.stream,
                            partial_images=task.task_params.partial_images,
                            advanced_params=task.task_params.advanced_params,
                        ),
                    )
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
                    by_index: dict[int, str] = {}

                    for idx, h in task.task_params.image_hashes.items():
                        assert h in self.image_cache
                        by_index[idx] = self.image_cache[h]
                        _log_image_payload_debug(
                            "Worker resolved cached VLM",
                            idx,
                            by_index[idx],
                            expected_b64_sha256=h,
                        )

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
        runner = RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    async def _maybe_evict_shard(self, shard: ShardMetadata | None) -> None:
        """Evict staged shard files after runner teardown if configured."""
        if (
            shard is None
            or self._store_client is None
            or self._staging_config is None
            or not self._staging_config.cleanup_on_deactivate
        ):
            return
        model_id = str(shard.model_card.model_id)
        # Skip eviction if another runner for the same model is still active —
        # MLX loads weights lazily, so tearing down a sibling's staged files
        # would corrupt the live runner.
        if any(
            str(r.shard_metadata.model_card.model_id) == model_id
            for r in self.runners.values()
        ):
            logger.debug(
                f"Worker: skipping eviction for {model_id} — another runner is still active"
            )
            return
        cache_path = Path(self._staging_config.node_cache_path).expanduser()
        try:
            await self._store_client.evict_shard(model_id, cache_path)
            logger.info(f"Worker: evicted staged files for {model_id}")
            # Reset download status so the dashboard no longer shows
            # this model as downloaded on this node.
            pending = DownloadPending(
                shard_metadata=shard,
                node_id=self.node_id,
                model_directory=str(cache_path / model_id.replace("/", "--")),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
        except Exception as exc:
            logger.warning(f"Worker: evict_shard failed for {model_id}: {exc}")

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
