"""Runner subprocess for the ``comfy`` video engine.

Same state machine as the test video engine (``RunnerIdle -> RunnerLoaded ->
RunnerReady``, one ``VideoGeneration`` at a time, progress ``VideoChunk``
frames, a terminal chunk carrying the output manifest) with a headless
ComfyUI server behind it. ``LoadModel`` resolves the staged artifact,
exposes it to ComfyUI through an ``extra_model_paths.yaml``, and starts the
server; ComfyUI itself loads weights lazily on the first prompt and keeps
them resident between prompts. A request the engine cannot honor fails that
task only; a server that dies fails the runner so the supervisor restarts it.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Final, cast, final

from anyio import WouldBlock
from loguru import logger
from PIL import Image

from skulk.shared.backends import COMFY_BIN_ENV, COMFY_ROOT_ENV
from skulk.shared.constants import (
    SKULK_CACHE_HOME,
    SKULK_VIDEO_INPUT_DIR,
    SKULK_VIDEO_OUTPUT_DIR,
)
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.chunks import ErrorChunk, VideoChunk
from skulk.shared.types.common import CommandId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    VideoGeneration,
)
from skulk.shared.types.video import (
    VIDEO_OUTPUT_FILENAME,
    VIDEO_THUMBNAIL_FILENAME,
    VideoGenerationStats,
    VideoGenerationTaskParams,
    VideoOutputManifest,
    VideoStage,
)
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.comfy.graph import (
    NODE_SAVE_THUMBNAIL,
    NODE_SAVE_VIDEO,
    ComfyRenderPlan,
    bind_references,
    build_prompt,
    extra_model_paths_yaml,
    plan_comfy_render,
    resolve_model_files,
)
from skulk.worker.runner.comfy.server import (
    SKULK_LAUNCH_MARKER_DIR,
    ComfyRenderError,
    ComfyServer,
    RenderResult,
    run_prompt,
)

_HASH_CHUNK: Final = 1 << 20
_THUMBNAIL_QUALITY: Final = 85


def launch_flags(resolved_backend: str | None) -> tuple[str, ...]:
    """Extra ComfyUI flags for the node's compute backend.

    CUDA needs nothing beyond the headless defaults. The ROCm lane records its
    validated flags with the ROCm provisioning variant; until then a ROCm
    node launches with the same defaults.
    """
    del resolved_backend
    return ()


def _configured_install() -> tuple[Path, Path]:
    """The interpreter and checkout this node's facts exported."""
    interpreter = os.environ.get(COMFY_BIN_ENV, "").strip()
    root = os.environ.get(COMFY_ROOT_ENV, "").strip()
    if not interpreter or not root:
        raise RuntimeError(
            f"{COMFY_BIN_ENV} and {COMFY_ROOT_ENV} are not both set; the comfy runner cannot start "
            "ComfyUI. This node should not have been a placement candidate for a comfy card."
        )
    interpreter_path = Path(interpreter)
    root_path = Path(root)
    if not (interpreter_path.is_file() and os.access(interpreter_path, os.X_OK)):
        raise RuntimeError(f"{COMFY_BIN_ENV}={interpreter!r} is not an executable interpreter")
    if not (root_path / "main.py").is_file():
        raise RuntimeError(f"{COMFY_ROOT_ENV}={root!r} holds no ComfyUI main.py")
    return interpreter_path, root_path


def _model_directory(card: ModelCard) -> Path:
    from skulk.download.download_utils import (
        artifact_install_directory,
        build_model_path,
    )

    root = card.artifact_bundle.root if card.artifact_bundle is not None else None
    load_directory = build_model_path(ModelId(card.model_id), card.source_revision, root)
    return artifact_install_directory(load_directory, root)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _saved_file(outputs: dict[str, Any], node_id: str, output_root: Path) -> Path:
    """Locate the file a save node wrote, refusing anything outside the output root."""
    node_output = cast("dict[str, Any] | None", outputs.get(node_id))
    saved = cast("list[dict[str, Any]]", (node_output or {}).get("images") or [])
    if not saved:
        raise ComfyRenderError(f"ComfyUI completed the render but node {node_id!r} saved nothing")
    entry = saved[0]
    root = output_root.resolve()
    candidate = (root / str(entry.get("subfolder") or "") / str(entry.get("filename") or "")).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ComfyRenderError(f"ComfyUI reported {entry} for {node_id!r} but no such file exists under the output root")
    return candidate


@final
class Runner:
    """ComfyUI-backed video runner; same constructor shape as the image runner."""

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ) -> None:
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance
        self.runner_id = bound_instance.bound_runner_id
        self.shard_metadata = bound_instance.bound_shard
        self.card: ModelCard = self.shard_metadata.model_card
        self.model_id = self.card.model_id
        self.server: ComfyServer | None = None
        self.work_dir: Path = SKULK_CACHE_HOME / SKULK_LAUNCH_MARKER_DIR / str(self.runner_id)
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self.current_status: RunnerStatus = RunnerIdle()
        self.update_status(RunnerIdle())

    # --- events ---------------------------------------------------------------

    def update_status(self, status: RunnerStatus) -> None:
        """Record and publish a runner status."""
        self.current_status = status
        self.event_sender.send(RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status))

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Publish a task status transition."""
        self.event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=status))

    def acknowledge_task(self, task: Task) -> None:
        """Tell the supervisor the task was received."""
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    # --- cancellation ---------------------------------------------------------

    def _drain_cancellations(self) -> None:
        while True:
            try:
                cancelled = self.cancel_receiver.receive_nowait()
            except WouldBlock:
                return
            self.cancelled_tasks.add(cancelled)

    def _is_cancelled(self, task_id: TaskId) -> bool:
        self._drain_cancellations()
        return task_id in self.cancelled_tasks or CANCEL_ALL_TASKS in self.cancelled_tasks

    # --- main loop ------------------------------------------------------------

    def main(self) -> None:
        """Serve tasks until shutdown, tearing the server down on every exit."""
        try:
            with self.task_receiver as tasks:
                for task in tasks:
                    if task.task_id in self.seen:
                        logger.warning("repeat task - potential error")
                    self.seen.add(task.task_id)
                    self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
                    self.send_task_status(task, TaskStatus.Running)
                    self.handle_task(task)
                    if self._is_cancelled(task.task_id):
                        self.send_task_status(task, TaskStatus.Cancelled)
                    else:
                        self.send_task_status(task, TaskStatus.Complete)
                    self.update_status(self.current_status)
                    if isinstance(self.current_status, RunnerShutdown):
                        break
        finally:
            self._teardown_server()

    def _teardown_server(self) -> None:
        if self.server is not None:
            self.server.teardown()
            self.server = None

    def _ensure_server_alive(self) -> None:
        if self.server is None or not self.server.alive():
            tail = self.server.log_tail() if self.server is not None else "(no server)"
            self._teardown_server()
            raise RuntimeError(f"the ComfyUI server is not running; log tail:\n{tail}")

    def handle_task(self, task: Task) -> None:
        """Apply one task to the runner state machine."""
        match task:
            case ConnectToGroup() if isinstance(self.current_status, (RunnerIdle, RunnerFailed)):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.current_status = RunnerConnected()
            case LoadModel() if isinstance(self.current_status, (RunnerIdle, RunnerConnected)):
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)
                self._load_model()
                self.current_status = RunnerLoaded()
            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                # ComfyUI loads weights on the first prompt and keeps them
                # resident; a warm-up prompt would be a full render.
                self._ensure_server_alive()
                self.current_status = RunnerReady()
            case VideoGeneration(task_params=params, command_id=command_id) if isinstance(
                self.current_status, RunnerReady
            ):
                self.update_status(RunnerRunning())
                self.acknowledge_task(task)
                self._ensure_server_alive()
                try:
                    self._render(task.task_id, command_id, params)
                except Exception as error:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(model=self.model_id, finish_reason="error", error_message=str(error)),
                        )
                    )
                    shutil.rmtree(SKULK_VIDEO_OUTPUT_DIR / str(command_id), ignore_errors=True)
                    if not isinstance(error, ValueError):
                        raise
                    logger.warning(f"comfy engine rejected {command_id}: {error}")
                self.current_status = RunnerReady()
            case Shutdown():
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self._teardown_server()
                self.current_status = RunnerShutdown()
            case _:
                raise ValueError(f"unexpected task {type(task).__name__} in state {self.current_status}")

    # --- model load -----------------------------------------------------------

    def _load_model(self) -> None:
        if self.card.video is None:
            raise RuntimeError(f"{self.model_id} has no [video] section")
        interpreter, root = _configured_install()
        model_dir = _model_directory(self.card)
        files = resolve_model_files(self.card)
        for folder, name in (
            ("diffusion_models", files.diffusion_model),
            ("text_encoders", files.text_encoder),
            ("vae", files.video_vae),
            ("vae", files.audio_vae),
        ):
            if name is not None and not (model_dir / folder / name).is_file():
                raise RuntimeError(f"{self.model_id}: {folder}/{name} is missing from {model_dir}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        extra_paths = self.work_dir / "extra_model_paths.yaml"
        extra_paths.write_text(extra_model_paths_yaml(model_dir))
        server = ComfyServer(
            interpreter=interpreter,
            root=root,
            extra_model_paths=extra_paths,
            input_dir=SKULK_VIDEO_INPUT_DIR,
            output_dir=SKULK_VIDEO_OUTPUT_DIR,
            temp_dir=self.work_dir / "temp",
            user_dir=self.work_dir / "user",
            log_path=self.work_dir / "server.log",
            extra_args=launch_flags(self.shard_metadata.resolved_backend),
        )
        server.start()
        self.server = server
        logger.info(f"ComfyUI serving {self.model_id} from {model_dir} at {server.base_url}")

    # --- render ---------------------------------------------------------------

    def _render(self, task_id: TaskId, command_id: CommandId, params: VideoGenerationTaskParams) -> None:
        server = self.server
        assert server is not None and server.base_url is not None
        started = time.monotonic()
        render = plan_comfy_render(params, self.card)
        bindings = bind_references(params.references, SKULK_VIDEO_INPUT_DIR)
        prompt = build_prompt(render, params, bindings, str(command_id))
        output_dir = SKULK_VIDEO_OUTPUT_DIR / str(command_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"comfy engine rendering {render.mode.value} {render.plan.frame_count} frames at "
            f"{render.plan.width}x{render.plan.height} over {render.plan.steps} steps for {command_id}"
        )

        def progress(stage: VideoStage, step: int | None, total_steps: int | None, fraction: float) -> None:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=VideoChunk(
                        model=self.model_id, stage=stage, step=step, total_steps=total_steps, progress=fraction
                    ),
                )
            )

        result: RenderResult = asyncio.run(
            run_prompt(
                server.base_url,
                prompt,
                total_steps=render.plan.steps,
                on_progress=progress,
                is_cancelled=lambda: self._is_cancelled(task_id),
                server_alive=server.alive,
            )
        )
        if result.state == "cancelled":
            logger.info(f"comfy engine render cancelled for {command_id}")
            shutil.rmtree(output_dir, ignore_errors=True)
            return
        manifest = self._collect_output(result, render, output_dir)
        steps_done = result.steps_observed or render.plan.steps
        stats = VideoGenerationStats(
            steps=render.plan.steps,
            seconds_per_step=result.sampling_seconds / max(1, steps_done),
            total_generation_time=time.monotonic() - started,
        )
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=VideoChunk(
                    model=self.model_id,
                    stage="muxing",
                    progress=1.0,
                    output=manifest,
                    stats=stats,
                    finish_reason="stop",
                ),
            )
        )

    def _collect_output(self, result: RenderResult, render: ComfyRenderPlan, output_dir: Path) -> VideoOutputManifest:
        """Move ComfyUI's files onto the names the worker streams, and describe them."""
        container = _saved_file(result.outputs, NODE_SAVE_VIDEO, SKULK_VIDEO_OUTPUT_DIR)
        final = output_dir / VIDEO_OUTPUT_FILENAME
        container.replace(final)
        sha256, size = _sha256_and_size(final)
        frame = _saved_file(result.outputs, NODE_SAVE_THUMBNAIL, SKULK_VIDEO_OUTPUT_DIR)
        thumbnail = output_dir / VIDEO_THUMBNAIL_FILENAME
        with Image.open(frame) as image:
            image.convert("RGB").save(thumbnail, format="JPEG", quality=_THUMBNAIL_QUALITY)
        frame.unlink(missing_ok=True)
        thumbnail_sha256, thumbnail_size = _sha256_and_size(thumbnail)
        plan = render.plan
        return VideoOutputManifest(
            sha256=sha256,
            size_bytes=size,
            width=plan.width,
            height=plan.height,
            frame_count=plan.frame_count,
            fps=plan.fps,
            seconds=plan.seconds,
            audio_sample_rate=plan.sample_rate if plan.audio else None,
            audio_channels=plan.channels if plan.audio else None,
            thumbnail_sha256=thumbnail_sha256,
            thumbnail_size_bytes=thumbnail_size,
        )
