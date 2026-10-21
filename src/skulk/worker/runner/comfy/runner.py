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
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast, final

import psutil
from loguru import logger
from PIL import Image

from skulk.shared.backends import (
    COMFY_BIN_ENV,
    COMFY_ROOT_ENV,
    ENGINE_UNAVAILABLE_FAILURE_MARKER,
)
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
    ConnectToGroup,
    LoadModel,
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
from skulk.shared.types.video_motion import MAX_MOTION_TRACK_BYTES, VideoMotionTrackData
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
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.comfy.graph import (
    NODE_SAVE_THUMBNAIL,
    NODE_SAVE_VIDEO,
    ComfyModelSearchRoot,
    ComfyRenderPlan,
    bind_references,
    build_prompt,
    companion_file,
    companion_model_folder,
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
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch
from skulk.worker.runner.video_input import (
    MAX_SOURCE_VIDEO_BYTES,
    MediaInputRequest,
    VerifiedMediaInput,
)
from skulk.worker.runner.video_input_process import prepare_media_input

_HASH_CHUNK: Final = 1 << 20
_THUMBNAIL_QUALITY: Final = 85


ROCM_LAUNCH_FLAGS: Final[tuple[str, ...]] = ("--bf16-vae",)
"""ComfyUI flags for the ROCm lane, validated for H3 on Strix Halo (gfx1151).

``--bf16-vae`` because the fp32 VAE decode of a 768p clip does not fit beside
the transformer. Weights are memory-mapped (below ``ROCM_MMAP_CEILING_BYTES``)
and models stay resident between renders under ComfyUI's RAM-pressure cache,
with the headroom from ``ROCM_CACHE_RAM_HEADROOM_FRACTION`` added at launch.
H3 keeps ComfyUI's default SDPA path, which already selects the wheel set's
flash-class kernel on gfx1151. Family-specific attention policy is added
separately by ``launch_flags``.
"""


ROCM_CACHE_RAM_HEADROOM_FRACTION: Final = 0.4
"""Share of host RAM ComfyUI's cache keeps free on the ROCm lane (``--cache-ram``).

A HIP allocation on a unified-memory APU comes out of host RAM, so the cached
models and the running render compete for it. At ComfyUI's default headroom
(10%) a new prompt made the text encoder and the transformer evict each other:
on gfx1151 with 61 GiB of host RAM a warm 480x480 four-step H3 render took 180
to 300 s, worse than rebuilding every model per prompt (about 210 s). Keeping
40% free (24 GiB there) holds the models resident without that fight: a warm
render with a new prompt takes about 105 s, the turbo LoRA is applied once
rather than every step, and host memory peaks lower. Outputs agree with the
rebuilt path as closely as that path agrees with itself run to run.
"""


ROCM_MMAP_CEILING_BYTES: Final = 64 * 1024**3
"""Largest weight file the ROCm lane memory-maps.

Memory-mapping a safetensors file above 64 GB through the unified-memory
path is pathologically slow, so a card with such a file launches with
``--disable-mmap``; the pruned H3 files (the largest is 21 GB) map.
"""


CUDA_LAUNCH_FLAGS: Final[tuple[str, ...]] = ("--disable-cuda-malloc",)
"""ComfyUI flags for the CUDA lane: PyTorch's own caching allocator.

ComfyUI defaults CUDA devices to the ``cudaMallocAsync`` allocator backend.
With it, a MiniMax H3 render that applies the Fun ControlNet patch aborts
the server on its first sampling step on the GB10 (``cuMemFreeAsync``
returns ``CUDA_ERROR_INVALID_VALUE`` inside the model's forward, where
ComfyUI's allocation graph hands back memory the async backend did not
record). The native allocator renders the same graph, and a plain render
measured the same on both (43.2 s against 42.1 s for 480x480 at four
steps), so the CUDA lane uses it for every render rather than only for
guided ones: one server serves consecutive renders.
"""


def _local_comfy_backend() -> str | None:
    """The node's own advertised ``comfy-<compute>`` tag, for an unstamped shard."""
    from skulk.shared.backends import probe_node_backends

    tags = sorted(tag for tag in probe_node_backends() if tag.startswith("comfy-"))
    return tags[0] if tags else None


def _is_rocm_lane(resolved_backend: str | None) -> bool:
    """Whether this launch is the ROCm lane.

    The backend is the stamped ``comfy-<compute>`` tag; an unstamped shard
    (telemetry still warming, or a manual launch), or one stamped with the
    bare engine tag by a card that declares no compute, uses the tag this
    node advertises itself, so a managed ROCm install never launches without
    the settings it was qualified with.
    """
    backend = resolved_backend if resolved_backend is not None and "-" in resolved_backend else _local_comfy_backend()
    return backend is not None and backend.endswith("-rocm")


def launch_flags(
    resolved_backend: str | None,
    weight_files: Iterable[Path] = (),
    total_ram_bytes: int | None = None,
    *,
    model_family: str | None = None,
) -> tuple[str, ...]:
    """Return launch flags for the backend, selected weights and model family.

    The H3 ROCm lane adds ``ROCM_LAUNCH_FLAGS``. Every ROCm family adds the RAM-pressure cache headroom
    (``ROCM_CACHE_RAM_HEADROOM_FRACTION`` of ``total_ram_bytes``, the host's
    RAM when omitted, in whole GiB), and ``--disable-mmap`` when any of
    ``weight_files`` is larger than ``ROCM_MMAP_CEILING_BYTES``; every other
    lane, CUDA, adds ``CUDA_LAUNCH_FLAGS``. An ``ltx-2.5`` family CUDA server
    also uses disk-backed dynamic loading so CPU weight staging competes less
    with GPU allocations on unified-memory systems. It still executes on the
    GPU and retains ordinary memory admission. LTX on ROCm instead selects the
    engine's split attention path to preserve prompt subject adherence and
    leaves VAE precision to each component's supported dtypes. This also
    executes on the GPU. No files or settings are changed.
    """
    if not _is_rocm_lane(resolved_backend):
        if model_family == "ltx-2.5":
            # Keeping the large LTX encoder and transformer in RAM offload
            # storage can exhaust a unified-memory host before sampling.
            # The pinned engine's disk-backed path avoids that residency;
            # low-VRAM mode would instead move text encoding onto the CPU.
            return (*CUDA_LAUNCH_FLAGS, "--fast-disk")
        return CUDA_LAUNCH_FLAGS
    ram = total_ram_bytes if total_ram_bytes is not None else psutil.virtual_memory().total
    headroom_gib = max(1, int(ram * ROCM_CACHE_RAM_HEADROOM_FRACTION / 1024**3))
    # Forcing BF16 overrides the LTX audio VAE's FP32-only working dtype;
    # its mel preprocessor returns FP32 and source encoding then fails in
    # the first BF16 convolution. Let Comfy select each LTX VAE's dtype.
    vae_flags = () if model_family == "ltx-2.5" else ROCM_LAUNCH_FLAGS
    flags = (*vae_flags, "--cache-ram", str(headroom_gib))
    if model_family == "ltx-2.5":
        # The pinned ROCm default produced unrelated subjects in matched LTX
        # renders; split attention recovered the requested subjects with the
        # same weights, graph and seed while keeping inference on the GPU.
        flags = (*flags, "--use-split-cross-attention")
    if any(path.stat().st_size > ROCM_MMAP_CEILING_BYTES for path in weight_files):
        return (*flags, "--disable-mmap")
    return flags


def _configured_install(resolved_backend: str | None = None) -> tuple[Path, Path]:
    """The interpreter and checkout this runner starts ComfyUI from.

    An operator override (``SKULK_COMFY_BIN`` with ``SKULK_COMFY_ROOT``) wins.
    Without one, the managed install on disk is used: the runner process
    starts before an on-demand install finishes, so it cannot rely on the
    environment the worker exported afterwards. A half-set override stays an
    error rather than silently falling back.
    """
    interpreter = os.environ.get(COMFY_BIN_ENV, "").strip()
    root = os.environ.get(COMFY_ROOT_ENV, "").strip()
    if not interpreter and not root:
        from skulk.provisioning.comfy import installed_managed_comfy

        managed = installed_managed_comfy(resolved_backend)
        if managed is not None:
            return managed
        raise RuntimeError(
            f"{ENGINE_UNAVAILABLE_FAILURE_MARKER}: the video engine is not "
            "installed on this node, so the comfy runner cannot start ComfyUI."
        )
    if not interpreter or not root:
        raise RuntimeError(
            f"{COMFY_BIN_ENV} and {COMFY_ROOT_ENV} are not both set; the comfy runner cannot start "
            "ComfyUI. Set both to an install, or unset both to use the managed one."
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


def _companion_search_roots(card: ModelCard) -> tuple[ComfyModelSearchRoot, ...]:
    """Resolve each external companion's exact staged path and loader search root.

    Every file the card names from such a repository must be present: the
    downloader fetches them with the card and treats them as load-bearing, so
    one missing means an incomplete install, reported by name rather than
    left for ComfyUI to fail on mid-render.
    """
    from skulk.download.download_utils import build_sidecar_path

    if card.video is None:
        return ()
    roots: list[ComfyModelSearchRoot] = []
    for repository, revision in card.external_video_companions():
        for item in card.video.companions:
            if item.repo is None or str(item.repo) != repository or item.revision != revision:
                continue
            found = build_sidecar_path(ModelId(repository), item.path, revision)
            if found is None:
                raise RuntimeError(
                    f"{card.model_id}: companion {item.name} ({repository} {item.path}) is not staged"
                )
            # Root files and conventional loras/foo layouts must produce the
            # same loader name while retaining their immutable upstream paths.
            name = companion_file(item, card.artifact_repository)
            root = ComfyModelSearchRoot(
                folder=companion_model_folder(item),
                directory=found.parents[len(PurePosixPath(name).parts) - 1],
            )
            if root not in roots:
                roots.append(root)
    return tuple(roots)


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
class Runner(ServedConcurrentDispatch):
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
        # Width 1: one ComfyUI server renders one prompt at a time, but the
        # shared loop acknowledges every render on admission and holds the
        # next ones in its queue, oldest first, so the worker plans on (a
        # cancel included) while a render waits or runs, however many are
        # waiting. The queue has no bound of its own: the API admits a
        # bounded number of jobs per node, and a bound here would block the
        # loop on the render past it (the very hold this fixes) and let that
        # render take the slot ahead of the queue.
        self._init_concurrent_dispatch(1, "comfy-render", queue_admitted=True)
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

    # --- main loop ------------------------------------------------------------

    _generation_kinds = (VideoGeneration,)

    def main(self) -> None:
        """Serve tasks until shutdown through the shared served-dispatch loop.

        Renders stay strictly serial (width 1), but a render is acknowledged the
        moment it is admitted rather than when the server gets to it. The
        worker's control loop waits on that acknowledgement before it plans
        again, so without this a second render queued behind a running one
        froze the worker until the first finished, and no cancel could reach
        either of them. Shutdown and the liveness poll are the loop's as well;
        the loop tears the server down on every exit.
        """
        self.run_dispatch_loop()

    def _teardown_server(self) -> None:
        if self.server is not None:
            self.server.teardown()
            self.server = None

    def _ensure_server_alive(self) -> None:
        """Raise if the server the runner should have is gone.

        Before ``LoadModel`` there is no server to check and the loop's idle
        poll must not end the runner; once the runner has been ready, a
        missing or dead server is fatal, and the supervisor restarts it.
        """
        if self.server is None:
            if isinstance(self.current_status, (RunnerReady, RunnerRunning)):
                raise RuntimeError("the ComfyUI server is not running; log tail:\n(no server)")
            return
        if not self.server.alive():
            tail = self.server.log_tail()
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
            case _:
                # VideoGeneration and Shutdown are the dispatch loop's.
                raise ValueError(f"unexpected task {type(task).__name__} in state {self.current_status}")

    def _generate(self, task: Task) -> None:
        """Pool-worker body for one admitted render.

        Acknowledgement and the Ready/Running transitions belong to the dispatch
        loop. A render cancelled while it waited for the server is skipped here
        without touching ComfyUI; one cancelled while it runs is interrupted by
        ``run_prompt``, which polls the cancel set between events.
        """
        assert isinstance(task, VideoGeneration)
        command_id = task.command_id
        if self._is_cancelled(task.task_id):
            logger.info(f"comfy engine render skipped (cancelled before it started) for {command_id}")
            return
        self._ensure_server_alive()
        try:
            self._render(task.task_id, command_id, task.task_params)
            # A cancel that landed while the container was collected sits in
            # the pipe; read it now so the terminal status says cancelled.
            self._is_cancelled(task.task_id)
        except ValueError as error:
            # The request or the plan was refused (ComfyRenderError included):
            # the task fails with its reason and the runner lives on.
            self._mark_generation_failed(task.task_id)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(model=self.model_id, finish_reason="error", error_message=str(error)),
                )
            )
            shutil.rmtree(SKULK_VIDEO_OUTPUT_DIR / str(command_id), ignore_errors=True)
            logger.warning(f"comfy engine rejected {command_id}: {error}")
        except Exception:
            # Anything else is the server or the runner failing. The pool
            # keeps the loop alive past this thread, so the server goes here
            # and the loop's next liveness check, on the next task or the idle
            # poll, ends the runner for the supervisor to restart.
            self._mark_generation_failed(task.task_id)
            shutil.rmtree(SKULK_VIDEO_OUTPUT_DIR / str(command_id), ignore_errors=True)
            self._teardown_server()
            raise
        finally:
            (SKULK_VIDEO_INPUT_DIR / f"_prepared_{command_id}.wav").unlink(missing_ok=True)

    # --- model load -----------------------------------------------------------

    def _load_model(self) -> None:
        if self.card.video is None:
            raise RuntimeError(f"{self.model_id} has no [video] section")
        interpreter, root = _configured_install(self.shard_metadata.resolved_backend)
        from skulk.provisioning.comfy_extension import verified_comfy_extension

        try:
            extension = verified_comfy_extension(self.shard_metadata.resolved_engine_build, interpreter, root)
        except ValueError as error:
            # Load cannot repair missing engine bytes. Mark this terminal so
            # the worker explains the failed engine rather than crash-looping.
            raise RuntimeError(f"{ENGINE_UNAVAILABLE_FAILURE_MARKER}: {error}") from error
        model_dir = _model_directory(self.card)
        files = resolve_model_files(self.card)
        weight_files: list[Path] = []
        for folder, name in (
            ("diffusion_models", files.diffusion_model),
            ("text_encoders", files.text_encoder),
            ("vae", files.video_vae),
            ("vae", files.audio_vae),
        ):
            if name is None:
                continue
            if not (model_dir / folder / name).is_file():
                raise RuntimeError(f"{self.model_id}: {folder}/{name} is missing from {model_dir}")
            weight_files.append(model_dir / folder / name)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        extra_paths = self.work_dir / "extra_model_paths.yaml"
        extra_paths.write_text(extra_model_paths_yaml(model_dir, _companion_search_roots(self.card)))
        server = ComfyServer(
            interpreter=interpreter,
            root=root,
            extra_model_paths=extra_paths,
            input_dir=SKULK_VIDEO_INPUT_DIR,
            output_dir=SKULK_VIDEO_OUTPUT_DIR,
            temp_dir=self.work_dir / "temp",
            user_dir=self.work_dir / "user",
            log_path=self.work_dir / "server.log",
            trusted_extension=extension,
            extra_args=launch_flags(
                self.shard_metadata.resolved_backend,
                weight_files,
                model_family=self.card.family,
            ),
        )
        server.start()
        self.server = server
        logger.info(f"ComfyUI serving {self.model_id} from {model_dir} at {server.base_url}")

    # --- render ---------------------------------------------------------------

    def _render(self, task_id: TaskId, command_id: CommandId, params: VideoGenerationTaskParams) -> None:
        server = self.server
        assert server is not None and server.base_url is not None
        started = time.monotonic()
        render = plan_comfy_render(params, self.card, engine_build=self.shard_metadata.resolved_engine_build)
        bindings = bind_references(params.references, SKULK_VIDEO_INPUT_DIR)
        output_dir = SKULK_VIDEO_OUTPUT_DIR / str(command_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        if render.motion_tracks is not None:
            source = next(binding for binding in bindings if binding.spec.role == "motion_tracks")
            path = SKULK_VIDEO_INPUT_DIR / source.input_name
            if path.stat().st_size > MAX_MOTION_TRACK_BYTES:
                raise ValueError("Motion Track data exceeds the four-MiB bound")
            tracks = VideoMotionTrackData.model_validate_json(path.read_bytes())
            if tracks.frame_count != render.plan.frame_count:
                raise ValueError("Motion Track data must match the complete requested frame count")
        if render.union_control is not None:
            source = next(binding for binding in bindings if binding.spec.role == "control")
            # Validate every frame before Comfy materializes the IMAGE batch.
            # Exact alignment avoids an implicit retime, loop or held last frame.
            prepared = prepare_media_input(server.interpreter, MediaInputRequest(
                operation="video", source=str(SKULK_VIDEO_INPUT_DIR / source.input_name),
                duration_seconds=render.plan.frame_count / render.plan.fps,
                width=render.plan.width, height=render.plan.height, fps=render.plan.fps,
                maximum_frames=render.plan.frame_count, exact_frames=True,
            ), output_dir, is_cancelled=lambda: self._is_cancelled(task_id))
            if prepared is None:
                shutil.rmtree(output_dir, ignore_errors=True)
                return
        if render.editing is not None:
            source = next(binding for binding in bindings if binding.spec.kind == "video" and binding.spec.role == "source")
            if render.plan.width * render.plan.height * render.plan.frame_count * 12 > MAX_SOURCE_VIDEO_BYTES:
                raise ValueError("editing output exceeds the bounded decoded-pixel budget; reduce its size or duration")
            prepared = prepare_media_input(server.interpreter, MediaInputRequest(
                operation="video", source=str(SKULK_VIDEO_INPUT_DIR / source.input_name),
                duration_seconds=render.plan.frame_count / render.plan.fps,
                width=render.plan.width, height=render.plan.height, fps=render.plan.fps,
                maximum_frames=render.plan.frame_count,
                exact_frames=render.editing.operation == "retake",
                require_audio=render.plan.audio and not render.editing.regenerate_audio,
            ), output_dir, is_cancelled=lambda: self._is_cancelled(task_id))
            if prepared is None:
                shutil.rmtree(output_dir, ignore_errors=True)
                return
            if prepared.audio_channels not in (None, 1, 2):
                raise ValueError("LTX temporal editing requires mono or stereo source audio")
            if render.editing.operation == "extend" and prepared.frames >= render.plan.frame_count:
                raise ValueError("extension output must exceed the complete source clip")
            render = replace(render, source_frames=prepared.frames)
        if render.inpainting is not None:
            from skulk.worker.runner.comfy.inpaint_input import prepare_inpaint_media

            def prepare_inpaint(request: MediaInputRequest) -> VerifiedMediaInput | None:
                return prepare_media_input(server.interpreter, request, output_dir, is_cancelled=lambda: self._is_cancelled(task_id))

            facts = prepare_inpaint_media(render.plan, render.inpainting, bindings, SKULK_VIDEO_INPUT_DIR, prepare=prepare_inpaint)
            if facts is None:
                shutil.rmtree(output_dir, ignore_errors=True)
                return
            render = replace(render, inpaint_inputs=facts)
        if params.audio_conditioning is not None:
            source = next(binding for binding in bindings if binding.spec.kind == "audio")
            destination = SKULK_VIDEO_INPUT_DIR / f"_prepared_{command_id}.wav"
            prepared = prepare_media_input(server.interpreter, MediaInputRequest(
                operation="audio", source=str(SKULK_VIDEO_INPUT_DIR / source.input_name),
                destination=str(destination), start_seconds=params.audio_conditioning.start_seconds,
                duration_seconds=render.plan.frame_count / render.plan.fps,
            ), output_dir, is_cancelled=lambda: self._is_cancelled(task_id))
            if prepared is None:
                shutil.rmtree(output_dir, ignore_errors=True)
                return
            bindings = tuple(replace(binding, input_name=destination.name) if binding is source else binding for binding in bindings)
        prompt = build_prompt(render, params, bindings, str(command_id))
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
                total_steps=render.steps,
                sampler_steps=render.sampler_steps,
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
        steps_done = result.steps_observed or render.steps
        stats = VideoGenerationStats(
            steps=render.steps,
            seconds_per_step=result.sampling_seconds / max(1, steps_done),
            total_generation_time=time.monotonic() - started,
            engine=render.engine_settings().model_copy(update={"frame_count": manifest.frame_count}),
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
        actual_frames, actual_seconds = plan.frame_count, plan.seconds
        if render.duration == "auto":
            from skulk.worker.runner.video_probe import probe_video_output

            server = self.server
            assert server is not None
            actual = probe_video_output(final, plan, codec=render.codec, interpreter=server.interpreter, minimum_frames=render.minimum_frames)
            actual_frames, actual_seconds = actual.frame_count, actual.seconds
        return VideoOutputManifest(
            sha256=sha256,
            size_bytes=size,
            width=plan.width,
            height=plan.height,
            frame_count=actual_frames,
            fps=plan.fps,
            seconds=actual_seconds,
            audio_sample_rate=plan.sample_rate if plan.audio else None,
            audio_channels=plan.channels if plan.audio else None,
            thumbnail_sha256=thumbnail_sha256,
            thumbnail_size_bytes=thumbnail_size,
        )
