"""Serialize native image tasks through one resident, supervised GPU engine."""

from __future__ import annotations

import base64
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import cast, final

import httpx

from skulk.api.types.api import ImageEditsTaskParams, ImageGenerationStats
from skulk.provisioning.stable_diffusion_cpp import BINARY_ENV
from skulk.shared.constants import SKULK_CACHE_HOME
from skulk.shared.types.chunks import ErrorChunk
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import (
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
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
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.image_models.output import send_image_chunks
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch
from skulk.worker.runner.stable_diffusion_cpp.adapter import (
    native_component_arguments,
    native_image_request,
    validate_native_output,
)
from skulk.worker.runner.stable_diffusion_cpp.server import (
    NativeImageCancelledError,
    NativeImageServer,
    NativePreview,
    NativeResult,
)


@final
class Runner(ServedConcurrentDispatch):
    """Own one exact image model, bounded serial admission, and native job cleanup."""

    _generation_kinds = (ImageGeneration, ImageEdits)

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ) -> None:
        """Initialize the standard served lifecycle without loading any MLX modules."""
        self.bound_instance = bound_instance
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.shard_metadata = bound_instance.bound_shard
        self.runner_id = bound_instance.bound_runner_id
        self.card = self.shard_metadata.model_card
        self.server: NativeImageServer | None = None
        self.current_status: RunnerStatus = RunnerIdle()
        self.seen: set[TaskId] = set()
        self.cancelled_tasks: set[TaskId] = set()
        self.work_dir = SKULK_CACHE_HOME / "stable_diffusion_cpp" / str(self.runner_id)
        self._init_concurrent_dispatch(
            1, "native-image", queue_admitted=True, max_admitted_queue=8
        )
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus) -> None:
        """Publish a runner lifecycle transition to its supervising worker."""
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
        )

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Publish one authoritative task transition."""
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def acknowledge_task(self, task: Task) -> None:
        """Acknowledge bounded admission before generation starts."""
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self) -> None:
        """Run serial lifecycle and image dispatch until shutdown or producer failure."""
        self.run_dispatch_loop()

    def _teardown_server(self) -> None:
        server, self.server = self.server, None
        if server is not None:
            server.close()

    def _ensure_server_alive(self) -> None:
        if self.server is not None:
            self.server.check_alive()
        elif isinstance(self.current_status, (RunnerReady, RunnerRunning)):
            raise RuntimeError("native image producer is unavailable")

    def handle_task(self, task: Task) -> None:
        """Connect, load complete immutable components, then announce readiness."""
        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.current_status = RunnerConnected()
            case LoadModel() if isinstance(
                self.current_status, (RunnerIdle, RunnerConnected)
            ):
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)
                if self.shard_metadata.world_size != 1 or self.card.image is None:
                    raise ValueError(
                        "native image models require one host and a complete image contract"
                    )
                backend, build = (
                    self.shard_metadata.resolved_backend,
                    self.shard_metadata.resolved_engine_build,
                )
                binary = os.environ.get(BINARY_ENV, "").strip()
                if backend is None or build is None or not binary:
                    raise ValueError(
                        "native image placement lacks a prepared exact engine build"
                    )
                self.work_dir.mkdir(parents=True, exist_ok=True)
                self.server = NativeImageServer(
                    Path(binary),
                    build,
                    backend,
                    native_component_arguments(self.card),
                    self.work_dir / "native.log",
                )
                self.current_status = RunnerLoaded()
            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                self._ensure_server_alive()
                self.current_status = RunnerReady()
            case _:
                raise ValueError("unexpected native image lifecycle task")

    def _generate(self, task: Task) -> None:
        if not isinstance(task, (ImageGeneration, ImageEdits)):
            raise ValueError("native image runner accepts only image tasks")
        server = self.server
        if server is None:
            raise RuntimeError("native image producer is not loaded")
        cancelled, done = threading.Event(), threading.Event()
        started = time.monotonic()
        peak_resident_bytes = server.resident_memory_bytes() or 0

        def watch_cancel() -> None:
            nonlocal peak_resident_bytes
            while not done.wait(0.05):
                peak_resident_bytes = max(
                    peak_resident_bytes, server.resident_memory_bytes() or 0
                )
                if self._is_cancelled(task.task_id):
                    cancelled.set()
                    return

        watcher = threading.Thread(
            target=watch_cancel, name="native-image-cancel", daemon=True
        )
        watcher.start()
        params = task.task_params
        try:
            references: tuple[bytes, ...] = ()
            if isinstance(params, ImageEditsTaskParams):
                encoded = params.reference_image_data or (
                    (params.image_data,) if params.image_data else ()
                )
                references = tuple(
                    base64.b64decode(value, validate=True) for value in encoded
                )
                if len(references) != params.input_image_count:
                    raise ValueError(
                        "native edit input does not match its admitted reference slots"
                    )
            advanced = params.advanced_params
            seed = (
                advanced.seed
                if advanced and advanced.seed is not None
                else secrets.randbits(32)
            )
            count = params.n or 1
            if not 1 <= count <= 4:
                raise ValueError(
                    "native image requests require between one and four images"
                )
            partial_limit = params.partial_images or 0 if params.stream else 0
            for index in range(count):
                if self._is_cancelled(task.task_id):
                    cancelled.set()
                body = native_image_request(
                    self.card,
                    params,
                    seed + index,
                    references,
                    preview=partial_limit > 0,
                )
                with closing(server.generate(body, cancelled)) as outputs:

                    def stats(
                        image_index: int = index, request_body: dict[str, object] = body
                    ) -> ImageGenerationStats | None:
                        if image_index != count - 1 or peak_resident_bytes == 0:
                            return None
                        sampling = request_body["sample_params"]
                        assert isinstance(sampling, dict)
                        steps = cast("dict[str, object]", sampling)["sample_steps"]
                        assert isinstance(steps, int)
                        elapsed = time.monotonic() - started
                        return ImageGenerationStats(
                            total_generation_time=elapsed,
                            seconds_per_step=elapsed / (steps * count),
                            num_inference_steps=steps,
                            num_images=count,
                            image_width=int(str(request_body["width"])),
                            image_height=int(str(request_body["height"])),
                            peak_memory_usage=Memory.from_bytes(peak_resident_bytes),
                            memory_scope="native_process_rss",
                        )

                    self._send_outputs(outputs, task, index, partial_limit, stats)
        except NativeImageCancelledError:
            return
        except (
            OSError,
            ValueError,
            RuntimeError,
            TimeoutError,
            httpx.HTTPError,
        ) as error:
            from loguru import logger

            self._mark_generation_failed(task.task_id)
            logger.opt(exception=error).error("Native image generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.card.model_id,
                        error_message="Native image generation failed; see runner diagnostics.",
                    ),
                )
            )
        finally:
            done.set()
            watcher.join(timeout=1)

    def _send_outputs(
        self,
        outputs: Iterator[NativePreview | NativeResult],
        task: ImageGeneration | ImageEdits,
        index: int,
        partial_limit: int,
        stats: Callable[[], ImageGenerationStats | None],
    ) -> None:
        partial_index = 0
        for output in outputs:
            if self._is_cancelled(task.task_id):
                return
            is_partial = isinstance(output, NativePreview)
            if isinstance(output, NativePreview):
                if partial_index >= partial_limit:
                    continue
                image_format = "png"
                encoded = output.b64_json
            else:
                image_format = output.output_format
                encoded = output.images[0].b64_json
            profile = self.card.image
            assert profile is not None
            params = task.task_params
            if params.size == "auto":
                expected_canvas = (profile.default_width, profile.default_height)
            else:
                width_text, height_text = params.size.split("x", 1)
                expected_canvas = (int(width_text), int(height_text))
            validate_native_output(
                encoded, image_format, profile, expected_canvas, partial=is_partial
            )
            send_image_chunks(
                encoded,
                task.command_id,
                self.card.model_id,
                self.event_sender,
                index,
                is_partial,
                partial_index=partial_index if is_partial else None,
                total_partials=partial_limit if is_partial else None,
                image_format=image_format,
                stats=None if is_partial else stats(),
            )
            if is_partial:
                partial_index += 1
