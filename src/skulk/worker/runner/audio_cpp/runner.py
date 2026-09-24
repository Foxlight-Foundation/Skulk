"""Single-model music runner over one supervised audio.cpp server process."""

from __future__ import annotations

import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import final

from loguru import logger

from skulk.facts.probe import probe_audio_cpp
from skulk.shared.backends import AUDIO_CPP_BIN_ENV, probe_node_backends
from skulk.shared.constants import SKULK_CACHE_HOME, SKULK_MUSIC_OUTPUT_DIR
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.chunks import ErrorChunk, MusicChunk
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.music import MusicGenerationTaskParams, MusicOutputManifest
from skulk.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    MusicGeneration,
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
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.audio_cpp.adapter import audio_cpp_music_request
from skulk.worker.runner.audio_cpp.server import AudioCppServer
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch


def model_directory(card: ModelCard) -> Path:
    """Resolve the verified immutable artifact root that audio.cpp loads."""
    from skulk.download.download_utils import build_model_path

    root = card.artifact_bundle.root if card.artifact_bundle is not None else None
    return build_model_path(ModelId(card.model_id), card.source_revision, root)


def _compute_backend(resolved_backend: str | None) -> str:
    """Use the placed compute lane, or fail closed for an unstamped shard."""
    if resolved_backend is None or resolved_backend == "audio_cpp":
        tags = probe_node_backends()
        for compute in ("cuda", "rocm", "metal", "vulkan", "cpu"):
            if f"audio_cpp-{compute}" in tags:
                return "hip" if compute == "rocm" else compute
        raise RuntimeError("this node has no ready audio.cpp compute lane")
    if not resolved_backend.startswith("audio_cpp-"):
        raise RuntimeError(f"invalid music placement backend {resolved_backend}")
    if resolved_backend not in probe_node_backends():
        raise RuntimeError(f"placed audio.cpp lane {resolved_backend} is no longer ready")
    compute = resolved_backend.removeprefix("audio_cpp-")
    return "hip" if compute == "rocm" else compute


def verify_audio_cpp_launch(binary: Path, expected_build: str | None, backend: str) -> None:
    """Require the placed build and compute lane on the executable about to launch."""
    if expected_build is None:
        raise RuntimeError("music placement has no stamped audio.cpp build")
    probe = probe_audio_cpp(str(binary))
    compute = "rocm" if backend == "hip" else backend
    if probe.outcome != "ready" or compute not in probe.computes:
        raise RuntimeError(
            f"placed audio.cpp lane {compute} is no longer usable: {probe.detail or 'device missing'}"
        )
    digest = sha256()
    try:
        with binary.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise RuntimeError("audio.cpp executable cannot be verified at load") from error
    observed = f"audio.cpp@sha256:{digest.hexdigest()}"
    if observed != expected_build:
        raise RuntimeError("audio.cpp executable changed since music placement")


@final
class Runner(ServedConcurrentDispatch):
    """Serve one music model with strict serial generation and cancel teardown."""

    _generation_kinds = (MusicGeneration,)

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ) -> None:
        self.bound_instance = bound_instance
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.runner_id = bound_instance.bound_runner_id
        self.shard_metadata = bound_instance.bound_shard
        self.card = self.shard_metadata.model_card
        self.model_id = self.card.model_id
        self.server: AudioCppServer | None = None
        self.work_dir = SKULK_CACHE_HOME / "audio_cpp" / str(self.runner_id)
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self.current_status: RunnerStatus = RunnerIdle()
        self._init_concurrent_dispatch(1, "audio-cpp-music", queue_admitted=True)
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus) -> None:
        """Publish one runner lifecycle transition."""
        self.current_status = status
        self.event_sender.send(RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status))

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Publish one task lifecycle transition."""
        self.event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=status))

    def acknowledge_task(self, task: Task) -> None:
        """Tell the worker this task entered the serial dispatch queue."""
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self) -> None:
        """Run the bounded served-engine dispatch loop until shutdown."""
        self.run_dispatch_loop()

    def _teardown_server(self) -> None:
        server = self.server
        self.server = None
        if server is not None:
            server.teardown()

    def _ensure_server_alive(self) -> None:
        server = self.server
        with self._status_lock:
            generation_active = self._inflight > 0
        if server is None:
            if generation_active:
                return
            if isinstance(self.current_status, (RunnerReady, RunnerRunning)):
                raise RuntimeError("audio.cpp server stopped; runner restart is required")
            return
        if not server.alive():
            # Cancellation stops this sidecar while the active generation is
            # still unwinding. Its worker restores the sidecar before releasing
            # the serial dispatch permit; do not crash the dispatch loop here.
            if generation_active:
                return
            tail = server.log_tail()
            self._teardown_server()
            raise RuntimeError(f"audio.cpp server exited: {tail}")

    def handle_task(self, task: Task) -> None:
        """Connect, load, and warm one model before generations are admitted."""
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
                self._ensure_server_alive()
                self.current_status = RunnerReady()
            case _:
                raise ValueError(
                    f"unexpected music task {type(task).__name__} in {self.current_status}"
                )

    def _load_model(self) -> None:
        music = self.card.music
        bundle = self.card.artifact_bundle
        if music is None or bundle is None:
            raise RuntimeError("music runner requires a typed [music] artifact card")
        binary_value = os.environ.get(AUDIO_CPP_BIN_ENV, "").strip()
        if not binary_value:
            raise RuntimeError("audio.cpp package has not been prepared on this node")
        binary = Path(binary_value)
        if not binary.is_file():
            raise RuntimeError("audio.cpp package is missing its server")
        from skulk.provisioning.audio_cpp import audio_cpp_model_specs

        specs = audio_cpp_model_specs(binary)
        model_dir = model_directory(self.card)
        from skulk.download.download_utils import resolve_artifact_file

        for component in bundle.files:
            resolve_artifact_file(model_dir, bundle.root, component.path)
        backend = _compute_backend(self.shard_metadata.resolved_backend)
        verify_audio_cpp_launch(
            binary, self.shard_metadata.resolved_engine_build, backend,
        )
        server = AudioCppServer(
            binary=binary,
            model_specs=specs,
            model_dir=model_dir,
            music=music,
            backend=backend,
            work_dir=self.work_dir,
        )
        server.start()
        self.server = server
        logger.info(f"audio.cpp serving {self.model_id} from {model_dir}")

    def _generate(self, task: Task) -> None:
        """Run a fixed family request and publish only a measured WAV manifest."""
        assert isinstance(task, MusicGeneration)
        if self._is_cancelled(task.task_id):
            return
        server = self.server
        self._ensure_server_alive()
        assert server is not None
        output_dir = SKULK_MUSIC_OUTPUT_DIR / str(task.command_id)
        try:
            self._render(task, task.task_params, server, output_dir)
        except ValueError as error:
            shutil.rmtree(output_dir, ignore_errors=True)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(model=self.model_id, error_message=str(error)),
                )
            )
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            self._teardown_server()
            raise
        if (
            not isinstance(self.current_status, (RunnerShuttingDown, RunnerShutdown))
            and not server.alive()
        ):
            # Cancellation and an oversized response both terminate the
            # sidecar. Restore it before the completion callback releases the
            # serial permit to another admitted generation.
            self._teardown_server()
            self._load_model()

    def _render(
        self,
        task: MusicGeneration,
        params: MusicGenerationTaskParams,
        server: AudioCppServer,
        output_dir: Path,
    ) -> None:
        music = self.card.music
        assert music is not None
        body = audio_cpp_music_request(
            family=music,
            prompt=params.prompt,
            lyrics=params.lyrics,
            seconds=params.seconds,
            seed=params.seed,
        )
        wav = server.generate(body, is_cancelled=lambda: self._is_cancelled(task.task_id))
        if wav is None or self._is_cancelled(task.task_id):
            shutil.rmtree(output_dir, ignore_errors=True)
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        staging = output_dir / "output.wav.part"
        final = output_dir / "output.wav"
        try:
            staging.write_bytes(wav.data)
            staging.replace(final)
        finally:
            staging.unlink(missing_ok=True)
        self.event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=MusicChunk(
                    model=self.model_id,
                    finish_reason="stop",
                    output=MusicOutputManifest(
                        size_bytes=wav.size_bytes,
                        sha256=wav.sha256,
                        duration_seconds=wav.duration_seconds,
                        sample_rate=wav.sample_rate,
                        channels=wav.channels,
                    ),
                ),
            )
        )
