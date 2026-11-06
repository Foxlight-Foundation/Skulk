"""Worker-only spawn context; native model construction waits for LoadModel."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

from skulk.download.download_utils import build_model_path
from skulk.provisioning.demucs_runtime import PreparedDemucsRuntime
from skulk.shared.types.audio_separation import (
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationServeConfig,
)
from skulk.shared.types.tasks import AudioSeparation
from skulk.shared.types.worker.instances import BoundInstance
from skulk.worker.runner.demucs.model import GeneratedSeparation, SeparationModel
from skulk.worker.runner.demucs.resident import ResidentSeparator, spawn_resident
from skulk.worker.runner.demucs.source_fence import SourceReadFence


@final
@dataclass(frozen=True)
class SeparationRunnerContext:
    """Private prepared runtime, owned roots and an inherited reader fence.

    This context travels only through same-version local process spawning. The
    worker retains its fence until the runner exits, while native descendants
    retain another duplicate. Paths and descriptors never enter State/events.
    """

    runtime: PreparedDemucsRuntime
    root: Path
    source_root: Path
    source_fence: SourceReadFence
    load_directory: Path | None = None

    def model(self, bound: BoundInstance) -> PreparedSeparationModel:
        """Bind placement now, deferring staged-weight resolution until LoadModel."""
        if self.source_fence.descriptor < 0:
            raise RuntimeError("separation source reader fence is closed")
        return PreparedSeparationModel(bound, self)

    def bind(self, bound: BoundInstance, load_directory: Path) -> SeparationModel:
        """Construct the actual resident binding once the normal download gate passes."""

        def create_resident(
            interpreter: Path,
            root: Path,
            configuration: NativeSeparationServeConfig,
            *,
            payload_sha256: str,
            weights_sha256: str,
            resource_guard: Callable[[int], str | None] | None = None,
        ) -> ResidentSeparator:
            def spawn(
                interpreter: Path,
                configuration: Path,
                descriptor: int,
                environment: dict[str, str],
            ) -> subprocess.Popen[bytes]:
                return spawn_resident(
                    interpreter,
                    configuration,
                    descriptor,
                    environment,
                    source_fence=self.source_fence.descriptor,
                )

            return ResidentSeparator(
                interpreter,
                root,
                configuration,
                payload_sha256=payload_sha256,
                weights_sha256=weights_sha256,
                resource_guard=resource_guard,
                spawn=spawn,
            )

        return SeparationModel(
            bound,
            self.runtime,
            load_directory,
            self.root,
            self.source_root,
            create_resident=create_resident,
        )


@final
class PreparedSeparationModel:
    """Adapt normal create/download/load ordering to the retained native binding.

    CreateRunner precedes DownloadModel. Resolving a staged file while bootstrap
    constructs its idle runner would fail first-time mounts before downloading.
    Keep failed cleanup handles until close succeeds; readiness is always live.
    """

    def __init__(self, bound: BoundInstance, context: SeparationRunnerContext) -> None:
        """Retain private placement/context without path resolution or native effects."""
        self.bound, self.context = bound, context
        self._model: SeparationModel | None = None

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeSeparationReady:
        """Resolve staged or canonical Store weights after the normal download gate."""
        if is_cancelled():
            raise InterruptedError("native model load was cancelled")
        if self._model is None:
            card = self.bound.bound_shard.model_card
            directory = self.context.load_directory
            if directory is None:
                directory = build_model_path(
                    card.model_id,
                    card.source_revision,
                    card.artifact_bundle.root
                    if card.artifact_bundle is not None
                    else None,
                    expected_card=card,
                )
            self._model = self.context.bind(self.bound, directory)
        return self._model.start(
            is_cancelled=is_cancelled, timeout_seconds=timeout_seconds
        )

    def is_ready(self) -> bool:
        """Return actual retained native liveness, never package availability."""
        return self._model is not None and self._model.is_ready()

    def separate(
        self,
        task: AudioSeparation,
        *,
        progress: Callable[[NativeSeparationProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 1800.0,
    ) -> GeneratedSeparation | None:
        """Run only a previously loaded binding with its exact source/policy checks."""
        if self._model is None:
            raise RuntimeError("native separation model is not loaded")
        return self._model.separate(
            task,
            progress=progress,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        """Drain the native binding before forgetting it; failed cleanup stays owned."""
        if self._model is not None:
            self._model.close()
            self._model = None
