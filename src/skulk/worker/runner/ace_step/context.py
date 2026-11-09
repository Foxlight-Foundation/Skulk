"""Local spawn binding for native music; staged data is resolved only at load."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import final

from skulk.download.download_utils import resolve_native_music_source_directory
from skulk.provisioning.ace_step_runtime import PreparedAceStepRuntime
from skulk.shared.types.native_music import (
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicServeConfig,
)
from skulk.shared.types.tasks import MusicEditing, MusicGeneration
from skulk.shared.types.worker.instances import BoundInstance
from skulk.worker.runner.ace_step.model import GeneratedNativeMusic, NativeMusicModel
from skulk.worker.runner.ace_step.resident import (
    MusicResidentProcess,
    ResidentMusic,
    spawn_music_resident,
)
from skulk.worker.runner.ace_step.workspace import NativeMusicWorkspace
from skulk.worker.runner.demucs.source_fence import SourceReadFence


@final
@dataclass(frozen=True)
class NativeMusicRunnerContext:
    """Prepared package, private roots, inherited custody and live admission guard.

    This is a same-version local process argument, never replicated metadata.
    A Worker must supply an actual resource guard before constructing it. The
    descriptor is duplicated by the existing spawn transfer and retained by the
    native guardian until its entire process group drains.
    """

    runtime: PreparedAceStepRuntime
    root: Path
    source_root: Path
    source_fence: SourceReadFence
    resource_guard: Callable[[int], str | None]

    def require_reader(self) -> None:
        """Refuse a released source lifetime before resolving or spawning anything."""
        if self.source_fence.descriptor < 0:
            raise RuntimeError("native music source reader fence is closed")

    def model(self, bound: BoundInstance) -> PreparedNativeMusicModel:
        """Create an unloaded lifetime without resolving absent first-mount weights."""
        self.require_reader()
        return PreparedNativeMusicModel(bound, self)

    def bind(
        self, bound: BoundInstance, sources: Mapping[tuple[str, str], Path]
    ) -> NativeMusicModel:
        """Bind complete staged sources to actual native supervision after download."""
        self.require_reader()

        def create_resident(
            interpreter: Path,
            workspace: NativeMusicWorkspace,
            configuration: NativeMusicServeConfig,
            *,
            payload_sha256: str,
            resource_guard: Callable[[int], str | None] | None = None,
        ) -> ResidentMusic:
            def spawn(
                interpreter: Path,
                configuration: Path,
                descriptor: int,
                environment: dict[str, str],
            ) -> MusicResidentProcess:
                self.require_reader()
                return spawn_music_resident(
                    interpreter,
                    configuration,
                    descriptor,
                    environment,
                    source_fence=self.source_fence.descriptor,
                )

            return ResidentMusic(
                interpreter,
                workspace,
                configuration,
                payload_sha256=payload_sha256,
                resource_guard=resource_guard,
                spawn=spawn,
            )

        return NativeMusicModel(
            bound,
            self.runtime,
            sources,
            self.root,
            self.source_root,
            create_resident=create_resident,
            resource_guard=self.resource_guard,
        )


@final
class PreparedNativeMusicModel:
    """Preserve create/download/load ordering and retire each cancelled lifetime.

    Resolving a complete source at CreateRunner would reject a valid first
    mount before DownloadModel. Keep the binding lazy while requiring exact
    installed-card identity for its primary source. A failed drain remains
    owned for a later close attempt; successful close never reactivates this
    wrapper, because the runner creates a fresh unloaded replacement instead.
    """

    def __init__(self, bound: BoundInstance, context: NativeMusicRunnerContext) -> None:
        """Retain authoritative placement without file lookup or native effects."""
        self.bound, self.context = bound, context
        self._model: NativeMusicModel | None = None
        self._closed = threading.Event()
        self._binding_actor = threading.Lock()

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeMusicReady:
        """Resolve every exact source and load under one cancellation/deadline budget."""
        if not self._binding_actor.acquire(blocking=False):
            raise RuntimeError("native music already owns an active model load")
        try:
            return self._start_owned(
                is_cancelled=is_cancelled, timeout_seconds=timeout_seconds
            )
        finally:
            self._binding_actor.release()

    def _start_owned(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float
    ) -> NativeMusicReady:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 7200:
            raise ValueError("native music load requires a finite bounded deadline")
        deadline = time.monotonic() + timeout_seconds

        def checkpoint() -> None:
            if self._closed.is_set():
                raise RuntimeError("native music requires a new unloaded lifetime")
            self.context.require_reader()
            if is_cancelled():
                raise InterruptedError("native music model load was cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "native music source resolution exceeded its deadline"
                )

        checkpoint()
        if self._model is None:
            card = self.bound.bound_shard.model_card
            selected = card.native_music_artifact_sources()
            if not selected:
                raise ValueError(
                    "native music placement has no complete data selection"
                )
            primary = str(card.artifact_repository), card.source_revision
            sources: dict[tuple[str, str], Path] = {}
            for source in selected:
                checkpoint()
                identity = source.repository, source.revision
                directory = resolve_native_music_source_directory(
                    card,
                    *identity,
                    require_card_identity=identity == primary,
                )
                if directory is None:
                    raise FileNotFoundError(
                        "native music selected source is not installed"
                    )
                sources[identity] = directory
            checkpoint()
            self._model = self.context.bind(self.bound, sources)
        checkpoint()
        return self._model.start(
            is_cancelled=is_cancelled,
            timeout_seconds=max(0.001, deadline - time.monotonic()),
        )

    def is_ready(self) -> bool:
        """Observe loaded native liveness only while this source lifetime is owned."""
        return (
            not self._closed.is_set()
            and self.context.source_fence.descriptor >= 0
            and self._model is not None
            and self._model.is_ready()
        )

    def run(
        self,
        task: MusicGeneration | MusicEditing,
        *,
        progress: Callable[[NativeMusicProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 7200.0,
    ) -> GeneratedNativeMusic | None:
        """Execute only a loaded, retained lifetime with the shared native checks."""
        self.context.require_reader()
        model = self._model
        if self._closed.is_set() or model is None:
            raise RuntimeError("native music model is not loaded in this lifetime")
        return model.run(
            task,
            progress=progress,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        """Retire before draining, retaining failed cleanup handles for a retry."""
        self._closed.set()
        observed = self._model
        if observed is not None:
            # Stop native loading before joining the outer actor: waiting for
            # that actor first would prevent close from interrupting inference.
            observed.close()
        with self._binding_actor:
            # Source resolution may have been constructing the unloaded model
            # when close observed None. Join it before releasing this lifetime;
            # its closed checkpoint prevents a late native spawn.
            if self._model is not None and self._model is not observed:
                self._model.close()
            self._model = None
