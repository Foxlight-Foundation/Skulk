"""Bind actual native generation/editing to the admitted model and owned files.

This adapter does not activate a backend or advertise readiness. Worker bootstrap
must supply the prepared exact runtime, complete source roots, inherited reader
fence and shared resource assessment before connecting it to normal placement.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

from skulk.provisioning.ace_step_runtime import PreparedAceStepRuntime
from skulk.shared.models.model_cards import MusicLyricRequirement
from skulk.shared.models.native_music_memory import estimate_native_music_peak
from skulk.shared.models.native_music_runtime import resolve_native_music_policy
from skulk.shared.types.native_music import (
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
)
from skulk.shared.types.tasks import MusicEditing, MusicGeneration
from skulk.shared.types.worker.instances import BoundInstance
from skulk.worker.runner.ace_step.artifacts import observe_native_music_artifacts
from skulk.worker.runner.ace_step.process import verify_music_source
from skulk.worker.runner.ace_step.resident import ResidentMusic
from skulk.worker.runner.ace_step.result_custody import (
    MusicResultPreparationError,
    PendingMusicResult,
    music_result_directory,
    prepare_result_directory,
    reclaim_unsubmitted_result,
)
from skulk.worker.runner.ace_step.workspace import NativeMusicWorkspace


class MusicModelResident(Protocol):
    """Inject process effects without importing native dependencies into Core."""

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeMusicReady:
        """Load and independently verify actual model/device/component precision."""
        ...

    def is_ready(self) -> bool:
        """Observe actual native liveness rather than retaining a stale receipt."""
        ...

    def run(
        self,
        request: NativeMusicRequest,
        *,
        progress: Callable[[NativeMusicProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 7200.0,
    ) -> NativeMusicResult | None:
        """Return independently verified output; drain failure/cancel before cleanup."""
        ...

    def close(self) -> None:
        """Drain actual descendants before releasing native file custody."""
        ...


class MusicModelResidentFactory(Protocol):
    """Construct the supervisor, taking the already-opened workspace without spawn."""

    def __call__(
        self,
        interpreter: Path,
        workspace: NativeMusicWorkspace,
        configuration: NativeMusicServeConfig,
        *,
        payload_sha256: str,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> MusicModelResident:
        """Retain fixed native paths/policy; execution begins only during start."""
        ...


@final
@dataclass(frozen=True)
class GeneratedNativeMusic:
    """Verified completion plus its retained private path, never a State payload."""

    result: NativeMusicResult
    directory: Path


@final
class NativeMusicModel:
    """Own one complete admitted model, with one active generation or edit.

    Invalid/changed placement cannot fall back to another engine. The owning
    runner must recreate this binding after cancellation or process failure;
    completed output remains in independent custody until addressed delivery.
    """

    def __init__(
        self,
        bound: BoundInstance,
        runtime: PreparedAceStepRuntime,
        sources: Mapping[tuple[str, str], Path],
        root: Path,
        source_root: Path,
        *,
        create_resident: MusicModelResidentFactory = ResidentMusic,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> None:
        """Validate a complete placement without resolving files, loading or spawning.

        Args:
            bound: Authoritative complete single-host rank-zero assignment.
            runtime: Worker-prepared exact payload and actual library/device probe.
            sources: Complete selected repository/revision installation roots.
            root: Exclusive private lifetime/results root owned by the worker.
            source_root: Private command-original store, protected by reader custody.
            create_resident: Process-effect injection; production uses ResidentMusic.
            resource_guard: Live resource refusal shared with admitted engine policy.
        """
        shard = bound.bound_shard
        assignments = bound.instance.shard_assignments
        policy = shard.native_music_policy
        if (
            len(assignments.node_to_runner) != 1
            or len(assignments.runner_to_shard) != 1
            or assignments.node_to_runner.get(bound.bound_node_id)
            != bound.bound_runner_id
            or shard.device_rank != 0
            or shard.world_size != 1
            or shard.start_layer != 0
            or shard.end_layer != shard.model_card.n_layers
            or shard.n_layers != shard.model_card.n_layers
            or policy is None
            or shard.resolved_backend != policy.backend
        ):
            raise ValueError(
                "native music requires its complete admitted rank-zero model"
            )
        expected = resolve_native_music_policy(
            shard.model_card,
            policy.backend,
            runtime.probe,
            observed_layouts=policy.tensor_layouts,
        )
        if (
            policy != expected
            or shard.resolved_engine_build != runtime.wheel.build_identity
        ):
            raise ValueError(
                "prepared native music runtime differs from admitted model policy"
            )
        if not root.is_absolute() or not source_root.is_absolute():
            raise ValueError("native music requires worker-owned absolute roots")
        self.bound, self.runtime, self.policy = bound, runtime, policy
        self.memory_estimate = estimate_native_music_peak(shard.model_card, policy)
        self.sources, self.root, self.source_root = dict(sources), root, source_root
        self._create_resident, self._resource_guard = create_resident, resource_guard
        self._resident: MusicModelResident | None = None
        self._workspace: NativeMusicWorkspace | None = None
        self._actor = threading.Lock()
        self._readiness_actor = threading.Lock()
        self._ready_observed = threading.Event()
        self._stop_requested = threading.Event()
        self._retired = False
        self._unsubmitted_output: PendingMusicResult | None = None

    @staticmethod
    def _deadline(timeout_seconds: float) -> float:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 7200:
            raise ValueError(
                "native music requires a finite bounded operation deadline"
            )
        return time.monotonic() + timeout_seconds

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeMusicReady:
        """Rehash complete objects, verify actual headers, then load the admitted model.

        Start runs off the event loop. Complete file inspection and native load
        share the same caller deadline. Header facts are checked against the
        admitted policy before creating any loader view or starting native code.
        """
        deadline = self._deadline(timeout_seconds)
        if not self._actor.acquire(blocking=False):
            raise RuntimeError("native music model already owns an active operation")

        def checkpoint() -> None:
            if is_cancelled() or self._stop_requested.is_set():
                raise InterruptedError("native music model load was cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "native music model inspection exceeded its deadline"
                )

        try:
            if self._retired:
                raise RuntimeError(
                    "native music model requires normal runner recreation"
                )
            checkpoint()
            if self._resident is None:
                observed = observe_native_music_artifacts(
                    self.bound.bound_shard.model_card,
                    self.sources,
                    checkpoint=checkpoint,
                )
                actual = resolve_native_music_policy(
                    self.bound.bound_shard.model_card,
                    self.policy.backend,
                    self.runtime.probe,
                    observed_layouts=observed.tensor_layouts,
                )
                # Optional missing geometry is an assessed unknown, not a
                # declared fact. New verified observations may refine it while
                # all admitted data/execution/workload settings stay identical.
                comparison = (
                    actual
                    if self.policy.tensor_layouts is not None
                    else actual.model_copy(update={"tensor_layouts": None})
                )
                if comparison != self.policy:
                    raise ValueError(
                        "observed native music files differ from admitted tensor policy"
                    )
                observed_memory = estimate_native_music_peak(
                    self.bound.bound_shard.model_card, actual
                )
                if any(
                    measured > admitted
                    for measured, admitted in (
                        (
                            observed_memory.host_load_bytes,
                            self.memory_estimate.host_load_bytes,
                        ),
                        (
                            observed_memory.host_forward_bytes,
                            self.memory_estimate.host_forward_bytes,
                        ),
                        (
                            observed_memory.device_load_bytes,
                            self.memory_estimate.device_load_bytes,
                        ),
                        (
                            observed_memory.device_forward_bytes,
                            self.memory_estimate.device_forward_bytes,
                        ),
                    )
                ):
                    # Refining an assessed unknown may lower a reservation;
                    # it cannot spend more than placement held. Re-admit before
                    # creating a loader view or allocating native model state.
                    raise ValueError(
                        "observed native music memory exceeds admitted phase reservations"
                    )
                workspace = NativeMusicWorkspace(self.root, observed.weights)
                workspace.open()
                self._workspace = workspace
                configuration = NativeMusicServeConfig(
                    model_identity=self.policy.model_identity,
                    weights=observed.weights,
                    directory=str(workspace.directory),
                    directory_identity=workspace.directory_identity,
                    requests_identity=workspace.requests_identity,
                    loader_identity=workspace.loader_identity,
                    device=self.policy.execution.device,
                    precision=self.policy.execution.precision,
                    memory_budget=self.memory_estimate.process_budget(),
                    threads=self.policy.threads,
                )
                try:
                    self._resident = self._create_resident(
                        self.runtime.interpreter,
                        workspace,
                        configuration,
                        payload_sha256=self.policy.payload_sha256,
                        resource_guard=self._resource_guard,
                    )
                except BaseException:
                    workspace.close()
                    self._workspace = None
                    raise
            checkpoint()
            with self._readiness_handover(deadline):
                ready = self._resident.start(
                    is_cancelled=lambda: is_cancelled()
                    or self._stop_requested.is_set(),
                    timeout_seconds=max(0.001, deadline - time.monotonic()),
                )
            if (ready.model_identity, ready.payload_sha256, ready.execution) != (
                self.policy.model_identity,
                self.policy.payload_sha256,
                self.policy.execution,
            ):
                raise ValueError(
                    "native music loaded readiness differs from its admitted model"
                )
            checkpoint()
            self._ready_observed.set()
            return ready
        except BaseException:
            self._retired = True
            self._ready_observed.clear()
            if self._resident is not None:
                self._resident.close()
            raise
        finally:
            self._actor.release()

    def is_ready(self) -> bool:
        """Poll idle liveness without racing native submission or waiting for inference.

        Active operations run their own actual liveness/resource checks. They
        hand over any in-flight idle poll before calling the resident; later
        polls return the last observed readiness while the operation owns it.
        """
        if self._retired or self._stop_requested.is_set():
            return False
        if self._actor.locked() or not self._readiness_actor.acquire(blocking=False):
            return self._ready_observed.is_set()
        try:
            if self._actor.locked():
                return self._ready_observed.is_set()
            return self._is_ready()
        finally:
            self._readiness_actor.release()

    def _is_ready(self) -> bool:
        resident = self._resident
        try:
            ready = (
                not self._retired
                and not self._stop_requested.is_set()
                and resident is not None
                and resident.is_ready()
            )
        except BaseException:
            self._ready_observed.clear()
            raise
        if ready and not self._retired and not self._stop_requested.is_set():
            self._ready_observed.set()
            return True
        self._ready_observed.clear()
        return False

    @contextmanager
    def _readiness_handover(self, deadline: float) -> Iterator[None]:
        # Readiness uses a separate short-lived gate so a poll can never look
        # like another generation, nor block a poll behind long inference.
        if not self._readiness_actor.acquire(
            timeout=max(0.0, min(20.0, deadline - time.monotonic()))
        ):
            raise TimeoutError("native music idle readiness check remains pending")
        try:
            yield
        finally:
            self._readiness_actor.release()

    def _request(
        self, task: MusicGeneration | MusicEditing, checkpoint: Callable[[], None]
    ) -> NativeMusicRequest:
        card = self.bound.bound_shard.model_card
        profile = card.music
        workspace = self._workspace
        if workspace is None or profile is None or profile.native is None:
            raise RuntimeError("native music model has no loaded workspace")
        if (
            task.instance_id != self.bound.instance.instance_id
            or task.task_params.model != card.model_id
        ):
            raise ValueError("music task does not belong to this mounted native model")
        params = task.task_params
        if not params.prompt.strip():
            raise ValueError("native music caption must contain non-whitespace text")
        directory = music_result_directory(self.root, task.command_id)
        if isinstance(task, MusicGeneration):
            params = task.task_params
            if params.seed is None:
                raise ValueError(
                    "native music generation requires a once-resolved seed"
                )
            if (
                not profile.min_seconds
                <= params.seconds
                <= min(120, profile.max_seconds)
            ):
                raise ValueError(
                    "native music generation exceeds the selected duration bounds"
                )
            if params.lyrics is not None and not params.lyrics.strip():
                raise ValueError("native music lyrics must contain non-whitespace text")
            if (
                profile.lyrics == MusicLyricRequirement.Required
                and params.lyrics is None
                or profile.lyrics == MusicLyricRequirement.Unsupported
                and params.lyrics is not None
            ):
                raise ValueError(
                    "native music lyrics differ from the selected card contract"
                )
            return NativeMusicRequest(
                request_id=str(task.command_id),
                operation="generate",
                prompt=params.prompt,
                lyrics=params.lyrics if params.lyrics is not None else "[Instrumental]",
                seed=params.seed,
                target_frames=params.seconds * 48000,
                output_directory=str(directory),
                output_directory_identity=workspace.requests_identity,
            )
        params = task.task_params
        if (
            params.task not in profile.native.editing_modes
            or params.part_class is not None
            and params.part_class not in profile.native.part_classes
        ):
            raise ValueError(
                "native music card does not implement the selected edit or part"
            )
        if params.target_frames > self.policy.max_target_frames:
            raise ValueError("native music edit exceeds its admitted workload ceiling")
        source = self.source_root / str(task.command_id) / "original.wav"
        verify_music_source(source, params.source, checkpoint=checkpoint)
        return NativeMusicRequest(
            request_id=str(task.command_id),
            operation=params.task,
            prompt=params.prompt,
            lyrics=params.lyrics,
            seed=params.seed,
            target_frames=params.target_frames,
            source=params.source,
            source_path=str(source),
            region_start_frame=params.region_start_frame,
            region_end_frame=params.region_end_frame,
            part_class=params.part_class,
            output_directory=str(directory),
            output_directory_identity=workspace.requests_identity,
        )

    def run(
        self,
        task: MusicGeneration | MusicEditing,
        *,
        progress: Callable[[NativeMusicProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 7200.0,
    ) -> GeneratedNativeMusic | None:
        """Run one closed intent, retaining verified output for acknowledged delivery.

        The original source is verified again from its fixed worker path. No
        caller can submit native options, paths, mutable seeds or execution dtype.
        Invalid intent creates no output; cancellation/failure after submission
        retires the native lifetime through the real resident supervisor.
        """
        deadline = self._deadline(timeout_seconds)
        if not self._actor.acquire(blocking=False):
            raise RuntimeError("native music model already owns an active operation")

        def checkpoint() -> None:
            if is_cancelled() or self._stop_requested.is_set():
                raise InterruptedError("native music operation was cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("native music operation exceeded its deadline")

        try:
            if is_cancelled():
                return None
            with self._readiness_handover(deadline):
                if (
                    not self._is_ready()
                    or self._resident is None
                    or self._workspace is None
                ):
                    raise RuntimeError("native music model is not ready")
            request = self._request(task, checkpoint)
            checkpoint()
            directory = Path(request.output_directory)
            parent_identity = self._workspace.requests_identity
            try:
                identity = prepare_result_directory(directory, parent_identity)
            except MusicResultPreparationError as error:
                self._unsubmitted_output = error.pending
                self._retired = True
                raise
            self._unsubmitted_output = PendingMusicResult(
                directory, parent_identity, identity
            )
            try:
                request = NativeMusicRequest.model_validate(
                    {**request.model_dump(), "output_directory_identity": identity}
                )
                checkpoint()
            except BaseException:
                if not reclaim_unsubmitted_result(self._unsubmitted_output):
                    self._retired = True
                    raise RuntimeError(
                        "native music unsubmitted result cleanup remains pending"
                    ) from None
                self._unsubmitted_output = None
                raise
            try:
                # Keep the empty-result fallback until successful completion:
                # the resident may refuse before recording its writer custody.
                # Close drains it before attempting this exact empty rollback.
                result = self._resident.run(
                    request,
                    progress=progress,
                    is_cancelled=lambda: is_cancelled()
                    or self._stop_requested.is_set(),
                    timeout_seconds=max(0.001, deadline - time.monotonic()),
                )
            except BaseException:
                self._retired = True
                raise
            if result is None:
                self._retired = True
                return None
            self._unsubmitted_output = None
            return GeneratedNativeMusic(result, directory)
        except InterruptedError:
            self._retired = True
            if self._resident is not None:
                self._resident.close()
            return None
        finally:
            self._actor.release()

    def close(self) -> None:
        """Retire and drain; retain failed-cleanup handles until a later close succeeds."""
        self._retired = True
        self._ready_observed.clear()
        self._stop_requested.set()
        # The model actor covers header inspection and pre-submit filesystem
        # effects as well as inference. Waiting here prevents close from
        # returning while another thread can still create a new native child.
        if not self._actor.acquire(timeout=20):
            raise RuntimeError("native music model actor cleanup remains pending")
        try:
            if self._resident is not None:
                self._resident.close()
                self._resident = None
                self._workspace = None
            if self._unsubmitted_output is not None:
                if not reclaim_unsubmitted_result(self._unsubmitted_output):
                    raise RuntimeError(
                        "native music unsubmitted result cleanup remains pending"
                    )
                self._unsubmitted_output = None
        finally:
            self._actor.release()
