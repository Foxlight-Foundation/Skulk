"""Bind an admitted complete native placement to one owned resident-model lifetime.

Only worker-owned staged paths enter this boundary. Source/result bytes remain
private files; returned metadata can be emitted separately over the data plane.
This module does not register a bootstrap runner or advertise node resources.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

from skulk.download.download_utils import resolve_artifact_file
from skulk.provisioning.demucs_runtime import PreparedDemucsRuntime
from skulk.shared.models.separation_runtime import resolve_demucs_policy
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_SOURCE_BYTES,
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    source_from_wav,
)
from skulk.shared.types.common import CommandId
from skulk.shared.types.tasks import AudioSeparation
from skulk.shared.types.worker.instances import BoundInstance
from skulk.worker.runner.demucs.resident import ResidentSeparator


class ResidentModel(Protocol):
    """Inject process effects while production retains the real resident supervisor."""

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeSeparationReady:
        """Return an actual model/device identity after load or fail without readiness."""
        ...

    def is_ready(self) -> bool:
        """Check current process liveness instead of trusting a remembered receipt."""
        ...

    def run(
        self, job: NativeSeparationJob, *, progress: Callable[[NativeSeparationProgress], None],
        is_cancelled: Callable[[], bool], timeout_seconds: float = 6 * 60 * 60,
    ) -> NativeSeparationResult | None:
        """Return a verified full set or drain the failed/cancelled lifetime."""
        ...

    def close(self) -> None:
        """Drain owned descendants before releasing lifetime/file ownership."""
        ...


class ResidentFactory(Protocol):
    """Construct the real supervisor without importing native dependencies into Core."""

    def __call__(
        self, interpreter: Path, root: Path, configuration: NativeSeparationServeConfig,
        *, payload_sha256: str, weights_sha256: str,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> ResidentModel:
        """Create one fixed-path/policy lifetime, without starting a native process."""
        ...


def separation_result_directory(root: Path, command_id: CommandId) -> Path:
    """Derive a fixed private native job path for runner and output producer.

    Native protocol jobs use a 32-hex directory; command IDs remain opaque and
    never become path components. The mapping is local, not replicated metadata.
    """
    identifier = str(command_id)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier) is None:
        raise ValueError("invalid native separation command identity")
    return root / "jobs" / hashlib.sha256(identifier.encode()).hexdigest()[:32]


@final
@dataclass(frozen=True)
class GeneratedSeparation:
    """Verified result plus private local directory; never a State/event payload."""

    result: NativeSeparationResult
    directory: Path


@final
class SeparationModel:
    """Own a resident model bound to exact placement, package, weights and policy."""

    def __init__(
        self, bound: BoundInstance, runtime: PreparedDemucsRuntime,
        load_directory: Path, root: Path, source_root: Path,
        *, create_resident: ResidentFactory = ResidentSeparator,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> None:
        """Validate placement and staged weights without starting native execution.

        Args:
            bound: Authoritative complete single-host rank-zero placement.
            runtime: Worker-prepared exact package/device receipt, not caller data.
            load_directory: Worker-staged loader directory for the exact artifact.
            root: Exclusive private model lifetime/result root.
            source_root: Owner-verified input root with command-private WAVs.
            create_resident: Injectable process effect; production uses the real supervisor.
            resource_guard: Live admitted-process resource refusal callback.
        """
        shard = bound.bound_shard
        assignments = bound.instance.shard_assignments
        backend = shard.resolved_backend
        if (
            not bound.is_separation_model or backend is None
            or len(assignments.node_to_runner) != 1
            or len(assignments.runner_to_shard) != 1
            or assignments.node_to_runner.get(bound.bound_node_id) != bound.bound_runner_id
            or shard.device_rank != 0 or shard.world_size != 1
            or shard.start_layer != 0 or shard.end_layer != shard.model_card.n_layers
            or shard.n_layers != shard.model_card.n_layers
        ):
            raise ValueError("native separation needs its complete bound rank-zero model")
        policy = resolve_demucs_policy(shard.model_card, backend)
        if (
            shard.separation_policy != policy
            or shard.resolved_engine_build != runtime.wheel.build_identity
            or policy.device != runtime.device
        ):
            raise ValueError("prepared native runtime differs from admitted model policy")
        bundle = shard.model_card.artifact_bundle
        assert bundle is not None
        weights = resolve_artifact_file(load_directory, bundle.root, policy.weights_path)
        self.bound, self.runtime, self.policy = bound, runtime, policy
        self.root, self.source_root = root, source_root
        self.configuration = NativeSeparationServeConfig(
            weights_path=str(weights), device=policy.device, threads=policy.threads
        )
        self._create_resident = create_resident
        self._resource_guard = resource_guard
        self._resident: ResidentModel | None = None

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeSeparationReady:
        """Load the exact native model; never infer readiness from installed libraries."""
        if self._resident is None:
            self._resident = self._create_resident(
                self.runtime.interpreter, self.root, self.configuration,
                payload_sha256=self.runtime.wheel.payload_sha256,
                weights_sha256=self.policy.weights_sha256,
                resource_guard=self._resource_guard,
            )
        return self._resident.start(is_cancelled=is_cancelled, timeout_seconds=timeout_seconds)

    def is_ready(self) -> bool:
        """Observe current native liveness; a dead lifetime cannot remain ready."""
        return self._resident is not None and self._resident.is_ready()

    def separate(
        self, task: AudioSeparation, *, progress: Callable[[NativeSeparationProgress], None],
        is_cancelled: Callable[[], bool], timeout_seconds: float = 1800.0,
    ) -> GeneratedSeparation | None:
        """Run one correlated private source; the real supervisor verifies all outputs.

        No caller supplies file paths or native options. Independently verify the
        worker-owned source again before inference. Cancellation/failure drains
        the resident; the normal runner owner must recreate/reload that lifetime.
        Successful files remain for the separately acknowledged output producer.
        """
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 6 * 60 * 60:
            raise ValueError("native separation requires a finite complete job budget")
        deadline = time.monotonic() + timeout_seconds
        if is_cancelled():
            return None
        resident = self._resident
        if resident is None or not resident.is_ready():
            raise RuntimeError("native separation model is not ready")
        if (
            task.instance_id != self.bound.instance.instance_id
            or task.task_params.model != self.bound.bound_shard.model_card.model_id
        ):
            raise ValueError("separation task does not belong to this mounted model")
        directory = separation_result_directory(self.root, task.command_id)
        source = self.source_root / str(task.command_id) / "original.wav"
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as original:
            info = os.fstat(original.fileno())
            if (
                not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size > MAX_SEPARATION_SOURCE_BYTES
            ):
                raise ValueError("native source is not a bounded private original")
            measured = source_from_wav(original.read(MAX_SEPARATION_SOURCE_BYTES + 1))
        if measured != task.task_params.source:
            raise ValueError("native source differs from the admitted original")
        job = NativeSeparationJob(
            weights_path=self.configuration.weights_path, source_path=str(source),
            output_dir=str(directory), source=measured, device=self.policy.device,
            threads=self.policy.threads, segment_seconds=self.policy.segment_seconds,
        )
        if is_cancelled():
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native separation source verification exceeded its budget")
        result = resident.run(
            job, progress=progress, is_cancelled=is_cancelled, timeout_seconds=remaining
        )
        return GeneratedSeparation(result, directory) if result is not None else None

    def close(self) -> None:
        """Drain before dropping the original handle; failed cleanup remains retryable."""
        if self._resident is not None:
            self._resident.close()
            self._resident = None
