"""Bounded process-local observability for streaming capability providers."""

import time
from dataclasses import dataclass, field

from skulk.extensions import CapabilityStreamFrame, InlineMediaAttachment
from skulk.shared.types.diagnostics import (
    ProviderCapabilityDiagnostics,
    ProviderDiagnostics,
)


@dataclass
class _MetricAccumulator:
    count: int = 0
    total: float = 0.0
    maximum: float | None = None

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    @property
    def average(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class _CapabilityMetrics:
    active_streams: int = 0
    max_active_streams: int = 0
    admitted_streams: int = 0
    rejected_streams: int = 0
    overloaded_rejections: int = 0
    input_queue_depth: int = 0
    max_input_queue_depth: int = 0
    input_frames: int = 0
    input_media_bytes: int = 0
    output_frames: int = 0
    output_media_bytes: int = 0
    completed_streams: int = 0
    failed_streams: int = 0
    cancelled_streams: int = 0
    missing_terminal_streams: int = 0
    cancellation_requests: int = 0
    first_output: _MetricAccumulator = field(default_factory=_MetricAccumulator)
    duration: _MetricAccumulator = field(default_factory=_MetricAccumulator)


@dataclass(frozen=True)
class _ActiveStream:
    capability_id: str
    admitted_at: float
    first_output_recorded: bool = False


class ProviderObserver:
    """Track bounded provider pressure without retaining payloads or call history."""

    def __init__(self) -> None:
        """Create an empty process-lifetime provider observer."""

        self._capabilities: dict[str, _CapabilityMetrics] = {}
        self._active: dict[str, _ActiveStream] = {}
        self._input_queue_depths: dict[str, int] = {}
        self._max_active_streams = 0
        self._max_input_queue_depth = 0

    def record_rejected(self, capability_id: str, *, overloaded: bool = False) -> None:
        """Record one resolved capability stream rejected before execution.

        Args:
            capability_id: Stable qualified capability identifier.
            overloaded: Whether the process-wide stream bound caused rejection.
        """

        metrics = self._metrics(capability_id)
        metrics.rejected_streams += 1
        if overloaded:
            metrics.overloaded_rejections += 1

    def record_admitted(
        self,
        call_id: str,
        capability_id: str,
        *,
        observed_at: float | None = None,
    ) -> None:
        """Record one stream admitted inside the process-wide concurrency bound.

        Args:
            call_id: Active call identifier, retained only until termination.
            capability_id: Stable qualified capability identifier.
            observed_at: Optional monotonic timestamp for deterministic tests.
        """

        now = time.monotonic() if observed_at is None else observed_at
        metrics = self._metrics(capability_id)
        metrics.admitted_streams += 1
        metrics.active_streams += 1
        metrics.max_active_streams = max(
            metrics.max_active_streams, metrics.active_streams
        )
        self._active[call_id] = _ActiveStream(
            capability_id=capability_id,
            admitted_at=now,
        )
        self._max_active_streams = max(self._max_active_streams, len(self._active))

    def record_input(
        self,
        call_id: str,
        frame: CapabilityStreamFrame,
        *,
        queue_depth: int,
    ) -> None:
        """Record one validated input frame delivered to a provider queue.

        Args:
            call_id: Active call receiving the frame.
            frame: Validated caller-to-provider lifecycle frame.
            queue_depth: Buffered frame count after enqueue.
        """

        active = self._active.get(call_id)
        if active is None:
            return
        metrics = self._metrics(active.capability_id)
        metrics.input_frames += 1
        metrics.input_media_bytes += self._inline_media_bytes(frame)
        self.record_input_queue_depth(call_id, queue_depth)

    def record_input_queue_depth(self, call_id: str, queue_depth: int) -> None:
        """Update current and high-water caller input queue depth.

        Args:
            call_id: Active call whose queue changed.
            queue_depth: Current buffered frame count for that call.
        """

        active = self._active.get(call_id)
        if active is None:
            return
        metrics = self._metrics(active.capability_id)
        self._input_queue_depths[call_id] = max(0, queue_depth)
        metrics.input_queue_depth = sum(
            depth
            for active_call_id, depth in self._input_queue_depths.items()
            if self._active.get(active_call_id) is not None
            and self._active[active_call_id].capability_id == active.capability_id
        )
        metrics.max_input_queue_depth = max(
            metrics.max_input_queue_depth, metrics.input_queue_depth
        )
        total_depth = sum(
            capability.input_queue_depth
            for capability in self._capabilities.values()
        )
        self._max_input_queue_depth = max(self._max_input_queue_depth, total_depth)

    def record_output(
        self,
        call_id: str,
        frame: CapabilityStreamFrame,
        *,
        observed_at: float | None = None,
    ) -> None:
        """Record one output frame after it is accepted by provider DATA egress.

        Args:
            call_id: Active call producing the frame.
            frame: Provider-to-caller lifecycle frame accepted for egress.
            observed_at: Optional monotonic timestamp for deterministic tests.
        """

        active = self._active.get(call_id)
        if active is None:
            return
        now = time.monotonic() if observed_at is None else observed_at
        metrics = self._metrics(active.capability_id)
        metrics.output_frames += 1
        metrics.output_media_bytes += self._inline_media_bytes(frame)
        carries_output = frame.kind == "chunk" or (
            frame.kind == "completed"
            and (frame.payload is not None or frame.media is not None)
        )
        if carries_output and not active.first_output_recorded:
            metrics.first_output.record(max(0.0, now - active.admitted_at))
            self._active[call_id] = _ActiveStream(
                capability_id=active.capability_id,
                admitted_at=active.admitted_at,
                first_output_recorded=True,
            )
        if frame.is_terminal:
            if frame.kind == "completed":
                metrics.completed_streams += 1
            elif frame.kind == "failed":
                metrics.failed_streams += 1
            else:
                metrics.cancelled_streams += 1
            self._finish(call_id, metrics, now)

    def record_cancellation_request(self, call_id: str) -> None:
        """Record one authorized cancellation of an active provider stream.

        Args:
            call_id: Active call for which cancellation was requested.
        """

        active = self._active.get(call_id)
        if active is not None:
            self._metrics(active.capability_id).cancellation_requests += 1

    def finalize(self, call_id: str, *, observed_at: float | None = None) -> None:
        """Remove a stream when egress did not observe a terminal frame.

        Args:
            call_id: Active call being removed.
            observed_at: Optional monotonic timestamp for deterministic tests.
        """

        active = self._active.get(call_id)
        if active is None:
            return
        metrics = self._metrics(active.capability_id)
        metrics.missing_terminal_streams += 1
        self._finish(
            call_id,
            metrics,
            time.monotonic() if observed_at is None else observed_at,
        )

    def snapshot(
        self,
        *,
        active_unary_calls: int,
        stream_slots_in_use: int,
        unary_concurrency_limit: int,
        stream_concurrency_limit: int,
    ) -> ProviderDiagnostics:
        """Return aggregate and stable per-capability provider metrics.

        Args:
            active_unary_calls: Generic unary handlers currently executing.
            stream_slots_in_use: Admission or execution slots currently reserved.
            unary_concurrency_limit: Process-wide unary call bound.
            stream_concurrency_limit: Process-wide streaming call bound.

        Returns:
            Immutable response data for the node diagnostics surface.
        """

        capabilities = {
            capability_id: self._snapshot_metrics(metrics)
            for capability_id, metrics in sorted(self._capabilities.items())
        }
        aggregate = _CapabilityMetrics()
        for metrics in self._capabilities.values():
            for name in (
                "active_streams",
                "max_active_streams",
                "admitted_streams",
                "rejected_streams",
                "overloaded_rejections",
                "input_queue_depth",
                "max_input_queue_depth",
                "input_frames",
                "input_media_bytes",
                "output_frames",
                "output_media_bytes",
                "completed_streams",
                "failed_streams",
                "cancelled_streams",
                "missing_terminal_streams",
                "cancellation_requests",
            ):
                setattr(aggregate, name, getattr(aggregate, name) + getattr(metrics, name))
            aggregate.first_output.count += metrics.first_output.count
            aggregate.first_output.total += metrics.first_output.total
            aggregate.first_output.maximum = self._maximum(
                aggregate.first_output.maximum, metrics.first_output.maximum
            )
            aggregate.duration.count += metrics.duration.count
            aggregate.duration.total += metrics.duration.total
            aggregate.duration.maximum = self._maximum(
                aggregate.duration.maximum, metrics.duration.maximum
            )
        aggregate.max_active_streams = self._max_active_streams
        aggregate.max_input_queue_depth = self._max_input_queue_depth
        aggregate_snapshot = self._snapshot_metrics(aggregate)
        return ProviderDiagnostics.model_validate(
            {
                **aggregate_snapshot.model_dump(),
                "active_unary_calls": active_unary_calls,
                "stream_slots_in_use": stream_slots_in_use,
                "unary_concurrency_limit": unary_concurrency_limit,
                "stream_concurrency_limit": stream_concurrency_limit,
                "capabilities": capabilities,
            }
        )

    def _finish(
        self, call_id: str, metrics: _CapabilityMetrics, observed_at: float
    ) -> None:
        active = self._active.pop(call_id, None)
        if active is None:
            return
        self._input_queue_depths.pop(call_id, None)
        metrics.active_streams = max(0, metrics.active_streams - 1)
        metrics.input_queue_depth = sum(
            depth
            for active_call_id, depth in self._input_queue_depths.items()
            if self._active.get(active_call_id) is not None
            and self._active[active_call_id].capability_id == active.capability_id
        )
        metrics.duration.record(max(0.0, observed_at - active.admitted_at))

    def _metrics(self, capability_id: str) -> _CapabilityMetrics:
        return self._capabilities.setdefault(capability_id, _CapabilityMetrics())

    @staticmethod
    def _inline_media_bytes(frame: CapabilityStreamFrame) -> int:
        return len(frame.media.data) if isinstance(frame.media, InlineMediaAttachment) else 0

    @staticmethod
    def _maximum(left: float | None, right: float | None) -> float | None:
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)

    @staticmethod
    def _snapshot_metrics(metrics: _CapabilityMetrics) -> ProviderCapabilityDiagnostics:
        return ProviderCapabilityDiagnostics(
            active_streams=metrics.active_streams,
            max_active_streams=metrics.max_active_streams,
            admitted_streams=metrics.admitted_streams,
            rejected_streams=metrics.rejected_streams,
            overloaded_rejections=metrics.overloaded_rejections,
            input_queue_depth=metrics.input_queue_depth,
            max_input_queue_depth=metrics.max_input_queue_depth,
            input_frames=metrics.input_frames,
            input_media_bytes=metrics.input_media_bytes,
            output_frames=metrics.output_frames,
            output_media_bytes=metrics.output_media_bytes,
            completed_streams=metrics.completed_streams,
            failed_streams=metrics.failed_streams,
            cancelled_streams=metrics.cancelled_streams,
            missing_terminal_streams=metrics.missing_terminal_streams,
            cancellation_requests=metrics.cancellation_requests,
            first_output_samples=metrics.first_output.count,
            first_output_seconds_average=metrics.first_output.average,
            first_output_seconds_max=metrics.first_output.maximum,
            duration_samples=metrics.duration.count,
            duration_seconds_average=metrics.duration.average,
            duration_seconds_max=metrics.duration.maximum,
        )
