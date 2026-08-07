"""Bounded local observability for ordered DATA stream delivery."""

import time
from collections import deque
from dataclasses import dataclass

from skulk.shared.types.chunks import DataChunk
from skulk.shared.types.common import CommandId
from skulk.shared.types.diagnostics import (
    DataPlaneDiagnostics,
    DataPlaneEgressDiagnostics,
    DataPlaneTransport,
)

_TERMINAL_TOMBSTONE_LIMIT = 4096


@dataclass
class _MetricAccumulator:
    count: int = 0
    total: float = 0.0
    last: float | None = None
    maximum: float | None = None

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.last = value
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    @property
    def average(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class _StreamObservation:
    started_at: float | None
    first_payload_at: float | None = None


class DataPlaneObserver:
    """Track bounded process-local delivery health for DATA stream frames."""

    def __init__(
        self,
        *,
        transport: DataPlaneTransport,
        reorder_buffer_enabled: bool,
    ) -> None:
        """Create an observer for one API node's configured DATA transport."""

        self._transport: DataPlaneTransport = transport
        self._reorder_buffer_enabled = reorder_buffer_enabled
        self._streams: dict[CommandId, _StreamObservation] = {}
        self._terminal_commands: set[CommandId] = set()
        self._terminal_order: deque[CommandId] = deque()
        self._first_byte = _MetricAccumulator()
        self._stream_span = _MetricAccumulator()
        self.frames_received = 0
        self.frames_dispatched = 0
        self.started_frames = 0
        self.chunk_frames = 0
        self.completed_frames = 0
        self.failed_frames = 0
        self.cancelled_frames = 0
        self.duplicate_frames = 0
        self.out_of_order_frames = 0
        self.skipped_sequences = 0
        self.late_frames = 0
        self.missing_started_streams = 0
        self.missing_terminal_streams = 0
        self.idle_timeouts = 0
        self.transport_failures = 0

    def record_received(self) -> None:
        """Record one DATA frame before ordering and deduplication."""

        self.frames_received += 1

    def record_duplicate(self) -> None:
        """Record one duplicate or late-sequence frame."""

        self.duplicate_frames += 1

    def record_out_of_order(self) -> None:
        """Record one frame received above the next expected sequence."""

        self.out_of_order_frames += 1

    def record_skipped_sequences(self, count: int) -> None:
        """Record sequence positions skipped to release a bounded reorder gap."""

        self.skipped_sequences += max(0, count)

    def record_late(self) -> None:
        """Record one frame dropped because its command no longer has a queue."""

        self.late_frames += 1

    def record_idle_timeout(self) -> None:
        """Record one API receive that exceeded its DATA stream idle deadline."""

        self.idle_timeouts += 1

    def record_transport_failure(self, command_id: CommandId) -> None:
        """Record a receiver-synthesized terminal caused by DATA delivery."""

        self.transport_failures += 1
        self._streams.pop(command_id, None)
        self._remember_terminal(command_id)

    def record_dispatched(
        self,
        frame: DataChunk,
        *,
        observed_at: float | None = None,
    ) -> None:
        """Record an ordered, unique frame accepted for a live command."""

        now = time.monotonic() if observed_at is None else observed_at
        self.frames_dispatched += 1
        if frame.kind == "started":
            self.started_frames += 1
            self._streams[frame.command_id] = _StreamObservation(started_at=now)
            self._terminal_commands.discard(frame.command_id)
            return

        observation = self._streams.get(frame.command_id)
        if observation is None:
            self.missing_started_streams += 1
            observation = _StreamObservation(started_at=None)
            self._streams[frame.command_id] = observation

        if frame.chunk is not None and observation.first_payload_at is None:
            observation.first_payload_at = now
            if observation.started_at is not None:
                self._first_byte.record(max(0.0, now - observation.started_at))

        if frame.kind == "chunk":
            self.chunk_frames += 1
            return
        if frame.kind == "completed":
            self.completed_frames += 1
        elif frame.kind == "failed":
            self.failed_frames += 1
        else:
            self.cancelled_frames += 1

        if observation.first_payload_at is not None:
            self._stream_span.record(max(0.0, now - observation.first_payload_at))
        self._streams.pop(frame.command_id, None)
        self._remember_terminal(frame.command_id)

    def finalize(self, command_id: CommandId) -> None:
        """Forget one local command, counting an absent terminal lifecycle frame."""

        if command_id in self._streams:
            self.missing_terminal_streams += 1
            self._streams.pop(command_id, None)
        self._remember_terminal(command_id)

    def snapshot(
        self,
        egress: DataPlaneEgressDiagnostics | None = None,
    ) -> DataPlaneDiagnostics:
        """Return diagnostics joined with an optional router egress snapshot."""

        return DataPlaneDiagnostics(
            transport=self._transport,
            reorder_buffer_enabled=self._reorder_buffer_enabled,
            active_streams=len(self._streams),
            frames_received=self.frames_received,
            frames_dispatched=self.frames_dispatched,
            started_frames=self.started_frames,
            chunk_frames=self.chunk_frames,
            completed_frames=self.completed_frames,
            failed_frames=self.failed_frames,
            cancelled_frames=self.cancelled_frames,
            duplicate_frames=self.duplicate_frames,
            out_of_order_frames=self.out_of_order_frames,
            skipped_sequences=self.skipped_sequences,
            late_frames=self.late_frames,
            missing_started_streams=self.missing_started_streams,
            missing_terminal_streams=self.missing_terminal_streams,
            idle_timeouts=self.idle_timeouts,
            transport_failures=self.transport_failures,
            first_byte_samples=self._first_byte.count,
            first_byte_seconds_last=self._first_byte.last,
            first_byte_seconds_average=self._first_byte.average,
            first_byte_seconds_max=self._first_byte.maximum,
            stream_span_samples=self._stream_span.count,
            stream_span_seconds_last=self._stream_span.last,
            stream_span_seconds_average=self._stream_span.average,
            stream_span_seconds_max=self._stream_span.maximum,
            egress=egress or DataPlaneEgressDiagnostics.empty(),
        )

    def _remember_terminal(self, command_id: CommandId) -> None:
        if command_id in self._terminal_commands:
            return
        self._terminal_commands.add(command_id)
        self._terminal_order.append(command_id)
        while len(self._terminal_order) > _TERMINAL_TOMBSTONE_LIMIT:
            self._terminal_commands.discard(self._terminal_order.popleft())
