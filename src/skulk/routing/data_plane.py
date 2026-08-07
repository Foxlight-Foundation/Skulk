"""Router-side observability for bounded DATA stream egress."""

from dataclasses import dataclass

from skulk.shared.types.diagnostics import (
    DataPlaneEgressDiagnostics,
    DataPlaneOwnerDiagnostics,
)

_MAX_TRACKED_OWNERS = 256
_OVERFLOW_OWNER_KEY = "__other__"


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
class _OwnerMetrics:
    queue_depth: int = 0
    active_streams: int = 0
    max_queue_depth: int = 0
    frames_enqueued: int = 0
    frames_published: int = 0
    frames_dropped: int = 0
    publish_failures: int = 0
    idle_stream_reclaims: int = 0


class DataPlaneEgressObserver:
    """Track bounded per-owner and per-command router egress pressure."""

    def __init__(self) -> None:
        """Create a zeroed process-local router observer."""

        self._owners: dict[str, _OwnerMetrics] = {}
        self._overflow_owner = _OwnerMetrics()
        self._overflow_active_owners: dict[str, int] = {}
        self._enqueue_latency = _MetricAccumulator()
        self._publish_latency = _MetricAccumulator()
        self.active_stream_queues = 0
        self.queue_depth = 0
        self.max_queue_depth = 0
        self.local_short_circuits = 0
        self.remote_frames_enqueued = 0
        self.remote_frames_published = 0
        self.remote_frames_dropped = 0
        self.remote_publish_failures = 0
        self.remote_bytes_published = 0
        self.idle_stream_reclaims = 0

    def record_local_short_circuit(self) -> None:
        """Record one frame whose owner is the producing node."""

        self.local_short_circuits += 1

    def record_enqueue_latency(self, seconds: float) -> None:
        """Record time spent enqueueing a remote frame from the TopicRouter."""

        self._enqueue_latency.record(max(0.0, seconds))

    def record_stream_opened(self, owner: str) -> None:
        """Record creation of one independent remote command queue."""

        metrics = self._owner_for_open(owner)
        metrics.active_streams += 1
        self.active_stream_queues += 1

    def record_stream_closed(self, owner: str) -> None:
        """Record removal of one independent remote command queue."""

        overflow_streams = self._overflow_active_owners.get(owner)
        if overflow_streams is not None:
            metrics = self._overflow_owner
            if overflow_streams == 1:
                del self._overflow_active_owners[owner]
            else:
                self._overflow_active_owners[owner] = overflow_streams - 1
        else:
            metrics = self._owner(owner)
        metrics.active_streams = max(0, metrics.active_streams - 1)
        self.active_stream_queues = max(0, self.active_stream_queues - 1)

    def record_enqueued(self, owner: str) -> None:
        """Record one frame accepted into a remote command queue."""

        metrics = self._owner(owner)
        metrics.queue_depth += 1
        metrics.max_queue_depth = max(metrics.max_queue_depth, metrics.queue_depth)
        metrics.frames_enqueued += 1
        self.queue_depth += 1
        self.max_queue_depth = max(self.max_queue_depth, self.queue_depth)
        self.remote_frames_enqueued += 1

    def record_dequeued(self, owner: str) -> None:
        """Record one frame removed from a remote command queue for publish."""

        metrics = self._owner(owner)
        metrics.queue_depth = max(0, metrics.queue_depth - 1)
        self.queue_depth = max(0, self.queue_depth - 1)

    def record_dropped(self, owner: str) -> None:
        """Record one frame rejected by a bounded queue or admission cap."""

        self._owner(owner).frames_dropped += 1
        self.remote_frames_dropped += 1

    def record_published(self, owner: str, size_bytes: int, seconds: float) -> None:
        """Record one successful remote Zenoh publish."""

        self._owner(owner).frames_published += 1
        self.remote_frames_published += 1
        self.remote_bytes_published += size_bytes
        self._publish_latency.record(max(0.0, seconds))

    def record_publish_failure(self, owner: str, seconds: float) -> None:
        """Record one failed or deadline-expired remote Zenoh publish."""

        self._owner(owner).publish_failures += 1
        self.remote_publish_failures += 1
        self._publish_latency.record(max(0.0, seconds))

    def record_stream_idle_reclaimed(self, owner: str) -> None:
        """Record one command queue reclaimed after its idle lease expired."""

        self._owner(owner).idle_stream_reclaims += 1
        self.idle_stream_reclaims += 1

    def snapshot(self) -> DataPlaneEgressDiagnostics:
        """Return the current immutable egress diagnostics model."""

        return DataPlaneEgressDiagnostics(
            active_stream_queues=self.active_stream_queues,
            queue_depth=self.queue_depth,
            max_queue_depth=self.max_queue_depth,
            local_short_circuits=self.local_short_circuits,
            remote_frames_enqueued=self.remote_frames_enqueued,
            remote_frames_published=self.remote_frames_published,
            remote_frames_dropped=self.remote_frames_dropped,
            remote_publish_failures=self.remote_publish_failures,
            remote_bytes_published=self.remote_bytes_published,
            idle_stream_reclaims=self.idle_stream_reclaims,
            enqueue_latency_samples=self._enqueue_latency.count,
            enqueue_latency_seconds_average=self._enqueue_latency.average,
            enqueue_latency_seconds_max=self._enqueue_latency.maximum,
            publish_latency_samples=self._publish_latency.count,
            publish_latency_seconds_average=self._publish_latency.average,
            publish_latency_seconds_max=self._publish_latency.maximum,
            owners=self._owner_snapshots(),
        )

    def _owner_snapshots(self) -> dict[str, DataPlaneOwnerDiagnostics]:
        owners = {
            owner: DataPlaneOwnerDiagnostics(
                queue_depth=metrics.queue_depth,
                active_streams=metrics.active_streams,
                max_queue_depth=metrics.max_queue_depth,
                frames_enqueued=metrics.frames_enqueued,
                frames_published=metrics.frames_published,
                frames_dropped=metrics.frames_dropped,
                publish_failures=metrics.publish_failures,
                idle_stream_reclaims=metrics.idle_stream_reclaims,
            )
            for owner, metrics in self._owners.items()
        }
        overflow = self._overflow_owner
        if any(
            (
                overflow.queue_depth,
                overflow.active_streams,
                overflow.max_queue_depth,
                overflow.frames_enqueued,
                overflow.frames_published,
                overflow.frames_dropped,
                overflow.publish_failures,
                overflow.idle_stream_reclaims,
            )
        ):
            owners[_OVERFLOW_OWNER_KEY] = DataPlaneOwnerDiagnostics(
                queue_depth=overflow.queue_depth,
                active_streams=overflow.active_streams,
                max_queue_depth=overflow.max_queue_depth,
                frames_enqueued=overflow.frames_enqueued,
                frames_published=overflow.frames_published,
                frames_dropped=overflow.frames_dropped,
                publish_failures=overflow.publish_failures,
                idle_stream_reclaims=overflow.idle_stream_reclaims,
            )
        return owners

    def _owner_for_open(self, owner: str) -> _OwnerMetrics:
        if owner in self._overflow_active_owners:
            self._overflow_active_owners[owner] += 1
            return self._overflow_owner
        metrics = self._tracked_owner(owner)
        if metrics is not None:
            return metrics
        self._overflow_active_owners[owner] = 1
        return self._overflow_owner

    def _owner(self, owner: str) -> _OwnerMetrics:
        if owner in self._overflow_active_owners:
            return self._overflow_owner
        return self._tracked_owner(owner) or self._overflow_owner

    def _tracked_owner(self, owner: str) -> _OwnerMetrics | None:
        existing = self._owners.get(owner)
        if existing is not None:
            return existing
        if len(self._owners) >= _MAX_TRACKED_OWNERS:
            evictable_owner = next(
                (
                    tracked_owner
                    for tracked_owner, metrics in self._owners.items()
                    if metrics.active_streams == 0 and metrics.queue_depth == 0
                ),
                None,
            )
            if evictable_owner is None:
                return None
            del self._owners[evictable_owner]
        metrics = _OwnerMetrics()
        self._owners[owner] = metrics
        return metrics
