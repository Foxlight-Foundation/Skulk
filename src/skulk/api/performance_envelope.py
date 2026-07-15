"""Observe-only performance-envelope registry (adaptive concurrency, Phase 0).

Skulk records, for each ``(hardware class x model x engine+backend x quant)``,
how throughput and latency behave as a function of the number of requests the
serving instance handles at once. That per-concurrency rollup is the
throughput-versus-concurrency curve; its *knee* (the concurrency past which
aggregate throughput stops rising) is the number a later admission controller
would target. This module is the data asset. It changes no behavior.

The registry lives on the API node and is fed one observation per completed
generation by a stream tap (mirroring the field-telemetry tap). Each
observation carries the envelope key, the in-flight concurrency the instance was
already serving when this request was admitted, and the request's time-to-first
-token and steady-state decode rate. The registry keeps a bounded reservoir of
recent samples per concurrency bucket and, on read, computes summary statistics
and a simple knee estimate.

Bounds are load-bearing: the sample reservoirs and the envelope count are capped
so a long-lived API node cannot grow this without limit. Nothing here touches
the event log, State, or the telemetry gossip plane; the rollup is exposed only
through a read-only diagnostics endpoint, and the cluster view is a fan-out over
peer endpoints.

Concurrency note: the in-flight count is measured from THIS API node's
outstanding requests to the instance. With one API node that is exactly the
instance's concurrency; across several API nodes each sees only its own share.
The engine-reported batch size is the fully accurate signal and is a documented
follow-on, not part of this observe-only phase.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Literal, final

from skulk.utils.pydantic_ext import CamelCaseModel

#: Outcome of one observed generation. ``cancelled`` (client disconnect) is
#: recorded separately so it never pollutes the latency/throughput distributions.
GenerationOutcome = Literal["success", "error", "cancelled"]

#: Recent (ttft_seconds, decode_tps) samples kept per concurrency bucket. Bounded
#: so percentiles stay meaningful without unbounded memory.
_MAX_SAMPLES_PER_BUCKET = 256
#: Cap on distinct envelope keys tracked at once (a busy mixed fleet is far
#: below this; the cap is a runaway backstop, not an expected limit).
_MAX_ENVELOPES = 512


@final
class ConcurrencyBucketSummary(CamelCaseModel):
    """Aggregated performance at one in-flight-concurrency level.

    Attributes:
        concurrency: In-flight requests the instance was already serving when a
            request in this bucket was admitted (1 for the single-stream engines).
        request_count: Observations folded into this bucket's reservoir.
        success_count: Observations that finished cleanly.
        error_count: Observations that ended in a generation error.
        ttft_seconds_p50: Median time-to-first-token over the reservoir.
        ttft_seconds_p90: 90th-percentile time-to-first-token.
        decode_tps_mean: Mean steady-state decode tokens/second.
        decode_tps_p50: Median steady-state decode tokens/second.
        aggregate_decode_tps: ``concurrency * decode_tps_mean`` -- the instance's
            total useful throughput at this concurrency, the quantity whose knee
            an admission controller targets.
    """

    concurrency: int
    request_count: int
    success_count: int
    error_count: int
    ttft_seconds_p50: float | None
    ttft_seconds_p90: float | None
    decode_tps_mean: float | None
    decode_tps_p50: float | None
    aggregate_decode_tps: float | None


@final
class PerformanceEnvelopeSummary(CamelCaseModel):
    """One model+engine+hardware envelope: its per-concurrency curve and knee.

    Attributes:
        hardware_class: Canonical class of the serving node(s), e.g.
            ``nvidia-a100-80gb`` or ``apple-m4-24gb``.
        model_id: The served model identifier.
        backend: Resolved engine+backend tag, e.g. ``vllm-cuda``,
            ``llama_server-rocm``, ``mlx``.
        quantization: Model quantization label (``4bit``, ``Q4_K_M``, ...), or
            empty when unquantized/unknown.
        buckets: Per-concurrency summaries, ascending by concurrency.
        knee_concurrency: Concurrency past which aggregate decode throughput
            stops rising meaningfully, or ``None`` when too few concurrency
            levels have been observed to estimate one.
        observation_count: Total observations across all buckets.
    """

    hardware_class: str
    model_id: str
    backend: str
    quantization: str
    buckets: list[ConcurrencyBucketSummary]
    knee_concurrency: int | None
    observation_count: int


@final
class PerformanceEnvelopeReport(CamelCaseModel):
    """Read-only snapshot of one node's performance envelopes.

    Attributes:
        generated_at: UTC ISO-8601 time the snapshot was computed.
        envelopes: One summary per observed (hardware x model x engine x quant).
    """

    generated_at: str
    envelopes: list[PerformanceEnvelopeSummary]


@final
class NodePerformanceEnvelopes(CamelCaseModel):
    """One cluster member's envelope report, or the reason it is unavailable.

    Attributes:
        node_id: The member's node identifier.
        url: The peer API base URL, or ``None`` for the local node / unreachable.
        ok: Whether the report was collected.
        report: The member's envelope report when ``ok``.
        error: The failure summary when not ``ok``.
    """

    node_id: str
    url: str | None
    ok: bool
    report: PerformanceEnvelopeReport | None = None
    error: str | None = None


@final
class ClusterPerformanceEnvelopes(CamelCaseModel):
    """Read-only performance envelopes gathered from every reachable member.

    Attributes:
        generated_at: UTC ISO-8601 time the fan-out was assembled.
        nodes: One entry per cluster member (local first), each carrying its
            report or an explicit failure so an unreachable member never
            silently vanishes.
    """

    generated_at: str
    nodes: list[NodePerformanceEnvelopes]


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = fraction * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@final
class _Bucket:
    """Mutable per-concurrency reservoir. Not serialized directly."""

    def __init__(self) -> None:
        self.ttft_seconds: deque[float] = deque(maxlen=_MAX_SAMPLES_PER_BUCKET)
        self.decode_tps: deque[float] = deque(maxlen=_MAX_SAMPLES_PER_BUCKET)
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

    def record(
        self,
        ttft_seconds: float | None,
        decode_tps: float | None,
        outcome: GenerationOutcome,
    ) -> None:
        self.request_count += 1
        if outcome == "success":
            self.success_count += 1
        elif outcome == "error":
            self.error_count += 1
        # Only clean samples inform the latency/throughput distributions; an
        # errored or cancelled request's partial timings would skew them.
        if outcome != "success":
            return
        if ttft_seconds is not None and ttft_seconds >= 0.0:
            self.ttft_seconds.append(ttft_seconds)
        if decode_tps is not None and decode_tps > 0.0:
            self.decode_tps.append(decode_tps)

    def summary(self, concurrency: int) -> ConcurrencyBucketSummary:
        ttft_sorted = sorted(self.ttft_seconds)
        tps_sorted = sorted(self.decode_tps)
        decode_mean = sum(tps_sorted) / len(tps_sorted) if tps_sorted else None
        aggregate = decode_mean * concurrency if decode_mean is not None else None
        return ConcurrencyBucketSummary(
            concurrency=concurrency,
            request_count=self.request_count,
            success_count=self.success_count,
            error_count=self.error_count,
            ttft_seconds_p50=_percentile(ttft_sorted, 0.50) if ttft_sorted else None,
            ttft_seconds_p90=_percentile(ttft_sorted, 0.90) if ttft_sorted else None,
            decode_tps_mean=decode_mean,
            decode_tps_p50=_percentile(tps_sorted, 0.50) if tps_sorted else None,
            aggregate_decode_tps=aggregate,
        )


def _knee(buckets: list[ConcurrencyBucketSummary]) -> int | None:
    """Estimate the concurrency knee from aggregate throughput per bucket.

    The knee is the concurrency at which aggregate decode throughput peaks: past
    it, adding concurrency no longer buys throughput (and typically costs
    latency). Needs at least two concurrency levels with a throughput reading;
    returns ``None`` otherwise. Deliberately simple -- a faithful curve with a
    plain argmax beats a clever fit over sparse data.
    """
    scored = [
        (b.concurrency, b.aggregate_decode_tps)
        for b in buckets
        if b.aggregate_decode_tps is not None
    ]
    if len(scored) < 2:
        return None
    return max(scored, key=lambda pair: pair[1])[0]


@final
class PerformanceEnvelopeRegistry:
    """Bounded, in-memory registry of performance envelopes on the API node.

    Thread-unsafe by design: the API records from its single asyncio loop. If a
    threaded caller is ever added, guard :meth:`record` and :meth:`snapshot`
    with a lock.
    """

    def __init__(
        self,
        *,
        now: Callable[[], str],
        max_envelopes: int = _MAX_ENVELOPES,
    ) -> None:
        self._now = now
        self._max_envelopes = max_envelopes
        # (hardware_class, model_id, backend, quantization) -> concurrency -> bucket
        self._envelopes: dict[tuple[str, str, str, str], dict[int, _Bucket]] = {}

    def record(
        self,
        *,
        hardware_class: str,
        model_id: str,
        backend: str,
        quantization: str,
        concurrency: int,
        ttft_seconds: float | None,
        decode_tps: float | None,
        outcome: GenerationOutcome,
    ) -> None:
        """Fold one completed generation into its envelope's concurrency bucket.

        ``concurrency`` is clamped to at least 1 (a request always includes
        itself). A new envelope key is dropped silently once ``max_envelopes`` is
        reached, so a pathological fan-out of keys cannot grow memory without
        bound; existing envelopes keep updating.
        """
        concurrency = max(1, concurrency)
        key = (hardware_class, model_id, backend, quantization)
        envelope = self._envelopes.get(key)
        if envelope is None:
            if len(self._envelopes) >= self._max_envelopes:
                return
            envelope = {}
            self._envelopes[key] = envelope
        bucket = envelope.get(concurrency)
        if bucket is None:
            bucket = _Bucket()
            envelope[concurrency] = bucket
        bucket.record(ttft_seconds, decode_tps, outcome)

    def snapshot(self) -> PerformanceEnvelopeReport:
        """Compute the read-only report: per-envelope curves, sorted, with knees."""
        summaries: list[PerformanceEnvelopeSummary] = []
        for (hardware, model_id, backend, quant), buckets in self._envelopes.items():
            bucket_summaries = [
                buckets[concurrency].summary(concurrency)
                for concurrency in sorted(buckets)
            ]
            observation_count = sum(b.request_count for b in bucket_summaries)
            summaries.append(
                PerformanceEnvelopeSummary(
                    hardware_class=hardware,
                    model_id=model_id,
                    backend=backend,
                    quantization=quant,
                    buckets=bucket_summaries,
                    knee_concurrency=_knee(bucket_summaries),
                    observation_count=observation_count,
                )
            )
        # Stable, human-friendly ordering: busiest envelopes first.
        summaries.sort(key=lambda s: s.observation_count, reverse=True)
        return PerformanceEnvelopeReport(
            generated_at=self._now(), envelopes=summaries
        )
