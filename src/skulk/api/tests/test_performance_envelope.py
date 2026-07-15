# pyright: reportPrivateUsage=false
"""Tests for the observe-only performance-envelope registry."""

from __future__ import annotations

from skulk.api.performance_envelope import (
    _MAX_CONCURRENCY_BUCKET,
    _MAX_ENVELOPES,
    PerformanceEnvelopeRegistry,
    _knee,
    _percentile,
)


def _registry() -> PerformanceEnvelopeRegistry:
    return PerformanceEnvelopeRegistry(now=lambda: "2026-07-15T00:00:00Z")


def test_percentile_basic() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(sorted(values), 0.0) == 1.0
    assert _percentile(sorted(values), 1.0) == 4.0
    # Median of 1..4 by linear interpolation over ranks is 2.5.
    assert _percentile(sorted(values), 0.5) == 2.5


def test_single_observation_forms_one_bucket() -> None:
    reg = _registry()
    reg.record(
        hardware_class="nvidia-a100-80gb",
        model_id="m",
        backend="vllm-cuda",
        quantization="",
        concurrency=1,
        ttft_seconds=0.2,
        decode_tps=50.0,
        outcome="success",
        batches=True,
    )
    report = reg.snapshot()
    assert len(report.envelopes) == 1
    env = report.envelopes[0]
    assert env.hardware_class == "nvidia-a100-80gb"
    assert env.backend == "vllm-cuda"
    assert env.observation_count == 1
    assert len(env.buckets) == 1
    bucket = env.buckets[0]
    assert bucket.concurrency == 1
    assert bucket.decode_tps_mean == 50.0
    assert bucket.aggregate_decode_tps == 50.0
    # Too few concurrency levels to estimate a knee.
    assert env.knee_concurrency is None


def test_concurrency_clamped_to_one() -> None:
    reg = _registry()
    reg.record(
        hardware_class="hw",
        model_id="m",
        backend="mlx",
        quantization="4bit",
        concurrency=0,
        ttft_seconds=0.1,
        decode_tps=10.0,
        outcome="success",
        batches=True,
    )
    assert reg.snapshot().envelopes[0].buckets[0].concurrency == 1


def test_serial_backend_aggregate_uses_effective_concurrency_one() -> None:
    reg = _registry()
    # A serial backend (batches=False) queues concurrent requests, so its
    # aggregate must NOT scale with the observed concurrency bucket: at
    # concurrency 4 with full-speed decode, aggregate stays at the per-request
    # rate, and the knee is not biased upward.
    for concurrency in (1, 2, 4):
        for _ in range(3):
            reg.record(hardware_class="apple-m4-24gb", model_id="m", backend="mlx",
                       quantization="4bit", concurrency=concurrency,
                       ttft_seconds=0.2, decode_tps=20.0, outcome="success",
                       batches=False)
    env = reg.snapshot().envelopes[0]
    assert env.batches is False
    aggregates = {b.concurrency: b.aggregate_decode_tps for b in env.buckets}
    assert aggregates == {1: 20.0, 2: 20.0, 4: 20.0}
    # Flat aggregate -> the knee is the lowest concurrency (no upward bias).
    assert env.knee_concurrency == 1


def test_curve_and_knee_across_concurrency_levels() -> None:
    reg = _registry()
    # Aggregate throughput rises then plateaus: per-request tps 40 at c=1 (agg
    # 40), 30 at c=2 (agg 60), 15 at c=4 (agg 60). Knee is c=2: argmax over
    # ascending concurrency returns the LOWEST concurrency that hits the peak.
    for _ in range(5):
        reg.record(hardware_class="hw", model_id="m", backend="vllm-cuda",
                   quantization="", concurrency=1, ttft_seconds=0.2,
                   decode_tps=40.0, outcome="success", batches=True)
        reg.record(hardware_class="hw", model_id="m", backend="vllm-cuda",
                   quantization="", concurrency=2, ttft_seconds=0.3,
                   decode_tps=30.0, outcome="success", batches=True)
        reg.record(hardware_class="hw", model_id="m", backend="vllm-cuda",
                   quantization="", concurrency=4, ttft_seconds=0.6,
                   decode_tps=15.0, outcome="success", batches=True)
    env = reg.snapshot().envelopes[0]
    assert [b.concurrency for b in env.buckets] == [1, 2, 4]
    aggregates = {b.concurrency: b.aggregate_decode_tps for b in env.buckets}
    assert aggregates[1] == 40.0
    assert aggregates[2] == 60.0
    assert aggregates[4] == 60.0
    # c=2 and c=4 tie at 60; argmax returns the first max encountered (c=2).
    assert env.knee_concurrency == 2


def test_errors_and_cancellations_excluded_from_distributions() -> None:
    reg = _registry()
    reg.record(hardware_class="hw", model_id="m", backend="b", quantization="",
               concurrency=1, ttft_seconds=0.2, decode_tps=50.0, outcome="success", batches=True)
    reg.record(hardware_class="hw", model_id="m", backend="b", quantization="",
               concurrency=1, ttft_seconds=99.0, decode_tps=0.1, outcome="error", batches=True)
    reg.record(hardware_class="hw", model_id="m", backend="b", quantization="",
               concurrency=1, ttft_seconds=88.0, decode_tps=0.2, outcome="cancelled", batches=True)
    bucket = reg.snapshot().envelopes[0].buckets[0]
    assert bucket.request_count == 3
    assert bucket.success_count == 1
    assert bucket.error_count == 1
    # The error/cancelled timings must not pollute the success distribution.
    assert bucket.decode_tps_mean == 50.0
    assert bucket.ttft_seconds_p50 == 0.2


def test_distinct_keys_form_distinct_envelopes() -> None:
    reg = _registry()
    for backend in ("vllm-cuda", "llama_server-cuda"):
        reg.record(hardware_class="hw", model_id="m", backend=backend,
                   quantization="", concurrency=1, ttft_seconds=0.2,
                   decode_tps=10.0, outcome="success", batches=True)
    assert len(reg.snapshot().envelopes) == 2


def test_envelope_cap_drops_new_keys_but_keeps_existing() -> None:
    reg = _registry()
    for i in range(_MAX_ENVELOPES):
        reg.record(hardware_class="hw", model_id=f"m{i}", backend="b",
                   quantization="", concurrency=1, ttft_seconds=0.2,
                   decode_tps=10.0, outcome="success", batches=True)
    assert len(reg.snapshot().envelopes) == _MAX_ENVELOPES
    # A new key past the cap is dropped...
    reg.record(hardware_class="hw", model_id="overflow", backend="b",
               quantization="", concurrency=1, ttft_seconds=0.2,
               decode_tps=10.0, outcome="success", batches=True)
    assert len(reg.snapshot().envelopes) == _MAX_ENVELOPES
    # ...but an existing key still updates.
    reg.record(hardware_class="hw", model_id="m0", backend="b",
               quantization="", concurrency=2, ttft_seconds=0.2,
               decode_tps=10.0, outcome="success", batches=True)
    m0 = next(e for e in reg.snapshot().envelopes if e.model_id == "m0")
    assert {b.concurrency for b in m0.buckets} == {1, 2}


def test_concurrency_clamped_to_max_bucket() -> None:
    reg = _registry()
    # A single hot model hammered above the cap must not create unbounded
    # buckets: everything above _MAX_CONCURRENCY_BUCKET folds into the top bucket.
    for concurrency in (_MAX_CONCURRENCY_BUCKET, _MAX_CONCURRENCY_BUCKET + 1, 5000):
        reg.record(hardware_class="hw", model_id="m", backend="b", quantization="",
                   concurrency=concurrency, ttft_seconds=0.2, decode_tps=10.0,
                   outcome="success", batches=True)
    buckets = reg.snapshot().envelopes[0].buckets
    assert [b.concurrency for b in buckets] == [_MAX_CONCURRENCY_BUCKET]
    assert buckets[0].request_count == 3


def test_empty_registry_snapshot() -> None:
    report = _registry().snapshot()
    assert report.envelopes == []
    assert report.generated_at == "2026-07-15T00:00:00Z"


def test_knee_helper_needs_two_levels() -> None:
    reg = _registry()
    reg.record(hardware_class="hw", model_id="m", backend="b", quantization="",
               concurrency=1, ttft_seconds=0.2, decode_tps=10.0, outcome="success", batches=True)
    env = reg.snapshot().envelopes[0]
    assert _knee(env.buckets) is None
