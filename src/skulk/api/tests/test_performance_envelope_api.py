# pyright: reportPrivateUsage=false, reportAny=false, reportExplicitAny=false, reportUnknownLambdaType=false
"""API-level tests for the observe-only performance-envelope surface.

Covers the diagnostics endpoint reflecting the registry and the stream tap that
feeds it (concurrency capture, outcome classification, aborted-stream skip),
without standing up a full cluster.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel


def _build_api(node_id: str = "test-node") -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId(node_id),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


def _chunk(text: str | None, finish_reason: str | None, tps: float | None) -> Any:
    stats = SimpleNamespace(generation_tps=tps) if tps is not None else None
    return SimpleNamespace(text=text, finish_reason=finish_reason, stats=stats)


def _served_chunk(
    text: str | None,
    finish_reason: str | None,
    tps: float | None,
    *,
    node: str | None,
    backend: str | None,
    in_flight: int | None,
    batches: bool | None,
) -> Any:
    """A chunk whose stats carry the runner-reported serving fields (#596)."""
    stats = SimpleNamespace(
        generation_tps=tps,
        serving_node=node,
        serving_backend=backend,
        in_flight_at_admission=in_flight,
        serving_batches=batches,
    )
    return SimpleNamespace(text=text, finish_reason=finish_reason, stats=stats)


async def _drain(gen: Any) -> None:
    async for _ in gen:
        pass


def test_endpoint_reflects_registry() -> None:
    api = _build_api()
    client = TestClient(api.app)

    empty = client.get("/v1/diagnostics/performance-envelopes")
    assert empty.status_code == 200
    assert empty.json()["envelopes"] == []

    api._performance_envelopes.record(
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
    populated = client.get("/v1/diagnostics/performance-envelopes")
    body = cast(dict[str, Any], populated.json())
    assert len(body["envelopes"]) == 1
    envelope = body["envelopes"][0]
    assert envelope["hardwareClass"] == "nvidia-a100-80gb"
    assert envelope["backend"] == "vllm-cuda"
    assert envelope["buckets"][0]["concurrency"] == 1


async def test_tap_records_on_completion() -> None:
    api = _build_api()
    # Stub context resolution so the test needs no live instance/telemetry.
    object.__setattr__(
        api,
        "_resolve_envelope_context",
        lambda _model: ("nvidia-a100-80gb", "vllm-cuda", "4bit"),
    )
    stream = _one_stream(
        [_chunk("hi", None, None), _chunk("", "stop", 42.0)]
    )
    await _drain(api._tap_performance_envelope(ModelId("m"), stream))

    report = api._performance_envelopes.snapshot()
    assert len(report.envelopes) == 1
    bucket = report.envelopes[0].buckets[0]
    assert bucket.concurrency == 1
    assert bucket.success_count == 1
    assert bucket.decode_tps_mean == 42.0
    # In-flight is released after completion.
    assert api._envelope_inflight.get("m", 0) == 0


async def test_tap_skips_aborted_stream() -> None:
    api = _build_api()
    object.__setattr__(
        api,
        "_resolve_envelope_context",
        lambda _model: ("hw", "b", ""),
    )
    # No terminal chunk => client-abort; must not be recorded.
    stream = _one_stream([_chunk("partial", None, None)])
    await _drain(api._tap_performance_envelope(ModelId("m"), stream))

    assert api._performance_envelopes.snapshot().envelopes == []
    assert api._envelope_inflight.get("m", 0) == 0


async def test_tap_records_error_outcome() -> None:
    api = _build_api()
    object.__setattr__(
        api, "_resolve_envelope_context", lambda _model: ("hw", "b", "")
    )
    stream = _one_stream([_chunk("", "error", None)])
    await _drain(api._tap_performance_envelope(ModelId("m"), stream))

    bucket = api._performance_envelopes.snapshot().envelopes[0].buckets[0]
    assert bucket.error_count == 1
    assert bucket.success_count == 0


async def test_tap_prefers_runner_reported_context() -> None:
    api = _build_api()
    # Runner-reported path (#596): the served runner stamps the serving node,
    # backend, in-flight-at-admission, and batching mode onto the terminal
    # stats. The tap must key the envelope from those, NOT from the API's own
    # dispatch-time offered count -- so it is correct across replicas and when
    # several API nodes drive one instance. Stub the two resolvers so the test
    # needs no live telemetry/state.
    object.__setattr__(api, "_hardware_class_for_node", lambda _n: "nvidia-a100-80gb")
    object.__setattr__(api, "_model_quantization", lambda _m: "4bit")

    def _fallback_boom(_model: Any) -> Any:
        raise AssertionError("fallback context must not be used when the runner reports")

    object.__setattr__(api, "_resolve_envelope_context", _fallback_boom)

    stream = _one_stream(
        [
            _served_chunk(
                "hi", None, None,
                node="n1", backend="vllm-cuda", in_flight=5, batches=True,
            ),
            _served_chunk(
                "", "stop", 42.0,
                node="n1", backend="vllm-cuda", in_flight=5, batches=True,
            ),
        ]
    )
    await _drain(api._tap_performance_envelope(ModelId("m"), stream))

    env = api._performance_envelopes.snapshot().envelopes[0]
    assert env.hardware_class == "nvidia-a100-80gb"
    assert env.backend == "vllm-cuda"
    assert env.quantization == "4bit"
    assert env.batches is True
    bucket = env.buckets[0]
    # Concurrency is the RUNNER's in-flight (5), not the API's offered count (1).
    assert bucket.concurrency == 5
    assert bucket.success_count == 1


async def test_tap_runner_reported_unblinds_replicas() -> None:
    api = _build_api()
    # With two instances serving the same model the fallback path skips (it
    # records only when EXACTLY one instance serves, to avoid mis-keying). The
    # runner-reported path is per-instance, so it records anyway.
    object.__setattr__(api, "_hardware_class_for_node", lambda _n: "hw")
    object.__setattr__(api, "_model_quantization", lambda _m: "4bit")
    object.__setattr__(api, "_resolve_envelope_context", lambda _model: None)

    stream = _one_stream(
        [
            _served_chunk(
                "", "stop", 30.0,
                node="n2", backend="llama_server-vulkan", in_flight=1, batches=False,
            )
        ]
    )
    await _drain(api._tap_performance_envelope(ModelId("m"), stream))

    env = api._performance_envelopes.snapshot().envelopes[0]
    assert env.backend == "llama_server-vulkan"
    assert env.batches is False
    assert env.buckets[0].concurrency == 1


async def test_tap_runner_reported_skips_when_node_hardware_unknown() -> None:
    api = _build_api()
    # A stamped serving node whose telemetry has not populated yet must be
    # skipped rather than recorded under an incomplete key.
    object.__setattr__(api, "_hardware_class_for_node", lambda _n: None)
    object.__setattr__(api, "_model_quantization", lambda _m: "4bit")
    object.__setattr__(api, "_resolve_envelope_context", lambda _model: None)

    stream = _one_stream(
        [
            _served_chunk(
                "", "stop", 30.0,
                node="n3", backend="vllm-cuda", in_flight=2, batches=True,
            )
        ]
    )
    await _drain(api._tap_performance_envelope(ModelId("m"), stream))

    assert api._performance_envelopes.snapshot().envelopes == []
    assert api._envelope_inflight.get("m", 0) == 0


async def test_cluster_fanout_reports_local_first_and_peer_failure() -> None:
    api = _build_api("local-node")

    async def _fake_peers(fail_fast: bool = False) -> dict[str, str]:
        # An unreachable peer (discard port) exercises the ok=false failure path.
        return {"peer-1": "http://127.0.0.1:9"}

    object.__setattr__(api, "_reachable_peer_api_urls", _fake_peers)
    result = await api.get_cluster_performance_envelopes()

    assert result.nodes[0].node_id == "local-node"
    assert result.nodes[0].ok
    peer = next(n for n in result.nodes if n.node_id == "peer-1")
    assert not peer.ok
    assert peer.error


async def test_concurrency_incremented_eagerly_at_dispatch() -> None:
    api = _build_api()
    object.__setattr__(
        api, "_resolve_envelope_context", lambda _model: ("hw", "b", "")
    )
    # Eager increment: a request admitted during the window before an earlier
    # response iterates must still observe it, or concurrent bursts under-count.
    gen1 = api._tap_performance_envelope(
        ModelId("m"), _one_stream([_chunk("", "stop", 10.0)])
    )
    assert api._envelope_inflight["m"] == 1
    gen2 = api._tap_performance_envelope(
        ModelId("m"), _one_stream([_chunk("", "stop", 10.0)])
    )
    assert api._envelope_inflight["m"] == 2

    await _drain(gen1)
    await _drain(gen2)
    assert api._envelope_inflight.get("m", 0) == 0
    concurrencies = {
        b.concurrency
        for b in api._performance_envelopes.snapshot().envelopes[0].buckets
    }
    assert 2 in concurrencies


async def test_never_started_generation_does_not_leak_inflight() -> None:
    import gc

    api = _build_api()
    object.__setattr__(
        api, "_resolve_envelope_context", lambda _model: ("hw", "b", "")
    )
    gen = api._tap_performance_envelope(
        ModelId("m"), _one_stream([_chunk("", "stop", 10.0)])
    )
    # Eager increment took effect at dispatch.
    assert api._envelope_inflight["m"] == 1
    # Discard the response without ever iterating (client disconnect before the
    # first payload): the weakref finalizer must decrement on GC, so no leak and
    # no bogus record.
    del gen
    gc.collect()
    assert api._envelope_inflight.get("m", 0) == 0
    assert api._performance_envelopes.snapshot().envelopes == []


async def _one_stream(chunks: list[Any]) -> AsyncGenerator[Any, None]:
    for chunk in chunks:
        yield chunk
