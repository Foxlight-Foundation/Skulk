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
    assert api._envelope_inflight["m"] == 0


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
    assert api._envelope_inflight["m"] == 0


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


async def _one_stream(chunks: list[Any]) -> AsyncGenerator[Any, None]:
    for chunk in chunks:
        yield chunk
