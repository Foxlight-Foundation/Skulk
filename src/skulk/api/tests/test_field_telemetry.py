"""Unit tests for the opt-in field-telemetry collector. Fully offline."""

from __future__ import annotations

from typing import cast, final

import pytest

from skulk.api.field_telemetry import (
    TELEMETRY_KILL_SWITCH,
    FieldTelemetryCollector,
    TelemetrySample,
    hardware_class,
    prepare_telemetry_config_update,
    tap_generation_stream,
)
from skulk.shared.types.profiling import AcceleratorMetrics, SystemPerformanceProfile
from skulk.store.config import TelemetryConfig

_INSTALL_ID = "0f5b0a1e-6f5c-4b2a-9d3e-1a2b3c4d5e6f"


def _consented() -> TelemetryConfig:
    return TelemetryConfig(
        consent="enabled",
        install_id=_INSTALL_ID,
        ingest_url="https://ingest.example",
    )


def _profile(vendor: str, name: str) -> SystemPerformanceProfile:
    return SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(vendor=vendor, name=name)  # type: ignore[arg-type]
    )


@final
class _PostSpy:
    """Injectable HTTP effect capturing payloads; scriptable status/raise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.status = 201
        self.raise_error = False

    async def __call__(self, url: str, payload: dict[str, object]) -> int:
        if self.raise_error:
            raise RuntimeError("network down")
        self.calls.append((url, payload))
        return self.status


def _collector(
    config: TelemetryConfig | None,
    hardware: dict[str, tuple[SystemPerformanceProfile | None, int | None]],
    post: _PostSpy | None = None,
) -> tuple[FieldTelemetryCollector, _PostSpy]:
    spy = post if post is not None else _PostSpy()
    collector = FieldTelemetryCollector(
        config_provider=lambda: config,
        hardware_provider=lambda: dict(hardware),
        post=spy,
    )
    return collector, spy


def test_hardware_class_shapes() -> None:
    assert hardware_class("apple", "M4", 24 * 2**30) == "apple-m4-24gb"
    assert hardware_class("amd", "Radeon 8060S", 61 * 2**30) == "amd-radeon-8060s-64gb"
    assert hardware_class("amd", None, 30 * 2**30) == "amd-32gb"
    assert hardware_class(None, None, None) == "unknown"


def test_nothing_collected_without_consent() -> None:
    for config in (
        None,
        TelemetryConfig(),  # unasked
        TelemetryConfig(consent="disabled", install_id=_INSTALL_ID),
        TelemetryConfig(consent="enabled"),  # no install id
    ):
        collector, _ = _collector(config, {})
        collector.record_generation(
            "org/model",
            ttft_s=0.5,
            decode_tps=50.0,
            prompt_tokens=10,
            output_tokens=20,
            node_count=1,
            error_class=None,
        )
        assert collector.preview()["pending"] == []


def test_kill_switch_overrides_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TELEMETRY_KILL_SWITCH, "1")
    collector, _ = _collector(_consented(), {})
    assert collector.enabled is False


def test_generation_sample_shape_matches_wire_contract() -> None:
    hardware: dict[str, tuple[SystemPerformanceProfile | None, int | None]] = {
        "node-a": (_profile("apple", "M4"), 16 * 2**30),
        "node-b": (_profile("amd", "Radeon 8060S"), 61 * 2**30),
    }
    collector, _ = _collector(_consented(), hardware)
    collector.record_generation(
        "org/model",
        ttft_s=0.42,
        decode_tps=51.3,
        prompt_tokens=512,
        output_tokens=200,
        node_count=2,
        error_class=None,
    )
    pending: list[dict[str, object]] = collector.preview()["pending"]
    assert isinstance(pending, list) and len(pending) == 1
    sample = pending[0]
    assert isinstance(sample, dict)
    assert sample["kind"] == "generation"
    assert sample["model_id"] == "org/model"
    assert sample["hardware"] == ["amd-radeon-8060s-64gb", "apple-m4-16gb"]
    assert sample["decode_tps"] == 51.3
    # Content-free: no field may carry text or identity.
    assert set(sample) <= {
        "kind",
        "at",
        "model_id",
        "engine",
        "quantization",
        "hardware",
        "node_count",
        "ttft_s",
        "decode_tps",
        "prompt_tokens",
        "output_tokens",
        "mtp_accept_ratio",
        "error_class",
    }


async def test_flush_posts_batch_and_clears() -> None:
    collector, spy = _collector(_consented(), {})
    collector.record_generation(
        "org/model",
        ttft_s=0.1,
        decode_tps=40.0,
        prompt_tokens=1,
        output_tokens=2,
        node_count=1,
        error_class=None,
    )
    assert await collector.flush_once() is True
    url, payload = spy.calls[0]
    assert url == "https://ingest.example/v1/telemetry"
    assert payload["install_id"] == _INSTALL_ID
    samples = cast("list[dict[str, object]]", payload["samples"])
    assert len(samples) == 1
    assert collector.preview()["pending"] == []


async def test_flush_failure_retains_batch() -> None:
    collector, spy = _collector(_consented(), {})
    spy.raise_error = True
    collector.record_generation(
        "org/model",
        ttft_s=None,
        decode_tps=None,
        prompt_tokens=None,
        output_tokens=None,
        node_count=None,
        error_class="generation-error",
    )
    assert await collector.flush_once() is False
    pending: list[dict[str, object]] = collector.preview()["pending"]
    assert isinstance(pending, list) and len(pending) == 1
    # Rejected status keeps the batch too.
    spy.raise_error = False
    spy.status = 429
    assert await collector.flush_once() is False
    pending = collector.preview()["pending"]
    assert isinstance(pending, list) and len(pending) == 1


def test_bounded_queue_counts_drops() -> None:
    collector, _ = _collector(_consented(), {})
    for _ in range(1100):
        collector.record_generation(
            "org/model",
            ttft_s=None,
            decode_tps=None,
            prompt_tokens=None,
            output_tokens=None,
            node_count=None,
            error_class=None,
        )
    preview = collector.preview()
    pending: list[dict[str, object]] = preview["pending"]
    assert isinstance(pending, list) and len(pending) == 1000
    assert preview["dropped_since_start"] == 100


def test_peer_observed_node_death() -> None:
    hardware: dict[str, tuple[SystemPerformanceProfile | None, int | None]] = {
        "node-a": (_profile("apple", "M4"), 16 * 2**30),
        "node-b": (_profile("amd", "Radeon 8060S"), 61 * 2**30),
    }
    collector, _ = _collector(_consented(), hardware)
    collector.observe_node_set()  # baseline
    del hardware["node-b"]
    collector.observe_node_set()
    pending: list[dict[str, object]] = collector.preview()["pending"]
    assert isinstance(pending, list)
    deaths = [s for s in pending if s["kind"] == "node-death"]
    assert len(deaths) == 1
    assert deaths[0]["hardware"] == ["amd-radeon-8060s-64gb"]


def test_node_death_baseline_updates_even_without_consent() -> None:
    hardware: dict[str, tuple[SystemPerformanceProfile | None, int | None]] = {
        "node-a": (_profile("apple", "M4"), 16 * 2**30),
    }
    config_holder: list[TelemetryConfig | None] = [None]
    collector = FieldTelemetryCollector(
        config_provider=lambda: config_holder[0],
        hardware_provider=lambda: dict(hardware),
    )
    collector.observe_node_set()  # baseline while unconsented
    del hardware["node-a"]
    collector.observe_node_set()  # death while unconsented: not recorded
    config_holder[0] = _consented()
    collector.observe_node_set()  # enabling later must not emit stale deaths
    assert collector.preview()["pending"] == []


@final
class _Chunk:
    def __init__(
        self,
        text: str = "",
        finish_reason: str | None = None,
        stats: object | None = None,
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.stats = stats


@final
class _Stats:
    def __init__(self, tps: float, prompt: int, generated: int) -> None:
        self.generation_tps = tps
        self.prompt_tokens = prompt
        self.generation_tokens = generated


async def test_tap_records_one_sample_per_stream() -> None:
    collector, _ = _collector(_consented(), {})

    async def stream():  # noqa: ANN202
        yield _Chunk(text="hel")
        yield _Chunk(text="lo", finish_reason="stop", stats=_Stats(45.0, 100, 30))

    chunks = [c async for c in tap_generation_stream(collector, "org/model", None, stream())]
    assert len(chunks) == 2
    pending: list[dict[str, object]] = collector.preview()["pending"]
    assert isinstance(pending, list) and len(pending) == 1
    sample = pending[0]
    assert isinstance(sample, dict)
    assert sample["decode_tps"] == 45.0
    assert sample["prompt_tokens"] == 100
    assert sample["output_tokens"] == 30
    assert isinstance(sample["ttft_s"], float)  # first chunk carried text
    assert "error_class" not in sample  # exclude_none drops the clean case


async def test_tap_records_error_class_on_error_stream() -> None:
    collector, _ = _collector(_consented(), {})

    async def stream():  # noqa: ANN202
        yield _Chunk(text="par")
        yield _Chunk(finish_reason="error")

    _ = [c async for c in tap_generation_stream(collector, "org/model", None, stream())]
    pending: list[dict[str, object]] = collector.preview()["pending"]
    assert isinstance(pending, list)
    sample = pending[0]
    assert isinstance(sample, dict)
    assert sample["error_class"] == "generation-error"


async def test_tap_skips_aborted_streams() -> None:
    collector, _ = _collector(_consented(), {})

    async def stream():  # noqa: ANN202
        yield _Chunk(text="par")
        yield _Chunk(text="tial")  # no terminal finish_reason ever arrives

    agen = tap_generation_stream(collector, "org/model", None, stream())
    _ = await agen.__anext__()
    await agen.aclose()  # client disconnect
    assert collector.preview()["pending"] == []


def test_sample_model_is_frozen() -> None:
    sample = TelemetrySample(kind="generation", at="2026-07-11T00:00:00Z")
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen raises ValidationError
        sample.kind = "other"  # pyright: ignore[reportAttributeAccessIssue]


def test_config_update_preserves_omitted_telemetry() -> None:
    config: dict[str, object] = {"logging": {"enabled": False}}
    existing: dict[str, object] = {"telemetry": {"consent": "enabled", "install_id": "x"}}
    prepare_telemetry_config_update(config, existing)
    section = config["telemetry"]
    assert isinstance(section, dict)
    # Preserved AND normalized: the carried-forward decided consent gains a
    # version stamp exactly as if it had been sent explicitly.
    assert section["consent"] == "enabled"
    assert section["install_id"] == "x"
    assert section["consented_version"]


def test_config_update_never_stamps_version_while_unasked() -> None:
    config: dict[str, object] = {
        "telemetry": {"consent": "unasked", "diagnostics_consent": "unasked"}
    }
    prepare_telemetry_config_update(config, None)
    section = config["telemetry"]
    assert isinstance(section, dict)
    assert "consented_version" not in section
    assert "install_id" not in section


def test_config_update_stamps_version_and_backfills_id_once_decided() -> None:
    config: dict[str, object] = {
        "telemetry": {"consent": "enabled", "diagnostics_consent": "disabled", "install_id": ""}
    }
    prepare_telemetry_config_update(config, None)
    section = config["telemetry"]
    assert isinstance(section, dict)
    assert section["consented_version"]  # stamped from the running Skulk
    assert section["install_id"]  # backfilled UUID

    # Disabled-only decisions stamp the version but never mint an id.
    config2: dict[str, object] = {
        "telemetry": {"consent": "disabled", "diagnostics_consent": "disabled"}
    }
    prepare_telemetry_config_update(config2, None)
    section2 = cast("dict[str, object]", config2["telemetry"])
    assert section2["consented_version"]
    assert not section2.get("install_id")
