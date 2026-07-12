# pyright: reportPrivateUsage=false
"""Deterministic coverage for the built-in VAD provider."""

from collections.abc import AsyncIterator, Callable, Iterator

import pytest

from skulk.extensions import (
    VAD_CAPABILITY_DESCRIPTOR,
    BuiltinVadProvider,
    CapabilityCall,
    CapabilityStreamFrame,
    InlineMediaAttachment,
    descriptor_revision,
)
from skulk.extensions.vad import VadConfig, VoiceActivityDetector


def _classifier(decisions: list[bool]) -> Callable[[bytes, int], bool]:
    iterator: Iterator[bool] = iter(decisions)

    def classify(frame: bytes, sample_rate: int) -> bool:
        assert frame
        assert sample_rate == 16000
        return next(iterator)

    return classify


def _frame(config: VadConfig) -> bytes:
    return bytes(config.frame_bytes)


def test_vad_ignores_silence_and_short_speech_bursts() -> None:
    config = VadConfig(sample_rate=16000, minimum_speech_ms=60, frame_ms=20)
    detector = VoiceActivityDetector(
        config,
        classify=_classifier([False, True, True, False]),
    )

    assert [detector.process(_frame(config)) for _ in range(4)] == [(), (), (), ()]
    assert detector.finish() == ()


def test_vad_emits_prerolled_start_and_hangover_stop() -> None:
    config = VadConfig(
        sample_rate=16000,
        frame_ms=20,
        minimum_speech_ms=40,
        silence_hangover_ms=60,
        preroll_ms=40,
    )
    detector = VoiceActivityDetector(
        config,
        classify=_classifier([False, False, True, True, False, False, False]),
    )

    events = tuple(
        event
        for _ in range(7)
        for event in detector.process(_frame(config))
    )

    assert [event.kind for event in events] == ["speech_started", "speech_stopped"]
    assert events[0].timestamp_ms == 0
    assert events[0].preroll_ms == 40
    assert events[1].timestamp_ms == 80
    assert events[1].reason == "silence"


def test_vad_stops_at_maximum_utterance_and_can_start_again() -> None:
    config = VadConfig(
        sample_rate=16000,
        frame_ms=20,
        minimum_speech_ms=20,
        maximum_utterance_ms=100,
        preroll_ms=0,
    )
    detector = VoiceActivityDetector(
        config,
        classify=_classifier([True, True, True, True, True, True]),
    )

    events = tuple(
        event
        for _ in range(6)
        for event in detector.process(_frame(config))
    )

    assert [event.kind for event in events] == [
        "speech_started",
        "speech_stopped",
        "speech_started",
    ]
    assert events[1].reason == "maximum_duration"


def test_vad_finish_closes_active_turn() -> None:
    config = VadConfig(
        sample_rate=16000,
        frame_ms=20,
        minimum_speech_ms=20,
    )
    detector = VoiceActivityDetector(
        config,
        classify=_classifier([True]),
    )

    assert detector.process(_frame(config))[0].kind == "speech_started"
    stopped = detector.finish()
    assert stopped[0].kind == "speech_stopped"
    assert stopped[0].reason == "input_completed"


def test_vad_rejects_unsupported_rates_and_partial_frames() -> None:
    def silence(frame: bytes, sample_rate: int) -> bool:
        del frame, sample_rate
        return False

    with pytest.raises(ValueError, match="8, 16, 32, or 48"):
        VoiceActivityDetector(VadConfig(sample_rate=24000), classify=silence)

    config = VadConfig(sample_rate=16000)
    detector = VoiceActivityDetector(config, classify=silence)
    with pytest.raises(ValueError, match="exactly"):
        detector.process(b"too short")


@pytest.mark.anyio
async def test_vad_provider_emits_typed_turn_lifecycle() -> None:
    config = VadConfig(
        sample_rate=16000,
        frame_ms=20,
        minimum_speech_ms=40,
        silence_hangover_ms=40,
        preroll_ms=20,
    )
    provider = BuiltinVadProvider(
        detector_factory=lambda value: VoiceActivityDetector(
            value,
            classify=_classifier([False, True, True, False, False]),
        )
    )
    call = CapabilityCall(
        call_id="vad-call",
        capability_id="vad",
        version="1.0.0",
        descriptor_revision=descriptor_revision(VAD_CAPABILITY_DESCRIPTOR),
        caller_node="caller",
        target_node="provider",
        payload=config.model_dump(),
    )
    pcm = bytes(config.frame_bytes * 5)

    async def input_frames() -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="caller_to_provider",
            sequence=1,
            kind="chunk",
            payload={
                "format": "pcm_s16le",
                "sample_rate": config.sample_rate,
                "channels": 1,
            },
            media=InlineMediaAttachment(
                data=pcm,
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=config.sample_rate,
                channels=1,
            ),
        )

    frames = [frame async for frame in provider._stream(call, input_frames())]

    assert [frame.kind for frame in frames] == ["chunk", "chunk", "completed"]
    assert [frame.payload.get("event") for frame in frames[:-1] if frame.payload] == [
        "speech_started",
        "speech_stopped",
    ]
    assert frames[-1].payload == {"turns": 1}


@pytest.mark.anyio
async def test_vad_provider_rejects_partial_terminal_pcm() -> None:
    provider = BuiltinVadProvider()
    call = CapabilityCall(
        call_id="partial-vad-call",
        capability_id="vad",
        version="1.0.0",
        descriptor_revision=descriptor_revision(VAD_CAPABILITY_DESCRIPTOR),
        caller_node="caller",
        target_node="provider",
        payload={"sample_rate": 16000},
    )

    async def input_frames() -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="caller_to_provider",
            sequence=1,
            kind="chunk",
            payload={"format": "pcm_s16le", "sample_rate": 16000, "channels": 1},
            media=InlineMediaAttachment(
                data=b"\x00\x00",
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=16000,
                channels=1,
            ),
        )

    frames = [frame async for frame in provider._stream(call, input_frames())]

    assert len(frames) == 1
    assert frames[0].kind == "failed"
    assert frames[0].error is not None
    assert frames[0].error.code == "invalid_frame"
    assert "partial PCM frame" in frames[0].error.message
