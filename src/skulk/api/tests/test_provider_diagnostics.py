"""Tests for bounded provider lifecycle and media diagnostics."""

from skulk.api.provider_diagnostics import ProviderObserver
from skulk.extensions import CapabilityStreamFrame, InlineMediaAttachment


def _frame(
    *,
    call_id: str,
    sequence: int,
    kind: str,
    media: bytes | None = None,
    payload: dict[str, object] | None = None,
) -> CapabilityStreamFrame:
    """Build one provider output frame for observer tests."""

    return CapabilityStreamFrame.model_validate(
        {
            "call_id": call_id,
            "direction": "provider_to_caller",
            "sequence": sequence,
            "kind": kind,
            "payload": payload,
            "media": (
                InlineMediaAttachment(
                    data=media,
                    media_type="audio/mpeg",
                    codec="mp3",
                )
                if media is not None
                else None
            ),
        }
    )


def test_provider_observer_records_speech_media_latency_and_terminal() -> None:
    """Speech streams expose pressure, byte volume, latency, and completion."""

    observer = ProviderObserver()
    observer.record_admitted("call-1", "tts@1.0.0", observed_at=10.0)
    observer.record_output(
        "call-1", _frame(call_id="call-1", sequence=0, kind="started"), observed_at=10.1
    )
    observer.record_output(
        "call-1",
        _frame(call_id="call-1", sequence=1, kind="chunk", media=b"audio"),
        observed_at=10.4,
    )
    observer.record_output(
        "call-1",
        _frame(call_id="call-1", sequence=2, kind="completed"),
        observed_at=10.8,
    )

    diagnostics = observer.snapshot(
        active_unary_calls=2,
        stream_slots_in_use=0,
        unary_concurrency_limit=32,
        stream_concurrency_limit=16,
    )
    tts = diagnostics.capabilities["tts@1.0.0"]
    assert diagnostics.active_unary_calls == 2
    assert diagnostics.active_streams == 0
    assert diagnostics.max_active_streams == 1
    assert tts.admitted_streams == 1
    assert tts.output_frames == 3
    assert tts.output_media_bytes == 5
    assert tts.completed_streams == 1
    assert tts.first_output_samples == 1
    assert tts.first_output_seconds_average is not None
    assert abs(tts.first_output_seconds_average - 0.4) < 1e-9
    assert tts.duration_seconds_average is not None
    assert abs(tts.duration_seconds_average - 0.8) < 1e-9


def test_provider_observer_records_input_rejection_cancellation_and_missing_terminal() -> None:
    """STT input and abnormal lifecycle outcomes remain distinguishable."""

    observer = ProviderObserver()
    observer.record_rejected("stt.realtime@1.0.0", overloaded=True)
    observer.record_admitted("call-2", "stt.realtime@1.0.0", observed_at=20.0)
    input_frame = CapabilityStreamFrame(
        call_id="call-2",
        direction="caller_to_provider",
        sequence=1,
        kind="chunk",
        payload={"format": "pcm_s16le"},
        media=InlineMediaAttachment(
            data=b"\x00\x01\x02\x03",
            media_type="audio/pcm",
            codec="pcm_s16le",
            sample_rate=16000,
            channels=1,
        ),
    )
    observer.record_input("call-2", input_frame, queue_depth=3)
    observer.record_input_queue_depth("call-2", 1)
    observer.record_cancellation_request("call-2")
    observer.finalize("call-2", observed_at=21.5)

    diagnostics = observer.snapshot(
        active_unary_calls=0,
        stream_slots_in_use=0,
        unary_concurrency_limit=32,
        stream_concurrency_limit=16,
    )
    stt = diagnostics.capabilities["stt.realtime@1.0.0"]
    assert stt.rejected_streams == 1
    assert stt.overloaded_rejections == 1
    assert stt.input_frames == 1
    assert stt.input_media_bytes == 4
    assert stt.input_queue_depth == 0
    assert stt.max_input_queue_depth == 3
    assert stt.cancellation_requests == 1
    assert stt.missing_terminal_streams == 1
    assert stt.duration_seconds_average == 1.5


def test_provider_input_queue_depth_sums_concurrent_calls_and_clears() -> None:
    """Per-capability pressure is the sum of live call queues, not last-writer state."""

    observer = ProviderObserver()
    observer.record_admitted("call-a", "stt.realtime@1.0.0", observed_at=1.0)
    observer.record_admitted("call-b", "stt.realtime@1.0.0", observed_at=1.0)
    observer.record_input_queue_depth("call-a", 2)
    observer.record_input_queue_depth("call-b", 3)

    active = observer.snapshot(
        active_unary_calls=0,
        stream_slots_in_use=2,
        unary_concurrency_limit=8,
        stream_concurrency_limit=8,
    )
    assert active.input_queue_depth == 5
    assert active.max_input_queue_depth == 5
    assert active.capabilities["stt.realtime@1.0.0"].input_queue_depth == 5

    observer.finalize("call-a", observed_at=2.0)
    after_finalize = observer.snapshot(
        active_unary_calls=0,
        stream_slots_in_use=1,
        unary_concurrency_limit=8,
        stream_concurrency_limit=8,
    )
    assert after_finalize.input_queue_depth == 3
