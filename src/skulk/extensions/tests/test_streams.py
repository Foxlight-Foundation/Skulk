"""Provider-stream contract, framing, and receive-state coverage."""

import anyio
import pytest

from skulk.extensions.streams import (
    MAX_INLINE_MEDIA_BYTES,
    BlobMediaAttachment,
    CapabilityStreamError,
    CapabilityStreamFrame,
    CapabilityStreamInput,
    CapabilityStreamReceiver,
    InlineMediaAttachment,
    decode_capability_stream_frame,
    encode_capability_stream_frame,
)


@pytest.mark.asyncio
async def test_input_sink_owns_sequence_and_half_close() -> None:
    sent: list[CapabilityStreamFrame] = []

    async def send(frame: CapabilityStreamFrame) -> None:
        sent.append(frame)

    stream = CapabilityStreamInput(
        call_id="stt-call",
        deadline_at=anyio.current_time() + 1.0,
        send_frame=send,
    )
    await stream.start()
    await stream.send_chunk(
        payload={"duration_ms": 20},
        media=InlineMediaAttachment(
            data=b"\x00\x01",
            media_type="audio/pcm",
            codec="pcm_s16le",
            sample_rate=16000,
            channels=1,
        ),
    )
    await stream.complete()

    assert stream.closed is True
    assert [frame.direction for frame in sent] == ["caller_to_provider"] * 3
    assert [frame.sequence for frame in sent] == [0, 1, 2]
    assert [frame.kind for frame in sent] == ["started", "chunk", "completed"]
    with pytest.raises(RuntimeError, match="closed"):
        await stream.send_chunk(payload={"late": True})


def _started(call_id: str = "call-1") -> CapabilityStreamFrame:
    return CapabilityStreamFrame(
        call_id=call_id,
        direction="provider_to_caller",
        sequence=0,
        kind="started",
    )


def _chunk(sequence: int, call_id: str = "call-1") -> CapabilityStreamFrame:
    return CapabilityStreamFrame(
        call_id=call_id,
        direction="provider_to_caller",
        sequence=sequence,
        kind="chunk",
        payload={"text": f"part-{sequence}"},
    )


def test_inline_binary_media_round_trips_outside_json_header() -> None:
    media_bytes = bytes(range(256))
    frame = CapabilityStreamFrame(
        call_id="tts-call",
        direction="provider_to_caller",
        sequence=1,
        kind="chunk",
        payload={"codec": "pcm_s16le"},
        media=InlineMediaAttachment(
            data=media_bytes,
            media_type="audio/pcm",
            codec="pcm_s16le",
            sample_rate=24000,
            channels=1,
            duration_seconds=0.02,
        ),
    )

    header, attachment = encode_capability_stream_frame(frame)
    restored = decode_capability_stream_frame(header, attachment)

    assert attachment == media_bytes
    assert media_bytes not in header
    assert restored == frame


def test_blob_media_reference_requires_integrity_metadata() -> None:
    blob = BlobMediaAttachment(
        blob_id="staged:audio/result.wav",
        size_bytes=4096,
        sha256="a" * 64,
        media_type="audio/wav",
    )
    frame = CapabilityStreamFrame(
        call_id="tts-blob-call",
        direction="provider_to_caller",
        sequence=1,
        kind="completed",
        media=blob,
    )
    header, attachment = encode_capability_stream_frame(frame)
    assert attachment is None
    assert decode_capability_stream_frame(header, None) == frame

    with pytest.raises(ValueError):
        BlobMediaAttachment(
            blob_id="bad",
            size_bytes=1,
            sha256="not-a-digest",
            media_type="audio/wav",
        )


def test_frame_validation_enforces_lifecycle_and_binary_boundary() -> None:
    with pytest.raises(ValueError, match="sequence 0 must be started"):
        _chunk(0)
    with pytest.raises(ValueError, match="requires payload or media"):
        CapabilityStreamFrame(
            call_id="call-1",
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
        )
    with pytest.raises(ValueError, match="JSON-serializable"):
        CapabilityStreamFrame(
            call_id="call-1",
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={"audio": b"binary-belongs-in-media"},
        )
    with pytest.raises(ValueError, match="at most 1048576 bytes"):
        InlineMediaAttachment(
            data=b"x" * (MAX_INLINE_MEDIA_BYTES + 1),
            media_type="audio/pcm",
        )


def test_receiver_repairs_reordering_and_enforces_one_terminal() -> None:
    receiver = CapabilityStreamReceiver(
        call_id="call-1",
        direction="provider_to_caller",
    )
    assert receiver.accept(_started(), observed_at=1.0).ready == (_started(),)

    buffered = receiver.accept(_chunk(2), observed_at=2.0)
    assert buffered.ready == ()
    assert buffered.out_of_order_frames == 1

    released = receiver.accept(_chunk(1), observed_at=2.1)
    assert [frame.sequence for frame in released.ready] == [1, 2]

    terminal = CapabilityStreamFrame(
        call_id="call-1",
        direction="provider_to_caller",
        sequence=3,
        kind="completed",
    )
    assert receiver.accept(terminal, observed_at=3.0).ready == (terminal,)
    assert receiver.terminal == terminal
    assert receiver.accept(terminal, observed_at=3.1).duplicate_frames == 1
    assert receiver.accept(_chunk(4), observed_at=3.2).late_frames == 1


def test_receiver_synthesizes_transport_terminal_for_unresolved_gap() -> None:
    receiver = CapabilityStreamReceiver(
        call_id="call-1",
        direction="provider_to_caller",
        gap_timeout_seconds=5.0,
    )
    receiver.accept(_started(), observed_at=1.0)
    receiver.accept(_chunk(2), observed_at=2.0)

    outcome = receiver.expire(observed_at=7.0)
    terminal = outcome.synthesized_terminal
    assert terminal is not None
    assert terminal.kind == "failed"
    assert terminal.synthetic is True
    assert terminal.error is not None
    assert terminal.error.code == "transport_error"
    assert receiver.terminal == terminal


def test_later_frames_do_not_refresh_an_existing_gap_deadline() -> None:
    receiver = CapabilityStreamReceiver(
        call_id="call-1",
        direction="provider_to_caller",
        gap_timeout_seconds=5.0,
    )
    receiver.accept(_started(), observed_at=1.0)
    receiver.accept(_chunk(2), observed_at=2.0)
    receiver.accept(_chunk(3), observed_at=6.0)

    terminal = receiver.expire(observed_at=7.0).synthesized_terminal
    assert terminal is not None
    assert terminal.error is not None
    assert terminal.error.code == "transport_error"


def test_receiver_idle_timeout_and_cancellation_are_typed_and_idempotent() -> None:
    receiver = CapabilityStreamReceiver(
        call_id="call-timeout",
        direction="provider_to_caller",
        idle_timeout_seconds=10.0,
    )
    receiver.accept(_started("call-timeout"), observed_at=5.0)
    timeout = receiver.expire(observed_at=15.0).synthesized_terminal
    assert timeout is not None and timeout.error is not None
    assert timeout.error.code == "timeout"

    cancelled_receiver = CapabilityStreamReceiver(
        call_id="call-cancel",
        direction="provider_to_caller",
    )
    cancelled = cancelled_receiver.cancel("caller disconnected")
    assert cancelled is not None
    assert cancelled.kind == "cancelled"
    assert cancelled.error == CapabilityStreamError(
        code="cancelled", message="caller disconnected"
    )
    assert cancelled_receiver.cancel() is None
