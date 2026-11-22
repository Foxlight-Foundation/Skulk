"""Control protocol framing, strict validation and version negotiation."""

import io
import json
import struct

import pytest

from skulk.engines.protocol import (
    MAX_CONTROL_FRAME_BYTES,
    ControlProtocolError,
    DescribeResponse,
    HelloRequest,
    LoadSettings,
    ModelIdentity,
    decode_request,
    decode_response,
    encode_frame,
    negotiate_version,
    read_frame_body,
    write_frame,
)

DIGEST = "a" * 64
SETTINGS = LoadSettings(
    context_tokens=32768, parallel_streams=4, cache_precision="f16", drafting=False
)
MODEL = ModelIdentity(model_id="Qwen/Example", source_revision="b" * 40)


def _body(frame: bytes) -> bytes:
    assert int.from_bytes(frame[:4], "big") == len(frame) - 4
    return frame[4:]


def test_request_round_trips_through_a_frame() -> None:
    """A request encodes to one length-prefixed frame and decodes to an equal message."""
    request = HelloRequest(sequence=1, build_digest=DIGEST, versions=(1, 2))
    stream = io.BytesIO()
    write_frame(stream, request)
    stream.seek(0)
    body = read_frame_body(stream)
    assert body is not None
    assert decode_request(body) == request
    assert read_frame_body(stream) is None


def test_response_round_trips_with_nested_settings() -> None:
    """Nested settings survive encoding and validate strictly on the way back."""
    describe = DescribeResponse(
        sequence=7,
        build_digest=DIGEST,
        engine_version="1.3.2",
        model_families=("qwen", "gemma"),
    )
    assert decode_response(_body(encode_frame(describe))) == describe


def test_validated_messages_cannot_exceed_the_frame_limit() -> None:
    """Every field is bounded, so no validated message reaches the frame limit."""
    with pytest.raises(ValueError):
        DescribeResponse(
            sequence=1,
            build_digest=DIGEST,
            engine_version="1",
            model_families=("x" * 1025,),
        )


def test_encoding_refuses_a_frame_over_the_limit() -> None:
    """The encoder refuses an oversize body even if validation was bypassed."""
    oversized = DescribeResponse.model_construct(
        sequence=1,
        build_digest=DIGEST,
        type="describe",
        engine_version="1",
        model_families=tuple("x" * 1000 for _ in range(80)),
    )
    with pytest.raises(ControlProtocolError) as refused:
        encode_frame(oversized)
    assert refused.value.code == "frame_too_large"


def test_reader_refuses_an_announced_oversize_before_reading_the_body() -> None:
    """A hostile length prefix is refused without consuming the body."""
    stream = io.BytesIO(struct.pack(">I", MAX_CONTROL_FRAME_BYTES + 1) + b"x" * 16)
    with pytest.raises(ControlProtocolError) as refused:
        read_frame_body(stream)
    assert refused.value.code == "frame_too_large"
    assert stream.tell() == 4


def test_reader_reports_a_frame_cut_short() -> None:
    """End of stream inside a frame is a truncation, not a clean end."""
    stream = io.BytesIO(struct.pack(">I", 10) + b"abc")
    with pytest.raises(ControlProtocolError) as refused:
        read_frame_body(stream)
    assert refused.value.code == "truncated"


def test_reader_treats_an_empty_stream_as_a_clean_end() -> None:
    """No bytes at a frame boundary means the peer finished normally."""
    assert read_frame_body(io.BytesIO(b"")) is None


def _with(message: dict[str, object]) -> bytes:
    return json.dumps(message).encode()


_BASE: dict[str, object] = {"type": "describe", "sequence": 1, "build_digest": DIGEST}


def test_unknown_field_is_refused() -> None:
    """An extra field means a newer or different contract: refuse, do not guess."""
    with pytest.raises(ControlProtocolError) as refused:
        decode_request(_with({**_BASE, "surprise": True}))
    assert refused.value.code == "invalid_frame"


def test_unknown_message_type_is_refused() -> None:
    """A type this build does not know is refused rather than ignored."""
    with pytest.raises(ControlProtocolError) as refused:
        decode_request(_with({**_BASE, "type": "teleport"}))
    assert refused.value.code == "invalid_frame"


def test_types_are_strict() -> None:
    """A sequence sent as a string is not coerced into a number."""
    with pytest.raises(ControlProtocolError):
        decode_request(_with({**_BASE, "sequence": "1"}))


def test_digest_must_be_sha256_hex() -> None:
    """The build digest is a 64-character lowercase hex string."""
    with pytest.raises(ControlProtocolError):
        decode_request(_with({**_BASE, "build_digest": "A" * 64}))


def test_settings_bounds_are_enforced() -> None:
    """Zero context or streams cannot be admitted by any message."""
    with pytest.raises(ValueError):
        LoadSettings(
            context_tokens=0, parallel_streams=1, cache_precision="f16", drafting=False
        )
    with pytest.raises(ValueError):
        ModelIdentity(model_id="x", source_revision="main")


def test_negotiation_picks_the_highest_common_version() -> None:
    """Both sides' windows intersect at their highest shared version."""
    assert negotiate_version((1, 3, 5), supported=(1, 2, 3)) == 3
    assert negotiate_version((1,)) == 1


def test_disjoint_windows_are_refused() -> None:
    """No shared version means the package does not install on this build."""
    with pytest.raises(ControlProtocolError) as refused:
        negotiate_version((2,), supported=(1,))
    assert refused.value.code == "version_unsupported"
