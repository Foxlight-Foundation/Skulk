"""Wire and validation coverage for ephemeral speech media."""

import hashlib

import pytest
from pydantic import ValidationError

from skulk.routing.speech_media import SpeechMediaPacket
from skulk.routing.topics import SPEECH_MEDIA
from skulk.shared.types.common import CommandId, NodeId


def _chunk() -> SpeechMediaPacket:
    return SpeechMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=CommandId("speech-command"),
        sequence=0,
        kind="chunk",
        filename="voice.wav",
        content_type="audio/wav",
        data=b"RIFF-reference",
    )


def test_speech_media_topic_round_trips_binary_without_base64() -> None:
    packet = _chunk()

    restored = SPEECH_MEDIA.deserialize(SPEECH_MEDIA.serialize(packet))

    assert restored == packet
    assert SPEECH_MEDIA.routing_key is not None
    assert SPEECH_MEDIA.routing_key(restored) == "worker-node"
    assert SPEECH_MEDIA.stream_key is not None
    assert SPEECH_MEDIA.stream_key(restored) == "speech-command"


def test_speech_media_completion_requires_digest() -> None:
    with pytest.raises(ValidationError, match="requires sha256"):
        SpeechMediaPacket(
            source_node=NodeId("api-node"),
            target_node=NodeId("worker-node"),
            command_id=CommandId("speech-command"),
            sequence=1,
            kind="completed",
        )


def test_speech_media_terminal_carries_digest_without_payload() -> None:
    digest = hashlib.sha256(b"RIFF-reference").hexdigest()
    packet = SpeechMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=CommandId("speech-command"),
        sequence=1,
        kind="completed",
        sha256=digest,
    )

    restored = SPEECH_MEDIA.deserialize(SPEECH_MEDIA.serialize(packet))

    assert restored.sha256 == digest
    assert restored.is_terminal is True


def test_speech_media_transport_failure_reverses_route() -> None:
    rejection = _chunk().transport_failure("queue full")

    assert rejection.source_node == NodeId("worker-node")
    assert rejection.target_node == NodeId("api-node")
    assert rejection.kind == "transport_failed"
    assert rejection.error_message == "queue full"
