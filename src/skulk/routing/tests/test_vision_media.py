import hashlib

import pytest
from pydantic import ValidationError

from skulk.routing.topics import VISION_MEDIA
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import CommandId, NodeId


def _chunk(data: bytes = b"aGVsbG8=") -> VisionMediaPacket:
    return VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=CommandId("vision-command"),
        model=ModelId("org/vlm"),
        sequence=1,
        kind="chunk",
        data=data,
        image_index=0,
        total_chunks=1,
    )


def test_vision_media_binary_round_trip_preserves_payload_and_routing() -> None:
    packet = _chunk()

    restored = VISION_MEDIA.deserialize(VISION_MEDIA.serialize(packet))

    assert restored == packet
    assert VISION_MEDIA.routing_key is not None
    assert VISION_MEDIA.routing_key(restored) == "worker-node"
    assert VISION_MEDIA.stream_key is not None
    assert VISION_MEDIA.stream_key(restored) == "vision-command:api-node"


def test_vision_media_completion_carries_integrity_contract() -> None:
    payload = b"aGVsbG8="
    packet = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=CommandId("vision-command"),
        model=ModelId("org/vlm"),
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert VISION_MEDIA.deserialize(VISION_MEDIA.serialize(packet)) == packet
    assert packet.is_terminal is True


def test_vision_media_transport_failure_reverses_route() -> None:
    failed = _chunk().transport_failure("capacity exhausted")

    assert failed.source_node == NodeId("worker-node")
    assert failed.target_node == NodeId("api-node")
    assert failed.kind == "transport_failed"
    assert failed.error_message == "capacity exhausted"


def test_vision_media_acceptance_failure_preserves_api_target() -> None:
    accepted = VisionMediaPacket(
        source_node=NodeId("worker-node"),
        target_node=NodeId("api-node"),
        command_id=CommandId("vision-command"),
        model=ModelId("org/vlm"),
        sequence=2,
        kind="accepted",
    )

    failed = accepted.transport_failure("publish failed")

    assert failed.source_node == NodeId("worker-node")
    assert failed.target_node == NodeId("api-node")
    assert failed.kind == "transport_failed"


def test_vision_media_open_and_acceptance_define_explicit_lifecycle() -> None:
    opened = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=CommandId("vision-command"),
        model=ModelId("org/vlm"),
        sequence=0,
        kind="opened",
        total_chunks=1,
        image_count=1,
    )
    completion = VisionMediaPacket(
        source_node=opened.source_node,
        target_node=opened.target_node,
        command_id=opened.command_id,
        model=opened.model,
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256="0" * 64,
    )

    accepted = completion.accepted()

    assert opened.is_terminal is False
    assert accepted.kind == "accepted"
    assert accepted.source_node == NodeId("worker-node")
    assert accepted.target_node == NodeId("api-node")
    assert accepted.is_terminal is True


def test_vision_media_rejects_inconsistent_chunk_metadata() -> None:
    with pytest.raises(ValidationError, match="sequence exceeds"):
        VisionMediaPacket(
            source_node=NodeId("api-node"),
            target_node=NodeId("worker-node"),
            command_id=CommandId("vision-command"),
            model=ModelId("org/vlm"),
            sequence=2,
            kind="chunk",
            data=b"aGVsbG8=",
            image_index=0,
            total_chunks=1,
        )
