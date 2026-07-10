"""Provider DATA packet framing and transport-failure coverage."""

import json

import anyio
import pytest

from skulk.extensions import (
    MAX_CALL_PAYLOAD_BYTES,
    CapabilityStreamFrame,
    InlineMediaAttachment,
)
from skulk.routing.provider_streams import (
    ProviderStreamPacket,
    decode_provider_stream_packet,
    encode_provider_stream_packet,
    provider_stream_rejection_packets,
)
from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import PROVIDER_DATA
from skulk.shared.types.common import NodeId
from skulk.utils.channels import channel


def _media_packet() -> ProviderStreamPacket:
    return ProviderStreamPacket(
        owner_node=NodeId("caller-node"),
        frame=CapabilityStreamFrame(
            call_id="tts-call",
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={"format": "pcm_s16le"},
            media=InlineMediaAttachment(
                data=bytes(range(256)),
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=24000,
                channels=1,
            ),
        ),
    )


def test_provider_topic_preserves_arbitrary_inline_media_bytes() -> None:
    packet = _media_packet()

    wire = PROVIDER_DATA.serialize(packet)
    restored = PROVIDER_DATA.deserialize(wire)

    assert restored == packet
    assert PROVIDER_DATA.routing_key is not None
    assert PROVIDER_DATA.routing_key(restored) == "caller-node"
    assert PROVIDER_DATA.stream_key is not None
    assert PROVIDER_DATA.stream_key(restored) == "tts-call"


def test_provider_packet_decoder_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated"):
        decode_provider_stream_packet((100).to_bytes(4, "big") + b"{}")


def test_full_size_valid_payload_fits_outer_provider_envelope() -> None:
    text = "x" * (MAX_CALL_PAYLOAD_BYTES - 32)
    payload: dict[str, object] = {"text": text}
    assert len(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    ) <= MAX_CALL_PAYLOAD_BYTES
    packet = ProviderStreamPacket(
        owner_node=NodeId("caller-node"),
        frame=CapabilityStreamFrame(
            call_id="large-payload",
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload=payload,
        ),
    )

    wire = encode_provider_stream_packet(packet)

    assert len(wire) > MAX_CALL_PAYLOAD_BYTES
    assert decode_provider_stream_packet(wire) == packet


def test_provider_admission_rejection_is_started_then_failed() -> None:
    packet = ProviderStreamPacket(
        owner_node=NodeId("caller-node"),
        frame=CapabilityStreamFrame(
            call_id="tts-call",
            direction="provider_to_caller",
            sequence=0,
            kind="started",
        ),
    )

    rejected = provider_stream_rejection_packets(packet, include_started=True)

    assert [item.frame.kind for item in rejected] == ["started", "failed"]
    assert [item.frame.sequence for item in rejected] == [0, 1]
    assert rejected[-1].frame.error is not None
    assert rejected[-1].frame.error.code == "transport_error"


async def test_remote_provider_packet_egresses_with_call_isolation_key() -> None:
    networking_sender, networking_receiver = channel[OutboundPacket]()
    router = TopicRouter[ProviderStreamPacket](
        PROVIDER_DATA,
        networking_sender,
        local_routing_key="provider-node",
    )
    input_sender = router.new_sender()
    packet = _media_packet()
    outbound: OutboundPacket | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await input_sender.send(packet)
        with anyio.fail_after(0.5):
            outbound = await networking_receiver.receive()
        task_group.cancel_scope.cancel()

    assert outbound is not None
    assert outbound.topic == PROVIDER_DATA.topic
    assert outbound.routing_key == "caller-node"
    assert outbound.stream_key == "tts-call"
    assert PROVIDER_DATA.deserialize(outbound.data) == packet


async def test_malformed_provider_wire_does_not_tear_down_topic_router() -> None:
    networking_sender, _ = channel[OutboundPacket]()
    router = TopicRouter[ProviderStreamPacket](PROVIDER_DATA, networking_sender)

    await router.publish_bytes(b"\x00\x00\x00\x10{}", origin="peer")
