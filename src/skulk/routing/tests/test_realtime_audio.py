"""Realtime audio DATA packet framing and node routing coverage."""

import anyio
import pytest

from skulk.routing.realtime_audio import RealtimeAudioPacket
from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import REALTIME_AUDIO
from skulk.shared.types.common import CommandId, NodeId
from skulk.utils.channels import channel


def _chunk(*, target: str = "worker-node") -> RealtimeAudioPacket:
    return RealtimeAudioPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId(target),
        command_id=CommandId("realtime-command"),
        sequence=1,
        kind="chunk",
        data=b"\x00\x00\x01\x00",
    )


def test_realtime_audio_wire_preserves_pcm_and_routing() -> None:
    packet = _chunk()

    restored = REALTIME_AUDIO.deserialize(REALTIME_AUDIO.serialize(packet))

    assert restored == packet
    assert restored.data == b"\x00\x00\x01\x00"
    assert REALTIME_AUDIO.routing_key is not None
    assert REALTIME_AUDIO.routing_key(restored) == "worker-node"
    assert REALTIME_AUDIO.stream_key is not None
    assert REALTIME_AUDIO.stream_key(restored) == "realtime-command"


def test_realtime_audio_transport_failure_routes_to_source() -> None:
    failure = _chunk().transport_failure("capacity exhausted")

    assert failure.kind == "transport_failed"
    assert failure.target_node == NodeId("api-node")
    assert failure.source_node == NodeId("worker-node")
    assert failure.is_terminal is True


def test_realtime_audio_rejects_malformed_pcm() -> None:
    with pytest.raises(ValueError, match="complete 16-bit samples"):
        RealtimeAudioPacket(
            source_node=NodeId("api-node"),
            target_node=NodeId("worker-node"),
            command_id=CommandId("realtime-command"),
            sequence=1,
            kind="chunk",
            data=b"\x00",
        )


def test_realtime_audio_rejects_truncated_wire_header() -> None:
    with pytest.raises(ValueError, match="truncated"):
        REALTIME_AUDIO.deserialize(b"\x00\x00\x00\x10{}")


async def test_realtime_audio_remote_target_egresses_without_local_publish() -> None:
    networking_sender, networking_receiver = channel[OutboundPacket]()
    router = TopicRouter[RealtimeAudioPacket](
        REALTIME_AUDIO,
        networking_sender,
        local_routing_key="api-node",
    )
    input_sender = router.new_sender()
    outbound: OutboundPacket | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await input_sender.send(_chunk())
        with anyio.fail_after(0.5):
            outbound = await networking_receiver.receive()
        task_group.cancel_scope.cancel()

    assert outbound is not None
    assert outbound.topic == REALTIME_AUDIO.topic
    assert outbound.routing_key == "worker-node"
    assert outbound.stream_key == "realtime-command"
    assert REALTIME_AUDIO.deserialize(outbound.data) == _chunk()
