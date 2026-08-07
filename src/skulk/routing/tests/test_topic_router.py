"""Tests for TopicRouter wire-format robustness."""

import anyio

from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import DATA, MessagePlane, PublishPolicy, TypedTopic
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import DataChunk, TokenChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.utils.channels import channel
from skulk.utils.pydantic_ext import CamelCaseModel


class _SchemaV1(CamelCaseModel):
    """Old wire-format schema with one field."""

    name: str


class _SchemaV2(CamelCaseModel):
    """New wire-format schema that adds a field — emulates a version skew."""

    name: str
    extra: list[str] = []


_TOPIC_V1 = TypedTopic(
    "schema_compat_test", PublishPolicy.Always, MessagePlane.Control, _SchemaV1
)
_TOPIC_V2 = TypedTopic(
    "schema_compat_test", PublishPolicy.Always, MessagePlane.Control, _SchemaV2
)


async def test_publish_bytes_drops_unknown_field_payload_without_raising():
    """An older receiver must survive a payload that contains an unknown field.

    Reproduces the rolling-upgrade scenario where a 1.0.3 sender publishes a
    `PlaceInstance` carrying `excluded_nodes` to a 1.0.2 master. The master's
    strict (extra="forbid") deserializer would otherwise raise out of
    `publish_bytes` and tear down the gossipsub receive loop. After the fix,
    the bad message is dropped silently and the router stays alive to process
    the next valid message.
    """
    networking_send, _networking_recv = channel[OutboundPacket]()
    router_v1 = TopicRouter[_SchemaV1](_TOPIC_V1, networking_send)

    incompatible_payload = _TOPIC_V2.serialize(_SchemaV2(name="hi", extra=["x"]))
    valid_payload = _TOPIC_V1.serialize(_SchemaV1(name="ok"))

    # Must NOT raise — pre-fix this propagated ValidationError up to the
    # gossipsub receive loop and terminated it.
    await router_v1.publish_bytes(incompatible_payload, origin=None)

    # After dropping the bad message, the router must remain functional for
    # the next valid message.
    await router_v1.publish_bytes(valid_payload, origin=None)


async def test_outbound_serialization_failure_does_not_stop_topic_router() -> None:
    """One invalid outbound item cannot terminate all later topic egress."""

    def serialize(item: _SchemaV1) -> bytes:
        if item.name == "bad":
            raise ValueError("synthetic serializer failure")
        return item.model_dump_json().encode()

    topic = TypedTopic(
        "serializer_containment_test",
        PublishPolicy.Always,
        MessagePlane.Control,
        _SchemaV1,
        serializer=serialize,
    )
    networking_send, networking_recv = channel[OutboundPacket](1)
    router = TopicRouter[_SchemaV1](topic, networking_send)
    input_send = router.new_sender()
    outbound: OutboundPacket | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await input_send.send(_SchemaV1(name="bad"))
        await input_send.send(_SchemaV1(name="good"))
        with anyio.fail_after(0.5):
            outbound = await networking_recv.receive()
        task_group.cancel_scope.cancel()

    assert outbound is not None
    assert outbound.data == b'{"name":"good"}'


def _data_chunk(sequence: int, owner_node: str) -> DataChunk:
    return DataChunk(
        command_id=CommandId("local-first-data"),
        sequence=sequence,
        owner_node=NodeId(owner_node),
        chunk=TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"chunk-{sequence}",
            token_id=sequence,
            usage=None,
        ),
    )


async def test_local_data_topic_does_not_wait_for_blocked_network_egress() -> None:
    """DATA chunks must reach local API receivers before network backpressure.

    Same-node serving publishes DATA through the same TopicRouter as cross-node
    output. Local DATA does not need a network self-loopback copy because the
    owning API node is already local. If the router awaits outbound DATA egress,
    a blocked egress channel can hold subsequent audio/token chunks and make a
    stream appear as one late burst.
    """

    networking_send, networking_recv = channel[OutboundPacket](
        max_buffer_size=1
    )
    occupied = OutboundPacket(
        topic="occupied",
        routing_key=None,
        stream_key=None,
        is_terminal=False,
        data=b"held",
    )
    await networking_send.send(occupied)

    router = TopicRouter[DataChunk](
        DATA, networking_send, local_routing_key="api-node"
    )
    local_send, local_recv = channel[DataChunk]()
    router.senders.add(local_send)
    input_send = router.new_sender()

    first_chunk = _data_chunk(sequence=0, owner_node="api-node")
    second_chunk = _data_chunk(sequence=1, owner_node="api-node")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await input_send.send(first_chunk)
        await input_send.send(second_chunk)

        with anyio.fail_after(0.5):
            assert await local_recv.receive() == first_chunk
            assert await local_recv.receive() == second_chunk

        assert await networking_recv.receive() == occupied
        with anyio.move_on_after(0.1) as scope:
            await networking_recv.receive()
        assert scope.cancel_called
        task_group.cancel_scope.cancel()


async def test_remote_data_topic_still_egresses_to_owner_key() -> None:
    """Cross-node DATA egresses without dispatching on the producing API."""

    networking_send, networking_recv = channel[OutboundPacket]()

    router = TopicRouter[DataChunk](
        DATA, networking_send, local_routing_key="api-node"
    )
    local_send, local_recv = channel[DataChunk]()
    router.senders.add(local_send)
    input_send = router.new_sender()

    chunk = _data_chunk(sequence=0, owner_node="remote-node")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await input_send.send(chunk)

        with anyio.fail_after(0.5):
            packet = await networking_recv.receive()
        with anyio.move_on_after(0.1) as scope:
            await local_recv.receive()
        assert scope.cancel_called
        assert packet.topic == DATA.topic
        assert packet.routing_key == "remote-node"
        assert packet.stream_key == "local-first-data"
        assert DATA.deserialize(packet.data) == chunk
        task_group.cancel_scope.cancel()
