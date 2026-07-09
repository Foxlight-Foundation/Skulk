"""Tests for TopicRouter wire-format robustness."""

import anyio

from skulk.routing.router import TopicRouter
from skulk.routing.topics import DATA, PublishPolicy, TypedTopic
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


_TOPIC_V1 = TypedTopic("schema_compat_test", PublishPolicy.Always, _SchemaV1)
_TOPIC_V2 = TypedTopic("schema_compat_test", PublishPolicy.Always, _SchemaV2)


async def test_publish_bytes_drops_unknown_field_payload_without_raising():
    """An older receiver must survive a payload that contains an unknown field.

    Reproduces the rolling-upgrade scenario where a 1.0.3 sender publishes a
    `PlaceInstance` carrying `excluded_nodes` to a 1.0.2 master. The master's
    strict (extra="forbid") deserializer would otherwise raise out of
    `publish_bytes` and tear down the gossipsub receive loop. After the fix,
    the bad message is dropped silently and the router stays alive to process
    the next valid message.
    """
    networking_send, _networking_recv = channel[tuple[str, str | None, bytes]]()
    router_v1 = TopicRouter[_SchemaV1](_TOPIC_V1, networking_send)

    incompatible_payload = _TOPIC_V2.serialize(_SchemaV2(name="hi", extra=["x"]))
    valid_payload = _TOPIC_V1.serialize(_SchemaV1(name="ok"))

    # Must NOT raise — pre-fix this propagated ValidationError up to the
    # gossipsub receive loop and terminated it.
    await router_v1.publish_bytes(incompatible_payload, origin=None)

    # After dropping the bad message, the router must remain functional for
    # the next valid message.
    await router_v1.publish_bytes(valid_payload, origin=None)


async def test_data_topic_publishes_locally_before_blocked_network_egress() -> None:
    """DATA chunks must reach local API receivers before network backpressure.

    Same-node serving publishes DATA through the same TopicRouter as cross-node
    output. If the router awaits outbound DATA egress before local publish, a
    blocked egress channel can hold every audio/token chunk and make a stream
    appear as one late burst. The API's DATA consumer already dedupes the later
    Zenoh self-loopback copy by sequence, so local-first DATA delivery is safe.
    """

    networking_send, networking_recv = channel[tuple[str, str | None, bytes]](
        max_buffer_size=1
    )
    await networking_send.send(("occupied", None, b"held"))

    router = TopicRouter[DataChunk](DATA, networking_send)
    local_send, local_recv = channel[DataChunk]()
    router.senders.add(local_send)
    input_send = router.new_sender()

    chunk = DataChunk(
        command_id=CommandId("local-first-data"),
        sequence=0,
        owner_node=NodeId("api-node"),
        chunk=TokenChunk(
            model=ModelId("mlx-community/test"),
            text="hello",
            token_id=1,
            usage=None,
        ),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router.run)
        await input_send.send(chunk)

        with anyio.fail_after(0.5):
            assert await local_recv.receive() == chunk

        assert await networking_recv.receive() == ("occupied", None, b"held")
        with anyio.fail_after(0.5):
            topic, routing_key, payload = await networking_recv.receive()
        assert topic == DATA.topic
        assert routing_key == "api-node"
        assert DATA.deserialize(payload) == chunk
        task_group.cancel_scope.cancel()
