"""Tests for TopicRouter wire-format robustness."""

from exo.routing.router import TopicRouter
from exo.routing.topics import PublishPolicy, TypedTopic
from exo.utils.channels import channel
from exo.utils.pydantic_ext import CamelCaseModel


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
    networking_send, _networking_recv = channel[tuple[str, bytes]]()
    router_v1 = TopicRouter[_SchemaV1](_TOPIC_V1, networking_send)

    incompatible_payload = _TOPIC_V2.serialize(_SchemaV2(name="hi", extra=["x"]))
    valid_payload = _TOPIC_V1.serialize(_SchemaV1(name="ok"))

    # Must NOT raise — pre-fix this propagated ValidationError up to the
    # gossipsub receive loop and terminated it.
    await router_v1.publish_bytes(incompatible_payload, origin=None)

    # After dropping the bad message, the router must remain functional for
    # the next valid message.
    await router_v1.publish_bytes(valid_payload, origin=None)
