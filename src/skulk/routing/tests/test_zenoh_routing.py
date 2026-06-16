"""Unit tests for the Router's data-plane transport selection (Zenoh, #279).

The Router routes the DATA topic over Zenoh only when a ZenohHandle is present
(the SKULK_ZENOH_DATA_PLANE flag); every other topic, and all topics when the
flag is off, stay on libp2p gossipsub. These tests pin that decision without
opening a network session.
"""

from typing import cast

from skulk_pyo3_bindings import NetworkingHandle, ZenohHandle

from skulk.routing.router import Router
from skulk.routing.topics import COMMANDS, DATA, GLOBAL_EVENTS


def _router(*, zenoh: bool) -> Router:
    # uses_zenoh only inspects topic identity and whether a handle is present;
    # the handles are never called here, so opaque stand-ins are sufficient.
    fake_net = cast(NetworkingHandle, object())
    fake_zenoh = cast(ZenohHandle, object()) if zenoh else None
    return Router(handle=fake_net, zenoh=fake_zenoh)


def test_data_routes_over_zenoh_when_enabled() -> None:
    router = _router(zenoh=True)
    assert router.uses_zenoh(DATA.topic) is True
    # Control/telemetry/election planes stay on gossipsub even with zenoh on.
    assert router.uses_zenoh(COMMANDS.topic) is False
    assert router.uses_zenoh(GLOBAL_EVENTS.topic) is False


def test_nothing_routes_over_zenoh_when_disabled() -> None:
    router = _router(zenoh=False)
    assert router.uses_zenoh(DATA.topic) is False
    assert router.uses_zenoh(COMMANDS.topic) is False


def test_data_owner_key_addresses_owning_node() -> None:
    """The DATA topic's routing key is the chunk's owner node id (#279 Phase 2).

    Unset owner falls back to None (the bare topic), preserving the gossipsub
    broadcast path.
    """
    from skulk.routing.topics import (
        _data_owner_key,  # pyright: ignore[reportPrivateUsage]
    )
    from skulk.shared.models.model_cards import ModelId
    from skulk.shared.types.chunks import DataChunk, TokenChunk
    from skulk.shared.types.common import CommandId, NodeId

    def _chunk(owner: NodeId | None) -> DataChunk:
        return DataChunk(
            command_id=CommandId("c"),
            chunk=TokenChunk(
                model=ModelId("m"), text="x", token_id=0, usage=None
            ),
            sequence=0,
            owner_node=owner,
        )

    assert DATA.routing_key is not None
    assert _data_owner_key(_chunk(NodeId("api-3"))) == "api-3"
    assert _data_owner_key(_chunk(None)) is None


def test_zenoh_publish_keys_by_owner_and_subscribe_keys_by_self() -> None:
    """The serving worker publishes to data/<owner>; a node subscribes to its own.

    This is the unicast that kills the fan-out (#279 Phase 2): output reaches
    only the owning API node's subscription, and an inbound sample's owner suffix
    is stripped back to the bare topic so it routes to the DATA TopicRouter.
    """
    import anyio

    from skulk.routing.topics import DATA as DATA_TOPIC
    from skulk.shared.models.model_cards import ModelId
    from skulk.shared.types.chunks import DataChunk, TokenChunk
    from skulk.shared.types.common import CommandId, NodeId

    class _RecordingZenoh:
        def __init__(self) -> None:
            self.subscribed: list[str] = []
            self.published: list[tuple[str, bytes]] = []

        async def zenoh_subscribe(self, key: str) -> None:
            self.subscribed.append(key)

        async def zenoh_publish(self, key: str, data: bytes) -> None:
            self.published.append((key, data))

    async def _run() -> None:
        fake_net = cast(NetworkingHandle, object())
        zenoh = _RecordingZenoh()
        router = Router(
            handle=fake_net,
            zenoh=cast(ZenohHandle, cast(object, zenoh)),
            node_id="self-node",
        )

        # Subscription keys the data plane to this node's own id.
        await router._networking_subscribe(DATA_TOPIC.topic)  # pyright: ignore[reportPrivateUsage]
        assert zenoh.subscribed == ["data/self-node"]

        # Publishing a chunk addressed to another owner keys to data/<owner>.
        chunk = DataChunk(
            command_id=CommandId("c"),
            chunk=TokenChunk(
                model=ModelId("m"), text="hi", token_id=0, usage=None
            ),
            sequence=0,
            owner_node=NodeId("owner-9"),
        )
        topic, routing_key, payload = (
            DATA_TOPIC.topic,
            DATA_TOPIC.routing_key(chunk) if DATA_TOPIC.routing_key else None,
            DATA_TOPIC.serialize(chunk),
        )
        send = router.networking_receiver.clone_sender()
        await send.send((topic, routing_key, payload))
        send.close()
        with anyio.move_on_after(1):
            await router._networking_publish()  # pyright: ignore[reportPrivateUsage]
        assert zenoh.published == [("data/owner-9", payload)]

    anyio.run(_run)
