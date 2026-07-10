"""Unit tests for the Router's data-plane transport selection (Zenoh, #279).

The Router routes the DATA topic over Zenoh only when a ZenohHandle is present
(the SKULK_ZENOH_DATA_PLANE flag); every other topic, and all topics when the
flag is off, stay on libp2p gossipsub. These tests pin that decision without
opening a network session.
"""

from typing import cast

import pytest
from skulk_pyo3_bindings import NetworkingHandle, ZenohHandle

from skulk.routing.router import (
    _ZENOH_DATA_OUTBOUND_BUFFER,  # pyright: ignore[reportPrivateUsage]
    OutboundPacket,
    Router,
)
from skulk.routing.topics import COMMANDS, DATA, GLOBAL_EVENTS, PROVIDER_DATA


def _router(*, zenoh: bool) -> Router:
    # uses_zenoh only inspects topic identity and whether a handle is present;
    # the handles are never called here, so opaque stand-ins are sufficient.
    fake_net = cast(NetworkingHandle, object())
    fake_zenoh = cast(ZenohHandle, object()) if zenoh else None
    # node_id is required when zenoh is enabled (the data/<node_id> sub key).
    return Router(handle=fake_net, zenoh=fake_zenoh, node_id="test-node")


def test_data_routes_over_zenoh_when_enabled() -> None:
    router = _router(zenoh=True)
    assert router.uses_zenoh(DATA.topic) is True
    assert router.uses_zenoh(PROVIDER_DATA.topic) is True
    # Control/telemetry/election planes stay on gossipsub even with zenoh on.
    assert router.uses_zenoh(COMMANDS.topic) is False
    assert router.uses_zenoh(GLOBAL_EVENTS.topic) is False


def test_nothing_routes_over_zenoh_when_disabled() -> None:
    router = _router(zenoh=False)
    assert router.uses_zenoh(DATA.topic) is False
    assert router.uses_zenoh(COMMANDS.topic) is False


def test_zenoh_outbound_channel_is_bounded() -> None:
    """The DATA egress channel must be bounded so a stalled subscriber
    backpressures the producer instead of growing memory without limit and
    OOMing the node (#312 review)."""
    router = _router(zenoh=True)
    assert router._zenoh_out_send is not None  # pyright: ignore[reportPrivateUsage]
    stats = router._zenoh_out_send.statistics()  # pyright: ignore[reportPrivateUsage]
    assert stats.max_buffer_size == _ZENOH_DATA_OUTBOUND_BUFFER
    # Sanity: a finite (non-inf) bound.
    assert stats.max_buffer_size < float("inf")


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
        # DATA egresses via the dedicated Zenoh outbound loop (#309), not the
        # shared control-plane loop, so feed its channel and drain it.
        assert router._zenoh_out_send is not None  # pyright: ignore[reportPrivateUsage]
        send = router._zenoh_out_send.clone()  # pyright: ignore[reportPrivateUsage]
        await send.send(
            OutboundPacket(
                topic=topic,
                routing_key=routing_key,
                stream_key="c",
                is_terminal=True,
                data=payload,
            )
        )
        send.close()
        with anyio.move_on_after(1):
            await router._zenoh_networking_publish()  # pyright: ignore[reportPrivateUsage]
        assert zenoh.published == [("data/owner-9", payload)]

    anyio.run(_run)


def test_blocked_command_does_not_stall_another_command_for_same_owner() -> None:
    """Per-command egress workers isolate streams behind one slow owner."""

    import anyio

    from skulk.routing.router import OutboundPacket
    from skulk.shared.types.chunks import DataChunk
    from skulk.shared.types.common import CommandId, NodeId

    class _CommandBlockingZenoh:
        def __init__(self) -> None:
            self.slow_started = anyio.Event()
            self.release_slow = anyio.Event()
            self.fast_published = anyio.Event()

        async def zenoh_publish(self, _key: str, data: bytes) -> None:
            frame = DATA.deserialize(data)
            if frame.command_id == CommandId("slow-command"):
                self.slow_started.set()
                await self.release_slow.wait()
            else:
                self.fast_published.set()

    def _terminal_packet(command_id: str) -> OutboundPacket:
        frame = DataChunk(
            command_id=CommandId(command_id),
            kind="completed",
            sequence=0,
            owner_node=NodeId("shared-owner"),
        )
        return OutboundPacket(
            topic=DATA.topic,
            routing_key="shared-owner",
            stream_key=command_id,
            is_terminal=True,
            data=DATA.serialize(frame),
        )

    async def _run() -> None:
        zenoh = _CommandBlockingZenoh()
        router = Router(
            handle=cast(NetworkingHandle, object()),
            zenoh=cast(ZenohHandle, cast(object, zenoh)),
            node_id="self-node",
        )
        assert router._zenoh_out_send is not None  # pyright: ignore[reportPrivateUsage]

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                router._zenoh_networking_publish  # pyright: ignore[reportPrivateUsage]
            )
            await router._zenoh_out_send.send(  # pyright: ignore[reportPrivateUsage]
                _terminal_packet("slow-command")
            )
            with anyio.fail_after(0.5):
                await zenoh.slow_started.wait()

            await router._zenoh_out_send.send(  # pyright: ignore[reportPrivateUsage]
                _terminal_packet("fast-command")
            )
            with anyio.fail_after(0.5):
                await zenoh.fast_published.wait()

            diagnostics = router.data_plane_egress_diagnostics()
            assert diagnostics.remote_frames_published == 1
            assert diagnostics.active_stream_queues == 1
            zenoh.release_slow.set()
            task_group.cancel_scope.cancel()

    anyio.run(_run)


def test_matching_ids_on_data_topics_use_distinct_egress_queues() -> None:
    """DATA and PROVIDER_DATA ids cannot terminate each other's queues."""

    import anyio

    from skulk.extensions import CapabilityStreamFrame
    from skulk.routing.provider_streams import ProviderStreamPacket
    from skulk.shared.types.chunks import DataChunk
    from skulk.shared.types.common import CommandId, NodeId

    class _BlockingDataZenoh:
        def __init__(self) -> None:
            self.data_started = anyio.Event()
            self.release_data = anyio.Event()
            self.provider_published = anyio.Event()

        async def zenoh_publish(self, key: str, _data: bytes) -> None:
            if key.startswith("data/"):
                self.data_started.set()
                await self.release_data.wait()
            elif key.startswith("provider_data/"):
                self.provider_published.set()

    async def _run() -> None:
        owner = NodeId("shared-owner")
        shared_id = "same-stream-id"
        data_frame = DataChunk(
            command_id=CommandId(shared_id),
            kind="completed",
            sequence=0,
            owner_node=owner,
        )
        provider_frame = ProviderStreamPacket(
            owner_node=owner,
            frame=CapabilityStreamFrame(
                call_id=shared_id,
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            ),
        )
        zenoh = _BlockingDataZenoh()
        router = Router(
            handle=cast(NetworkingHandle, object()),
            zenoh=cast(ZenohHandle, cast(object, zenoh)),
            node_id="self-node",
        )
        assert router._zenoh_out_send is not None  # pyright: ignore[reportPrivateUsage]

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                router._zenoh_networking_publish  # pyright: ignore[reportPrivateUsage]
            )
            await router._zenoh_out_send.send(  # pyright: ignore[reportPrivateUsage]
                OutboundPacket(
                    topic=DATA.topic,
                    routing_key=str(owner),
                    stream_key=shared_id,
                    is_terminal=True,
                    data=DATA.serialize(data_frame),
                )
            )
            with anyio.fail_after(0.5):
                await zenoh.data_started.wait()

            await router._zenoh_out_send.send(  # pyright: ignore[reportPrivateUsage]
                OutboundPacket(
                    topic=PROVIDER_DATA.topic,
                    routing_key=str(owner),
                    stream_key=shared_id,
                    is_terminal=True,
                    data=PROVIDER_DATA.serialize(provider_frame),
                )
            )
            with anyio.fail_after(0.5):
                await zenoh.provider_published.wait()
            zenoh.release_data.set()
            task_group.cancel_scope.cancel()

    anyio.run(_run)


def test_stream_admission_limit_sends_terminal_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-cap command must fail at its API instead of hanging forever."""

    import anyio

    import skulk.routing.router as router_module
    from skulk.shared.types.chunks import DataChunk, ErrorChunk
    from skulk.shared.types.common import CommandId, NodeId

    monkeypatch.setattr(router_module, "_ZENOH_DATA_MAX_STREAMS_PER_OWNER", 0)

    class _RecordingZenoh:
        def __init__(self) -> None:
            self.frames: list[DataChunk] = []
            self.complete = anyio.Event()

        async def zenoh_publish(self, _key: str, data: bytes) -> None:
            self.frames.append(DATA.deserialize(data))
            if len(self.frames) == 2:
                self.complete.set()

    async def _run() -> None:
        zenoh = _RecordingZenoh()
        router = Router(
            handle=cast(NetworkingHandle, object()),
            zenoh=cast(ZenohHandle, cast(object, zenoh)),
            node_id="self-node",
        )
        assert router._zenoh_out_send is not None  # pyright: ignore[reportPrivateUsage]
        started = DataChunk(
            command_id=CommandId("rejected-command"),
            kind="started",
            sequence=0,
            owner_node=NodeId("remote-owner"),
        )
        packet = OutboundPacket(
            topic=DATA.topic,
            routing_key="remote-owner",
            stream_key="rejected-command",
            is_terminal=False,
            data=DATA.serialize(started),
        )

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                router._zenoh_networking_publish  # pyright: ignore[reportPrivateUsage]
            )
            await router._zenoh_out_send.send(packet)  # pyright: ignore[reportPrivateUsage]
            with anyio.fail_after(0.5):
                await zenoh.complete.wait()
            task_group.cancel_scope.cancel()

        assert [frame.kind for frame in zenoh.frames] == ["started", "failed"]
        assert [frame.sequence for frame in zenoh.frames] == [0, 1]
        assert isinstance(zenoh.frames[1].chunk, ErrorChunk)
        diagnostics = router.data_plane_egress_diagnostics()
        assert diagnostics.remote_frames_dropped == 1
        assert diagnostics.remote_frames_published == 2

    anyio.run(_run)


def test_full_stream_queue_replaces_terminal_and_ignores_closed_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full queue emits failure and cannot tear down sibling egress."""

    import anyio

    import skulk.routing.router as router_module
    from skulk.shared.models.model_cards import ModelId
    from skulk.shared.types.chunks import DataChunk, ErrorChunk, TokenChunk
    from skulk.shared.types.common import CommandId, NodeId

    monkeypatch.setattr(router_module, "_ZENOH_DATA_STREAM_BUFFER", 1)

    class _BlockingZenoh:
        def __init__(self) -> None:
            self.started = anyio.Event()
            self.release_started = anyio.Event()
            self.failed = anyio.Event()
            self.frames: list[DataChunk] = []

        async def zenoh_publish(self, _key: str, data: bytes) -> None:
            frame = DATA.deserialize(data)
            self.frames.append(frame)
            if frame.kind == "started":
                self.started.set()
                await self.release_started.wait()
            elif frame.kind == "failed":
                self.failed.set()

    def _packet(frame: DataChunk) -> OutboundPacket:
        return OutboundPacket(
            topic=DATA.topic,
            routing_key="remote-owner",
            stream_key="full-command",
            is_terminal=frame.is_terminal,
            data=DATA.serialize(frame),
        )

    async def _run() -> None:
        zenoh = _BlockingZenoh()
        router = Router(
            handle=cast(NetworkingHandle, object()),
            zenoh=cast(ZenohHandle, cast(object, zenoh)),
            node_id="self-node",
        )
        assert router._zenoh_out_send is not None  # pyright: ignore[reportPrivateUsage]
        command_id = CommandId("full-command")
        owner = NodeId("remote-owner")
        frames = (
            DataChunk(
                command_id=command_id,
                kind="started",
                sequence=0,
                owner_node=owner,
            ),
            DataChunk(
                command_id=command_id,
                kind="chunk",
                chunk=TokenChunk(
                    model=ModelId("model"),
                    text="payload",
                    token_id=1,
                    usage=None,
                ),
                sequence=1,
                owner_node=owner,
            ),
            DataChunk(
                command_id=command_id,
                kind="completed",
                sequence=2,
                owner_node=owner,
            ),
        )

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                router._zenoh_networking_publish  # pyright: ignore[reportPrivateUsage]
            )
            await router._zenoh_out_send.send(_packet(frames[0]))  # pyright: ignore[reportPrivateUsage]
            with anyio.fail_after(0.5):
                await zenoh.started.wait()
            await router._zenoh_out_send.send(_packet(frames[1]))  # pyright: ignore[reportPrivateUsage]
            await router._zenoh_out_send.send(_packet(frames[2]))  # pyright: ignore[reportPrivateUsage]
            await router._zenoh_out_send.send(_packet(frames[1]))  # pyright: ignore[reportPrivateUsage]
            with anyio.fail_after(0.5):
                await zenoh.failed.wait()
            zenoh.release_started.set()
            task_group.cancel_scope.cancel()

        failed_frames = [frame for frame in zenoh.frames if frame.kind == "failed"]
        assert len(failed_frames) == 1
        assert failed_frames[0].sequence == 2
        assert isinstance(failed_frames[0].chunk, ErrorChunk)
        diagnostics = router.data_plane_egress_diagnostics()
        assert diagnostics.remote_frames_dropped >= 2

    anyio.run(_run)
