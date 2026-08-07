"""Regression coverage for election egress isolation under control-plane load."""

from typing import cast

import anyio
from skulk_pyo3_bindings import NetworkingHandle

from skulk.routing.router import OutboundPacket, Router
from skulk.routing.topics import ELECTION_MESSAGES, GLOBAL_EVENTS


class _BlockingOrdinaryNetwork:
    """A transport whose ordinary publish cannot make forward progress."""

    def __init__(self) -> None:
        self.ordinary_started = anyio.Event()
        self.release_ordinary = anyio.Event()
        self.election_published = anyio.Event()

    async def gossipsub_publish(self, topic: str, data: bytes) -> None:
        if topic == ELECTION_MESSAGES.topic:
            self.election_published.set()
            return
        self.ordinary_started.set()
        await self.release_ordinary.wait()


def _packet(topic: str) -> OutboundPacket:
    return OutboundPacket(
        topic=topic,
        routing_key=None,
        stream_key=None,
        is_terminal=False,
        data=topic.encode(),
    )


async def test_blocked_ordinary_publish_does_not_block_election_egress() -> None:
    """Election publication must bypass a stalled ordinary publish loop."""

    network = _BlockingOrdinaryNetwork()
    router = Router(
        handle=cast(NetworkingHandle, cast(object, network)),
        node_id="test-node",
    )
    ordinary_sender = router.networking_receiver.clone_sender()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router._networking_publish)  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(router._election_networking_publish)  # pyright: ignore[reportPrivateUsage]

        await ordinary_sender.send(_packet(GLOBAL_EVENTS.topic))
        with anyio.fail_after(0.5):
            await network.ordinary_started.wait()

        await router._election_out_send.send(  # pyright: ignore[reportPrivateUsage]
            _packet(ELECTION_MESSAGES.topic)
        )
        with anyio.fail_after(0.5):
            await network.election_published.wait()

        task_group.cancel_scope.cancel()
