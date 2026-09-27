"""A closed worker event channel ends the topology tasks without crashing the node.

On a master change the node shuts the worker's event router down before it
replaces the worker. A topology send in that window used to raise out of the
worker's task group and take the whole node down, dropping every instance it
hosted; a peer restart is exactly when those sends happen.
"""

from collections.abc import AsyncIterator, Callable

import anyio
import pytest
from skulk_pyo3_bindings import PyFromSwarm

from skulk.routing.connection_message import ConnectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import Event, IndexedEvent
from skulk.shared.types.topology import Connection, SocketConnection
from skulk.utils.channels import Receiver, Sender, channel
from skulk.worker import main as worker_main
from skulk.worker.main import Worker

_NODE = NodeId("node-a")
_PEER = NodeId("12D3KooWTestPeer")


def _session_edge() -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Worker._session_edge_multiaddr("100.95.14.7", 52416),  # pyright: ignore[reportPrivateUsage]
        session=True,
    )


def _worker(
    event_sender: Sender[Event],
    connection_receiver: Receiver[ConnectionMessage] | None = None,
) -> tuple[Worker, list[Callable[[], None]]]:
    """A worker on fresh channels, plus the closers for its unused senders."""
    indexed_sender, indexed_receiver = channel[IndexedEvent]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=_NODE,
        event_receiver=indexed_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        connection_message_receiver=connection_receiver,
    )
    return worker, [indexed_sender.close, command_sender.close, download_sender.close]


@pytest.mark.asyncio
async def test_session_edge_ingress_stops_quietly_when_the_event_channel_closes() -> None:
    """A disconnect arriving after the router shut down ends ingress quietly."""
    event_sender, event_receiver = channel[Event]()
    connection_sender, connection_receiver = channel[ConnectionMessage]()
    worker, others = _worker(event_sender, connection_receiver)
    # The peer's session edge was emitted, so its last disconnect sends a delete.
    worker._session_edge_counts[_PEER] = 1  # pyright: ignore[reportPrivateUsage]
    worker._session_emitted_edges[_PEER] = _session_edge()  # pyright: ignore[reportPrivateUsage]
    event_receiver.close()
    await connection_sender.send(
        ConnectionMessage.from_update(PyFromSwarm.Connection(str(_PEER), False, "", 0))
    )
    with anyio.fail_after(5):
        await worker._session_edge_ingress()  # pyright: ignore[reportPrivateUsage]
    connection_sender.close()
    for close in others:
        close()


@pytest.mark.asyncio
async def test_topology_probing_stops_quietly_when_the_event_channel_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-heal delete after the router shut down ends probing quietly."""
    event_sender, event_receiver = channel[Event]()
    worker, others = _worker(event_sender)
    stale = Connection(source=_NODE, sink=_PEER, edge=_session_edge())

    async def no_reachable_peers(
        *_args: object, **_kwargs: object
    ) -> AsyncIterator[tuple[str, NodeId]]:
        for item in ():
            yield item

    def nothing_missing() -> list[Connection]:
        return []

    def one_stale() -> list[Connection]:
        return [stale]

    monkeypatch.setattr(worker_main, "check_reachable", no_reachable_peers)
    monkeypatch.setattr(worker, "_session_edges_missing_from_state", nothing_missing)
    monkeypatch.setattr(worker, "_session_edges_stale_in_state", one_stale)
    event_receiver.close()
    with anyio.fail_after(5):
        await worker._poll_connection_updates()  # pyright: ignore[reportPrivateUsage]
    for close in others:
        close()
