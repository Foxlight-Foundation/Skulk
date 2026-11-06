"""Convergence after startup vote loss without another membership event."""

from collections.abc import Callable
from dataclasses import dataclass

import anyio
import pytest
from anyio import create_task_group, fail_after, move_on_after

from skulk.routing.connection_message import ConnectionMessage
from skulk.routing.topics import ELECTION_MESSAGES
from skulk.shared.election import Election, ElectionMessage, ElectionResult
from skulk.shared.types.commands import ForwarderCommand
from skulk.shared.types.common import NodeId, SessionId
from skulk.utils.channels import Receiver, Sender, channel


@dataclass(frozen=True)
class _Peer:
    election: Election
    incoming: Sender[ElectionMessage]
    outgoing: Receiver[ElectionMessage]
    connections: Sender[ConnectionMessage]
    commands: Sender[ForwarderCommand]
    results: Receiver[ElectionResult]


def _peer(name: str) -> _Peer:
    incoming, received = channel[ElectionMessage]()
    outgoing, published = channel[ElectionMessage]()
    results, completed = channel[ElectionResult]()
    connections, membership = channel[ConnectionMessage]()
    commands, observed = channel[ForwarderCommand]()
    return _Peer(
        Election(
            NodeId(name),
            election_message_receiver=received,
            election_message_sender=outgoing,
            election_result_sender=results,
            connection_message_receiver=membership,
            command_receiver=observed,
        ),
        incoming,
        published,
        connections,
        commands,
        completed,
    )


async def _until(predicate: Callable[[], bool]) -> None:
    with fail_after(2):
        while not predicate():
            await anyio.sleep(0.002)


@pytest.fixture(autouse=True)
def short_election_deadlines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten actor deadlines while retaining their production ordering."""
    monkeypatch.setattr("skulk.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.08)
    monkeypatch.setattr("skulk.shared.election.CONNECTION_SETTLE_DELAY", 0.005)
    monkeypatch.setattr(
        "skulk.shared.election.COMPLETED_ELECTION_REPUBLISH_INTERVAL_SECONDS", 0.025
    )


@pytest.mark.anyio
@pytest.mark.parametrize("newer_isolated_round", [False, True])
async def test_startup_vote_loss_converges_without_reconnection(
    newer_isolated_round: bool,
) -> None:
    """Heal a healthy graph after one node's startup publications were lost.

    Use real actors and the actual topic codec, including duplicate protocol
    copies. Restoring delivery emits no connection event. The timer must neither
    increase vote strength nor keep initiating campaigns once everyone agrees.
    """
    peers = {name: _peer(name) for name in ("Z", "B", "C")}
    results: dict[str, list[ElectionResult]] = {name: [] for name in peers}
    blackout = True
    dropped = 0

    async def relay(name: str) -> None:
        nonlocal dropped
        with peers[name].outgoing as published:
            async for message in published:
                if blackout and name == "Z":
                    dropped += 1
                    continue
                packet = ELECTION_MESSAGES.serialize(message)
                for other, peer in peers.items():
                    if name != other:
                        await peer.incoming.send(ELECTION_MESSAGES.deserialize(packet))
                        await peer.incoming.send(ELECTION_MESSAGES.deserialize(packet))

    async def collect(name: str) -> None:
        with peers[name].results as received:
            async for result in received:
                # Election updates its session before its result is consumed.
                # Exercise that observation gap rather than relying on task order.
                await anyio.sleep(0.01)
                results[name].append(result)

    def settled() -> bool:
        return all(
            results[name]
            and results[name][-1].won_clock == peer.election.clock
            and results[name][-1].session_id == peer.election.current_session
            and peer.election.clock >= 1
            for name, peer in peers.items()
        )

    async with create_task_group() as tasks:
        for name, peer in peers.items():
            tasks.start_soon(peer.election.run)
            tasks.start_soon(relay, name)
            tasks.start_soon(collect, name)
        await _until(lambda: all(results.values()))
        for name, peer in peers.items():
            for other in peers:
                if name != other:
                    await peer.connections.send(
                        ConnectionMessage(node_id=NodeId(other), connected=True)
                    )
        await _until(settled)
        assert (
            len(
                {
                    (
                        peer.election.current_session.master_node_id,
                        peer.election.current_session.election_clock,
                    )
                    for peer in peers.values()
                }
            )
            > 1
        )
        if newer_isolated_round:
            # One member sees a transient bootstrap peer. After that peer goes
            # away the physical graph is again the same three connected nodes,
            # but this member has completed a higher round that nobody heard.
            isolated = peers["Z"]
            for connected in (True, False):
                previous = isolated.election.clock
                await isolated.connections.send(
                    ConnectionMessage(node_id=NodeId("transient"), connected=connected)
                )
                await _until(
                    lambda previous=previous: isolated.election.clock > previous
                )
                await _until(settled)
        clocks = {name: peer.election.clock for name, peer in peers.items()}
        assert dropped >= 2
        blackout = False
        await _until(
            lambda: len(
                {
                    (
                        peer.election.current_session.master_node_id,
                        peer.election.current_session.election_clock,
                    )
                    for peer in peers.values()
                }
            )
            == 1
            and settled()
        )
        if not newer_isolated_round:
            assert clocks == {name: peer.election.clock for name, peer in peers.items()}
        snapshot = {
            name: (
                peer.election.clock,
                peer.election.seniority,
                peer.election.current_session,
                len(results[name]),
            )
            for name, peer in peers.items()
        }
        await anyio.sleep(0.15)
        assert snapshot == {
            name: (
                peer.election.clock,
                peer.election.seniority,
                peer.election.current_session,
                len(results[name]),
            )
            for name, peer in peers.items()
        }
        tasks.cancel_scope.cancel()


@pytest.mark.anyio
async def test_old_vote_traffic_does_not_starve_original_winner_republication() -> None:
    """Repair uses the immutable winning vote even with a continually busy inbox."""
    peer = _peer("ME")
    winning = ElectionMessage(
        clock=5,
        seniority=20,
        proposed_session=SessionId(master_node_id=NodeId("PEER"), election_clock=5),
        commands_seen=17,
    )
    older = winning.model_copy(update={"clock": 4})
    repeated: list[ElectionMessage] = []

    async def observe() -> None:
        with peer.outgoing as published:
            async for message in published:
                if message == winning:
                    repeated.append(message)

    async def old_traffic() -> None:
        while True:
            await peer.incoming.send(older)
            await anyio.sleep(0.001)

    async with create_task_group() as tasks:
        tasks.start_soon(peer.election.run)
        tasks.start_soon(observe)
        await peer.connections.send(
            ConnectionMessage(node_id=NodeId("PEER"), connected=True)
        )
        with fail_after(2):
            while (await peer.results.receive()).won_clock != 1:
                pass
        await peer.incoming.send(winning)
        with fail_after(2):
            while (await peer.results.receive()).won_clock != 5:
                pass
        tasks.start_soon(old_traffic)
        await _until(lambda: len(repeated) >= 2)
        assert peer.election.current_session == winning.proposed_session
        assert peer.election.clock == 5
        assert all(message == winning for message in repeated)
        tasks.cancel_scope.cancel()


@pytest.mark.anyio
async def test_republication_never_sends_previous_result_during_new_campaign() -> None:
    """A new round fences the previous completed vote until the new result commits."""
    peer = _peer("ME")
    async with create_task_group() as tasks:
        tasks.start_soon(peer.election.run)
        await peer.connections.send(
            ConnectionMessage(node_id=NodeId("PEER"), connected=True)
        )
        with fail_after(2):
            while (await peer.results.receive()).won_clock != 1:
                pass
        await peer.incoming.send(
            ElectionMessage(
                clock=6,
                seniority=0,
                proposed_session=SessionId(
                    master_node_id=NodeId("PEER"), election_clock=6
                ),
                commands_seen=0,
            )
        )
        with fail_after(2):
            while (await peer.outgoing.receive()).clock != 6:
                pass
        with move_on_after(0.05):
            unexpected = await peer.outgoing.receive()
            pytest.fail(f"previous vote escaped during a new campaign: {unexpected}")
        tasks.cancel_scope.cancel()


@pytest.mark.anyio
async def test_peerless_node_stays_quiet_and_closed_inputs_stop_cleanly() -> None:
    """A singleton does not create repair traffic or keep shutdown alive."""
    peer = _peer("ME")
    with fail_after(2):
        async with create_task_group() as tasks:
            tasks.start_soon(peer.election.run)
            result = await peer.results.receive()
            assert result.won_clock == 0
            # Startup sends the original vote twice, independently of repair.
            assert (await peer.outgoing.receive()).clock == 0
            assert (await peer.outgoing.receive()).clock == 0
            with move_on_after(0.1):
                unexpected = await peer.outgoing.receive()
                pytest.fail(f"peerless node republished a completed vote: {unexpected}")
            peer.incoming.close()
            peer.connections.close()
            peer.commands.close()
