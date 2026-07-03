"""Master-side de-dup of unchanged connectivity NodeGatheredInfo events.

Connectivity readings (network interfaces, thunderbolt) ride the ordered event
log and double as a node's liveness heartbeat. Emitting them every poll filled the
master's bounded replay tail with redundant churn; a joining node then replayed
that burst and saturated the AMD nodes' send queues into a flap livelock (the
5-node gossip storm). The master now drops an unchanged reading from the
log/broadcast/snapshot while still refreshing last_seen so liveness is preserved.
These tests pin that behaviour.
"""

from datetime import datetime, timezone

import anyio
import pytest

from skulk.master.main import Master
from skulk.routing.router import get_node_id_keypair
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    NodeGatheredInfo,
)
from skulk.shared.types.profiling import NetworkInterfaceInfo
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import NodeNetworkInterfaces


def _network(*addrs: str) -> NodeNetworkInterfaces:
    return NodeNetworkInterfaces(
        ifaces=[NetworkInterfaceInfo(name="en0", ip_address=a) for a in addrs]
    )


@pytest.mark.asyncio
async def test_master_dedups_unchanged_connectivity_but_refreshes_last_seen() -> None:
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, _global_event_receiver = channel[GlobalForwarderEvent]()
    _command_sender, co_receiver = channel[ForwarderCommand]()
    local_event_sender, le_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    fcds, _fcdr = channel[ForwarderDownloadCommand]()
    ev_send, _ev_recv = channel[Event]()

    master = Master(
        node_id,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=fcds,
    )

    # The master's MultiSourceBuffer orders/dedups by (origin, origin_idx), so a
    # real worker sends a monotonically increasing origin_idx; reuse would be
    # dropped as a replay.
    origin_idx = 0
    worker_origin = SystemId("Worker")

    async def send_network(reading: NodeNetworkInterfaces) -> None:
        nonlocal origin_idx
        await local_event_sender.send(
            LocalForwarderEvent(
                origin_idx=origin_idx,
                origin=worker_origin,
                session=session_id,
                event=NodeGatheredInfo(
                    when=str(datetime.now(tz=timezone.utc)),
                    node_id=node_id,
                    info=reading,
                ),
            )
        )
        origin_idx += 1

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # 1) First reading is a cache miss: it is applied and logged, and it sets
        #    last_seen for the node.
        await send_network(_network("192.168.0.10"))
        for _ in range(1000):
            if node_id in master.state.last_seen:
                break
            await anyio.sleep(0.001)
        assert node_id in master.state.last_seen
        seen_after_first = master.state.last_seen[node_id]
        log_len_after_first = len(master._event_log)  # pyright: ignore[reportPrivateUsage]
        assert log_len_after_first >= 1

        # Let the master's clock advance so the keepalive bump is observable.
        await anyio.sleep(0.02)

        # 2) Identical reading: de-duped. It must NOT be appended to the log, but it
        #    MUST still refresh last_seen (the liveness heartbeat).
        await send_network(_network("192.168.0.10"))
        for _ in range(1000):
            if master.state.last_seen[node_id] > seen_after_first:
                break
            await anyio.sleep(0.001)
        assert master.state.last_seen[node_id] > seen_after_first, (
            "an unchanged connectivity reading must still refresh last_seen"
        )
        assert len(master._event_log) == log_len_after_first, (  # pyright: ignore[reportPrivateUsage]
            "an unchanged connectivity reading must not be appended to the event log"
        )

        # 3) Changed reading: cache miss again, so it is logged.
        await send_network(_network("192.168.0.11"))
        for _ in range(1000):
            if len(master._event_log) > log_len_after_first:  # pyright: ignore[reportPrivateUsage]
                break
            await anyio.sleep(0.001)
        assert len(master._event_log) == log_len_after_first + 1, (  # pyright: ignore[reportPrivateUsage]
            "a changed connectivity reading must be logged"
        )

        tg.cancel_scope.cancel()
