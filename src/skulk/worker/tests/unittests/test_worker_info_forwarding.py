import pytest

from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import Event, IndexedEvent, NodeGatheredInfo
from skulk.shared.types.profiling import NetworkInterfaceInfo
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MiscData,
    NodeNetworkInterfaces,
)
from skulk.worker.main import Worker


@pytest.mark.asyncio
async def test_forward_info_ignores_closed_event_sender() -> None:
    indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    info_sender, info_receiver = channel[GatheredInfo]()

    worker = Worker(
        node_id=NodeId("node-a"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
    )

    event_sender.close()
    await info_sender.send(MiscData(friendly_name="kite3"))
    info_sender.close()

    await worker._forward_info(info_receiver)  # pyright: ignore[reportPrivateUsage]

    indexed_event_sender.close()
    command_sender.close()
    download_sender.close()


@pytest.mark.asyncio
async def test_forward_info_gates_connectivity_on_confirmed_echo() -> None:
    # Connectivity readings ride the ordered event log; forwarding an unchanged
    # one every poll filled the master's replay tail and stormed the AMD nodes on
    # join (the 5-node gossip storm). The worker must skip a reading only when
    # the master has CONFIRMED (echoed back indexed) that exact payload; an
    # unconfirmed reading keeps re-sending each poll, because the delivery retry
    # is bounded and a change dropped during a masterless window would otherwise
    # be lost forever (the node's topology would stay invisible until restart).
    indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, event_receiver = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    info_sender, info_receiver = channel[GatheredInfo]()

    worker = Worker(
        node_id=NodeId("node-a"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
    )

    def net(*addrs: str) -> NodeNetworkInterfaces:
        return NodeNetworkInterfaces(
            ifaces=[NetworkInterfaceInfo(name="en0", ip_address=a) for a in addrs]
        )

    def collected_net_events() -> list[NodeGatheredInfo]:
        return [
            e
            for e in event_receiver.collect()
            if isinstance(e, NodeGatheredInfo)
            and isinstance(e.info, NodeNetworkInterfaces)
        ]

    # Phase 1: nothing confirmed yet -> every poll re-sends, even unchanged.
    await info_sender.send(net("10.0.0.1"))  # unconfirmed -> emit
    await info_sender.send(net("10.0.0.1"))  # STILL unconfirmed -> re-emit
    info_sender.close()
    await worker._forward_info(info_receiver)  # pyright: ignore[reportPrivateUsage]
    assert len(collected_net_events()) == 2, (
        "an unconfirmed reading must re-send every poll until echoed"
    )

    # Master echoes the reading back as an indexed event -> confirmed.
    worker._confirmed_forwarded_info[NodeNetworkInterfaces] = net(  # pyright: ignore[reportPrivateUsage]
        "10.0.0.1"
    )

    # Phase 2: confirmed value skips; a changed value emits again.
    info_sender2, info_receiver2 = channel[GatheredInfo]()
    await info_sender2.send(net("10.0.0.1"))  # confirmed + unchanged -> skip
    await info_sender2.send(net("10.0.0.2"))  # changed -> emit
    await info_sender2.send(net("10.0.0.2"))  # changed vs confirmed -> re-emit
    info_sender2.close()
    await worker._forward_info(info_receiver2)  # pyright: ignore[reportPrivateUsage]
    phase2 = collected_net_events()
    assert len(phase2) == 2, (
        "a confirmed unchanged reading skips; an unconfirmed change re-sends"
    )
    assert all(
        isinstance(e.info, NodeNetworkInterfaces)
        and e.info.ifaces[0].ip_address == "10.0.0.2"
        for e in phase2
    )

    indexed_event_sender.close()
    command_sender.close()
    download_sender.close()
