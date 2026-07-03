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
async def test_forward_info_emits_connectivity_only_on_change() -> None:
    # Connectivity readings ride the ordered event log; forwarding an unchanged
    # one every poll filled the master's replay tail and stormed the AMD nodes on
    # join (the 5-node gossip storm). The worker must forward a connectivity
    # reading only when its value changed.
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

    await info_sender.send(net("10.0.0.1"))  # first value -> emit
    await info_sender.send(net("10.0.0.1"))  # unchanged -> skip
    await info_sender.send(net("10.0.0.1"))  # unchanged -> skip
    await info_sender.send(net("10.0.0.2"))  # changed -> emit
    await info_sender.send(net("10.0.0.2"))  # unchanged -> skip
    info_sender.close()

    await worker._forward_info(info_receiver)  # pyright: ignore[reportPrivateUsage]

    net_events = [
        e
        for e in event_receiver.collect()
        if isinstance(e, NodeGatheredInfo)
        and isinstance(e.info, NodeNetworkInterfaces)
    ]
    assert len(net_events) == 2, "only the two distinct connectivity values emit"

    indexed_event_sender.close()
    command_sender.close()
    download_sender.close()
