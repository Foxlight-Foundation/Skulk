import pytest

from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, IndexedEvent
from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, MiscData
from exo.worker.main import Worker


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
