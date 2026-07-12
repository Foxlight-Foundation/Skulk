import hashlib

import anyio
import pytest

from skulk.routing.realtime_audio import RealtimeAudioPacket
from skulk.routing.speech_media import SpeechMediaPacket
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    TaskCancelled,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import Event, IndexedEvent, NodeGatheredInfo
from skulk.shared.types.profiling import NetworkInterfaceInfo
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MiscData,
    NodeNetworkInterfaces,
)
from skulk.worker import main as worker_main
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
async def test_realtime_audio_janitor_expires_undispatched_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-dispatch input cannot remain retained after its admission stalls."""

    indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=NodeId("node-a"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
    )
    command_id = CommandId("never-dispatched")
    frame = RealtimeAudioInputFrame(
        command_id=command_id,
        sequence=0,
        kind="chunk",
        data=b"\x00\x00",
    )
    worker._realtime_audio_pending[command_id] = [frame]  # pyright: ignore[reportPrivateUsage]
    worker._realtime_audio_pending_bytes[command_id] = 2  # pyright: ignore[reportPrivateUsage]
    worker._realtime_audio_pending_since[command_id] = 0.0  # pyright: ignore[reportPrivateUsage]
    original_sleep = anyio.sleep
    sleep_count = 0

    async def immediate_first_sleep(_seconds: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count > 1:
            await original_sleep(60)

    monkeypatch.setattr(worker_main.anyio, "sleep", immediate_first_sleep)
    command: ForwarderCommand | None = None
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._realtime_audio_janitor)  # pyright: ignore[reportPrivateUsage]
        command = await command_receiver.receive()
        task_group.cancel_scope.cancel()

    assert command is not None
    assert isinstance(command.command, TaskCancelled)
    assert command.command.cancelled_command_id == command_id
    assert command_id not in worker._realtime_audio_pending  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._realtime_audio_pending_bytes  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._realtime_audio_pending_since  # pyright: ignore[reportPrivateUsage]
    indexed_event_sender.close()
    event_sender.close()
    command_sender.close()
    download_sender.close()


@pytest.mark.asyncio
async def test_remote_realtime_audio_packet_enters_shared_pending_buffer() -> None:
    """A node-addressed remote packet follows the existing bounded ingress path."""

    indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    packet_sender, packet_receiver = channel[RealtimeAudioPacket](4)
    worker = Worker(
        node_id=NodeId("worker-node"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        realtime_audio_packet_receiver=packet_receiver,
    )
    command_id = CommandId("remote-command")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._realtime_audio_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(
            RealtimeAudioPacket(
                source_node=NodeId("api-node"),
                target_node=NodeId("worker-node"),
                command_id=command_id,
                sequence=1,
                kind="chunk",
                data=b"\x00\x00",
            )
        )
        while command_id not in worker._realtime_audio_pending:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    pending = worker._realtime_audio_pending[command_id]  # pyright: ignore[reportPrivateUsage]
    assert len(pending) == 1
    assert pending[0].sequence == 1
    assert pending[0].data == b"\x00\x00"
    indexed_event_sender.close()
    event_sender.close()
    command_sender.close()
    download_sender.close()


@pytest.mark.asyncio
async def test_remote_realtime_audio_packet_for_other_node_is_ignored() -> None:
    """Broadcast fallback delivery cannot make a worker consume another target."""

    _, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    packet_sender, packet_receiver = channel[RealtimeAudioPacket](4)
    worker = Worker(
        node_id=NodeId("worker-node"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        realtime_audio_packet_receiver=packet_receiver,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._realtime_audio_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(
            RealtimeAudioPacket(
                source_node=NodeId("api-node"),
                target_node=NodeId("different-worker"),
                command_id=CommandId("wrong-target"),
                sequence=1,
                kind="chunk",
                data=b"\x00\x00",
            )
        )
        await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert worker._realtime_audio_pending == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_speech_media_packet_is_bounded_and_completed_off_state() -> None:
    """Reference media is retained only in worker-local request buffers."""

    _, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    packet_sender, packet_receiver = channel[SpeechMediaPacket](4)
    worker = Worker(
        node_id=NodeId("worker-node"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        speech_media_packet_receiver=packet_receiver,
    )
    command_id = CommandId("reference-command")
    payload = b"RIFF-reference"

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._speech_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(
            SpeechMediaPacket(
                source_node=NodeId("api-node"),
                target_node=NodeId("worker-node"),
                command_id=command_id,
                sequence=0,
                kind="chunk",
                data=payload,
            )
        )
        await packet_sender.send(
            SpeechMediaPacket(
                source_node=NodeId("api-node"),
                target_node=NodeId("worker-node"),
                command_id=command_id,
                sequence=1,
                kind="completed",
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        while command_id not in worker._speech_media_completed:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert worker._speech_media_chunks[command_id][0].data == payload  # pyright: ignore[reportPrivateUsage]
    assert command_id in worker._speech_media_completed  # pyright: ignore[reportPrivateUsage]
    assert worker.state.tasks == {}


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
