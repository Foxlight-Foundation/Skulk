import hashlib

import anyio
import pytest

import skulk.shared.types.tasks as task_types
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import InputImageChunk
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import Event, IndexedEvent, TaskFailed
from skulk.shared.types.state import State
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.channels import Receiver, Sender, channel
from skulk.worker.main import Worker


def _worker_with_vision_transport() -> tuple[
    Worker,
    Sender[VisionMediaPacket],
    Receiver[VisionMediaPacket],
]:
    _, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    packet_sender, packet_receiver = channel[VisionMediaPacket](8)
    failure_sender, failure_receiver = channel[VisionMediaPacket](8)
    worker = Worker(
        node_id=NodeId("worker-node"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        vision_media_packet_sender=failure_sender,
        vision_media_packet_receiver=packet_receiver,
    )
    return worker, packet_sender, failure_receiver


def _packet(
    *,
    command_id: CommandId,
    sequence: int,
    data: bytes,
    image_index: int,
    total_chunks: int,
) -> VisionMediaPacket:
    return VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=sequence,
        kind="chunk",
        data=data,
        image_index=image_index,
        total_chunks=total_chunks,
    )


def _open(command_id: CommandId, total_chunks: int) -> VisionMediaPacket:
    return VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=0,
        kind="opened",
        total_chunks=total_chunks,
        image_count=1,
    )


@pytest.mark.asyncio
async def test_vision_media_is_exposed_only_after_verified_completion() -> None:
    worker, packet_sender, _ = _worker_with_vision_transport()
    command_id = CommandId("verified-vision")
    chunks = (b"aGVs", b"bG8=")
    completion = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=3,
        kind="completed",
        total_chunks=2,
        image_count=1,
        sha256=hashlib.sha256(b"".join(chunks)).hexdigest(),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._vision_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(_open(command_id, 2))
        await packet_sender.send(completion)
        assert command_id not in worker.input_chunk_buffer
        await packet_sender.send(
            _packet(
                command_id=command_id,
                sequence=2,
                data=chunks[1],
                image_index=0,
                total_chunks=2,
            )
        )
        assert command_id not in worker.input_chunk_buffer
        await packet_sender.send(
            _packet(
                command_id=command_id,
                sequence=1,
                data=chunks[0],
                image_index=0,
                total_chunks=2,
            )
        )
        while command_id not in worker._vision_media_verified_chunks:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert command_id not in worker.input_chunk_buffer
    assert "".join(
        worker._vision_media_verified_chunks[command_id][index].data  # pyright: ignore[reportPrivateUsage]
        for index in range(2)
    ) == "aGVsbG8="
    assert command_id in worker._vision_media_verified  # pyright: ignore[reportPrivateUsage]
    assert worker.state.tasks == {}
    diagnostics = worker.collect_vision_media_ingress_diagnostics()
    assert diagnostics.active_streams == 1
    assert diagnostics.pending_frames == 0
    assert diagnostics.retained_bytes == len(b"".join(chunks))
    assert diagnostics.verified_streams == 1
    assert diagnostics.completed_streams == 1


@pytest.mark.asyncio
async def test_corrupt_vision_media_returns_source_routed_failure() -> None:
    worker, packet_sender, failure_receiver = _worker_with_vision_transport()
    command_id = CommandId("corrupt-vision")
    chunk = _packet(
        command_id=command_id,
        sequence=1,
        data=b"aGVsbG8=",
        image_index=0,
        total_chunks=1,
    )
    completion = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256="0" * 64,
    )
    failure: VisionMediaPacket | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._vision_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(_open(command_id, 1))
        await packet_sender.send(chunk)
        await packet_sender.send(completion)
        failure = await failure_receiver.receive()
        task_group.cancel_scope.cancel()

    assert failure is not None
    assert failure.kind == "transport_failed"
    assert failure.target_node == NodeId("api-node")
    assert "SHA-256" in (failure.error_message or "")
    assert command_id not in worker.input_chunk_buffer
    diagnostics = worker.collect_vision_media_ingress_diagnostics()
    assert diagnostics.active_streams == 0
    assert diagnostics.retained_bytes == 0
    assert diagnostics.rejected_streams == 1


@pytest.mark.asyncio
async def test_vision_media_rejection_race_emits_one_task_failure() -> None:
    worker, _, _ = _worker_with_vision_transport()
    command_id = CommandId("racing-rejection")
    task = task_types.TextGeneration(
        instance_id=InstanceId("vision-instance"),
        command_id=command_id,
        owner_node=NodeId("api-node"),
        task_params=TextGenerationTaskParams(
            model=ModelId("org/vlm"),
            input=[InputMessage(role="user", content="describe")],
            total_input_chunks=1,
            image_count=1,
        ),
    )
    event_sender, event_receiver = channel[Event](2)
    worker.event_sender = event_sender
    blocked_sender, blocked_receiver = channel[VisionMediaPacket](0)
    worker._vision_media_packet_sender = blocked_sender  # pyright: ignore[reportPrivateUsage]
    rejection_finished = anyio.Event()

    async def reject() -> None:
        await worker._reject_vision_media(  # pyright: ignore[reportPrivateUsage]
            _open(command_id, 1), "checksum mismatch"
        )
        rejection_finished.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(reject)
        while command_id not in worker._vision_media_failures:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0)

        worker.state = State(tasks={task.task_id: task})
        pending_failure = worker._vision_media_failures.pop(command_id)  # pyright: ignore[reportPrivateUsage]
        worker._vision_media_failure_since.pop(command_id)  # pyright: ignore[reportPrivateUsage]
        await event_sender.send(
            TaskFailed(
                task_id=task.task_id,
                error_type="invalid_vision_media",
                error_message=pending_failure,
            )
        )
        assert (await blocked_receiver.receive()).kind == "transport_failed"
        await rejection_finished.wait()
        task_group.cancel_scope.cancel()

    failures = event_receiver.collect()
    assert len(failures) == 1


@pytest.mark.asyncio
async def test_vision_media_rejects_declared_frame_count_over_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skulk.worker.main._VISION_MEDIA_PENDING_FRAMES", 1)
    worker, packet_sender, failure_receiver = _worker_with_vision_transport()
    command_id = CommandId("oversized-vision")
    failure: VisionMediaPacket | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._vision_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(_open(command_id, 2))
        failure = await failure_receiver.receive()
        task_group.cancel_scope.cancel()

    assert failure is not None
    assert failure.kind == "transport_failed"
    assert "open exceeds declared bounds" in (failure.error_message or "")
    assert worker.collect_vision_media_ingress_diagnostics().rejected_streams == 1


@pytest.mark.asyncio
async def test_vision_media_buffers_pre_open_frames_until_open_arrives() -> None:
    worker, packet_sender, _ = _worker_with_vision_transport()
    command_id = CommandId("reordered-open")
    payload = b"aGVsbG8="
    completion = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._vision_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(completion)
        await packet_sender.send(
            _packet(
                command_id=command_id,
                sequence=1,
                data=payload,
                image_index=0,
                total_chunks=1,
            )
        )
        assert command_id not in worker._vision_media_verified  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(_open(command_id, 1))
        while command_id not in worker._vision_media_verified_chunks:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert command_id in worker._vision_media_verified  # pyright: ignore[reportPrivateUsage]
    assert worker.collect_vision_media_ingress_diagnostics().rejected_streams == 0


@pytest.mark.asyncio
async def test_vision_media_acknowledges_authoritative_task_owner() -> None:
    worker, packet_sender, acknowledgement_receiver = (
        _worker_with_vision_transport()
    )
    command_id = CommandId("acknowledged-vision")
    payload = b"aGVsbG8="
    task = task_types.TextGeneration(
        instance_id=InstanceId("vision-instance"),
        command_id=command_id,
        owner_node=NodeId("api-node"),
        task_params=TextGenerationTaskParams(
            model=ModelId("org/vlm"),
            input=[InputMessage(role="user", content="describe")],
            total_input_chunks=1,
            image_count=1,
        ),
    )
    worker.state = State(tasks={task.task_id: task})
    acknowledgement: VisionMediaPacket | None = None
    completion = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._vision_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(_open(command_id, 1))
        await packet_sender.send(
            _packet(
                command_id=command_id,
                sequence=1,
                data=payload,
                image_index=0,
                total_chunks=1,
            )
        )
        await packet_sender.send(completion)
        acknowledgement = await acknowledgement_receiver.receive()
        task_group.cancel_scope.cancel()

    assert acknowledgement is not None
    assert acknowledgement.kind == "accepted"
    assert acknowledgement.source_node == NodeId("worker-node")
    assert acknowledgement.target_node == NodeId("api-node")
    assert command_id in worker._vision_media_accepted  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._vision_media_pending_since  # pyright: ignore[reportPrivateUsage]
    assert command_id in worker.input_chunk_buffer
    diagnostics = worker.collect_vision_media_ingress_diagnostics()
    assert diagnostics.active_streams == 1
    assert diagnostics.retained_bytes == len(payload)


@pytest.mark.asyncio
async def test_vision_media_accepts_sparse_new_index_with_cached_image() -> None:
    worker, packet_sender, acknowledgement_receiver = (
        _worker_with_vision_transport()
    )
    command_id = CommandId("cached-and-new-vision")
    payload = b"aGVsbG8="
    task = task_types.TextGeneration(
        instance_id=InstanceId("vision-instance"),
        command_id=command_id,
        owner_node=NodeId("api-node"),
        task_params=TextGenerationTaskParams(
            model=ModelId("org/vlm"),
            input=[InputMessage(role="user", content="compare")],
            image_hashes={0: "0" * 64},
            total_input_chunks=1,
            image_count=1,
        ),
    )
    worker.state = State(tasks={task.task_id: task})
    completion = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    acknowledgement: VisionMediaPacket | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._vision_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(_open(command_id, 1))
        await packet_sender.send(
            _packet(
                command_id=command_id,
                sequence=1,
                data=payload,
                image_index=1,
                total_chunks=1,
            )
        )
        await packet_sender.send(completion)
        acknowledgement = await acknowledgement_receiver.receive()
        task_group.cancel_scope.cancel()

    assert acknowledgement is not None
    assert acknowledgement.kind == "accepted"


@pytest.mark.asyncio
async def test_accepted_vision_media_ignores_late_duplicate_frames() -> None:
    worker, packet_sender, _ = _worker_with_vision_transport()
    command_id = CommandId("already-running")
    probe_command_id = CommandId("ingress-probe")
    payload = b"aGVsbG8="
    worker._vision_media_accepted.add(command_id)  # pyright: ignore[reportPrivateUsage]
    completion = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker._vision_media_packet_ingress)  # pyright: ignore[reportPrivateUsage]
        await packet_sender.send(_open(command_id, 1))
        await packet_sender.send(
            _packet(
                command_id=command_id,
                sequence=1,
                data=payload,
                image_index=0,
                total_chunks=1,
            )
        )
        await packet_sender.send(completion)
        await packet_sender.send(_open(probe_command_id, 1))
        while probe_command_id not in worker._vision_media_pending_since:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert command_id not in worker._vision_media_pending_since  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._vision_media_opened  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._vision_media_chunks  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._vision_media_completed  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._vision_media_verified  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_vision_media_acknowledgement_can_retry_after_send_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient full egress path does not strand verified media forever."""

    monkeypatch.setattr(
        "skulk.worker.main._VISION_MEDIA_ACK_SEND_TIMEOUT_SECONDS", 0.01
    )
    worker, _, _ = _worker_with_vision_transport()
    command_id = CommandId("retry-acknowledgement")
    task = task_types.TextGeneration(
        instance_id=InstanceId("vision-instance"),
        command_id=command_id,
        owner_node=NodeId("api-node"),
        task_params=TextGenerationTaskParams(
            model=ModelId("org/vlm"),
            input=[InputMessage(role="user", content="describe")],
            total_input_chunks=1,
            image_count=1,
        ),
    )
    completion = VisionMediaPacket(
        source_node=NodeId("api-node"),
        target_node=NodeId("worker-node"),
        command_id=command_id,
        model=ModelId("org/vlm"),
        sequence=2,
        kind="completed",
        total_chunks=1,
        image_count=1,
        sha256="0" * 64,
    )
    worker.state = State(tasks={task.task_id: task})
    worker._vision_media_verified[command_id] = completion  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_verified_chunks[command_id] = {  # pyright: ignore[reportPrivateUsage]
        0: InputImageChunk(
            model=ModelId("org/vlm"),
            command_id=command_id,
            data="aGVsbG8=",
            chunk_index=0,
            total_chunks=1,
            image_index=0,
        )
    }
    blocked_sender, _ = channel[VisionMediaPacket](0)
    worker._vision_media_packet_sender = blocked_sender  # pyright: ignore[reportPrivateUsage]

    await worker._acknowledge_vision_media_if_admitted(command_id)  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._vision_media_accepted  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker._vision_media_ack_inflight  # pyright: ignore[reportPrivateUsage]
    assert command_id not in worker.input_chunk_buffer

    retry_sender, retry_receiver = channel[VisionMediaPacket](1)
    worker._vision_media_packet_sender = retry_sender  # pyright: ignore[reportPrivateUsage]
    await worker._acknowledge_vision_media_if_admitted(command_id)  # pyright: ignore[reportPrivateUsage]

    assert (await retry_receiver.receive()).kind == "accepted"
    assert command_id in worker._vision_media_accepted  # pyright: ignore[reportPrivateUsage]
    assert command_id in worker.input_chunk_buffer
