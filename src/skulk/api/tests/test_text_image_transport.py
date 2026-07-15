import hashlib

import anyio
import pytest
from fastapi import HTTPException

import skulk.shared.types.tasks as task_types
from skulk.api.main import API
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    TaskCancelled,
    TextGeneration,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import Receiver, Sender, channel


def _build_api() -> tuple[
    API,
    Receiver[ForwarderCommand],
    Receiver[VisionMediaPacket],
    Sender[VisionMediaPacket],
]:
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    vision_sender, vision_receiver = channel[VisionMediaPacket]()
    result_sender, result_receiver = channel[VisionMediaPacket]()
    api = API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        vision_media_packet_sender=vision_sender,
        vision_media_packet_receiver=result_receiver,
    )
    return api, command_receiver, vision_receiver, result_sender


def _task_params(image: str) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        input=[InputMessage(role="user", content="describe this image")],
        images=[image],
    )


@pytest.mark.asyncio
async def test_text_image_transport_resends_images_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKULK_TEXT_IMAGE_HASH_CACHE", raising=False)
    api, receiver, vision_receiver, _ = _build_api()
    image = "aGVsbG8="

    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]
    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]

    messages = await receiver.receive_at_least(2)
    commands = [message.command for message in messages]
    assert isinstance(commands[0], TextGeneration)
    assert isinstance(commands[1], TextGeneration)

    for command in commands:
        pending = api._pending_vision_media[command.command_id]  # pyright: ignore[reportPrivateUsage]
        await api._send_vision_media_to_target(  # pyright: ignore[reportPrivateUsage]
            command.command_id,
            pending,
            NodeId("worker-node"),
        )

    packets = await vision_receiver.receive_at_least(6)
    openings = [packet for packet in packets if packet.kind == "opened"]
    chunks = [packet for packet in packets if packet.kind == "chunk"]
    completions = [packet for packet in packets if packet.kind == "completed"]
    assert len(openings) == 2
    assert [chunk.data.decode("ascii") for chunk in chunks] == [image, image]
    assert len(completions) == 2
    first_generation = commands[0].task_params
    second_generation = commands[1].task_params
    assert first_generation.image_hashes == {}
    assert second_generation.image_hashes == {}
    assert first_generation.total_input_chunks == 1
    assert second_generation.total_input_chunks == 1


def test_vision_media_admission_rejects_excess_pending_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skulk.api.main._VISION_MEDIA_PENDING_COMMANDS", 1)
    api, _, _, _ = _build_api()
    model = ModelId("org/vlm")
    api._stage_vision_media(  # pyright: ignore[reportPrivateUsage]
        CommandId("first-command"),
        model,
        [(0, "aGVsbG8=")],
        1,
    )
    diagnostics = api._vision_media_ingress_diagnostics()  # pyright: ignore[reportPrivateUsage]
    assert diagnostics.pending_api_commands == 1
    assert diagnostics.pending_api_bytes == len(b"aGVsbG8=")

    with pytest.raises(HTTPException, match="admission capacity is exhausted") as error:
        api._stage_vision_media(  # pyright: ignore[reportPrivateUsage]
            CommandId("second-command"),
            model,
            [(0, "d29ybGQ=")],
            1,
        )
    assert error.value.status_code == 503


def test_vision_media_admission_rejects_excess_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skulk.api.main._VISION_MEDIA_PENDING_FRAMES", 1)
    api, _, _, _ = _build_api()
    command_id = CommandId("too-many-frames")

    with pytest.raises(HTTPException, match="media frame limit") as error:
        api._stage_vision_media(  # pyright: ignore[reportPrivateUsage]
            command_id,
            ModelId("org/vlm"),
            [(0, "YQ=="), (1, "Yg==")],
            2,
        )

    assert error.value.status_code == 413
    assert command_id not in api._pending_vision_media  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_text_image_transport_targets_authoritative_task_participants() -> None:
    api, command_receiver, vision_receiver, _ = _build_api()
    image = "aGVsbG8="
    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]
    forwarded = await command_receiver.receive()
    assert isinstance(forwarded.command, TextGeneration)
    command = forwarded.command
    model_id = command.task_params.model
    runner_id = RunnerId("runner-1")
    runner_id_2 = RunnerId("runner-2")
    target_node = NodeId("selected-worker")
    target_node_2 = NodeId("selected-worker-2")
    instance_id = InstanceId("selected-instance")
    card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(100),
        n_layers=2,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    api.state = State(
        instances={
            instance_id: MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=ShardAssignments(
                    model_id=model_id,
                    runner_to_shard={
                        runner_id: PipelineShardMetadata(
                            model_card=card,
                            device_rank=0,
                            world_size=2,
                            start_layer=0,
                            end_layer=1,
                            n_layers=2,
                        ),
                        runner_id_2: PipelineShardMetadata(
                            model_card=card,
                            device_rank=1,
                            world_size=2,
                            start_layer=1,
                            end_layer=2,
                            n_layers=2,
                        ),
                    },
                    node_to_runner={
                        target_node: runner_id,
                        target_node_2: runner_id_2,
                    },
                ),
                hosts_by_node={target_node: [], target_node_2: []},
                ephemeral_port=52415,
            )
        }
    )
    task = task_types.TextGeneration(
        instance_id=instance_id,
        command_id=command.command_id,
        owner_node=NodeId("api-node"),
        task_params=command.task_params,
    )
    packets: list[VisionMediaPacket] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        api._dispatch_pending_vision_media(task)  # pyright: ignore[reportPrivateUsage]
        packets = await vision_receiver.receive_at_least(6)
        task_group.cancel_scope.cancel()

    assert {packet.target_node for packet in packets} == {
        target_node,
        target_node_2,
    }
    for target in (target_node, target_node_2):
        assert [packet.kind for packet in packets if packet.target_node == target] == [
            "opened",
            "chunk",
            "completed",
        ]


@pytest.mark.asyncio
async def test_vision_media_requires_acceptance_from_every_selected_worker() -> None:
    api, _, _, result_sender = _build_api()
    command_id = CommandId("acknowledged-command")
    model = ModelId("org/vlm")
    targets = (NodeId("worker-one"), NodeId("worker-two"))
    api._vision_media_commands.add(command_id)  # pyright: ignore[reportPrivateUsage]
    api._vision_media_models[command_id] = model  # pyright: ignore[reportPrivateUsage]
    api._vision_media_targets[command_id] = targets  # pyright: ignore[reportPrivateUsage]
    api._vision_media_pending_acks[command_id] = set(targets)  # pyright: ignore[reportPrivateUsage]
    api._vision_media_ack_deadlines[command_id] = 100.0  # pyright: ignore[reportPrivateUsage]
    api._active_vision_media_bytes[command_id] = 8  # pyright: ignore[reportPrivateUsage]
    api._active_vision_media_total_bytes = 8  # pyright: ignore[reportPrivateUsage]

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(api._apply_vision_media_transport)  # pyright: ignore[reportPrivateUsage]
        for target in targets:
            await result_sender.send(
                VisionMediaPacket(
                    source_node=target,
                    target_node=NodeId("api-node"),
                    command_id=command_id,
                    model=model,
                    sequence=2,
                    kind="accepted",
                )
            )
        while command_id in api._vision_media_pending_acks:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert command_id not in api._vision_media_ack_deadlines  # pyright: ignore[reportPrivateUsage]
    assert api._active_vision_media_total_bytes == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_vision_media_missing_worker_acknowledgement_fails_request() -> None:
    api, command_receiver, _, _ = _build_api()
    command_id = CommandId("unacknowledged-command")
    model = ModelId("org/vlm")
    target = NodeId("worker-node")
    api._vision_media_commands.add(command_id)  # pyright: ignore[reportPrivateUsage]
    api._vision_media_models[command_id] = model  # pyright: ignore[reportPrivateUsage]
    api._vision_media_targets[command_id] = (target,)  # pyright: ignore[reportPrivateUsage]
    api._vision_media_pending_acks[command_id] = {target}  # pyright: ignore[reportPrivateUsage]
    api._vision_media_ack_deadlines[command_id] = 1.0  # pyright: ignore[reportPrivateUsage]
    api._active_vision_media_bytes[command_id] = 8  # pyright: ignore[reportPrivateUsage]
    api._active_vision_media_total_bytes = 8  # pyright: ignore[reportPrivateUsage]

    await api._expire_stale_vision_media(2.0)  # pyright: ignore[reportPrivateUsage]

    cancelled = await command_receiver.receive()
    assert isinstance(cancelled.command, TaskCancelled)
    assert cancelled.command.cancelled_command_id == command_id
    assert command_id not in api._vision_media_commands  # pyright: ignore[reportPrivateUsage]
    assert command_id not in api._vision_media_pending_acks  # pyright: ignore[reportPrivateUsage]
    assert api._active_vision_media_total_bytes == 0  # pyright: ignore[reportPrivateUsage]
    failure = api._vision_media_failures[command_id]  # pyright: ignore[reportPrivateUsage]
    assert "worker verification" in failure.error_message


@pytest.mark.asyncio
async def test_text_image_hash_cache_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_TEXT_IMAGE_HASH_CACHE", "1")
    api, receiver, vision_receiver, _ = _build_api()
    image = "aGVsbG8="
    image_hash = hashlib.sha256(image.encode("ascii")).hexdigest()

    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]
    await api._send_text_generation_with_images(_task_params(image))  # pyright: ignore[reportPrivateUsage]

    messages = await receiver.receive_at_least(2)
    commands = [message.command for message in messages]
    assert isinstance(commands[0], TextGeneration)
    assert isinstance(commands[1], TextGeneration)
    pending = api._pending_vision_media[commands[0].command_id]  # pyright: ignore[reportPrivateUsage]
    await api._send_vision_media_to_target(  # pyright: ignore[reportPrivateUsage]
        commands[0].command_id,
        pending,
        NodeId("worker-node"),
    )
    packets = await vision_receiver.receive_at_least(3)
    assert packets[0].kind == "opened"
    assert packets[1].data.decode("ascii") == image
    assert packets[2].kind == "completed"
    assert commands[1].task_params.image_hashes == {0: image_hash}
    assert commands[1].task_params.total_input_chunks == 0
