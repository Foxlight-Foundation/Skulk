# pyright: reportPrivateUsage=false
"""A request an engine package cannot carry is refused with a 400 before dispatch.

A runner-side refusal reaches the caller as a 200 with an error object in the
body, because the status is committed before the body streams. The shared
dispatch point therefore refuses first, and only when nothing but engine
packages serves the model.
"""

import pytest
from fastapi import HTTPException

from skulk.api.main import API
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    TextGeneration,
)
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import Receiver, channel

_MODEL = ModelId("org/chat")
_TOOL = {"type": "function", "function": {"name": "get_time", "parameters": {}}}


def _instance(name: str, backend: str | None) -> MlxRingInstance:
    card = ModelCard(
        model_id=_MODEL,
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    runner = RunnerId(name)
    return MlxRingInstance(
        instance_id=InstanceId(name),
        shard_assignments=ShardAssignments(
            model_id=_MODEL,
            node_to_runner={NodeId(name): runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                    resolved_backend=backend,
                )
            },
        ),
        hosts_by_node={NodeId(name): []},
        ephemeral_port=52416,
    )


def _api(*instances: MlxRingInstance) -> tuple[API, Receiver[ForwarderCommand]]:
    command_sender, command_receiver = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    vision_sender, _ = channel[VisionMediaPacket]()
    _, result_receiver = channel[VisionMediaPacket]()
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
    api.state = State(
        instances={instance.instance_id: instance for instance in instances}
    )
    return api, command_receiver


def _params(**fields: object) -> TextGenerationTaskParams:
    return TextGenerationTaskParams.model_validate(
        {
            "model": _MODEL,
            "input": [InputMessage(role="user", content="hello")],
            **fields,
        }
    )


@pytest.mark.parametrize(
    ("fields", "word"),
    [
        ({"tools": [_TOOL]}, "tool calls"),
        ({"images": ["aGVsbG8="]}, "images"),
        ({"logprobs": True}, "logprobs"),
    ],
)
async def test_a_model_only_packages_serve_refuses_with_400_and_sends_nothing(
    fields: dict[str, object], word: str
) -> None:
    api, commands = _api(_instance("package", "tensorfold-rocm"))

    with pytest.raises(HTTPException) as refusal:
        await api._send_text_generation_with_images(_params(**fields))

    assert refusal.value.status_code == 400
    assert word in str(refusal.value.detail)
    assert commands.collect() == []


async def test_a_text_request_to_a_package_is_dispatched() -> None:
    api, commands = _api(_instance("package", "tensorfold-rocm"))

    await api._send_text_generation_with_images(_params())

    (sent,) = commands.collect()
    assert isinstance(sent.command, TextGeneration)


@pytest.mark.parametrize(
    "instances",
    [
        [],
        [("built-in", "mlx-metal")],
        [("built-in", "llama_server-vulkan"), ("package", "tensorfold-rocm")],
        [("unstamped", None)],
    ],
)
async def test_a_built_in_engine_or_no_instance_leaves_dispatch_unchanged(
    instances: list[tuple[str, str | None]],
) -> None:
    """With a built-in engine in reach, or nothing to judge by, nothing is refused."""
    api, commands = _api(*(_instance(name, backend) for name, backend in instances))

    await api._send_text_generation_with_images(_params(tools=[_TOOL]))

    (sent,) = commands.collect()
    assert isinstance(sent.command, TextGeneration)
    assert sent.command.task_params.tools == [_TOOL]


async def test_a_request_aimed_at_one_instance_is_judged_by_that_instance() -> None:
    api, commands = _api(
        _instance("built-in", "mlx-metal"), _instance("package", "tensorfold-rocm")
    )

    with pytest.raises(HTTPException) as refusal:
        await api._send_text_generation_with_images(
            _params(tools=[_TOOL]), target_instance_id=InstanceId("package")
        )
    assert refusal.value.status_code == 400
    assert commands.collect() == []

    await api._send_text_generation_with_images(
        _params(tools=[_TOOL]), target_instance_id=InstanceId("built-in")
    )
    (sent,) = commands.collect()
    assert isinstance(sent.command, TextGeneration)
    assert sent.command.target_instance_id == InstanceId("built-in")
