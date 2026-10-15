"""Video renders place only on instances whose card serves the resolved mode."""

from __future__ import annotations

import anyio
import pytest

from skulk.master.main import Master
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import ModelCard, ModelTask, VideoCardConfig
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    VideoGeneration,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TaskCreated,
    TaskFailed,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.tasks import TaskStatus
from skulk.shared.types.tasks import VideoGeneration as VideoGenerationTask
from skulk.shared.types.video import VideoCodecName, VideoGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerStatus,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel

MODEL = ModelId("org/video")


def _state(modes: list[str], status: RunnerStatus | None = None) -> State:
    if status is None:
        status = RunnerReady()
    card = ModelCard(
        model_id=MODEL,
        storage_size=Memory.from_mb(128),
        n_layers=4,
        hidden_size=256,
        supports_tensor=False,
        tasks=[
            ModelTask.TextToVideo if "t2va" in modes else ModelTask.ReferenceToVideo,
            *([ModelTask.ImageToVideo] if "fl2va" in modes else []),
        ],
        video=VideoCardConfig.model_validate(
            {
                "modes": modes,
                "reference_limits": {"max_images": 9} if "ref2va" in modes else None,
            }
        ),
    )
    runner_id = RunnerId("runner-video")
    shard = PipelineShardMetadata(
        model_card=card, device_rank=0, world_size=1, start_layer=0, end_layer=4, n_layers=4
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("video"),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            node_to_runner={NodeId("node-a"): runner_id},
            runner_to_shard={runner_id: shard},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    return State(instances={instance.instance_id: instance}, runners={runner_id: status})


async def _dispatch(state: State, params: VideoGenerationTaskParams, expected: int) -> list[Event]:
    node_id = NodeId(get_node_id_keypair().to_node_id())
    sender, receiver = channel[ForwarderCommand]()
    events, out = channel[Event]()
    global_sender, _ = channel[GlobalForwarderEvent]()
    _, local = channel[LocalForwarderEvent]()
    sync_sender, sync_receiver = channel[StateSyncMessage]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        event_sender=events,
        global_event_sender=global_sender,
        local_event_receiver=local,
        command_receiver=receiver,
        state_sync_receiver=sync_receiver,
        state_sync_sender=sync_sender,
        download_command_sender=downloads,
    )
    master.state = state
    command = VideoGeneration(task_params=params, owner_node=NodeId("api"))
    captured: list[Event] = []
    async with anyio.create_task_group() as group:
        group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await sender.send(ForwarderCommand(origin=SystemId("API"), command=command))
        with anyio.fail_after(2):
            captured = await out.receive_at_least(expected)
        group.cancel_scope.cancel()
    first = captured[0]
    assert isinstance(first, TaskCreated)
    assert isinstance(first.task, VideoGenerationTask)
    assert first.task.command_id == command.command_id
    assert first.task.owner_node == NodeId("api")
    return captured


async def test_text_render_places_on_serving_instance() -> None:
    params = VideoGenerationTaskParams(prompt="a fox", model=str(MODEL), seconds=5)
    events = await _dispatch(_state(["t2va", "fl2va"]), params, 1)
    created = events[0]
    assert isinstance(created, TaskCreated)
    task = created.task
    assert isinstance(task, VideoGenerationTask)
    assert task.instance_id == InstanceId("video")
    assert task.task_status == TaskStatus.Pending
    assert task.task_params.mode is not None and task.task_params.mode.value == "t2va"


async def test_unserved_mode_fails_terminally() -> None:
    params = VideoGenerationTaskParams(prompt="a fox", model=str(MODEL), seconds=5)
    events = await _dispatch(_state(["ref2va"]), params, 2)
    created = events[0]
    assert isinstance(created, TaskCreated)
    task = created.task
    assert isinstance(task, VideoGenerationTask)
    assert task.task_status == TaskStatus.Failed
    assert task.instance_id == InstanceId("video")
    failure = events[1]
    assert isinstance(failure, TaskFailed)
    assert failure.error_type == "video_mode_unavailable"
    assert "t2va" in failure.error_message


async def test_failed_runner_is_not_a_video_placement() -> None:
    """A failed rank cannot render; the job must end instead of queueing forever."""
    params = VideoGenerationTaskParams(prompt="a fox", model=str(MODEL), seconds=5)
    events = await _dispatch(_state(["t2va"], RunnerFailed()), params, 2)
    created = events[0]
    assert isinstance(created, TaskCreated)
    assert created.task.task_status == TaskStatus.Failed
    failure = events[1]
    assert isinstance(failure, TaskFailed)
    assert failure.error_type == "video_mode_unavailable"


async def test_multi_rank_instance_is_not_a_video_placement() -> None:
    state = _state(["t2va"])
    instance = state.instances[InstanceId("video")]
    runner_b = RunnerId("runner-b")
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    widened = instance.model_copy(
        update={
            "shard_assignments": ShardAssignments(
                model_id=MODEL,
                node_to_runner={NodeId("node-a"): RunnerId("runner-video"), NodeId("node-b"): runner_b},
                runner_to_shard={RunnerId("runner-video"): shard, runner_b: shard},
            )
        }
    )
    state = State(
        instances={widened.instance_id: widened},
        runners={RunnerId("runner-video"): RunnerReady(), runner_b: RunnerReady()},
    )
    params = VideoGenerationTaskParams(prompt="a fox", model=str(MODEL), seconds=5)
    events = await _dispatch(state, params, 2)
    failure = events[1]
    assert isinstance(failure, TaskFailed)
    assert failure.error_type == "video_mode_unavailable"


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("codec", [None, "h264", "av1"])
async def test_ltx_codec_is_matched_before_task_dispatch(
    native: bool, codec: VideoCodecName | None
) -> None:
    """Native rejects AV1 before dispatch while Comfy retains both codecs."""
    card = await load_fixture_card(ModelId(
        "dgrauet/LTX-2.5-distilled-mlx-q4" if native
        else "Lightricks/LTX-2.5-distilled-comfy-int8"
    ))
    assert card is not None
    runner_id = RunnerId("runner-ltx")
    instance = MlxRingInstance(
        instance_id=InstanceId("ltx"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={NodeId("node-a"): runner_id},
            runner_to_shard={runner_id: PipelineShardMetadata(
                model_card=card, device_rank=0, world_size=1,
                start_layer=0, end_layer=card.n_layers, n_layers=card.n_layers,
                resolved_backend="mlx_video-metal" if native else "comfy-cuda",
            )},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    state = State(instances={instance.instance_id: instance}, runners={runner_id: RunnerReady()})
    params = VideoGenerationTaskParams(
        prompt="a fox", model=str(card.model_id), seconds=2,
        size="768x512", codec=codec,
    )
    refused = native and codec == "av1"
    events = await _dispatch(state, params, 2 if refused else 1)
    created = events[0]
    assert isinstance(created, TaskCreated)
    assert isinstance(created.task, VideoGenerationTask)
    assert created.task.task_status == (TaskStatus.Failed if refused else TaskStatus.Pending)
    if refused:
        failure = events[1]
        assert isinstance(failure, TaskFailed)
        assert failure.error_type == "video_mode_unavailable"
    else:
        assert created.task.instance_id == instance.instance_id
        assert created.task.task_params.codec == codec
