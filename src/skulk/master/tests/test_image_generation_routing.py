# pyright: reportPrivateUsage=false
"""Image targets must survive command admission without replica fallback."""

import anyio
import pytest

from skulk.api.types import ImageGenerationTaskParams
from skulk.master.main import Master
from skulk.master.tests.test_text_generation_routing import _state
from skulk.routing.router import get_node_id_keypair
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageGeneration,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TaskCreated,
    TaskFailed,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.tasks import ImageGeneration as ImageGenerationTask
from skulk.shared.types.tasks import TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerFailed, RunnerLoading, RunnerReady
from skulk.utils.channels import channel


async def _dispatch(state: State, target: InstanceId | None, *, failed: bool) -> list[Event]:
    """Send a real forwarded command through the master's event boundary."""
    node = NodeId(get_node_id_keypair().to_node_id())
    commands, receiver = channel[ForwarderCommand]()
    events, output = channel[Event]()
    global_sender, _ = channel[GlobalForwarderEvent]()
    _, local = channel[LocalForwarderEvent]()
    sync_sender, sync_receiver = channel[StateSyncMessage]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    master = Master(node, SessionId(master_node_id=node, election_clock=0), event_sender=events,
        global_event_sender=global_sender, local_event_receiver=local, command_receiver=receiver,
        state_sync_receiver=sync_receiver, state_sync_sender=sync_sender, download_command_sender=downloads)
    master.state = state
    command = ImageGeneration(target_instance_id=target, task_params=ImageGenerationTaskParams(model="org/model", prompt="blue square"))
    result: list[Event] = []
    async with anyio.create_task_group() as group:
        group.start_soon(master._command_processor)
        await commands.send(ForwarderCommand(origin=SystemId("API"), command=command))
        with anyio.fail_after(2):
            result = await output.receive_at_least(2 if failed else 1)
        group.cancel_scope.cancel()
    first = result[0]
    assert isinstance(first, TaskCreated)
    assert isinstance(first.task, ImageGenerationTask)
    assert first.task.command_id == command.command_id
    return result


@pytest.mark.parametrize(("target", "failed"), [(None, False), ("ordinary", False), ("steward", True), ("absent", True)])
async def test_master_respects_exact_image_target(target: str | None, *, failed: bool) -> None:
    """A failed or missing target never borrows a ready sibling's execution."""
    events = await _dispatch(_state(RunnerFailed(), RunnerReady()), InstanceId(target) if target else None, failed=failed)
    first = events[0]
    assert isinstance(first, TaskCreated)
    assert first.task.instance_id == InstanceId(target or "ordinary")
    assert first.task.task_status == (TaskStatus.Failed if failed else TaskStatus.Pending)
    if failed:
        assert isinstance(events[1], TaskFailed)
        assert events[1].error_type == "instance_unavailable"


async def test_loading_image_target_waits_without_using_ready_replica() -> None:
    """Explicit cold pins retain their identity rather than falling back."""
    events = await _dispatch(_state(RunnerLoading(), RunnerReady()), InstanceId("steward"), failed=False)
    assert isinstance(events[0], TaskCreated)
    assert events[0].task.instance_id == InstanceId("steward")


async def test_image_target_model_mismatch_is_terminal() -> None:
    """An extant target for another model must not accept the image task."""
    state = _state(RunnerReady(), RunnerReady())
    target = state.instances[InstanceId("ordinary")]
    target = target.model_copy(update={"shard_assignments": target.shard_assignments.model_copy(update={"model_id": ModelId("org/other")})})
    state = state.model_copy(update={"instances": {**state.instances, target.instance_id: target}})
    events = await _dispatch(state, target.instance_id, failed=True)
    assert isinstance(events[1], TaskFailed)
