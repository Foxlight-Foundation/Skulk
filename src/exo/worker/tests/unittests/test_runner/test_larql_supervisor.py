import io
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    LarqlRunnerReadinessUpdated,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import LoadModel, Shutdown, TaskStatus
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.larql import LarqlExpertRange
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
)
from exo.shared.types.worker.shards import LarqlShardMetadata, ShardMetadata
from exo.utils.channels import channel
from exo.worker.runner.larql_supervisor import (
    LarqlRunnerSupervisor,
    build_larql_serve_command,
)
from exo.worker.tests.unittests.conftest import get_mlx_ring_instance


class _FakeProcess:
    pid = 12345

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")
        self._exit_event = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._exit_event.wait(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self._exit_event.set()

    def kill(self) -> None:
        self.returncode = -9
        self._exit_event.set()

    def exit(self, returncode: int) -> None:
        self.returncode = returncode
        self._exit_event.set()


class _HungProcess(_FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.terminate_called = False
        self.kill_called = False

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("larql", timeout or 0)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9
        self._exit_event.set()


def _larql_shard() -> LarqlShardMetadata:
    return LarqlShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("skulk/gemma-4-26b-a4b-expert-server-q4-k-vindex"),
            storage_size=Memory.from_mb(512),
            n_layers=46,
            hidden_size=4096,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=4,
        end_layer=12,
        n_layers=46,
        vindex_uri="hf://skulk/gemma-4-26b-a4b-expert-server-q4-k-vindex",
        preset="expert-server",
        local_vindex_path="/tmp/gemma-vindex",
        server_port=49152,
        expert_range=LarqlExpertRange(start_expert=0, end_expert=8),
    )


def _bound_instance(shard: ShardMetadata) -> BoundInstance:
    runner_id = RunnerId("runner-a")
    node_id = NodeId("node-a")
    instance = get_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=shard.model_card.model_id,
        node_to_runner={node_id: runner_id},
        runner_to_shard={runner_id: shard},
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )


def test_build_larql_serve_command_includes_slice_arguments() -> None:
    command = build_larql_serve_command(
        _larql_shard(),
        vindex_path=Path("/tmp/gemma-vindex"),
        port=49152,
    )

    assert command == (
        "larql",
        "serve",
        "/tmp/gemma-vindex",
        "--host",
        "127.0.0.1",
        "--port",
        "49152",
        "--ffn-only",
        "--layers",
        "4-12",
        "--preset",
        "expert-server",
        "--experts",
        "0-8",
    )


@pytest.mark.asyncio
async def test_larql_supervisor_load_model_starts_process_and_emits_readiness() -> None:
    shard = _larql_shard()
    event_sender, event_receiver = channel[Event]()
    commands: list[tuple[str, ...]] = []
    process = _FakeProcess()

    def process_factory(command: Sequence[str]) -> subprocess.Popen[str]:
        commands.append(tuple(str(part) for part in command))
        return cast(subprocess.Popen[str], cast(object, process))

    supervisor = LarqlRunnerSupervisor.create(
        bound_instance=_bound_instance(shard),
        event_sender=event_sender,
        process_factory=process_factory,
    )

    async def health_check() -> bool:
        return True

    supervisor._health_check = health_check  # pyright: ignore[reportPrivateUsage]

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.start_task(
            LoadModel(instance_id=supervisor.bound_instance.instance.instance_id)
        )
        task_group.cancel_scope.cancel()

    observed = [
        await event_receiver.receive(),
        await event_receiver.receive(),
        await event_receiver.receive(),
        await event_receiver.receive(),
    ]

    assert commands
    assert isinstance(observed[0], TaskAcknowledged)
    assert isinstance(observed[1], RunnerStatusUpdated)
    assert isinstance(observed[2], RunnerStatusUpdated)
    assert isinstance(observed[2].runner_status, RunnerReady)
    assert isinstance(observed[3], LarqlRunnerReadinessUpdated)
    assert observed[3].readiness.status == "ready"

    completion = await event_receiver.receive()
    assert isinstance(completion, TaskStatusUpdated)
    assert completion.task_status == TaskStatus.Complete


@pytest.mark.asyncio
async def test_larql_supervisor_marks_failed_when_ready_child_exits() -> None:
    shard = _larql_shard()
    event_sender, event_receiver = channel[Event]()
    process = _FakeProcess()

    def process_factory(_command: Sequence[str]) -> subprocess.Popen[str]:
        return cast(subprocess.Popen[str], cast(object, process))

    supervisor = LarqlRunnerSupervisor.create(
        bound_instance=_bound_instance(shard),
        event_sender=event_sender,
        process_factory=process_factory,
    )

    async def health_check() -> bool:
        return True

    supervisor._health_check = health_check  # pyright: ignore[reportPrivateUsage]
    failed_status: Event | None = None
    failed_readiness: Event | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.start_task(
            LoadModel(instance_id=supervisor.bound_instance.instance.instance_id)
        )
        for _ in range(5):
            await event_receiver.receive()

        process.exit(42)

        with anyio.fail_after(1):
            failed_status = await event_receiver.receive()
            failed_readiness = await event_receiver.receive()

        task_group.cancel_scope.cancel()

    assert isinstance(failed_status, RunnerStatusUpdated)
    assert isinstance(failed_status.runner_status, RunnerFailed)
    assert failed_status.runner_status.error_message == (
        "LARQL child exited unexpectedly with exit code 42"
    )
    assert isinstance(failed_readiness, LarqlRunnerReadinessUpdated)
    assert failed_readiness.readiness.status == "failed"
    assert failed_readiness.readiness.error_message == (
        "LARQL child exited unexpectedly with exit code 42"
    )


@pytest.mark.asyncio
async def test_larql_supervisor_shutdown_completes_without_failure() -> None:
    shard = _larql_shard()
    event_sender, event_receiver = channel[Event]()
    supervisor = LarqlRunnerSupervisor.create(
        bound_instance=_bound_instance(shard),
        event_sender=event_sender,
    )
    supervisor._process = cast(  # pyright: ignore[reportPrivateUsage]
        subprocess.Popen[str],
        cast(object, _FakeProcess()),
    )

    await supervisor.start_task(
        Shutdown(
            instance_id=supervisor.bound_instance.instance.instance_id,
            runner_id=supervisor.bound_instance.bound_runner_id,
        )
    )

    observed = [
        await event_receiver.receive(),
        await event_receiver.receive(),
        await event_receiver.receive(),
        await event_receiver.receive(),
    ]

    assert isinstance(observed[0], TaskAcknowledged)
    assert isinstance(observed[1], RunnerStatusUpdated)
    assert isinstance(observed[1].runner_status, RunnerShuttingDown)
    assert isinstance(observed[2], TaskStatusUpdated)
    assert observed[2].task_status == TaskStatus.Complete
    assert isinstance(observed[3], RunnerStatusUpdated)
    assert isinstance(observed[3].runner_status, RunnerShutdown)


@pytest.mark.asyncio
async def test_larql_supervisor_shutdown_kills_child_after_wait_timeout() -> None:
    """Hung child waits fall through to kill during supervisor shutdown."""

    shard = _larql_shard()
    event_sender, event_receiver = channel[Event]()
    process = _HungProcess()
    supervisor = LarqlRunnerSupervisor.create(
        bound_instance=_bound_instance(shard),
        event_sender=event_sender,
    )
    supervisor._process = cast(  # pyright: ignore[reportPrivateUsage]
        subprocess.Popen[str],
        cast(object, process),
    )

    await supervisor.start_task(
        Shutdown(
            instance_id=supervisor.bound_instance.instance.instance_id,
            runner_id=supervisor.bound_instance.bound_runner_id,
        )
    )

    observed = [
        await event_receiver.receive(),
        await event_receiver.receive(),
        await event_receiver.receive(),
        await event_receiver.receive(),
    ]

    assert process.terminate_called
    assert process.kill_called
    assert process.returncode == -9
    assert isinstance(observed[1], RunnerStatusUpdated)
    assert isinstance(observed[1].runner_status, RunnerShuttingDown)
    assert isinstance(observed[2], TaskStatusUpdated)
    assert observed[2].task_status == TaskStatus.Complete
