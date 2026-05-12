import io
import subprocess
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
from exo.shared.types.tasks import LoadModel, TaskStatus
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.larql import LarqlExpertRange
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.shared.types.worker.shards import LarqlShardMetadata, ShardMetadata
from exo.utils.channels import channel
from exo.worker.runner.larql_supervisor import (
    LarqlRunnerSupervisor,
    build_larql_serve_command,
)
from exo.worker.tests.unittests.conftest import get_mlx_ring_instance


class _FakeProcess:
    pid = 12345
    returncode: int | None = None
    stdout = io.StringIO("")
    stderr = io.StringIO("")

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, _timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


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

    def process_factory(command: Sequence[str]) -> subprocess.Popen[str]:
        commands.append(tuple(str(part) for part in command))
        return cast(subprocess.Popen[str], cast(object, _FakeProcess()))

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
