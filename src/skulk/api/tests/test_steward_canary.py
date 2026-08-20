"""Canary behavior: probe target selection and probe outcome semantics."""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import anyio
import pytest

import skulk.api.steward as steward_module
from skulk.api.steward import StewardHarness, canary_probe_target
from skulk.master.placement import place_instance
from skulk.master.tests.test_placement import fully_connected_three_nodes
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.chunks import TokenChunk
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import CommandId
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.runners import RunnerLoading, RunnerReady, RunnerRunning
from skulk.shared.types.worker.shards import Sharding

if TYPE_CHECKING:
    from skulk.api.main import API


def _steward_placement() -> dict[InstanceId, Instance]:
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=ModelCard(
            model_id=ModelId("canary-brain"),
            storage_size=Memory.from_gb(3),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            trust_remote_code=False,
        ),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        system_role="steward",
    )
    return place_instance(command, topology, {}, node_memory, node_network)


def test_probes_only_on_the_hosting_node_when_idle_ready() -> None:
    placed = _steward_placement()
    instance = next(iter(placed.values()))
    host = next(iter(instance.shard_assignments.node_to_runner))
    runner_id = instance.shard_assignments.node_to_runner[host]
    runners = {runner_id: RunnerReady()}

    assert (
        canary_probe_target(placed, runners, {}, host) == instance.instance_id
    )


def test_non_hosting_node_does_not_probe() -> None:
    placed = _steward_placement()
    instance = next(iter(placed.values()))
    host = next(iter(instance.shard_assignments.node_to_runner))
    runner_id = instance.shard_assignments.node_to_runner[host]
    runners = {runner_id: RunnerReady()}
    from skulk.shared.types.common import NodeId

    assert canary_probe_target(placed, runners, {}, NodeId()) is None


def test_busy_or_loading_runner_is_not_probed() -> None:
    placed = _steward_placement()
    instance = next(iter(placed.values()))
    host = next(iter(instance.shard_assignments.node_to_runner))
    runner_id = instance.shard_assignments.node_to_runner[host]

    assert canary_probe_target(placed, {runner_id: RunnerRunning()}, {}, host) is None
    assert (
        canary_probe_target(
            placed, {runner_id: RunnerLoading(layers_loaded=1, total_layers=2)}, {}, host
        )
        is None
    )


def test_in_flight_task_defers_the_probe() -> None:
    placed = _steward_placement()
    instance = next(iter(placed.values()))
    host = next(iter(instance.shard_assignments.node_to_runner))
    runner_id = instance.shard_assignments.node_to_runner[host]
    runners = {runner_id: RunnerReady()}
    task_id = TaskId()
    tasks = {
        task_id: TextGeneration(
            task_id=task_id,
            command_id=CommandId(),
            instance_id=instance.instance_id,
            task_status=TaskStatus.Running,
            task_params=TextGenerationTaskParams(
                model=ModelId("canary-brain"),
                input=[InputMessage(role="user", content="hi")],
            ),
        )
    }

    assert canary_probe_target(placed, runners, tasks, host) is None


def test_ordinary_instances_are_never_probed() -> None:
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=ModelCard(
            model_id=ModelId("ordinary-model"),
            storage_size=Memory.from_gb(3),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            trust_remote_code=False,
        ),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    host = next(iter(instance.shard_assignments.node_to_runner))
    runner_id = instance.shard_assignments.node_to_runner[host]

    assert canary_probe_target(placed, {runner_id: RunnerReady()}, {}, host) is None


def test_multi_node_steward_elects_one_prober() -> None:
    """Only the lexicographically smallest hosting node probes (election)."""
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=ModelCard(
            model_id=ModelId("canary-brain"),
            storage_size=Memory.from_gb(12),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            trust_remote_code=False,
        ),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
        system_role="steward",
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    hosting = sorted(instance.shard_assignments.node_to_runner, key=str)
    assert len(hosting) >= 2
    runners = {
        runner_id: RunnerReady()
        for runner_id in instance.shard_assignments.node_to_runner.values()
    }
    assert (
        canary_probe_target(placed, runners, {}, hosting[0])
        == instance.instance_id
    )
    assert canary_probe_target(placed, runners, {}, hosting[1]) is None


def test_terminal_lifecycle_tasks_do_not_mute_the_canary() -> None:
    """Completed startup tasks linger in state; only live work defers."""
    placed = _steward_placement()
    instance = next(iter(placed.values()))
    host = next(iter(instance.shard_assignments.node_to_runner))
    runner_id = instance.shard_assignments.node_to_runner[host]
    runners = {runner_id: RunnerReady()}
    task_id = TaskId()
    tasks = {
        task_id: TextGeneration(
            task_id=task_id,
            command_id=CommandId(),
            instance_id=instance.instance_id,
            task_status=TaskStatus.Complete,
            task_params=TextGenerationTaskParams(
                model=ModelId("canary-brain"),
                input=[InputMessage(role="user", content="done earlier")],
            ),
        )
    }

    assert (
        canary_probe_target(placed, runners, tasks, host) == instance.instance_id
    )


class _WedgedStreamApi:
    """Stub API whose generation emits text and then stalls forever."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    async def running_model_card(self, model_id: ModelId) -> ModelCard:
        return ModelCard(
            model_id=ModelId(self._model_id),
            storage_size=Memory.from_gb(3),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        )

    async def dispatch_text_generation(
        self,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> object:
        return SimpleNamespace(command_id=CommandId())

    def text_generation_chunk_stream(
        self,
        command: object,
        task_params: TextGenerationTaskParams,
        *,
        extension_tap: bool = True,
    ) -> "AsyncGenerator[TokenChunk, None]":
        async def _stream() -> "AsyncGenerator[TokenChunk, None]":
            yield TokenChunk(
                model=ModelId(self._model_id),
                text="O",
                token_id=-1,
                usage=None,
                finish_reason=None,
            )
            # The partial-output wedge: text arrived, the terminal never does.
            await anyio.Event().wait()

        return _stream()


async def test_partial_output_then_stall_is_a_failed_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled deadline must fail the probe even after text arrived.

    Counting the partial output as success would clear the consecutive
    failure run, so the exact wedge the canary exists to catch could never
    reach the three-failure teardown threshold (#830).
    """
    monkeypatch.setattr(steward_module, "CANARY_PROBE_TIMEOUT_SECONDS", 0.05)
    api = _WedgedStreamApi("canary-brain")
    harness = StewardHarness(cast("API", cast(object, api)))

    assert await harness.canary_probe(InstanceId(), "canary-brain") is False
