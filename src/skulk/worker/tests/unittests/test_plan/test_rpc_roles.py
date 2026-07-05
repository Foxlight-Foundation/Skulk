"""Role-aware plan gates for multi-node GGUF (RPC driver + donors, #328).

Donors never download, load, warm up, or serve; the driver loads once its own
node holds the model AND every donor's rpc-server reports Ready; nobody runs
ConnectToGroup. These gates are what keep a pooled placement from wedging in
the MLX lockstep machinery or planning a multi-GB download on a donor.
"""

import skulk.worker.plan as plan_mod
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import (
    ConnectToGroup,
    DownloadModel,
    LoadModel,
    StartWarmup,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.downloads import DownloadCompleted
from skulk.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    LlamaRpcInstance,
)
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerReady,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    RpcDonorShardMetadata,
)
from skulk.worker.tests.unittests.conftest import FakeRunnerSupervisor

_INSTANCE_ID = InstanceId()
_MODEL_ID = ModelId("test/pooled-gguf")
_DRIVER_NODE = NodeId("driver-node")
_DONOR_NODE = NodeId("donor-node")
_DRIVER_RUNNER = RunnerId()
_DONOR_RUNNER = RunnerId()


def _card() -> ModelCard:
    return ModelCard(
        model_id=_MODEL_ID,
        storage_size=Memory.from_gb(55),
        n_layers=36,
        hidden_size=2880,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _rpc_instance() -> LlamaRpcInstance:
    card = _card()
    driver_shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    donor_shard = RpcDonorShardMetadata(
        model_card=card,
        device_rank=1,
        world_size=2,
        start_layer=0,
        end_layer=0,
        n_layers=card.n_layers,
    )
    return LlamaRpcInstance(
        instance_id=_INSTANCE_ID,
        shard_assignments=ShardAssignments(
            model_id=_MODEL_ID,
            node_to_runner={_DRIVER_NODE: _DRIVER_RUNNER, _DONOR_NODE: _DONOR_RUNNER},
            runner_to_shard={
                _DRIVER_RUNNER: driver_shard,
                _DONOR_RUNNER: donor_shard,
            },
        ),
        driver_node=_DRIVER_NODE,
        donor_endpoints={_DONOR_NODE: "10.99.0.2:50052"},
    )


def _bound(instance: LlamaRpcInstance, node: NodeId, runner: RunnerId) -> BoundInstance:
    return BoundInstance(
        instance=instance, bound_runner_id=runner, bound_node_id=node
    )


def _driver_download_complete(instance: LlamaRpcInstance) -> dict[NodeId, list[DownloadCompleted]]:
    driver_shard = instance.shard_assignments.runner_to_shard[_DRIVER_RUNNER]
    return {
        _DRIVER_NODE: [
            DownloadCompleted(
                shard_metadata=driver_shard, node_id=_DRIVER_NODE, total=Memory()
            )
        ]
    }


def test_donor_never_plans_download() -> None:
    instance = _rpc_instance()
    donor = FakeRunnerSupervisor(
        bound_instance=_bound(instance, _DONOR_NODE, _DONOR_RUNNER),
        status=RunnerIdle(),
    )
    result = plan_mod.plan(
        node_id=_DONOR_NODE,
        runners={_DONOR_RUNNER: donor},  # type: ignore
        global_download_status={_DONOR_NODE: []},
        instances={_INSTANCE_ID: instance},
        all_runners={_DONOR_RUNNER: RunnerIdle(), _DRIVER_RUNNER: RunnerIdle()},
        tasks={},
    )
    assert not isinstance(result, (DownloadModel, LoadModel, ConnectToGroup))
    assert result is None


def test_rpc_instance_never_plans_connect_to_group() -> None:
    instance = _rpc_instance()
    driver = FakeRunnerSupervisor(
        bound_instance=_bound(instance, _DRIVER_NODE, _DRIVER_RUNNER),
        status=RunnerIdle(),
    )
    result = plan_mod.plan(
        node_id=_DRIVER_NODE,
        runners={_DRIVER_RUNNER: driver},  # type: ignore
        global_download_status={},
        instances={_INSTANCE_ID: instance},
        all_runners={_DRIVER_RUNNER: RunnerIdle(), _DONOR_RUNNER: RunnerIdle()},
        tasks={},
    )
    assert not isinstance(result, (ConnectToGroup, StartWarmup))


def test_driver_waits_for_donor_ready_before_load() -> None:
    instance = _rpc_instance()
    driver = FakeRunnerSupervisor(
        bound_instance=_bound(instance, _DRIVER_NODE, _DRIVER_RUNNER),
        status=RunnerIdle(),
    )
    downloads = _driver_download_complete(instance)
    # Donor not Ready yet: no LoadModel.
    result = plan_mod.plan(
        node_id=_DRIVER_NODE,
        runners={_DRIVER_RUNNER: driver},  # type: ignore
        global_download_status=downloads,
        instances={_INSTANCE_ID: instance},
        all_runners={_DRIVER_RUNNER: RunnerIdle(), _DONOR_RUNNER: RunnerIdle()},
        tasks={},
    )
    assert not isinstance(result, LoadModel)
    # Donor Ready: driver loads, with the download required on ITS node only.
    result = plan_mod.plan(
        node_id=_DRIVER_NODE,
        runners={_DRIVER_RUNNER: driver},  # type: ignore
        global_download_status=downloads,
        instances={_INSTANCE_ID: instance},
        all_runners={_DRIVER_RUNNER: RunnerIdle(), _DONOR_RUNNER: RunnerReady()},
        tasks={},
    )
    assert isinstance(result, LoadModel)
    assert result.instance_id == _INSTANCE_ID


def test_generation_tasks_never_reach_a_donor() -> None:
    instance = _rpc_instance()
    donor = FakeRunnerSupervisor(
        bound_instance=_bound(instance, _DONOR_NODE, _DONOR_RUNNER),
        status=RunnerReady(),
    )
    task = TextGeneration(
        task_id=TaskId(),
        instance_id=_INSTANCE_ID,
        command_id=CommandId(),
        task_status=TaskStatus.Pending,
        task_params=TextGenerationTaskParams(
            model=_MODEL_ID,
            input=[InputMessage(role="user", content="hi")],
        ),
    )
    result = plan_mod.plan(
        node_id=_DONOR_NODE,
        runners={_DONOR_RUNNER: donor},  # type: ignore
        global_download_status={},
        instances={_INSTANCE_ID: instance},
        all_runners={_DRIVER_RUNNER: RunnerReady(), _DONOR_RUNNER: RunnerReady()},
        tasks={task.task_id: task},
    )
    assert result is None
