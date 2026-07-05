"""Subprocess-death detection for the RPC donor runner (#328 hardening).

A donor whose ggml-rpc-server dies must crash the runner (so the supervisor's
peer-failure cascade tears down the pooled instance) instead of gossiping
RunnerReady over a dead port forever. Live repro on the kite4+kite5 pair:
killing the donor's rpc-server mid-generation aborted the driver's
llama-server, yet both runners stayed "Ready" and the instance wedged.
"""

import subprocess
import sys

import pytest

from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import Event
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    LlamaRpcInstance,
)
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerShuttingDown,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    RpcDonorShardMetadata,
)
from skulk.utils.channels import mp_channel
from skulk.worker.runner.rpc_donor.runner import Runner

_DRIVER_NODE = NodeId("driver-node")
_DONOR_NODE = NodeId("donor-node")
_DRIVER_RUNNER = RunnerId()
_DONOR_RUNNER = RunnerId()


def _donor_runner() -> Runner:
    card = ModelCard(
        model_id=ModelId("test/pooled-gguf"),
        storage_size=Memory.from_gb(55),
        n_layers=36,
        hidden_size=2880,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
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
    instance = LlamaRpcInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={_DRIVER_NODE: _DRIVER_RUNNER, _DONOR_NODE: _DONOR_RUNNER},
            runner_to_shard={
                _DRIVER_RUNNER: driver_shard,
                _DONOR_RUNNER: donor_shard,
            },
        ),
        driver_node=_DRIVER_NODE,
        donor_endpoints={_DONOR_NODE: "127.0.0.1:50052"},
    )
    event_sender, _event_receiver = mp_channel[Event]()
    _task_sender, task_receiver = mp_channel[Task]()
    _cancel_sender, cancel_receiver = mp_channel[TaskId]()
    return Runner(
        bound_instance=BoundInstance(
            instance=instance,
            bound_runner_id=_DONOR_RUNNER,
            bound_node_id=_DONOR_NODE,
        ),
        event_sender=event_sender,
        task_receiver=task_receiver,
        cancel_receiver=cancel_receiver,
    )


def test_dead_subprocess_raises() -> None:
    runner = _donor_runner()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    _ = proc.wait(timeout=10)
    runner.server_proc = proc
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        runner._ensure_server_alive()  # pyright: ignore[reportPrivateUsage]


def test_live_subprocess_passes() -> None:
    runner = _donor_runner()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        runner.server_proc = proc
        runner._ensure_server_alive()  # pyright: ignore[reportPrivateUsage]
    finally:
        proc.kill()
        _ = proc.wait(timeout=10)


def test_no_raise_during_shutdown() -> None:
    runner = _donor_runner()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    _ = proc.wait(timeout=10)
    runner.server_proc = proc
    runner.current_status = RunnerShuttingDown()
    runner._ensure_server_alive()  # pyright: ignore[reportPrivateUsage]
