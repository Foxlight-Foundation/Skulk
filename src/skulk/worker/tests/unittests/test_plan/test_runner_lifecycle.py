from typing import Any

import skulk.worker.plan as plan_mod
from skulk.shared.types.common import CommandId
from skulk.shared.types.tasks import Shutdown, TaskId, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from skulk.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerStatus,
)
from skulk.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from skulk.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def test_plan_kills_runner_when_instance_missing():
    """
    If a local runner's instance is no longer present in state,
    plan() should return a Shutdown for that runner.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances: dict[InstanceId, Instance] = {}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_prefers_shutdown_over_cancel_for_missing_instance() -> None:
    """
    If the authoritative instance has already been deleted, plan() should tear
    down the local runner immediately instead of spending cycles on task-level
    cancellation first.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    cancelled_task = TextGeneration(
        task_id=TaskId("cancelled-task"),
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Cancelled,
        command_id=CommandId("cmd-1"),
        task_params=TextGenerationTaskParams(
            model=MODEL_A_ID,
            input=[InputMessage(role="user", content="stop")],
        ),
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances={},
        all_runners={RUNNER_1_ID: RunnerReady()},
        tasks={cancelled_task.task_id: cancelled_task},
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_kills_runner_when_sibling_failed():
    """
    If a sibling runner in the same instance has failed, the local runner
    should be shut down.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerFailed(error_message="boom"),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_creates_runner_when_missing_for_node():
    """
    If shard_assignments specify a runner for this node but we don't have
    a local supervisor yet, plan() should emit a CreateRunner.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )

    runners: dict[Any, Any] = {}  # nothing local yet
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[Any, Any] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    # We patched plan_mod.CreateRunner → CreateRunner
    assert isinstance(result, plan_mod.CreateRunner)
    assert result.instance_id == INSTANCE_1_ID
    assert isinstance(result.bound_instance, BoundInstance)
    assert result.bound_instance.instance is instance
    assert result.bound_instance.bound_runner_id == RUNNER_1_ID


def test_plan_does_not_create_runner_when_supervisor_already_present():
    """
    If we already have a local supervisor for the runner assigned to this node,
    plan() should not emit a CreateRunner again.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None


def test_plan_does_not_create_runner_for_unassigned_node():
    """
    If this node does not appear in shard_assignments.node_to_runner,
    plan() should not try to create a runner on this node.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_2_ID: shard},
    )

    runners: dict[RunnerId, FakeRunnerSupervisor] = {}  # no local runners
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[RunnerId, RunnerStatus] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None


def _single_node_runner(status: RunnerStatus) -> tuple[FakeRunnerSupervisor, Instance]:
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    return FakeRunnerSupervisor(bound_instance=bound_instance, status=status), instance


def _plan_single_node(status: RunnerStatus) -> object:
    runner, instance = _single_node_runner(status)
    return plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: status},
        tasks={},
    )


def test_plan_shuts_down_a_single_node_runner_that_died() -> None:
    """A dead runner behind a live single-node instance is shut down.

    Nothing else would: the sibling check skips the runner itself, so the
    instance kept a dead supervisor, never relaunched and never failed. The
    shutdown hands the decision to the worker's crash breaker.
    """
    result = _plan_single_node(RunnerFailed(error_message="Terminated (signal=9)"))

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_leaves_terminal_failures_to_the_worker() -> None:
    """A GPU wedge or a trust refusal is given up directly, never relaunched."""
    from skulk.shared.models.remote_code_approval import MODEL_TRUST_FAILURE_MARKER
    from skulk.worker.runner.bootstrap import WEDGE_FAILURE_MARKER

    for marker in (WEDGE_FAILURE_MARKER, MODEL_TRUST_FAILURE_MARKER):
        result = _plan_single_node(RunnerFailed(error_message=f"Terminated ({marker})"))
        assert not isinstance(result, Shutdown), marker


def test_plan_does_not_shut_down_a_healthy_single_node_runner() -> None:
    assert not isinstance(_plan_single_node(RunnerReady()), Shutdown)

