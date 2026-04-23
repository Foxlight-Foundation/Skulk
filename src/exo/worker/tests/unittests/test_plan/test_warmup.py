import exo.worker.plan as plan_mod
from exo.shared.models.model_cards import ModelId
from exo.shared.types.tasks import StartWarmup
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerWarmingUp,
)
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    NODE_C,
    RUNNER_1_ID,
    RUNNER_2_ID,
    RUNNER_3_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

GEMMA4_MODEL_ID = ModelId("mlx-community/gemma-4-26b-a4b-it-4bit")


def test_plan_starts_warmup_for_accepting_rank_when_all_loaded_or_warming():
    """
    For non-zero device_rank shards, StartWarmup should be emitted when all
    shards in the instance are Loaded/WarmingUp.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=3)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=3)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=2, world_size=3)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID, NODE_C: RUNNER_3_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1, RUNNER_3_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_2_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoaded(),
        RUNNER_3_ID: RunnerWarmingUp(),
    }

    result = plan_mod.plan(
        node_id=NODE_B,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_start_warmup_for_accepting_rank_when_peer_is_ready():
    """
    Non-zero ranks should not start distributed warmup once any peer has
    already reached Ready because that implies the coordinated warmup phase has
    moved past the point where this runner can safely join.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=3)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=3)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=2, world_size=3)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID, NODE_C: RUNNER_3_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1, RUNNER_3_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_2_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerLoaded(),
        RUNNER_3_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_B,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None


def test_plan_starts_gemma4_warmup_for_accepting_rank_when_peer_is_ready():
    """
    Gemma 4 distributed warmup is independently skipped by the runner, so a
    fast peer reaching Ready must not strand remaining Loaded ranks.
    """
    shard0 = get_pipeline_shard_metadata(GEMMA4_MODEL_ID, device_rank=0, world_size=3)
    shard1 = get_pipeline_shard_metadata(GEMMA4_MODEL_ID, device_rank=1, world_size=3)
    shard2 = get_pipeline_shard_metadata(GEMMA4_MODEL_ID, device_rank=2, world_size=3)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=GEMMA4_MODEL_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID, NODE_C: RUNNER_3_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1, RUNNER_3_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    result = plan_mod.plan(
        node_id=NODE_B,
        runners={RUNNER_2_ID: local_runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={
            RUNNER_1_ID: RunnerReady(),
            RUNNER_2_ID: RunnerLoaded(),
            RUNNER_3_ID: RunnerLoaded(),
        },
        tasks={},
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_starts_warmup_for_rank_zero_after_others_warming():
    """
    For device_rank == 0, StartWarmup should only be emitted once all the
    other runners in the instance are already warming up.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerWarmingUp(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_start_warmup_for_rank_zero_after_other_rank_is_ready():
    """
    Rank zero should not start distributed warmup once another rank has already
    reached Ready because the synchronized warmup phase is no longer joinable.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerReady(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None


def test_plan_starts_gemma4_warmup_for_rank_zero_after_others_ready():
    """
    Rank zero can finish Gemma 4 skipped warmup after non-zero peers have
    already moved from WarmingUp to Ready.
    """
    shard0 = get_pipeline_shard_metadata(GEMMA4_MODEL_ID, device_rank=0, world_size=3)
    shard1 = get_pipeline_shard_metadata(GEMMA4_MODEL_ID, device_rank=1, world_size=3)
    shard2 = get_pipeline_shard_metadata(GEMMA4_MODEL_ID, device_rank=2, world_size=3)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=GEMMA4_MODEL_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID, NODE_C: RUNNER_3_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1, RUNNER_3_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: local_runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={
            RUNNER_1_ID: RunnerLoaded(),
            RUNNER_2_ID: RunnerReady(),
            RUNNER_3_ID: RunnerReady(),
        },
        tasks={},
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_start_warmup_for_non_zero_rank_until_all_loaded_or_warming():
    """
    Non-zero rank should not start warmup while any shard is not Loaded/WarmingUp.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_2_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerIdle(),
        RUNNER_2_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_B,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None


def test_plan_does_not_start_warmup_for_rank_zero_until_others_warming():
    """
    Rank-zero shard should not start warmup until all non-zero ranks are
    already WarmingUp.
    For accepting ranks (device_rank != 0), StartWarmup should be
    emitted when all shards in the instance are Loaded/WarmingUp.
    In a 2-node setup, rank 1 is the accepting rank.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # Rank 1 is the accepting rank
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None

    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerWarmingUp(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_starts_warmup_for_connecting_rank_after_others_warming():
    """
    For connecting rank (device_rank == world_size - 1), StartWarmup should
    only be emitted once all the other runners are already warming up.
    In a 2-node setup, rank 1 is the connecting rank.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # Rank 1 is the connecting rank
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_2_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerWarmingUp(),
        RUNNER_2_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_B,
        runners=runners,  # type: ignore
        global_download_status={NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_start_warmup_for_accepting_rank_until_all_loaded_or_warming():
    """
    Accepting rank should not start warmup while any shard is not Loaded/WarmingUp.
    In a 2-node setup, rank 0 is the accepting rank.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # Rank 0 is the accepting rank
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoading(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None


def test_plan_does_not_start_warmup_for_connecting_rank_until_others_warming():
    """
    Connecting rank (device_rank == 0) should not start warmup
    until all other ranks are already WarmingUp.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # Rank 1 is the connecting rank
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None
