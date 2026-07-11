# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from skulk.shared.models.capabilities import is_gemma4_family
from skulk.shared.types.chunks import AudioInputChunk, InputImageChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import (
    AudioTranscription,
    CancelTask,
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    RealtimeAudioTranscription,
    Shutdown,
    SpeechSynthesis,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextEmbedding,
    TextGeneration,
)
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    LlamaRpcInstance,
)
from skulk.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)
from skulk.shared.types.worker.shards import RpcDonorShardMetadata, ShardMetadata
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


def _is_rpc_donor(runner: RunnerSupervisor) -> bool:
    """Whether this runner is an RPC memory donor of a multi-node GGUF
    placement (#328). Donors never download, load, warm up, or serve
    requests; the plan gates below skip them accordingly."""
    return isinstance(runner.bound_instance.bound_shard, RpcDonorShardMetadata)


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]] | None = None,
    input_audio_chunk_buffer: Mapping[
        CommandId, Mapping[int, AudioInputChunk]
    ] | None = None,
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        # Kill orphaned or poisoned runners before issuing cooperative task
        # cancellation. Once an instance disappears from authoritative state we
        # should not strand local runner processes behind best-effort cancel
        # handshakes; shutting them down is the safest recovery path.
        _kill_runner(runners, all_runners, instances)
        or _cancel_tasks(runners, tasks)
        or _create_runner(node_id, runners, instances)
        or _model_needs_download(node_id, runners, global_download_status)
        or _init_distributed_backend(runners, all_runners)
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(
            runners,
            tasks,
            all_runners,
            input_chunk_buffer or {},
            input_audio_chunk_buffer or {},
        )
    )


def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    for runner in runners.values():
        runner_id = runner.bound_instance.bound_runner_id
        if (instance_id := runner.bound_instance.instance.instance_id) not in instances:
            return Shutdown(instance_id=instance_id, runner_id=runner_id)

        for (
            global_runner_id
        ) in runner.bound_instance.instance.shard_assignments.node_to_runner.values():
            if runner_id == global_runner_id:
                continue

            if isinstance(all_runners.get(global_runner_id, None), RunnerFailed):
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    instances: Mapping[InstanceId, Instance],
) -> CreateRunner | None:
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        shard = instance.shard(runner_id)
        assert shard is not None

        return CreateRunner(
            instance_id=instance.instance_id,
            bound_instance=BoundInstance(
                instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
            ),
        )


def _model_needs_download(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> DownloadModel | None:
    local_downloads = global_download_status.get(node_id, [])
    download_status = {
        dp.shard_metadata.model_card.model_id: dp for dp in local_downloads
    }

    for runner in runners.values():
        # An RPC donor never touches the model file: the driver reads the GGUF
        # and pushes tensors over the network. Without this skip a donor's brief
        # RunnerIdle window would plan a multi-GB download for nothing (#328).
        if _is_rpc_donor(runner):
            continue
        model_id = runner.bound_instance.bound_shard.model_card.model_id
        if isinstance(runner.status, RunnerIdle) and (
            model_id not in download_status
            or not isinstance(
                download_status[model_id],
                (DownloadOngoing, DownloadCompleted, DownloadFailed),
            )
        ):
            # We don't invalidate download_status randomly in case a file gets deleted on disk
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )


def _init_distributed_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
):
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        is_single_node_instance = len(shard_assignments.runner_to_shard) == 1
        if is_single_node_instance:
            continue

        # An RPC placement (#328) has no distributed group: the driver's
        # llama-server dials the donors' rpc-servers directly, so nobody runs
        # ConnectToGroup (an MLX ring init would wedge forever).
        if isinstance(instance, LlamaRpcInstance):
            continue

        runner_is_idle = isinstance(runner.status, RunnerIdle)
        all_runners_connecting = all(
            isinstance(
                all_runners.get(global_runner_id),
                (RunnerConnecting, RunnerIdle),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if not (runner_is_idle and all_runners_connecting):
            continue

        runner_id = runner.bound_instance.bound_runner_id

        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        world_size = shard.world_size

        assert device_rank < world_size
        assert device_rank >= 0

        accepting_ranks = device_rank < world_size - 1

        # Rank = n-1
        connecting_rank_ready = device_rank == world_size - 1 and all(
            isinstance(all_runners.get(global_runner_id, None), RunnerConnecting)
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if not (accepting_ranks or connecting_rank_ready):
            continue

        return ConnectToGroup(instance_id=instance.instance_id)

    return None


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        # RPC placements (#328) are role-asymmetric: donors never load (their
        # runner reports Ready by itself), and the DRIVER loads when the model
        # is downloaded on ITS node only (donors never hold the file) and every
        # donor's rpc-server is already Ready, so the endpoints answer before
        # llama-server dials them.
        if isinstance(instance, LlamaRpcInstance):
            if _is_rpc_donor(runner):
                continue
            driver_node = runner.bound_instance.bound_node_id
            driver_download_complete = driver_node in global_download_status and any(
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata.model_card.model_id == shard_assignments.model_id
                for dp in global_download_status[driver_node]
            )
            donors_ready = all(
                isinstance(all_runners.get(global_runner_id), RunnerReady)
                for global_runner_id, shard in shard_assignments.runner_to_shard.items()
                if isinstance(shard, RpcDonorShardMetadata)
            )
            if (
                isinstance(runner.status, RunnerIdle)
                and driver_download_complete
                and donors_ready
            ):
                return LoadModel(instance_id=instance.instance_id)
            continue

        all_local_downloads_complete = all(
            nid in global_download_status
            and any(
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata.model_card.model_id == shard_assignments.model_id
                for dp in global_download_status[nid]
            )
            for nid in shard_assignments.node_to_runner
        )
        if not all_local_downloads_complete:
            continue

        is_single_node_instance = len(instance.shard_assignments.runner_to_shard) == 1
        if is_single_node_instance and isinstance(runner.status, RunnerIdle):
            return LoadModel(instance_id=instance.instance_id)

        is_runner_waiting = isinstance(runner.status, RunnerConnected)

        all_ready_for_model = all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerConnected, RunnerLoading, RunnerLoaded),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if is_runner_waiting and all_ready_for_model:
            return LoadModel(instance_id=instance.instance_id)

    return None


def _ready_to_warmup(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> StartWarmup | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        # No warmup on RPC placements (#328): the served driver goes
        # Loading -> Ready internally and donors never load a model.
        if isinstance(instance, LlamaRpcInstance):
            continue
        shard_assignments = instance.shard_assignments
        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        runner_id = runner.bound_instance.bound_runner_id
        world_size = shard.world_size

        is_runner_loaded = isinstance(runner.status, RunnerLoaded)
        peer_warmup_statuses: tuple[type[RunnerStatus], ...]
        rank_zero_peer_warmup_statuses: tuple[type[RunnerStatus], ...]
        if _uses_independent_distributed_warmup(shard):
            # Gemma 4 distributed synthetic warmup is intentionally skipped by
            # the runner, so each shard can advance from Loaded to Ready without
            # needing peers to remain parked in WarmingUp.
            peer_warmup_statuses = (RunnerLoaded, RunnerWarmingUp, RunnerReady)
            rank_zero_peer_warmup_statuses = (RunnerWarmingUp, RunnerReady)
        else:
            peer_warmup_statuses = (RunnerLoaded, RunnerWarmingUp)
            rank_zero_peer_warmup_statuses = (RunnerWarmingUp,)

        assert device_rank < world_size
        assert device_rank >= 0

        # Rank != 0
        accepting_ranks_ready = device_rank > 0 and all(
            isinstance(
                all_runners.get(global_runner_id, None),
                peer_warmup_statuses,
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        # Rank = 0
        connecting_rank_ready = device_rank == 0 and all(
            isinstance(
                all_runners.get(global_runner_id, None),
                rank_zero_peer_warmup_statuses,
            )
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if is_runner_loaded and (accepting_ranks_ready or connecting_rank_ready):
            return StartWarmup(instance_id=instance.instance_id)

    return None


def _uses_independent_distributed_warmup(shard: ShardMetadata) -> bool:
    """Return whether distributed warmup is independently skippable per shard.

    Today the only family that opts out of synthetic distributed warmup is
    Gemma 4 — see ``Runner._should_skip_distributed_warmup`` for the matching
    runner-side bypass. Detection delegates to the consolidated
    ``is_gemma4_family`` predicate so the planner and the runner cannot drift.
    """
    if shard.world_size <= 1:
        return False
    return is_gemma4_family(shard.model_card)


def _pending_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
    all_runners: Mapping[RunnerId, RunnerStatus],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]] | None,
    input_audio_chunk_buffer: Mapping[
        CommandId, Mapping[int, AudioInputChunk]
    ] | None,
) -> Task | None:
    for task in tasks.values():
        # Forward inference tasks to runners
        if not isinstance(
            task,
            (
                TextGeneration,
                ImageGeneration,
                ImageEdits,
                TextEmbedding,
                SpeechSynthesis,
                AudioTranscription,
                RealtimeAudioTranscription,
            ),
        ):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        # For tasks with images, verify all input chunks have been received
        expected_image_chunks = 0
        if isinstance(task, (ImageEdits, TextGeneration)):
            expected_image_chunks = task.task_params.total_input_chunks
        if expected_image_chunks > 0:
            assert input_chunk_buffer is not None
            cmd_id = task.command_id
            received = len(input_chunk_buffer.get(cmd_id, {}))
            if received < expected_image_chunks:
                continue  # Wait for all chunks to arrive

        expected_audio_chunks = 0
        if isinstance(task, AudioTranscription):
            expected_audio_chunks = task.task_params.total_input_chunks
        if expected_audio_chunks > 0:
            assert input_audio_chunk_buffer is not None
            cmd_id = task.command_id
            received = len(input_audio_chunk_buffer.get(cmd_id, {}))
            if received < expected_audio_chunks:
                continue  # Wait for the complete uploaded audio payload

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            # Inference traffic never reaches an RPC donor (#328): the driver
            # is the only rank that serves requests; the donor's process is a
            # memory endpoint with no model.
            if _is_rpc_donor(runner):
                continue

            # the task status _should_ be set to completed by the LAST runner
            # it is currently set by the first
            # this is definitely a hack
            if task.task_id in runner.completed or task.task_id in runner.in_progress:
                continue

            if isinstance(runner.status, (RunnerReady, RunnerRunning)) and all(
                isinstance(all_runners[global_runner_id], (RunnerReady, RunnerRunning))
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            ):
                return task


def _cancel_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
) -> Task | None:
    for task in tasks.values():
        if task.task_status != TaskStatus.Cancelled:
            continue
        for runner_id, runner in runners.items():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue
            if task.task_id in runner.cancelled:
                continue
            return CancelTask(
                instance_id=task.instance_id,
                cancelled_task_id=task.task_id,
                runner_id=runner_id,
            )
