import multiprocessing as mp
from typing import cast

import anyio
import pytest
from anyio import to_thread

from skulk.api.types import ImageGenerationTaskParams
from skulk.routing.trace_data import TraceDataPacket
from skulk.shared.models.model_cards import ModelId, ModelTask
from skulk.shared.types.audio import (
    AudioTranscriptionTaskParams,
    RealtimeAudioInputFrame,
    RealtimeAudioTranscriptionTaskParams,
    SpeechSynthesisTaskParams,
)
from skulk.shared.types.chunks import ErrorChunk, ImageChunk, TokenChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.diagnostics import (
    RunnerDiagnosticContext,
    RunnerDiagnosticUpdate,
)
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from skulk.shared.types.tasks import (
    AudioTranscription,
    ImageGeneration,
    RealtimeAudioTranscription,
    SpeechSynthesis,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerRunning
from skulk.utils.channels import MpSender, channel, mp_channel
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.unittests.conftest import (
    get_bound_mlx_ring_instance,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


class _DeadProcess:
    pid = 123
    exitcode = -6

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def join(self, _timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


@pytest.mark.asyncio
async def test_trace_payload_routes_directly_to_owning_api() -> None:
    """Per-rank trace data must never fall back to the event sender."""

    trace_sender, trace_receiver = channel[TraceDataPacket](1)
    event_sender, event_receiver = channel[Event](1)
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, event_process_receiver = mp_channel[Event]()
    _, diagnostics_receiver = mp_channel[RunnerDiagnosticUpdate]()
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("trace-instance"),
        model_id=ModelId("mlx-community/trace-test"),
        runner_id=RunnerId("trace-runner"),
        node_id=NodeId("trace-worker"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=event_process_receiver,
        _diag_recv=diagnostics_receiver,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _trace_sender=trace_sender,
    )
    task_id = TaskId("trace-task")
    supervisor._trace_routes[task_id] = (  # pyright: ignore[reportPrivateUsage]
        NodeId("api-owner"),
        (0,),
        0.0,
    )
    trace = TraceEventData(
        name="decode",
        start_us=1,
        duration_us=2,
        rank=0,
        category="inference",
    )

    await supervisor._emit(  # pyright: ignore[reportPrivateUsage]
        TracesCollected(task_id=task_id, rank=0, traces=[trace])
    )
    packet = await trace_receiver.receive()

    assert packet.owner_node == NodeId("api-owner")
    assert packet.source_node == NodeId("trace-worker")
    assert packet.traces == (trace,)
    assert event_receiver.collect() == []


@pytest.mark.asyncio
async def test_generated_output_without_data_sender_never_uses_control_sender() -> None:
    """A local wiring defect must not put payload onto the control plane."""

    event_sender, event_receiver = channel[Event](1)
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, event_process_receiver = mp_channel[Event]()
    _, diagnostics_receiver = mp_channel[RunnerDiagnosticUpdate]()
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("missing-data-instance"),
        model_id=ModelId("mlx-community/data-test"),
        runner_id=RunnerId("missing-data-runner"),
        node_id=NodeId("missing-data-worker"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=event_process_receiver,
        _diag_recv=diagnostics_receiver,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    await supervisor._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(
            command_id=CommandId("missing-data-command"),
            chunk=TokenChunk(
                model=ModelId("mlx-community/data-test"),
                text="must not cross planes",
                token_id=0,
                usage=None,
                finish_reason=None,
            ),
        )
    )

    assert event_receiver.collect() == []


class _ReapTrackingProcess:
    """A live process stub that records whether it was reaped.

    Stays alive until ``join``/``terminate``/``kill`` is called, so the
    supervisor's watcher/forwarder subtasks block (keeping ``run()`` alive)
    until the test cancels it externally — mirroring a worker shutdown on a
    master-election transition.
    """

    pid = 123
    exitcode = 0

    def __init__(self) -> None:
        self.joins = 0
        self.closed = False
        self._alive = True

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, _timeout: float | None = None) -> None:
        self.joins += 1
        self._alive = False

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False

    def close(self) -> None:
        self.closed = True


class _FailingRealtimeSender:
    """IPC sender stub for multiprocessing queue-close race coverage."""

    def __init__(self, failure: Exception) -> None:
        self._failure = failure

    async def send_async(self, _frame: RealtimeAudioInputFrame) -> None:
        raise self._failure


@pytest.mark.asyncio
async def test_teardown_reaps_runner_process_under_cancellation() -> None:
    """The runner process must be reaped even when run() is cancelled.

    Regression: on a master-election transition the worker is torn down via
    ``worker.shutdown()``, which cancels the worker task group and so cancels
    each ``RunnerSupervisor.run()``. The teardown ``finally`` reaps the runner
    process (Metal reclaims its wired GPU memory on exit). Without shielding,
    the first ``await`` in that ``finally`` (the process join) re-raised
    CancelledError immediately, so the process was never reaped — it lingered
    holding GPU memory. The replacement worker then planned CreateRunner for the
    same shard, the pre-load memory guard saw the not-yet-reclaimed memory,
    falsely refused, and #290 deleted the carried instance (master failover
    silently killed a healthy serving instance). The teardown is now shielded,
    so the process is joined/closed before run() returns.
    """
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = _ReapTrackingProcess()
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, proc)),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(supervisor.run)
            # let run() start the process + its watcher/forwarder subtasks
            await anyio.sleep(0.2)
            assert proc.is_alive()
            # simulate worker.shutdown() cancelling the worker task group
            tg.cancel_scope.cancel()

    # despite the cancellation, the teardown must have reaped the process so its
    # GPU memory is reclaimed before the replacement worker admits a new runner.
    assert proc.joins >= 1, "process was not joined during cancelled teardown"
    assert not proc.is_alive()
    assert proc.closed, "process.close() was skipped during cancelled teardown"

    event_sender.close()


@pytest.mark.asyncio
async def test_emit_stamps_explicit_lifecycle_and_sequence() -> None:
    """Each DataChunk gets an explicit lifecycle kind and ordered sequence.

    Sequence zero is the lifecycle start; payload and terminal frames follow in
    generation order. Producer state stays available until terminal task status
    so a status-only terminal path can still close the stream.
    """
    from skulk.shared.types.chunks import DataChunk
    from skulk.shared.types.events import ChunkGenerated

    data_sender, data_recv = channel[DataChunk]()
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _data_sender=data_sender,
    )
    cmd = CommandId("cmd-seq")

    def chunk(text: str, finish: str | None) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=text,
            token_id=0,
            usage=None,
            finish_reason=finish,  # pyright: ignore[reportArgumentType]
        )

    await supervisor._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(command_id=cmd, chunk=chunk("a", None))
    )
    await supervisor._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(command_id=cmd, chunk=chunk("b", None))
    )
    assert supervisor._chunk_sequence[cmd] == 3  # pyright: ignore[reportPrivateUsage]
    await supervisor._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(command_id=cmd, chunk=chunk("c", "stop"))
    )
    assert supervisor._chunk_sequence[cmd] == 4  # pyright: ignore[reportPrivateUsage]
    assert cmd in supervisor._stream_terminal  # pyright: ignore[reportPrivateUsage]

    frames = [data_recv.receive_nowait() for _ in range(4)]
    assert [frame.sequence for frame in frames] == [0, 1, 2, 3]
    assert [frame.kind for frame in frames] == [
        "started",
        "chunk",
        "chunk",
        "completed",
    ]
    assert frames[0].chunk is None
    assert isinstance(frames[-1].chunk, TokenChunk)
    data_sender.close()
    event_sender.close()


@pytest.mark.asyncio
async def test_emit_stamps_owner_node_on_every_lifecycle_frame() -> None:
    """_emit addresses each DataChunk to the command's owning API node.

    #279 Phase 2: the owner is recorded when the serving task starts and stamped
    onto every output chunk so the Zenoh data plane keys to data/<owner_node>.
    The owner remains until terminal task status so a status-only terminal frame
    can still be addressed to the same API node.
    """
    from skulk.shared.types.chunks import DataChunk
    from skulk.shared.types.events import ChunkGenerated

    data_sender, data_recv = channel[DataChunk]()
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _data_sender=data_sender,
    )
    cmd = CommandId("cmd-owner")
    owner = NodeId("api-node-7")
    # The owner is recorded by start_task; set it directly here (start_task
    # blocks on the runner ack, which a dead process never sends).
    supervisor._command_owner[cmd] = owner  # pyright: ignore[reportPrivateUsage]

    def chunk(text: str, finish: str | None) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=text,
            token_id=0,
            usage=None,
            finish_reason=finish,  # pyright: ignore[reportArgumentType]
        )

    await supervisor._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(command_id=cmd, chunk=chunk("a", None))
    )
    await supervisor._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(command_id=cmd, chunk=chunk("b", "stop"))
    )
    assert supervisor._command_owner[cmd] == owner  # pyright: ignore[reportPrivateUsage]

    frames = [data_recv.receive_nowait() for _ in range(3)]
    assert [frame.owner_node for frame in frames] == [owner, owner, owner]
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    data_sender.close()
    event_sender.close()


@pytest.mark.asyncio
async def test_only_rank_zero_emits_multi_rank_data_lifecycle() -> None:
    """A non-output rank cannot terminate a remote stream before rank zero."""

    from skulk.shared.types.chunks import DataChunk

    data_sender, data_receiver = channel[DataChunk]()
    event_sender, _ = channel[Event]()
    model_id = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")
    rank_zero_bound = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-multi-rank"),
        model_id=model_id,
        runner_id=RunnerId("runner-zero"),
        node_id=NodeId("node-zero"),
    )
    rank_one_bound = BoundInstance(
        instance=rank_zero_bound.instance,
        bound_runner_id=RunnerId("other_runner"),
        bound_node_id=NodeId("other_node"),
    )

    def supervisor(bound_instance: BoundInstance) -> RunnerSupervisor:
        task_sender, _ = mp_channel[Task]()
        cancel_sender, _ = mp_channel[TaskId]()
        _, event_receiver = mp_channel[Event]()
        _, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
        return RunnerSupervisor(
            shard_metadata=bound_instance.bound_shard,
            bound_instance=bound_instance,
            runner_process=cast("mp.Process", cast(object, _DeadProcess())),
            initialize_timeout=400,
            _ev_recv=event_receiver,
            _diag_recv=diagnostic_receiver,
            _task_sender=task_sender,
            _event_sender=event_sender,
            _cancel_sender=cancel_sender,
            _data_sender=data_sender,
        )

    rank_zero = supervisor(rank_zero_bound)
    rank_one = supervisor(rank_one_bound)
    command_id = CommandId("cmd-multi-rank")
    owner_node = NodeId("remote-api-owner")

    def generation_task(task_id: str, bound: BoundInstance) -> TextGeneration:
        return TextGeneration(
            task_id=TaskId(task_id),
            instance_id=bound.instance.instance_id,
            command_id=command_id,
            owner_node=owner_node,
            task_params=TextGenerationTaskParams(
                model=model_id,
                input=[InputMessage(role="user", content="hi")],
                stream=True,
            ),
        )

    rank_zero_task = generation_task("task-zero", rank_zero_bound)
    rank_one_task = generation_task("task-one", rank_one_bound)
    rank_zero._command_owner[command_id] = owner_node  # pyright: ignore[reportPrivateUsage]

    await rank_one._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(
            command_id=command_id,
            chunk=TokenChunk(
                model=model_id,
                text="ignored",
                token_id=0,
                usage=None,
                finish_reason=None,
            ),
        )
    )
    await rank_one._finish_data_stream(  # pyright: ignore[reportPrivateUsage]
        rank_one_task, TaskStatus.Complete
    )
    await rank_zero._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(
            command_id=command_id,
            chunk=TokenChunk(
                model=model_id,
                text="blue",
                token_id=0,
                usage=None,
                finish_reason=None,
            ),
        )
    )
    await rank_zero._finish_data_stream(  # pyright: ignore[reportPrivateUsage]
        rank_zero_task, TaskStatus.Complete
    )

    frames = [data_receiver.receive_nowait() for _ in range(3)]
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert isinstance(frames[1].chunk, TokenChunk)
    assert frames[1].chunk.text == "blue"
    assert command_id not in rank_one._chunk_sequence  # pyright: ignore[reportPrivateUsage]
    data_sender.close()
    event_sender.close()


@pytest.mark.asyncio
async def test_image_data_lifecycle_uses_primary_output_rank() -> None:
    """Image DATA follows the terminal pipeline stage that emits image chunks."""

    from skulk.shared.types.chunks import DataChunk

    data_sender, data_receiver = channel[DataChunk]()
    event_sender, _ = channel[Event]()
    model_id = ModelId("exolabs/test-image-model")
    rank_zero_runner = RunnerId("image-runner-zero")
    output_runner = RunnerId("image-runner-output")
    rank_zero_node = NodeId("image-node-zero")
    output_node = NodeId("image-node-output")

    def image_shard(device_rank: int):
        shard = get_pipeline_shard_metadata(model_id, device_rank, world_size=2)
        image_card = shard.model_card.model_copy(
            update={"tasks": [ModelTask.TextToImage]}
        )
        return shard.model_copy(update={"model_card": image_card})

    instance = get_mlx_ring_instance(
        instance_id=InstanceId("image-instance"),
        model_id=model_id,
        node_to_runner={
            rank_zero_node: rank_zero_runner,
            output_node: output_runner,
        },
        runner_to_shard={
            rank_zero_runner: image_shard(0),
            output_runner: image_shard(1),
        },
    )
    rank_zero_bound = BoundInstance(
        instance=instance,
        bound_runner_id=rank_zero_runner,
        bound_node_id=rank_zero_node,
    )
    output_bound = BoundInstance(
        instance=instance,
        bound_runner_id=output_runner,
        bound_node_id=output_node,
    )

    def supervisor(bound_instance: BoundInstance) -> RunnerSupervisor:
        task_sender, _ = mp_channel[Task]()
        cancel_sender, _ = mp_channel[TaskId]()
        _, event_receiver = mp_channel[Event]()
        _, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
        return RunnerSupervisor(
            shard_metadata=bound_instance.bound_shard,
            bound_instance=bound_instance,
            runner_process=cast("mp.Process", cast(object, _DeadProcess())),
            initialize_timeout=400,
            _ev_recv=event_receiver,
            _diag_recv=diagnostic_receiver,
            _task_sender=task_sender,
            _event_sender=event_sender,
            _cancel_sender=cancel_sender,
            _data_sender=data_sender,
        )

    rank_zero = supervisor(rank_zero_bound)
    output_rank = supervisor(output_bound)
    command_id = CommandId("cmd-image-output-rank")
    owner_node = NodeId("remote-image-api-owner")

    def image_task(task_id: str, bound: BoundInstance) -> ImageGeneration:
        return ImageGeneration(
            task_id=TaskId(task_id),
            instance_id=bound.instance.instance_id,
            command_id=command_id,
            owner_node=owner_node,
            task_params=ImageGenerationTaskParams(
                prompt="a blue square",
                model=model_id,
            ),
        )

    rank_zero_task = image_task("image-task-zero", rank_zero_bound)
    output_task = image_task("image-task-output", output_bound)
    output_rank._command_owner[command_id] = owner_node  # pyright: ignore[reportPrivateUsage]

    def image_chunk(data: str) -> ImageChunk:
        return ImageChunk(
            model=model_id,
            data=data,
            chunk_index=0,
            total_chunks=1,
            image_index=0,
            format="png",
        )

    await rank_zero._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(command_id=command_id, chunk=image_chunk("ignored"))
    )
    await rank_zero._finish_data_stream(  # pyright: ignore[reportPrivateUsage]
        rank_zero_task, TaskStatus.Complete
    )
    await output_rank._emit(  # pyright: ignore[reportPrivateUsage]
        ChunkGenerated(command_id=command_id, chunk=image_chunk("image-data"))
    )
    await output_rank._finish_data_stream(  # pyright: ignore[reportPrivateUsage]
        output_task, TaskStatus.Complete
    )

    frames = [data_receiver.receive_nowait() for _ in range(3)]
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert isinstance(frames[1].chunk, ImageChunk)
    assert frames[1].chunk.data == "image-data"
    assert command_id not in rank_zero._chunk_sequence  # pyright: ignore[reportPrivateUsage]
    data_sender.close()
    event_sender.close()


@pytest.mark.asyncio
async def test_check_runner_emits_errors_for_inflight_generation_and_realtime() -> None:
    from skulk.shared.types.chunks import DataChunk

    event_sender, event_receiver = channel[Event]()
    data_sender, data_receiver = channel[DataChunk]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _data_sender=data_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content="hi")],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    realtime_command_id = CommandId("cmd-realtime")
    realtime_task = RealtimeAudioTranscription(
        task_id=TaskId("task-realtime"),
        instance_id=bound_instance.instance.instance_id,
        command_id=realtime_command_id,
        owner_node=NodeId("api-node"),
        task_params=RealtimeAudioTranscriptionTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input_sample_rate=16000,
        ),
    )
    supervisor.in_progress[realtime_task.task_id] = realtime_task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_status = await event_receiver.receive()
    frames = [data_receiver.receive_nowait() for _ in range(4)]

    assert [frame.command_id for frame in frames] == [
        command_id,
        command_id,
        realtime_command_id,
        realtime_command_id,
    ]
    assert [frame.kind for frame in frames] == [
        "started",
        "failed",
        "started",
        "failed",
    ]
    assert all(frame.chunk is None for frame in frames[::2])
    assert all(isinstance(frame.chunk, ErrorChunk) for frame in frames[1::2])
    generation_error = frames[1].chunk
    realtime_error = frames[3].chunk
    assert isinstance(generation_error, ErrorChunk)
    assert isinstance(realtime_error, ErrorChunk)
    assert (
        "Runner shutdown before completing command"
        in generation_error.error_message
    )
    assert (
        "Runner shutdown before completing command"
        in realtime_error.error_message
    )

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    data_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.asyncio
async def test_non_output_rank_crash_fails_task_without_publishing_data() -> None:
    """A non-output rank reports failure without competing on the DATA stream."""

    from skulk.shared.types.chunks import DataChunk

    event_sender, event_receiver = channel[Event]()
    data_sender, data_receiver = channel[DataChunk]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, event_pipe_receiver = mp_channel[Event]()
    _, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
    model_id = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")
    rank_zero_bound = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-multi-rank-failure"),
        model_id=model_id,
        runner_id=RunnerId("runner-zero"),
        node_id=NodeId("node-zero"),
    )
    rank_one_bound = BoundInstance(
        instance=rank_zero_bound.instance,
        bound_runner_id=RunnerId("other_runner"),
        bound_node_id=NodeId("other_node"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=rank_one_bound.bound_shard,
        bound_instance=rank_one_bound,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=event_pipe_receiver,
        _diag_recv=diagnostic_receiver,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _data_sender=data_sender,
    )
    command_id = CommandId("cmd-rank-failure")
    task = TextGeneration(
        task_id=TaskId("task-rank-failure"),
        instance_id=rank_one_bound.instance.instance_id,
        command_id=command_id,
        owner_node=NodeId("remote-api-owner"),
        task_params=TextGenerationTaskParams(
            model=model_id,
            input=[InputMessage(role="user", content="hi")],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    failure = await event_receiver.receive()
    status = await event_receiver.receive()
    assert isinstance(failure, TaskFailed)
    assert failure.task_id == task.task_id
    assert failure.error_type == "runner_terminated"
    assert "Runner shutdown before completing command" in failure.error_message
    assert isinstance(status, RunnerStatusUpdated)
    assert isinstance(status.runner_status, RunnerFailed)
    assert data_receiver.collect() == []
    data_sender.close()
    event_sender.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [anyio.ClosedResourceError(), ValueError("closed"), OSError("closed")],
)
async def test_realtime_audio_closed_ipc_invokes_runner_failure_handling(
    failure: Exception,
) -> None:
    """A closed audio pipe must not escape into the worker task group."""

    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _realtime_audio_sender=cast(
            "MpSender[RealtimeAudioInputFrame]",
            cast(object, _FailingRealtimeSender(failure)),
        ),
    )
    failures: list[Exception] = []

    async def check_runner(e: Exception) -> None:
        failures.append(e)

    supervisor._check_runner = check_runner  # pyright: ignore[reportPrivateUsage]
    await supervisor.send_realtime_audio(
        RealtimeAudioInputFrame(
            command_id=CommandId("realtime-command"),
            sequence=0,
            kind="chunk",
            data=b"\x00\x00",
        )
    )

    assert len(failures) == 1
    assert failures[0] is failure
    event_sender.close()


@pytest.mark.asyncio
async def test_start_task_records_speech_owner_and_terminal_status_clears_it() -> None:
    """Speech output keeps the command owner needed for Zenoh DATA routing."""

    from skulk.shared.types.chunks import DataChunk

    event_sender, event_receiver = channel[Event]()
    data_sender, data_receiver = channel[DataChunk]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _data_sender=data_sender,
    )
    supervisor.status = RunnerRunning()

    command_id = CommandId("cmd-speech-owner")
    owner = NodeId("api-node-speech")
    task = SpeechSynthesis(
        task_id=TaskId("task-speech-owner"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        owner_node=owner,
        task_params=SpeechSynthesisTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input_text="say this",
        ),
    )

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
            tg.start_soon(supervisor.start_task, task)

            dispatched = await to_thread.run_sync(
                task_receiver.receive_timeout,
                1,
            )
            assert isinstance(dispatched, SpeechSynthesis)
            assert dispatched.command_id == command_id
            assert supervisor._command_owner[command_id] == owner  # pyright: ignore[reportPrivateUsage]

            ev_send.send(TaskAcknowledged(task_id=task.task_id))
            ev_send.send(
                TaskStatusUpdated(
                    task_id=task.task_id,
                    task_status=TaskStatus.Complete,
                )
            )

            forwarded_status = await event_receiver.receive()
            assert isinstance(forwarded_status, TaskStatusUpdated)
            assert forwarded_status.task_id == task.task_id
            assert command_id not in supervisor._command_owner  # pyright: ignore[reportPrivateUsage]
            assert command_id not in supervisor._chunk_sequence  # pyright: ignore[reportPrivateUsage]
            assert task.task_id not in supervisor.in_progress

            frames = [data_receiver.receive_nowait() for _ in range(2)]
            assert [frame.sequence for frame in frames] == [0, 1]
            assert [frame.kind for frame in frames] == ["started", "completed"]
            assert all(frame.owner_node == owner for frame in frames)

            tg.cancel_scope.cancel()

    ev_send.close()
    data_sender.close()
    event_sender.close()


@pytest.mark.asyncio
async def test_start_task_records_transcription_owner_and_terminal_status_clears_it() -> None:
    """STT output keeps the command owner needed for Zenoh DATA routing."""

    event_sender, event_receiver = channel[Event]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerRunning()

    command_id = CommandId("cmd-transcription-owner")
    owner = NodeId("api-node-transcription")
    task = AudioTranscription(
        task_id=TaskId("task-transcription-owner"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        owner_node=owner,
        task_params=AudioTranscriptionTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            total_input_chunks=1,
            audio_sha256="abc123",
        ),
    )

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
            tg.start_soon(supervisor.start_task, task)

            dispatched = await to_thread.run_sync(
                task_receiver.receive_timeout,
                1,
            )
            assert isinstance(dispatched, AudioTranscription)
            assert dispatched.command_id == command_id
            assert supervisor._command_owner[command_id] == owner  # pyright: ignore[reportPrivateUsage]

            ev_send.send(TaskAcknowledged(task_id=task.task_id))
            ev_send.send(
                TaskStatusUpdated(
                    task_id=task.task_id,
                    task_status=TaskStatus.Complete,
                )
            )

            forwarded_status = await event_receiver.receive()
            assert isinstance(forwarded_status, TaskStatusUpdated)
            assert forwarded_status.task_id == task.task_id
            assert command_id not in supervisor._command_owner  # pyright: ignore[reportPrivateUsage]
            assert task.task_id not in supervisor.in_progress

            tg.cancel_scope.cancel()

    ev_send.close()
    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.asyncio
async def test_realtime_transcription_failure_emits_terminal_data_frame() -> None:
    """A status-only realtime failure must terminate DATA and clear ownership."""

    from skulk.shared.types.chunks import DataChunk

    event_sender, event_receiver = channel[Event]()
    data_sender, data_receiver = channel[DataChunk]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
        _data_sender=data_sender,
    )
    supervisor.status = RunnerRunning()
    command_id = CommandId("cmd-realtime-transcription")
    owner = NodeId("api-node-realtime")
    task = RealtimeAudioTranscription(
        task_id=TaskId("task-realtime-transcription"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        owner_node=owner,
        task_params=RealtimeAudioTranscriptionTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input_sample_rate=16000,
        ),
    )

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
            task_group.start_soon(supervisor.start_task, task)
            dispatched = await to_thread.run_sync(task_receiver.receive_timeout, 1)
            assert isinstance(dispatched, RealtimeAudioTranscription)
            ev_send.send(TaskAcknowledged(task_id=task.task_id))
            ev_send.send(
                TaskStatusUpdated(
                    task_id=task.task_id,
                    task_status=TaskStatus.Failed,
                )
            )

            forwarded_status = await event_receiver.receive()
            assert isinstance(forwarded_status, TaskStatusUpdated)
            frames = [data_receiver.receive_nowait() for _ in range(2)]
            assert [frame.kind for frame in frames] == ["started", "failed"]
            assert all(frame.owner_node == owner for frame in frames)
            assert command_id not in supervisor._command_owner  # pyright: ignore[reportPrivateUsage]
            assert task.task_id not in supervisor.in_progress
            task_group.cancel_scope.cancel()

    ev_send.close()
    data_sender.close()
    event_sender.close()


@pytest.mark.asyncio
async def test_cancelled_task_event_clears_in_progress_and_emits_delete() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerRunning()

    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-a"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content="hi")],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    forwarded_cancel: Event | None = None
    forwarded_delete: Event | None = None

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        ev_send.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Cancelled))
        forwarded_cancel = await event_receiver.receive()
        forwarded_delete = await event_receiver.receive()
        tg.cancel_scope.cancel()

    ev_send.close()

    assert isinstance(forwarded_cancel, TaskStatusUpdated)
    assert forwarded_cancel.task_status == TaskStatus.Cancelled
    assert isinstance(forwarded_delete, TaskDeleted)
    assert forwarded_delete.task_id == task.task_id
    assert task.task_id not in supervisor.in_progress
    assert task.task_id in supervisor.cancelled

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.asyncio
async def test_terminal_task_status_clears_chunk_sequence_counter() -> None:
    """A terminal task status (e.g. cancel) clears the per-command counter.

    #301 review (Codex P2): commands that end without a finish_reason chunk
    (image generation, or cancel/fail terminal TaskStatusUpdated that bypasses
    _emit) would otherwise leak one _chunk_sequence entry per command on a
    long-lived runner.
    """
    from skulk.shared.types.text_generation import (
        InputMessage,
        TextGenerationTaskParams,
    )

    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerRunning()

    cmd = CommandId("cmd-x")
    task = TextGeneration(
        task_id=TaskId("task-x"),
        instance_id=bound_instance.instance.instance_id,
        command_id=cmd,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content="hi")],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor._chunk_sequence[cmd] = 5  # pyright: ignore[reportPrivateUsage]  # mid-stream

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        ev_send.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Cancelled)
        )
        await event_receiver.receive()  # forwarded status
        await event_receiver.receive()  # TaskDeleted
        tg.cancel_scope.cancel()

    ev_send.close()
    assert cmd not in supervisor._chunk_sequence  # pyright: ignore[reportPrivateUsage]
    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.asyncio
async def test_runner_flight_recorder_retains_newest_entries() -> None:
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    context = RunnerDiagnosticContext(
        node_id="node-a",
        runner_id="runner-a",
        pid=123,
        instance_id="instance-a",
        model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
        rank=0,
        world_size=1,
        start_layer=0,
        end_layer=30,
        n_layers=30,
    )

    for idx in range(140):
        supervisor._apply_diagnostic_update(  # pyright: ignore[reportPrivateUsage]
            RunnerDiagnosticUpdate(
                at=f"2026-04-23T00:00:{idx:02d}+00:00",
                phase="decode_stream",
                event=f"token_{idx}",
                context=context,
                task_id="task-a",
            )
        )

    diagnostics = supervisor.diagnostics()
    assert diagnostics.phase == "decode_stream"
    assert len(diagnostics.flight_recorder) == 128
    assert diagnostics.flight_recorder[0].event == "token_12"
    assert diagnostics.flight_recorder[-1].event == "token_139"


@pytest.mark.asyncio
async def test_duplicate_terminal_status_is_forwarded_exactly_once() -> None:
    """A re-reported terminal status must not mint another event pair (#278).

    An idle SequentialGenerator used to re-report every ever-cancelled task
    on every step; the supervisor converted each report into a fresh
    TaskStatusUpdated(Cancelled) + TaskDeleted pair, flooding the cluster
    log. The supervisor now forwards a terminal status at most once.
    """
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    ev_send, ev_recv = mp_channel[Event]()
    _, diag_recv = mp_channel[RunnerDiagnosticUpdate]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=cast("mp.Process", cast(object, _DeadProcess())),
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _diag_recv=diag_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )
    supervisor.status = RunnerRunning()

    task_id = TaskId("task-a")
    forwarded: list[Event] = []

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        ev_send.send(
            TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Cancelled)
        )
        # first report: status + explicit delete
        forwarded.append(await event_receiver.receive())
        forwarded.append(await event_receiver.receive())
        # duplicate report: must be suppressed entirely
        ev_send.send(
            TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Cancelled)
        )
        with anyio.move_on_after(0.3):
            forwarded.append(await event_receiver.receive())
        tg.cancel_scope.cancel()

    ev_send.close()

    assert len(forwarded) == 2
    assert isinstance(forwarded[0], TaskStatusUpdated)
    assert isinstance(forwarded[1], TaskDeleted)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()
