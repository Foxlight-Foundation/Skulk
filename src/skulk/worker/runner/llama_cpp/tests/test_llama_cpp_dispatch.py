# pyright: reportPrivateUsage=false, reportAny=false
"""llama.cpp runner width-1 dispatch (#692).

The runner now routes through ``ServedConcurrentDispatch`` with
``max_concurrency=1``. These tests pin the properties that conversion was for:

- generations stay strictly serial (the ``Llama`` object requires it), with the
  second request admitted only after the first finishes;
- every generation's terminal stats carry the runner ground-truth stamp
  (#596) -- serving node/backend, ``in_flight_at_admission`` (always 1 at
  width 1), and ``serving_batches=False`` -- which this path never had, leaving
  the performance-envelope registry blind to the in-process engine;
- the model is released when the dispatch loop exits.

The mixin's generic behavior (backpressure, cancellation classification, status
ordering) is covered by ``test_served_concurrency.py`` and is not re-tested.
"""

import threading
from types import SimpleNamespace
from typing import Any, cast

from anyio import EndOfStream, WouldBlock

from skulk.api.types import GenerationStats
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.chunks import TokenChunk
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.events import ChunkGenerated, Event, TaskStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerId, RunnerReady
from skulk.worker.runner.llama_cpp.runner import Runner


class _CaptureSender:
    """Thread-safe stand-in for the runner's MpSender; records every event."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def send(self, item: Event) -> None:
        with self._lock:
            self.events.append(item)

    def snapshot(self) -> list[Event]:
        with self._lock:
            return list(self.events)


class _OneShotReceiver:
    """Serves a fixed task list via ``receive_timeout``, then ends the stream."""

    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def receive_timeout(self, _timeout: float) -> object:
        if self._items:
            return self._items.pop(0)
        raise EndOfStream


class _NoCancel:
    def receive_nowait(self) -> object:
        raise WouldBlock


def _make_runner(tasks: list[object]) -> tuple[Runner, _CaptureSender]:
    sender = _CaptureSender()
    card = ModelCard(
        model_id=ModelId("some/gguf-model"),
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model.gguf",
    )
    bound = SimpleNamespace(
        instance=SimpleNamespace(),
        bound_runner_id=RunnerId("r1"),
        bound_shard=SimpleNamespace(
            world_size=1,
            model_card=card,
            device_rank=0,
            resolved_backend="llama_cpp-vulkan",
        ),
        bound_node_id=NodeId("n1"),
    )
    runner = Runner(
        bound_instance=cast("object", bound),  # pyright: ignore[reportArgumentType]
        event_sender=cast("object", sender),  # pyright: ignore[reportArgumentType]
        task_receiver=cast("object", _OneShotReceiver(tasks)),  # pyright: ignore[reportArgumentType]
        cancel_receiver=cast("object", _NoCancel()),  # pyright: ignore[reportArgumentType]
    )
    runner.current_status = RunnerReady()
    return runner, sender


def _gen(n: int) -> TextGeneration:
    return TextGeneration(
        instance_id=InstanceId("i1"),
        command_id=CommandId(f"c{n}"),
        task_params=TextGenerationTaskParams.model_validate(
            {
                "model": ModelId("some/gguf-model"),
                "input": [InputMessage(role="user", content=f"prompt {n}")],
            }
        ),
    )


def _finish_chunk() -> dict[str, Any]:
    return {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}


def _terminal_stats(sender: _CaptureSender) -> list[GenerationStats]:
    return [
        e.chunk.stats
        for e in sender.snapshot()
        if isinstance(e, ChunkGenerated)
        and isinstance(e.chunk, TokenChunk)
        and e.chunk.finish_reason is not None
        and e.chunk.stats is not None
    ]


def test_second_generation_waits_for_the_first() -> None:
    """Width 1: task 2 must not enter the engine while task 1 is streaming."""
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[TaskId] = []
    calls_lock = threading.Lock()
    tasks = [_gen(1), _gen(2)]

    runner, sender = _make_runner(list(tasks))

    def fake_ccc(*, stream: bool, **_kw: object) -> Any:
        assert stream is True

        with calls_lock:
            index = len(calls)
            calls.append(tasks[index].task_id)

        def piece_stream():
            if index == 0:
                first_started.set()
                # A short guard only: the test releases this gate explicitly.
                assert release_first.wait(30), "test gate never released"
            yield _finish_chunk()

        return piece_stream()

    runner.model = SimpleNamespace(create_chat_completion=fake_ccc, n_tokens=0)

    loop = threading.Thread(target=runner.main, daemon=True)
    loop.start()

    assert first_started.wait(30)
    # The first generation is parked inside the engine. At width 1 the second
    # must not have been handed to the engine, however long the loop has run.
    with calls_lock:
        assert len(calls) == 1
    release_first.set()
    loop.join(timeout=30)
    assert not loop.is_alive()

    with calls_lock:
        assert calls == [tasks[0].task_id, tasks[1].task_id]
    completes = [
        e
        for e in sender.snapshot()
        if isinstance(e, TaskStatusUpdated) and e.task_status is TaskStatus.Complete
    ]
    assert len(completes) == 2


def test_terminal_stats_carry_runner_ground_truth() -> None:
    """Every generation is stamped (#596): this path used to stamp nothing."""
    runner, sender = _make_runner([_gen(1), _gen(2)])

    def fake_ccc(*, stream: bool, **_kw: object) -> Any:
        assert stream is True
        return iter([_finish_chunk()])

    runner.model = SimpleNamespace(create_chat_completion=fake_ccc, n_tokens=0)
    runner.main()

    stamped = _terminal_stats(sender)
    assert len(stamped) == 2
    for stats in stamped:
        assert stats.serving_node == "n1"
        assert stats.serving_backend == "llama_cpp-vulkan"
        # Width 1: each request is admitted alone, and the envelope must see
        # this engine as non-batching.
        assert stats.in_flight_at_admission == 1
        assert stats.serving_batches is False


def test_loop_exit_releases_the_model() -> None:
    """EndOfStream drains the loop and frees the in-process model."""
    runner, _sender = _make_runner([])
    runner.model = SimpleNamespace(create_chat_completion=None)
    runner.main()
    assert runner.model is None
