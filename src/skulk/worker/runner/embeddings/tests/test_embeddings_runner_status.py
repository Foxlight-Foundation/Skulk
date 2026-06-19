# pyright: reportPrivateUsage=false, reportMissingParameterType=false
"""Embedding runner status sequencing (#326).

The supervisor asserts the runner is in an *active* status (RunnerRunning /
WarmingUp / Loading / Connecting / ShuttingDown) when it forwards a task's
terminal TaskStatus. A TextEmbedding task that stays RunnerReady across the
forward pass trips that assert and aborts the event forwarder, so the embedding
task never reaches a clean terminal state. This test pins the fixed ordering:
the last runner status emitted before the terminal Complete is RunnerRunning.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import EmbeddingChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.embedding import TextEmbeddingTaskParams
from skulk.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import TaskStatus, TextEmbedding
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerRunning,
)
from skulk.worker.runner.embeddings import runner as embeddings_runner
from skulk.worker.runner.embeddings.runner import Runner


class _CaptureSender:
    """Stand-in for the runner's MpSender that records every emitted event."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, item: Event) -> None:
        self.events.append(item)


class _OneShotReceiver:
    """Stand-in MpReceiver that yields a fixed task list once, then stops."""

    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __enter__(self):
        return iter(self._items)

    def __exit__(self, *_: object) -> bool:
        return False


def _make_runner() -> tuple[Runner, _CaptureSender]:
    sender = _CaptureSender()
    bound = SimpleNamespace(
        instance=SimpleNamespace(),
        bound_runner_id=RunnerId("r1"),
        bound_shard=SimpleNamespace(
            world_size=1,
            model_card=SimpleNamespace(model_id=ModelId("some/embed-model")),
            device_rank=0,
        ),
        bound_node_id=NodeId("n1"),
    )
    runner = Runner(
        bound_instance=cast("object", bound),  # pyright: ignore[reportArgumentType]
        event_sender=cast("object", sender),  # pyright: ignore[reportArgumentType]
        task_receiver=cast("object", _OneShotReceiver([])),  # pyright: ignore[reportArgumentType]
        cancel_receiver=cast("object", SimpleNamespace()),  # pyright: ignore[reportArgumentType]
    )
    return runner, sender


def test_embedding_task_holds_active_status_until_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, sender = _make_runner()

    # Stand in for a loaded model so the TextEmbedding branch runs; the forward
    # pass itself is replaced so the test needs no torch model or weights.
    runner.model = object()
    runner.tokenizer = object()
    runner.current_status = RunnerReady()

    def _fake_forward(*_args: object, **_kwargs: object) -> tuple[list[list[float]], int]:
        return [[0.1, 0.2, 0.3]], 5

    monkeypatch.setattr(embeddings_runner, "_forward", _fake_forward)

    task = TextEmbedding(
        instance_id=InstanceId("i1"),
        command_id=CommandId("c1"),
        task_params=TextEmbeddingTaskParams(
            model=ModelId("some/embed-model"), input_texts=["hello world"]
        ),
    )
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]
    runner.main()

    # The terminal Complete must be preceded by an active runner status, and the
    # supervisor only accepts RunnerRunning among the active states a one-shot
    # embedding task can be in.
    complete_index = next(
        i
        for i, event in enumerate(sender.events)
        if isinstance(event, TaskStatusUpdated)
        and event.task_status == TaskStatus.Complete
    )
    status_before = [
        event.runner_status
        for event in sender.events[:complete_index]
        if isinstance(event, RunnerStatusUpdated)
    ]
    assert status_before, "expected a runner status before the terminal Complete"
    assert isinstance(status_before[-1], RunnerRunning)

    # And the work actually produced an embedding chunk.
    assert any(
        isinstance(getattr(event, "chunk", None), EmbeddingChunk)
        for event in sender.events
    )
