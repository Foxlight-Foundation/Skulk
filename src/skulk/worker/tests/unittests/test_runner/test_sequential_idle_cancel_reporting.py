"""Regression coverage for the idle cancel-reporting storm (#278).

An idle SequentialGenerator used to return a ``Cancelled`` result for every
ever-cancelled task id on EVERY step without pruning the set — the mint that
fed an unbounded TaskStatusUpdated+TaskDeleted event storm.
"""

# pyright: reportPrivateUsage=false

from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.types.common import ModelId
from skulk.shared.types.events import Event
from skulk.shared.types.mlx import Model
from skulk.shared.types.tasks import CANCEL_ALL_TASKS, TaskId
from skulk.utils.channels import mp_channel
from skulk.worker.runner.llm_inference.batch_generator import (
    Cancelled,
    SequentialGenerator,
)


def _idle_generator(monkeypatch: pytest.MonkeyPatch) -> SequentialGenerator:
    # The idle step path touches none of the model/tokenizer machinery; the
    # only collective (agree_on_tasks) is stubbed out so the test needs no
    # MLX distributed context.
    def _no_agreement(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(SequentialGenerator, "agree_on_tasks", _no_agreement)
    _cancel_sender, cancel_receiver = mp_channel[TaskId]()
    sender, _event_receiver = mp_channel[Event]()
    return SequentialGenerator(
        model=cast(Model, object()),
        tokenizer=cast(TokenizerWrapper, object()),
        group=cast("mx.distributed.Group | None", None),
        kv_prefix_cache=None,
        tool_parser=None,
        model_card=None,
        model_id=ModelId("test-model"),
        device_rank=0,
        cancel_receiver=cancel_receiver,
        event_sender=sender,
    )


def test_idle_step_reports_each_cancellation_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _idle_generator(monkeypatch)
    generator._cancelled_tasks.update({TaskId("task-a"), TaskId("task-b")})

    first = list(generator.step())
    assert sorted(task_id for task_id, _ in first) == ["task-a", "task-b"]
    assert all(isinstance(result, Cancelled) for _, result in first)

    # The storm: every subsequent idle step used to re-report both ids.
    assert list(generator.step()) == []
    assert list(generator.step()) == []


def test_idle_step_preserves_cancel_all_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _idle_generator(monkeypatch)
    generator._cancelled_tasks.update({TaskId("task-a"), CANCEL_ALL_TASKS})

    reported = list(generator.step())
    # CANCEL_ALL is a forward-looking marker (cleared by the next submit),
    # not a task: it is never reported and must survive the prune.
    assert [task_id for task_id, _ in reported] == ["task-a"]
    assert CANCEL_ALL_TASKS in generator._cancelled_tasks
    assert list(generator.step()) == []
    assert CANCEL_ALL_TASKS in generator._cancelled_tasks
