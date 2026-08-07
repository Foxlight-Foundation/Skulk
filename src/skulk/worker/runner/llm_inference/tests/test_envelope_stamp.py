# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false
"""MLX runner stamps its batching mode onto the terminal stats (#596).

The in-process MLX runner batches concurrent requests on the batch generator, so
it must report ``serving_batches`` / ``in_flight_at_admission`` like the served
runners do. Otherwise the API's performance-envelope tap guesses and classifies
every MLX instance as serial, biasing its throughput-vs-concurrency knee. These
tests exercise the generator contract and the runner stamp without a live model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from skulk.api.types import GenerationStats
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import TaskId
from skulk.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from skulk.worker.runner.llm_inference.runner import Runner


def _base_stats() -> GenerationStats:
    return GenerationStats(
        prompt_tps=1.0,
        generation_tps=2.0,
        prompt_tokens=3,
        generation_tokens=4,
        peak_memory_usage=Memory(in_bytes=0),
    )


def test_batch_generator_reports_batching_and_admission() -> None:
    gen: Any = object.__new__(BatchGenerator)
    gen._active_tasks = {}
    gen._admission_concurrency = {TaskId("t1"): 3}
    assert gen.batches is True
    # Reads the captured admission position.
    assert gen.admission_concurrency(TaskId("t1")) == 3
    # Missing capture falls back to the live batch size, never below 1.
    assert gen.admission_concurrency(TaskId("absent")) == 1
    gen._active_tasks = {0: object(), 1: object()}
    assert gen.admission_concurrency(TaskId("absent")) == 2


def test_sequential_generator_is_serial() -> None:
    gen: Any = object.__new__(SequentialGenerator)
    assert gen.batches is False
    assert gen.admission_concurrency(TaskId("t1")) == 1


def test_runner_stamps_batch_mode_onto_stats() -> None:
    runner: Any = object.__new__(Runner)
    gen: Any = object.__new__(BatchGenerator)
    gen._active_tasks = {}
    gen._admission_concurrency = {TaskId("t1"): 4}
    runner.generator = gen
    runner.bound_instance = SimpleNamespace(bound_node_id=NodeId("node-9"))
    runner.shard_metadata = SimpleNamespace(resolved_backend="mlx-metal")

    stamped = runner._stamp_stats(_base_stats(), TaskId("t1"))
    assert stamped is not None
    assert stamped.serving_node == "node-9"
    assert stamped.serving_backend == "mlx-metal"
    assert stamped.in_flight_at_admission == 4  # runner-reported batch position
    assert stamped.serving_batches is True  # the bug: MLX must NOT report serial
    # Base measurements survive the stamp.
    assert stamped.generation_tps == 2.0


def test_runner_stamps_serial_mode_for_sequential_generator() -> None:
    runner: Any = object.__new__(Runner)
    gen: Any = object.__new__(SequentialGenerator)
    runner.generator = gen
    runner.bound_instance = SimpleNamespace(bound_node_id=NodeId("node-2"))
    runner.shard_metadata = SimpleNamespace(resolved_backend="mlx-metal")

    stamped = runner._stamp_stats(_base_stats(), TaskId("t1"))
    assert stamped is not None
    assert stamped.serving_batches is False
    assert stamped.in_flight_at_admission == 1


def test_stamp_passes_through_none() -> None:
    runner: Any = object.__new__(Runner)
    runner.generator = object.__new__(SequentialGenerator)
    runner.bound_instance = SimpleNamespace(bound_node_id=NodeId("n"))
    runner.shard_metadata = SimpleNamespace(resolved_backend="mlx-metal")
    assert runner._stamp_stats(None, TaskId("t1")) is None
