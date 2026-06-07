# pyright: reportPrivateUsage=false, reportAny=false
"""Regression tests for the two launch-E2E smoke crashes (2026-06-05).

1. ``fix_unmatched_think_end_tokens`` crashed the runner during WARMUP when
   a tokenizer's thinking markers are multi-token (mlx-lm's
   ``think_start_id`` raises ValueError; observed on gemma-4-12B-it's
   unified tokenizer).
2. The runner crash then took down the WHOLE NODE: cancelling a task on the
   dead runner built a diagnostic context that read ``.pid`` off a CLOSED
   multiprocessing process object, and the ValueError propagated through
   the worker plan-step.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import cast
from unittest.mock import MagicMock

import mlx.core as mx

from skulk.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from skulk.worker.runner.runner_supervisor import RunnerSupervisor


class _MultiTokenThinkTokenizer:
    """Tokenizer stub mirroring mlx-lm's behavior for multi-token markers."""

    has_thinking = True

    @property
    def think_start_id(self) -> int:
        raise ValueError("The start thinking sequence is more than 1 token")

    @property
    def think_end_id(self) -> int:
        raise ValueError("The end thinking sequence is more than 1 token")


class _SingleTokenThinkTokenizer:
    has_thinking = True
    think_start_id = 7
    think_end_id = 9


def test_multi_token_think_markers_skip_fix_instead_of_crashing() -> None:
    tokens = mx.array([1, 2, 3])
    out = fix_unmatched_think_end_tokens(
        tokens, cast("object", _MultiTokenThinkTokenizer())  # type: ignore[arg-type]
    )
    assert out.tolist() == [1, 2, 3]


def test_single_token_think_markers_still_fix_unmatched_end() -> None:
    # An unmatched end marker gets a synthesized start inserted before it.
    tokens = mx.array([1, 9, 2])
    out = fix_unmatched_think_end_tokens(
        tokens, cast("object", _SingleTokenThinkTokenizer())  # type: ignore[arg-type]
    )
    assert out.tolist() == [1, 7, 9, 2]


def test_diagnostic_context_tolerates_closed_runner_process() -> None:
    # A reaped runner's process object raises ValueError on .pid access;
    # diagnostics must degrade to pid=None, not propagate.
    process = mp.get_context("spawn").Process(target=int)
    process.close()  # closed without ever starting — .pid now raises

    supervisor = MagicMock(spec=RunnerSupervisor)
    supervisor.runner_process = process
    supervisor.bound_instance = MagicMock()
    supervisor.bound_instance.bound_node_id = "node"
    supervisor.bound_instance.bound_runner_id = "runner"
    supervisor.bound_instance.instance.instance_id = "instance"
    supervisor.shard_metadata = MagicMock()
    supervisor.shard_metadata.model_card.model_id = "model"
    supervisor.shard_metadata.device_rank = 0
    supervisor.shard_metadata.world_size = 1
    supervisor.shard_metadata.start_layer = 0
    supervisor.shard_metadata.end_layer = 1
    supervisor.shard_metadata.n_layers = 1

    context = RunnerSupervisor._diagnostic_context(supervisor)
    assert context.pid is None
    assert context.runner_id == "runner"
