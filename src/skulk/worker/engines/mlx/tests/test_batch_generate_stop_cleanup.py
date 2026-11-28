# pyright: reportPrivateUsage=false
"""Regression tests for MLX SkulkBatchGenerator stop-sequence KV cleanup."""

from types import SimpleNamespace
from typing import cast

import mlx.core as mx
from mlx_lm.generate import BatchGenerator as MlxBatchGenerator
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from skulk.shared.types.common import ModelId
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.worker.engines.mlx.generator.batch_generate import (
    SkulkBatchGenerator,
    _EngineTask,
)


def test_stop_sequence_hit_removes_uid_from_mlx_generator() -> None:
    """User stop-sequence completion must release the MLX BatchGenerator UID."""
    removed_uids: list[list[int]] = []

    class _FakeDetokenizer:
        def __init__(self) -> None:
            self.last_segment = ""

        def add_token(self, _token: int) -> None:
            self.last_segment = "helloSTOP"

        def finalize(self) -> None:
            return None

    class _FakeMlxGen:
        prompt_time = 0.0
        _counters = SimpleNamespace(prompt_time=0.0, generation_steps=1)
        _needs_topk = False
        _unprocessed_sequences: list[object] = []
        _prompt_batch: list[object] = []
        _generation_batch: list[object] = [42]
        _decode_step_count = 1

        def next(self) -> tuple[list[object], list[object]]:
            return (
                [],
                [
                    SimpleNamespace(
                        uid=42,
                        token=7,
                        finish_reason=None,
                        logprobs=None,
                    )
                ],
            )

        def remove(self, uids: list[int]) -> None:
            removed_uids.append(list(uids))

    gen = object.__new__(SkulkBatchGenerator)
    gen._mlx_gen = cast(MlxBatchGenerator, cast(object, _FakeMlxGen()))
    gen._next_time_total = 0.0
    gen.tokenizer = cast(
        TokenizerWrapper,
        cast(object, SimpleNamespace(think_start=None, think_end=None)),
    )
    gen._active_tasks = {
        42: _EngineTask(
            uid=42,
            task_params=TextGenerationTaskParams(
                model=ModelId("org/model"),
                input=[],
                stop="STOP",
            ),
            all_prompt_tokens=mx.array([1, 2, 3]),
            prefix_hit_length=0,
            matched_index=None,
            cache_snapshots=None,
            detokenizer=cast(StreamingDetokenizer, cast(object, _FakeDetokenizer())),
            prefill_tps=10.0,
            generation_time_at_start=0.0,
        )
    }

    results = gen.step()

    assert len(results) == 1
    uid, response = results[0]
    assert uid == 42
    assert response.finish_reason == "stop"
    assert response.text == "hello"
    assert removed_uids == [[42]]
    assert 42 not in gen._active_tasks
