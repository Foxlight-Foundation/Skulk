from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PromptRendererType,
    RuntimeCapabilityCardConfig,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.events import Event
from skulk.shared.types.memory import Memory
from skulk.shared.types.mlx import Model
from skulk.shared.types.tasks import TaskId
from skulk.utils.channels import mp_channel
from skulk.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from skulk.worker.runner.llm_inference.runner import Builder


class _FakeTokenizer:
    has_tool_calling = False
    tool_call_start = None
    tool_call_end = None
    tool_parser = None


class _FakeModel:
    layers = []


class _FalseyModel(_FakeModel):
    def __bool__(self) -> bool:
        return False


@pytest.mark.parametrize(
    "kv_backend",
    ["mlx_quantized", "turboquant", "turboquant_adaptive"],
)
def test_builder_forces_sequential_for_quantized_kv_backends(
    monkeypatch: pytest.MonkeyPatch,
    kv_backend: str,
) -> None:
    _, cancel_recv = mp_channel[TaskId]()
    event_send, _ = mp_channel[Event]()

    monkeypatch.setattr(
        "skulk.worker.runner.llm_inference.runner.get_kv_cache_backend",
        lambda: kv_backend,
    )

    builder = Builder(
        model_id=ModelId("test-model"),
        event_sender=event_send,
        cancel_receiver=cancel_recv,
        inference_model=cast(Model, cast(object, _FakeModel())),
        tokenizer=cast(TokenizerWrapper, cast(object, _FakeTokenizer())),
        group=cast(mx.distributed.Group | None, None),
    )

    generator = builder.build()

    assert isinstance(generator, SequentialGenerator)
    assert not isinstance(generator, BatchGenerator)


def test_builder_accepts_falsey_but_present_inference_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cancel_recv = mp_channel[TaskId]()
    event_send, _ = mp_channel[Event]()

    monkeypatch.setattr(
        "skulk.worker.runner.llm_inference.runner.get_kv_cache_backend",
        lambda: "default",
    )

    builder = Builder(
        model_id=ModelId("test-model"),
        event_sender=event_send,
        cancel_receiver=cancel_recv,
        inference_model=cast(Model, cast(object, _FalseyModel())),
        tokenizer=cast(TokenizerWrapper, cast(object, _FakeTokenizer())),
        group=cast(mx.distributed.Group | None, None),
    )

    generator = builder.build()

    assert isinstance(generator, BatchGenerator)
    assert not isinstance(generator, SequentialGenerator)


def test_builder_forces_sequential_for_gemma4_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cancel_recv = mp_channel[TaskId]()
    event_send, _ = mp_channel[Event]()

    monkeypatch.setattr(
        "skulk.worker.runner.llm_inference.runner.get_kv_cache_backend",
        lambda: "default",
    )

    builder = Builder(
        model_id=ModelId("local/custom-gemma-runtime"),
        event_sender=event_send,
        cancel_receiver=cancel_recv,
        inference_model=cast(Model, cast(object, _FakeModel())),
        tokenizer=cast(TokenizerWrapper, cast(object, _FakeTokenizer())),
        group=cast(mx.distributed.Group | None, None),
        model_card=ModelCard(
            model_id=ModelId("local/custom-gemma-runtime"),
            storage_size=Memory(),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            runtime=RuntimeCapabilityCardConfig(
                prompt_renderer=PromptRendererType.Gemma4
            ),
        ),
    )

    generator = builder.build()

    assert isinstance(generator, SequentialGenerator)
    assert not isinstance(generator, BatchGenerator)
