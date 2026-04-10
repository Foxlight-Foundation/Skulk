from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PromptRendererType,
    RuntimeCapabilityCardConfig,
)
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import TaskId
from exo.utils.channels import mp_channel
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from exo.worker.runner.llm_inference.runner import Builder


class _FakeTokenizer:
    has_tool_calling = False
    tool_call_start = None
    tool_call_end = None
    tool_parser = None


class _FakeModel:
    layers = []


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
        "exo.worker.runner.llm_inference.runner.get_kv_cache_backend",
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


def test_builder_forces_sequential_for_gemma4_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cancel_recv = mp_channel[TaskId]()
    event_send, _ = mp_channel[Event]()

    monkeypatch.setattr(
        "exo.worker.runner.llm_inference.runner.get_kv_cache_backend",
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
