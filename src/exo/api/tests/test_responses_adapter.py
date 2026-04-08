"""Tests for OpenAI Responses adapter reasoning normalization."""

import pytest

from exo.api.adapters.responses import responses_request_to_text_generation
from exo.api.types.openai_responses import ResponsesRequest
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    ReasoningCardConfig,
)
from exo.shared.types.memory import Memory


@pytest.mark.anyio
async def test_responses_adapter_explicit_disable_uses_profile_disabled_effort() -> None:
    model_id = ModelId("custom/toggleable-responses-model")
    card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="custom",
        capabilities=["text", "thinking"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            default_effort="high",
            disabled_effort="minimal",
        ),
    )
    request = ResponsesRequest(
        model=model_id,
        input="Hello",
        enable_thinking=False,
    )

    params = await responses_request_to_text_generation(request, model_card=card)

    assert params.enable_thinking is False
    assert params.reasoning_effort == "minimal"


@pytest.mark.anyio
async def test_responses_adapter_non_toggleable_model_ignores_disable_inputs() -> None:
    model_id = ModelId("custom/non-toggle-responses-model")
    card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="custom",
        capabilities=["text", "thinking"],
        reasoning=ReasoningCardConfig(
            supports_toggle=False,
            default_effort="high",
            disabled_effort="minimal",
        ),
    )
    request = ResponsesRequest(
        model=model_id,
        input="Hello",
        enable_thinking=False,
    )

    params = await responses_request_to_text_generation(request, model_card=card)

    assert params.enable_thinking is None
    assert params.reasoning_effort is None
