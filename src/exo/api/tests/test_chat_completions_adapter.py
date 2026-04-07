"""Tests for Chat Completions request conversion edge cases."""

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import (
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionRequest,
)
from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    ReasoningCardConfig,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


@pytest.mark.anyio
async def test_single_image_object_builds_multimodal_message() -> None:
    request = ChatCompletionRequest(
        model=ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        messages=[
            ChatCompletionMessage(
                role="user",
                content=ChatCompletionMessageImageUrl(
                    image_url={"url": "data:image/png;base64,AAAA"}
                ),
            )
        ],
    )

    params = await chat_request_to_text_generation(request)

    assert params.images == ["AAAA"]
    assert params.chat_template_messages == [
        {"role": "user", "content": [{"type": "image"}]}
    ]


@pytest.mark.anyio
async def test_multimodal_list_preserves_text_and_image_order() -> None:
    request = ChatCompletionRequest(
        model=ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        messages=[
            ChatCompletionMessage(
                role="user",
                content=[
                    ChatCompletionMessageText(text="before"),
                    ChatCompletionMessageImageUrl(
                        image_url={"url": "data:image/png;base64,BBBB"}
                    ),
                    ChatCompletionMessageText(text="after"),
                ],
            )
        ],
    )

    params = await chat_request_to_text_generation(request)

    assert params.images == ["BBBB"]
    assert params.chat_template_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        }
    ]


@pytest.mark.anyio
async def test_toggleable_model_explicit_false_normalizes_to_disabled_effort() -> None:
    model_id = ModelId("custom/toggleable-model")
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
    request = ChatCompletionRequest(
        model=model_id,
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        enable_thinking=False,
    )

    params = await chat_request_to_text_generation(request, model_card=card)

    assert params.enable_thinking is False
    assert params.reasoning_effort == "minimal"


@pytest.mark.anyio
async def test_non_toggleable_model_ignores_explicit_disable() -> None:
    model_id = ModelId("custom/non-toggle-thinking-model")
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
    request = ChatCompletionRequest(
        model=model_id,
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        enable_thinking=False,
        reasoning_effort="none",
    )

    params = await chat_request_to_text_generation(request, model_card=card)

    assert params.enable_thinking is None
    assert params.reasoning_effort is None
