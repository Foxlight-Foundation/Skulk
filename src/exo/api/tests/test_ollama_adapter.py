"""Tests for Ollama adapter reasoning normalization."""

from exo.api.adapters.ollama import (
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from exo.api.types.ollama_api import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaMessage,
)
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    ReasoningCardConfig,
)
from exo.shared.types.memory import Memory


def test_ollama_chat_adapter_uses_toggleable_model_controls() -> None:
    model_id = ModelId("custom/toggleable-ollama-model")
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
    request = OllamaChatRequest(
        model=model_id,
        messages=[OllamaMessage(role="user", content="Hello")],
        think=False,
    )

    params = ollama_request_to_text_generation(request, model_card=card)

    assert params.enable_thinking is False
    assert params.reasoning_effort == "minimal"


def test_ollama_generate_adapter_ignores_non_toggleable_disable_request() -> None:
    model_id = ModelId("custom/non-toggle-ollama-model")
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
    request = OllamaGenerateRequest(
        model=model_id,
        prompt="Hello",
        think=False,
    )

    params = ollama_generate_request_to_text_generation(request, model_card=card)

    assert params.enable_thinking is None
    assert params.reasoning_effort is None
