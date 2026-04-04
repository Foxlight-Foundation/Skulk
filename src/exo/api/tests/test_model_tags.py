# pyright: reportPrivateUsage=false
"""Tests for derived model tags exposed by the API."""

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


def test_model_tags_include_vision() -> None:
    """Vision-capable models should expose a `vision` display tag."""
    card = ModelCard(
        model_id=ModelId("google/gemma-4b-it"),
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "vision", "thinking"],
        quantization="4bit",
    )

    assert API._model_tags(card) == ["thinking", "vision", "tensor"]
