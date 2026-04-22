"""Tests for model-card metadata used by the public API."""

import pytest
from anyio import Path

from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ModelCard


@pytest.mark.anyio
async def test_gemma4_builtin_card_declares_context_length() -> None:
    """Built-in Gemma 4 cards should publish their context length to the UI."""
    card_path = (
        Path(RESOURCES_DIR)
        / "inference_model_cards"
        / "mlx-community--gemma-4-26b-a4b-it-4bit.toml"
    )

    card = await ModelCard.load_from_path(card_path)

    assert card.context_length == 262144
