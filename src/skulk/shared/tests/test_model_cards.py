"""Tests for model-card metadata used by the public API."""

import pytest
from anyio import Path

from skulk.shared.constants import RESOURCES_DIR
from skulk.shared.models.model_cards import ModelCard, RuntimeCapabilityCardConfig


def test_vllm_spec_pairing_validator() -> None:
    """The vllm speculative fields must be internally consistent at card load.

    Operator-authored custom cards get the same loud failure the bundled
    invariant suite gives shipped cards: depth without a method, dflash
    without its drafter repo, and a drafter repo under mtp (whose drafter
    lives inside the target checkpoint) are all card-authoring errors that
    would otherwise surface as opaque serve-time failures.
    """
    with pytest.raises(ValueError, match="requires vllm_spec_method"):
        RuntimeCapabilityCardConfig(vllm_spec_num_tokens=2)
    with pytest.raises(ValueError, match="requires vllm_spec_draft_repo"):
        RuntimeCapabilityCardConfig(vllm_spec_method="dflash")
    with pytest.raises(ValueError, match="requires vllm_spec_method 'dflash'"):
        RuntimeCapabilityCardConfig(
            vllm_spec_method="mtp",
            vllm_spec_draft_repo="poolside/Laguna-XS-2.1-DFlash-FP8",
        )
    valid = RuntimeCapabilityCardConfig(
        vllm_spec_method="dflash",
        vllm_spec_num_tokens=15,
        vllm_spec_draft_repo="poolside/Laguna-XS-2.1-DFlash-FP8",
    )
    assert valid.vllm_spec_draft_repo == "poolside/Laguna-XS-2.1-DFlash-FP8"


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
