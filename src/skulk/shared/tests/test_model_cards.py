# pyright: reportPrivateUsage=false
"""Tests for model-card metadata used by the public API."""

import pathlib

import pytest
from anyio import Path

from skulk.shared.constants import RESOURCES_DIR
from skulk.shared.models import model_cards as model_cards_module
from skulk.shared.models.model_cards import ModelCard, RuntimeCapabilityCardConfig
from skulk.shared.types.common import ModelId

_MINIMAL_CARD = """\
model_id = "testorg/override-model"
n_layers = 4
hidden_size = 64
supports_tensor = false
tasks = ["TextGeneration"]
quantization = "{quantization}"

[storage_size]
in_bytes = 1024
"""


@pytest.mark.anyio
async def test_custom_card_overrides_bundled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom card with a bundled card's model id must win (#652).

    The custom directory exists for operator override; before the fix the
    first-wins cache silently kept the bundled card, so the operator's card
    appeared installed while the bundled card actually served.
    """
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "override-model.toml").write_text(
        _MINIMAL_CARD.format(quantization="fp16")
    )
    (custom_dir / "override-model.toml").write_text(
        _MINIMAL_CARD.format(quantization="int4")
    )
    # An invalid custom card must be skipped (with a warning), not abort the
    # load of the valid ones.
    (custom_dir / "broken.toml").write_text("model_id = [not, valid, toml")

    monkeypatch.setattr(model_cards_module, "_card_cache", {})
    await model_cards_module._load_cards_from_dir(
        Path(str(builtin_dir)), is_custom=False
    )
    await model_cards_module._load_cards_from_dir(
        Path(str(custom_dir)), is_custom=True
    )

    card = model_cards_module._card_cache[ModelId("testorg/override-model")]
    assert card.is_custom, "the custom card must replace the bundled card"
    assert card.quantization == "int4"

    # Within the builtin pass, first-wins is preserved: reloading the builtin
    # dir must NOT displace the custom card.
    await model_cards_module._load_cards_from_dir(
        Path(str(builtin_dir)), is_custom=False
    )
    assert model_cards_module._card_cache[
        ModelId("testorg/override-model")
    ].is_custom


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
