"""Pure-mapping tests for ``get_eos_token_ids_for_model``.

These assert the per-model EOS overrides without loading or downloading any
tokenizer, so they run in normal CI. They deliberately live in their own module
(no ``pytestmark = pytest.mark.slow``) because the tokenizer tests in
``test_tokenizers.py`` are marked slow and CI runs ``pytest -m "not slow"`` — a
regression test placed there would be silently deselected and protect nothing.
"""

from skulk.shared.models.model_cards import ModelId
from skulk.worker.engines.mlx.utils_mlx import get_eos_token_ids_for_model


def test_moonlight_eos_includes_im_end() -> None:
    """Moonlight must stop on <|im_end|> (163586), not just its [EOS] (163585).

    Moonlight's tokenizer eos_token is [EOS] but its chat template ends assistant
    turns with <|im_end|>. Without 163586 in the eos set the turn token leaks into
    output and generation runs past the turn boundary (Skulk #304).
    """
    eos = get_eos_token_ids_for_model(
        ModelId("mlx-community/Moonlight-16B-A3B-Instruct-4-bit")
    )
    assert eos is not None, "Moonlight should have explicit eos token ids"
    assert 163586 in eos, "Moonlight eos must include <|im_end|> (163586)"
    assert 163585 in eos, "Moonlight eos should retain [EOS] (163585)"


def test_known_family_eos_overrides() -> None:
    """The other explicit-EOS families keep their documented token ids."""
    assert get_eos_token_ids_for_model(ModelId("moonshotai/Kimi-K2-Instruct")) == [
        163586
    ]
    assert get_eos_token_ids_for_model(ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8")) == [
        200002,
        200012,
    ]
    assert get_eos_token_ids_for_model(ModelId("mlx-community/Qwen3.5-27B-4bit")) == [
        248046,
        248044,
    ]


def test_unknown_model_uses_tokenizer_config() -> None:
    """Models without an explicit override return None (use tokenizer config)."""
    assert (
        get_eos_token_ids_for_model(ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"))
        is None
    )
