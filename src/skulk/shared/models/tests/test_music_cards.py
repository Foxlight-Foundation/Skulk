"""The music task and its model truth remain separate from speech."""

import pytest
from pydantic import ValidationError

from skulk.api.types.api import MusicCapabilitySection
from skulk.shared.backends import platform_compatible_backends
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    MusicCardConfig,
    MusicLyricRequirement,
)
from skulk.shared.types.memory import Memory


def _card(**overrides: object) -> ModelCard:
    payload: dict[str, object] = {
        "model_id": ModelId("audio-cpp/example-music"),
        "storage_size": Memory.from_bytes(1024),
        "n_layers": 1,
        "hidden_size": 1,
        "supports_tensor": False,
        "tasks": ["TextToMusic"],
        "music": {
            "family": "minimax_music3",
            "lyrics": "required",
            "min_seconds": 5,
            "max_seconds": 120,
            "language_model_gguf": "language_model_q4_0.gguf",
            "rvq_depth_decoder_gguf": "rvq_depth_decoder_q8_0.gguf",
            "flow_transformer_gguf": "transformer_q4_0.gguf",
        },
    }
    payload.update(overrides)
    return ModelCard.model_validate(payload)


def test_music_section_round_trips_as_distinct_model_truth() -> None:
    card = _card()
    assert card.audio is None
    assert card.music is not None
    assert card.music.lyrics == MusicLyricRequirement.Required
    assert ModelCard.model_validate_json(card.model_dump_json()).music == card.music
    projected = MusicCapabilitySection.from_model_card(card)
    assert projected is not None
    assert projected.family == "minimax_music3"
    assert projected.lyrics == "required"
    assert projected.output_format == "wav"


def test_music_task_and_section_must_agree() -> None:
    with pytest.raises(ValidationError, match=r"TextToMusic requires a \[music\]"):
        _card(music=None)
    with pytest.raises(ValidationError, match=r"TextToMusic requires a \[music\]"):
        _card(tasks=["TextGeneration"])
    with pytest.raises(ValidationError, match="must remain separate"):
        _card(audio={"kind": "tts"})


def test_music_bounds_and_minimax_lyrics_are_validated() -> None:
    components = {
        "language_model_gguf": "language_model_q4_0.gguf",
        "rvq_depth_decoder_gguf": "rvq_depth_decoder_q8_0.gguf",
        "flow_transformer_gguf": "transformer_q4_0.gguf",
    }
    with pytest.raises(ValidationError, match="requires lyrics"):
        MusicCardConfig.model_validate(
            {"family": "minimax_music3", "lyrics": "optional", "min_seconds": 5, "max_seconds": 60, **components}
        )
    with pytest.raises(ValidationError, match="cannot exceed 120"):
        MusicCardConfig.model_validate(
            {"family": "ace_step_1_5", "lyrics": "optional", "min_seconds": 5, "max_seconds": 121}
        )
    with pytest.raises(ValidationError, match="cannot exceed max_seconds"):
        MusicCardConfig.model_validate(
            {"family": "ace_step_1_5", "lyrics": "optional", "min_seconds": 60, "max_seconds": 5}
        )


def test_music_platform_gate_accepts_only_audio_cpp() -> None:
    assert platform_compatible_backends(
        frozenset({"audio_cpp-cpu", "mlx_audio-metal", "llama_cpp-cpu"}),
        card_serves_vision=False,
        card_serves_music=True,
    ) == frozenset({"audio_cpp-cpu"})
