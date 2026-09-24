"""The music task and its model truth remain separate from speech."""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from skulk.api.types.api import MusicCapabilitySection
from skulk.shared.backends import platform_compatible_backends
from skulk.shared.constants import RESOURCES_DIR
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
        "source_revision": "a" * 40,
        "gguf_file": "language_model_q4_0.gguf",
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
        "artifact_bundle": {
            "bundle_id": "bundle_" + "a" * 52,
            "files": [
                {"path": name, "size_bytes": 1}
                for name in (
                    "language_model_q4_0.gguf",
                    "rvq_depth_decoder_q8_0.gguf",
                    "transformer_q4_0.gguf",
                    "condition_encoder.gguf",
                    "vocoder.gguf",
                    "config.json",
                    "config/condition_encoder.json",
                    "config/language_model.json",
                    "config/rvq_depth_decoder.json",
                    "config/transformer.json",
                    "config/vocoder.json",
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                )
            ],
            "download_size": 13,
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
    with pytest.raises(ValidationError, match="require an artifact bundle"):
        _card(artifact_bundle=None)
    with pytest.raises(ValidationError, match="require an artifact bundle"):
        _card(
            music={
                "family": "ace_step_1_5", "lyrics": "optional",
                "min_seconds": 5, "max_seconds": 60,
            },
            artifact_bundle=None,
        )


def test_curated_music_cards_have_verified_generator_geometry() -> None:
    """Prevent synthetic layer and width placeholders in signed music cards."""
    directory = Path(RESOURCES_DIR) / "music_model_cards"
    expected = {
        "audio-cpp--ACE-Step1.5-Turbo-BF16.toml": (24, 2048),
        "audio-cpp--MiniMax-Music3-GGUF-Q4.toml": (36, 4096),
    }
    for filename, geometry in expected.items():
        card = ModelCard.model_validate(
            tomllib.loads((directory / filename).read_text())
        )
        assert (card.n_layers, card.hidden_size) == geometry
        assert card.placement.backend_preference == ()


@pytest.mark.parametrize(
    "missing",
    [
        "condition_encoder.gguf",
        "config/vocoder.json",
        "tokenizer/tokenizer.json",
        "vocoder.gguf",
    ],
)
def test_minimax_bundle_requires_auxiliary_runtime_files(missing: str) -> None:
    card = _card()
    assert card.artifact_bundle is not None
    files = tuple(file for file in card.artifact_bundle.files if file.path != missing)
    incomplete = card.artifact_bundle.model_copy(
        update={"files": files, "download_size": sum(file.size_bytes for file in files)}
    )
    with pytest.raises(ValidationError, match="omits required runtime files"):
        _card(artifact_bundle=incomplete)


def test_minimax_bundle_checks_paths_relative_to_loader_root() -> None:
    card = _card()
    assert card.artifact_bundle is not None
    rooted = card.artifact_bundle.model_copy(
        update={
            "root": "models",
            "files": tuple(
                file.model_copy(update={"path": f"models/{file.path}"})
                for file in card.artifact_bundle.files
            ),
        }
    )
    assert _card(
        artifact_bundle=rooted,
        gguf_file="models/language_model_q4_0.gguf",
    ).music == card.music


def test_minimax_selected_file_matches_language_model() -> None:
    """Generic GGUF readers must see the same MiniMax component as the runner."""
    with pytest.raises(ValidationError, match="must select its language_model_gguf"):
        _card(gguf_file="vocoder.gguf")


def test_music_bundles_reject_unselected_quant_weights() -> None:
    """One signed music card must not download another quant variant."""
    minimax = _card()
    assert minimax.artifact_bundle is not None
    extra_minimax = minimax.artifact_bundle.files[0].model_copy(
        update={"path": "language_model_q8_0.gguf"}
    )
    minimax_bundle = minimax.artifact_bundle.model_copy(
        update={
            "files": (*minimax.artifact_bundle.files, extra_minimax),
            "download_size": minimax.artifact_bundle.download_size + extra_minimax.size_bytes,
        }
    )
    with pytest.raises(ValidationError, match="contains unselected files"):
        _card(artifact_bundle=minimax_bundle)

    path = Path(RESOURCES_DIR) / "music_model_cards" / "audio-cpp--ACE-Step1.5-Turbo-BF16.toml"
    ace = ModelCard.model_validate(tomllib.loads(path.read_text()))
    assert ace.artifact_bundle is not None
    extra_ace = ace.artifact_bundle.files[0].model_copy(
        update={"path": "ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-q8.gguf"}
    )
    ace_bundle = ace.artifact_bundle.model_copy(
        update={
            "files": (*ace.artifact_bundle.files, extra_ace),
            "download_size": ace.artifact_bundle.download_size + extra_ace.size_bytes,
        }
    )
    with pytest.raises(ValidationError, match="only its selected GGUF"):
        ModelCard.model_validate({**ace.model_dump(), "artifact_bundle": ace_bundle.model_dump()})


def test_ace_step_selected_file_must_be_gguf() -> None:
    """A static sidecar file cannot become the selected model payload."""
    path = Path(RESOURCES_DIR) / "music_model_cards" / "audio-cpp--ACE-Step1.5-Turbo-BF16.toml"
    body = tomllib.loads(path.read_text())
    selected = "ACE-Step1.5-GGUF/turbo/config.json"
    body["gguf_file"] = selected
    body["artifact_bundle"]["files"][0]["path"] = selected
    with pytest.raises(ValidationError, match="only its selected GGUF"):
        ModelCard.model_validate(body)


def test_music_bounds_and_minimax_lyrics_are_validated() -> None:
    components = {
        "language_model_gguf": "language_model_q4_0.gguf",
        "rvq_depth_decoder_gguf": "rvq_depth_decoder_q8_0.gguf",
        "flow_transformer_gguf": "transformer_q4_0.gguf",
    }
    with pytest.raises(ValidationError, match="requires lyrics"):
        MusicCardConfig.model_validate(
            {
                "family": "minimax_music3",
                "lyrics": "optional",
                "min_seconds": 5,
                "max_seconds": 60,
                **components,
            }
        )
    with pytest.raises(ValidationError, match="declared roles"):
        MusicCardConfig.model_validate(
            {
                "family": "minimax_music3",
                "lyrics": "required",
                "min_seconds": 5,
                "max_seconds": 60,
                **components,
                "language_model_gguf": components["rvq_depth_decoder_gguf"],
                "rvq_depth_decoder_gguf": components["language_model_gguf"],
            }
        )
    with pytest.raises(ValidationError, match="cannot exceed 120"):
        MusicCardConfig.model_validate(
            {
                "family": "ace_step_1_5",
                "lyrics": "optional",
                "min_seconds": 5,
                "max_seconds": 121,
            }
        )
    with pytest.raises(ValidationError, match="cannot exceed max_seconds"):
        MusicCardConfig.model_validate(
            {
                "family": "ace_step_1_5",
                "lyrics": "optional",
                "min_seconds": 60,
                "max_seconds": 5,
            }
        )


def test_music_platform_gate_accepts_only_audio_cpp() -> None:
    assert platform_compatible_backends(
        frozenset({"audio_cpp-cpu", "mlx_audio-metal", "llama_cpp-cpu"}),
        card_serves_vision=False,
        card_serves_music=True,
    ) == frozenset({"audio_cpp-cpu"})
