"""Fixed music option mapping and bounded audio.cpp response validation."""

import base64
import io
import json
import wave
from pathlib import Path
from typing import cast

import pytest

from skulk.shared.models.model_cards import MusicCardConfig
from skulk.worker.runner.audio_cpp.adapter import (
    MAX_AUDIO_CPP_RESPONSE_BYTES,
    audio_cpp_music_request,
    decode_audio_cpp_music_response,
)
from skulk.worker.runner.audio_cpp.server import server_config


def _family(name: str) -> MusicCardConfig:
    if name == "minimax_music3":
        return MusicCardConfig.model_validate(
            {
                "family": name,
                "lyrics": "required",
                "min_seconds": 10,
                "max_seconds": 60,
                "language_model_gguf": "language_model_q4_0.gguf",
                "rvq_depth_decoder_gguf": "rvq_depth_decoder_q8_0.gguf",
                "flow_transformer_gguf": "transformer_q4_0.gguf",
            }
        )
    return MusicCardConfig.model_validate(
        {"family": name, "lyrics": "optional", "min_seconds": 10, "max_seconds": 60}
    )


def _response() -> bytes:
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x01\x00\x02\x00" * 240)
    return json.dumps(
        {
            "audio": base64.b64encode(audio.getvalue()).decode(),
            "sample_rate": 24000,
            "channels": 2,
        }
    ).encode()


def test_minimax_maps_lyrics_and_budget_without_arbitrary_options() -> None:
    body = audio_cpp_music_request(
        family=_family("minimax_music3"),
        prompt="Warm strings",
        lyrics="We sing",
        seconds=30,
        seed=7,
    )
    assert body == {
        "model": "skulk-music",
        "request": {
            "text": "Warm strings",
            "options": {"lyrics": "We sing", "seed": "7", "duration_sec": "30"},
        },
    }


def test_ace_step_maps_text_to_music_route() -> None:
    body = audio_cpp_music_request(
        family=_family("ace_step_1_5"),
        prompt="Ambient piano",
        lyrics=None,
        seconds=45,
        seed=None,
    )
    assert body["request"] == {
        "text": "Ambient piano",
        "options": {"route": "text2music", "duration_seconds": "45"},
    }


def test_minimax_requires_lyrics_and_card_duration() -> None:
    with pytest.raises(ValueError, match="lyrics are required"):
        audio_cpp_music_request(
            family=_family("minimax_music3"),
            prompt="Piano",
            lyrics=None,
            seconds=30,
            seed=None,
        )


def test_maximum_public_text_fits_the_bounded_server_body() -> None:
    """Valid worst-case JSON escaping must reach the engine without a 413."""
    music = _family("minimax_music3")
    body = audio_cpp_music_request(
        family=music,
        prompt="\x00" * 8000,
        lyrics="\x00" * 20_000,
        seconds=60,
        seed=2**32 - 1,
    )
    config = server_config(
        port=10000,
        backend="cpu",
        model_dir=Path("/tmp/model"),
        music=music,
        model_specs=Path("/tmp/specs"),
    )
    assert len(json.dumps(body).encode()) < cast("int", config["max_request_body_bytes"])
    with pytest.raises(ValueError, match="seconds must lie"):
        audio_cpp_music_request(
            family=_family("ace_step_1_5"),
            prompt="Piano",
            lyrics=None,
            seconds=61,
            seed=None,
        )


def test_wav_result_uses_measured_metadata_and_digest() -> None:
    result = decode_audio_cpp_music_response(_response())
    assert result.sample_rate == 24000
    assert result.channels == 2
    assert result.duration_seconds == 0.01
    assert result.size_bytes == len(result.data)
    assert len(result.sha256) == 64


def test_invalid_and_oversized_results_fail_before_publication() -> None:
    with pytest.raises(ValueError, match="envelope limit"):
        decode_audio_cpp_music_response(b"x" * (MAX_AUDIO_CPP_RESPONSE_BYTES + 1))
    with pytest.raises(ValueError, match="metadata disagrees"):
        decode_audio_cpp_music_response(_response().replace(b"24000", b"22000"))
    with pytest.raises(ValueError, match="single WAV"):
        decode_audio_cpp_music_response(b"{}")
