"""Data plane coverage (#279 Phase 2): DATA topic wire round-trip.

A DataChunk must survive the gossip path (model_dump_json -> bytes ->
model_validate_json under the topic codec) or generation output never reaches
the owning API node. Mirrors the #287 lesson where an in-process-only test
missed a strict round-trip failure, and the slice-1/2 telemetry round-trip.
"""

import pytest

from skulk.routing.topics import DATA
from skulk.shared.models.model_cards import AudioResponseFormat, ModelId
from skulk.shared.types.chunks import (
    AudioChunk,
    DataChunk,
    ErrorChunk,
    TokenChunk,
    TranscriptionChunk,
)
from skulk.shared.types.common import CommandId


def test_data_chunk_token_survives_topic_codec_round_trip() -> None:
    msg = DataChunk(
        command_id=CommandId("cmd-1"),
        chunk=TokenChunk(
            model=ModelId("mlx-community/test"),
            text="hello",
            token_id=42,
            usage=None,
            finish_reason=None,
        ),
        sequence=7,
    )
    restored = DATA.deserialize(DATA.serialize(msg))
    assert restored.command_id == CommandId("cmd-1")
    assert restored.sequence == 7
    assert isinstance(restored.chunk, TokenChunk)
    assert restored.chunk.text == "hello"
    assert restored.chunk.token_id == 42


def test_data_chunk_error_survives_topic_codec_round_trip() -> None:
    # The error path (runner shutdown mid-stream) also rides the data plane, so
    # the terminal ErrorChunk must round-trip too.
    msg = DataChunk(
        command_id=CommandId("cmd-2"),
        chunk=ErrorChunk(
            model=ModelId("mlx-community/test"),
            error_message="runner shutdown before completing command",
        ),
        sequence=0,
    )
    restored = DATA.deserialize(DATA.serialize(msg))
    assert isinstance(restored.chunk, ErrorChunk)
    assert restored.chunk.finish_reason == "error"
    assert "runner shutdown" in restored.chunk.error_message


def test_data_chunk_lifecycle_frames_survive_topic_codec_round_trip() -> None:
    started = DataChunk(
        command_id=CommandId("cmd-lifecycle"),
        kind="started",
        sequence=0,
    )
    completed = DataChunk(
        command_id=CommandId("cmd-lifecycle"),
        kind="completed",
        sequence=1,
    )

    restored_started = DATA.deserialize(DATA.serialize(started))
    restored_completed = DATA.deserialize(DATA.serialize(completed))

    assert restored_started.kind == "started"
    assert restored_started.chunk is None
    assert not restored_started.is_terminal
    assert restored_completed.kind == "completed"
    assert restored_completed.chunk is None
    assert restored_completed.is_terminal


def test_data_chunk_rejects_ambiguous_lifecycle_payloads() -> None:
    error = ErrorChunk(
        model=ModelId("mlx-community/test"),
        error_message="cancelled",
    )
    with pytest.raises(ValueError, match="started.*must not carry"):
        DataChunk(
            command_id=CommandId("cmd-invalid-start"),
            kind="started",
            chunk=error,
            sequence=0,
        )
    with pytest.raises(ValueError, match="chunk.*must carry"):
        DataChunk(
            command_id=CommandId("cmd-invalid-chunk"),
            kind="chunk",
            sequence=0,
        )
    with pytest.raises(ValueError, match="cancelled.*ErrorChunk"):
        DataChunk(
            command_id=CommandId("cmd-invalid-cancel"),
            kind="cancelled",
            sequence=0,
        )


def test_data_chunk_speech_chunks_survive_topic_codec_round_trip() -> None:
    audio = DataChunk(
        command_id=CommandId("cmd-audio"),
        chunk=AudioChunk(
            model=ModelId("mlx-community/kokoro-test"),
            data="UklGRg==",
            chunk_index=0,
            total_chunks=1,
            format=AudioResponseFormat.Wav,
            sample_rate=24000,
            finish_reason="stop",
        ),
        sequence=0,
    )
    restored_audio = DATA.deserialize(DATA.serialize(audio))
    assert restored_audio.command_id == CommandId("cmd-audio")
    assert isinstance(restored_audio.chunk, AudioChunk)
    assert restored_audio.chunk.format == AudioResponseFormat.Wav
    assert restored_audio.chunk.sample_rate == 24000

    transcript = DataChunk(
        command_id=CommandId("cmd-transcript"),
        chunk=TranscriptionChunk(
            model=ModelId("mlx-community/whisper-test"),
            text="hello",
            segment_index=0,
            language="en",
            finish_reason="stop",
        ),
        sequence=1,
    )
    restored_transcript = DATA.deserialize(DATA.serialize(transcript))
    assert restored_transcript.command_id == CommandId("cmd-transcript")
    assert isinstance(restored_transcript.chunk, TranscriptionChunk)
    assert restored_transcript.chunk.text == "hello"
    assert restored_transcript.chunk.language == "en"
