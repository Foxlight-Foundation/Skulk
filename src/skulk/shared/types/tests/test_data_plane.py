"""Data plane coverage (#279 Phase 2): DATA topic wire round-trip.

A DataChunk must survive the gossip path (model_dump_json -> bytes ->
model_validate_json under the topic codec) or generation output never reaches
the owning API node. Mirrors the #287 lesson where an in-process-only test
missed a strict round-trip failure, and the slice-1/2 telemetry round-trip.
"""

from skulk.routing.topics import DATA
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import (
    DataChunk,
    ErrorChunk,
    TokenChunk,
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
