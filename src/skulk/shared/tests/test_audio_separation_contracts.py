"""Cross-field input/output checks at the native and admitted task boundary."""

import pytest
from pydantic import ValidationError

from skulk.routing.topics import DATA
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.audio_separation import (
    AudioSeparationOutput,
    AudioSeparationRequest,
    AudioSeparationTaskParams,
    SeparatedStem,
    SeparationSource,
)
from skulk.shared.types.chunks import DataChunk, SeparationChunk
from skulk.shared.types.common import CommandId, NodeId


def _source(**changes: object) -> SeparationSource:
    values: dict[str, object] = {
        "sha256": "a" * 64,
        "size_bytes": 16044,
        "frames": 8000,
        "sample_rate": 8000,
        "channels": 1,
        "bit_depth": 16,
        "encoding": "pcm",
    }
    return SeparationSource.model_validate(values | changes)


def _output() -> AudioSeparationOutput:
    return AudioSeparationOutput(
        source=_source(),
        common_gain=1.0,
        pre_gain_peak=0.2,
        stems=tuple(
            SeparatedStem(
                name=name,
                size_bytes=264644,
                sha256="b" * 64,
                sample_rate=44100,
                frames=44100,
                peak=0.2,
            )
            for name in ("drums", "bass", "other", "vocals")
        ),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"frames": 7999},
        {"frames": 960001},
        {"channels": 3},
        {"encoding": "float"},
        {"bit_depth": 64},
        {"size_bytes": 16043},
        {
            "frames": 192000 * 120,
            "sample_rate": 192000,
            "channels": 2,
            "size_bytes": 64 * 1024 * 1024,
        },
        {"frames": "8000"},
        {"local_path": "/tmp/source.wav"},
    ],
)
def test_source_rejects_inconsistent_or_unbounded_geometry(
    changes: dict[str, object],
) -> None:
    """A valid field in isolation cannot bypass complete source admission bounds."""
    with pytest.raises(ValidationError):
        _source(**changes)


def test_request_and_task_never_accept_audio_bytes_or_caller_paths() -> None:
    """Only measured metadata can enter replicated command/task records."""
    options: dict[str, object] = {}
    for extra in (
        {"options": options},
        {"source_path": "/tmp/audio.wav"},
        {"audio": "encoded"},
    ):
        with pytest.raises(ValidationError):
            AudioSeparationRequest.model_validate({"model": "demucs", **extra})
    with pytest.raises(ValidationError):
        AudioSeparationTaskParams(
            model="demucs", source=_source(), total_input_chunks=0
        )


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "reordered", "frames", "rate", "gain"]
)
def test_completion_requires_one_consistent_complete_set(mutation: str) -> None:
    """A single delivered stem or altered byte geometry is never a complete result."""
    output = _output()
    stems = list(output.stems)
    if mutation == "missing":
        stems.pop()
    elif mutation == "duplicate":
        stems[3] = stems[3].model_copy(update={"name": "drums"})
    elif mutation == "reordered":
        stems.reverse()
    elif mutation == "frames":
        stems[0] = stems[0].model_copy(update={"frames": stems[0].frames - 1})
    elif mutation == "rate":
        stems[0] = stems[0].model_copy(update={"sample_rate": 48000})
    else:
        output = output.model_copy(update={"pre_gain_peak": 2.0})
    output = output.model_copy(update={"stems": tuple(stems)})
    with pytest.raises(ValidationError):
        AudioSeparationOutput.model_validate_json(output.model_dump_json())


def test_json_round_trip_preserves_native_frames_and_shared_gain() -> None:
    """Native subprocess JSON and public job output keep identical immutable facts."""
    output = _output()
    assert AudioSeparationOutput.model_validate_json(output.model_dump_json()) == output
    shared = output.model_copy(update={"pre_gain_peak": 2.0, "common_gain": 0.49})
    assert (
        AudioSeparationOutput.model_validate_json(shared.model_dump_json()).common_gain
        == 0.49
    )
    with pytest.raises(ValidationError):
        field = "common_gain"
        setattr(output, field, 0.5)


@pytest.mark.parametrize("successful", [False, True])
def test_actual_data_topic_round_trip_preserves_separation_terminal(
    successful: bool,
) -> None:
    """Tagged network decoding must preserve a complete immutable stem manifest."""
    frame = DataChunk(
        command_id=CommandId("separation-wire"),
        owner_node=NodeId("accepting-api"),
        kind="completed",
        sequence=1,
        chunk=SeparationChunk(
            model=ModelId("adefossez/HTDemucs"),
            finish_reason="stop" if successful else "error",
            output=_output() if successful else None,
            error_message=None if successful else "Separation failed",
        ),
    )
    restored = DATA.deserialize(DATA.serialize(frame))
    assert restored == frame
    assert isinstance(restored.chunk, SeparationChunk)
    if successful:
        assert restored.chunk.output is not None
        assert isinstance(restored.chunk.output.stems, tuple)
        assert tuple(stem.name for stem in restored.chunk.output.stems) == (
            "drums", "bass", "other", "vocals"
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "reordered", "foreign", "coerced", "path"],
)
def test_decoded_stem_arrays_keep_complete_set_and_strict_field_validation(
    mutation: str,
) -> None:
    """Accepting decoded arrays must not relax roles, geometry or closed records."""
    output = _output()
    stems = [stem.model_dump(mode="json") for stem in output.stems]
    if mutation == "missing":
        stems.pop()
    elif mutation == "extra":
        stems.append(stems[0])
    elif mutation == "duplicate":
        stems[3] = stems[0]
    elif mutation == "reordered":
        stems.reverse()
    elif mutation == "foreign":
        stems[0]["name"] = "guitar"
    elif mutation == "coerced":
        stems[0]["sample_rate"] = "44100"
    else:
        stems[0]["path"] = "/tmp/foreign.wav"
    with pytest.raises(ValidationError):
        AudioSeparationOutput.model_validate(output.model_dump() | {"stems": stems})
