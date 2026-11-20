"""Public edit controls preserve the original clock, caption, seed and lineage."""

import json
from typing import Literal

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from skulk.shared.models.model_cards import MusicLyricRequirement
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.audio_separation import SeparationSource
from skulk.shared.types.music_edits import (
    MusicAddPartRequest,
    MusicContinueRequest,
    MusicEditingOutput,
    MusicEditingTaskParams,
    MusicEditRequest,
    MusicRepaintRequest,
    resolve_music_edit,
)
from skulk.shared.types.native_music import (
    NativeMusicExecution,
    NativeMusicResult,
    NativeMusicWav,
)

_REQUEST = TypeAdapter[MusicEditRequest](MusicEditRequest)


def _source(rate: int = 44100, *, frames: int | None = None) -> SeparationSource:
    selected = rate if frames is None else frames
    return SeparationSource(
        sha256="a" * 64, size_bytes=44 + selected * 4,
        sample_rate=rate, channels=2, frames=selected, encoding="pcm", bit_depth=16,
    )


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000, 96000, 192000])
def test_addition_preserves_original_and_fixes_seed_before_dispatch(rate: int) -> None:
    """One request cannot gain random seeds, transformed prompts or a new source."""
    card, _ = native_music_fixture()
    source = _source(rate)
    caption, lyrics = "  Sparse strings answering the guitar.\n", "[Verse]\nStay with me.\n"
    request = MusicAddPartRequest(
        task="add_part", model=str(card.model_id), prompt=caption, lyrics=lyrics,
        source_sample_rate=rate, part_class="strings", seed=0,
    )
    params = resolve_music_edit(request, source, card, resolved_seed=0, total_input_chunks=3)
    assert params.prompt == caption and params.lyrics == lyrics
    assert params.source == source and params.seed == 0
    assert (params.target_frames, params.region_start_frame, params.region_end_frame) == (48000, 0, 48000)
    assert MusicEditingTaskParams.model_validate_json(params.model_dump_json()) == params
    with pytest.raises(ValueError, match="explicit seed"):
        resolve_music_edit(request, source, card, resolved_seed=42, total_input_chunks=3)


def test_repaint_uses_integer_covering_boundaries_for_non_native_sample_clock() -> None:
    """The chosen source interval maps once; no seconds/float rounding is involved."""
    card, _ = native_music_fixture()
    source = _source(44100)
    request = MusicRepaintRequest(
        task="repaint", model=str(card.model_id), prompt="A sparse cello response",
        source_sample_rate=44100, region_start_frame=1, region_end_frame=22051,
    )
    params = resolve_music_edit(request, source, card, resolved_seed=17, total_input_chunks=1)
    assert params.region_start_frame == 1 and params.region_end_frame == 24002
    assert params.target_frames == 48000 and params.lyrics == "[Instrumental]"
    assert params.seed == 17


def test_continuation_starts_after_complete_measured_original_not_selected_region() -> None:
    """Continue covers only the appended tail and respects the native output limit."""
    card, _ = native_music_fixture()
    source = _source(44100, frames=44101)
    request = MusicContinueRequest(
        task="continue", model=str(card.model_id), prompt="Continue the guitar phrase",
        source_sample_rate=44100, target_frames=88200,
    )
    params = resolve_music_edit(request, source, card, resolved_seed=29, total_input_chunks=2)
    assert (params.region_start_frame, params.region_end_frame, params.target_frames) == (48002, 96000, 96000)
    with pytest.raises(ValueError, match="complete original"):
        resolve_music_edit(request.model_copy(update={"target_frames": 44100}), source, card, resolved_seed=29, total_input_chunks=2)


@pytest.mark.parametrize("change", [
    {"source_sample_rate": 48000},
    {"region_end_frame": 44101},
    {"region_start_frame": 44100},
])
def test_clock_and_range_are_checked_against_measured_source(change: dict[str, int]) -> None:
    """Request shape cannot claim a different WAV clock or extend outside its bytes."""
    card, _ = native_music_fixture()
    request = MusicAddPartRequest(
        task="add_part", model=str(card.model_id), prompt="Strings",
        source_sample_rate=44100, part_class="strings",
    ).model_copy(update=change)
    with pytest.raises(ValueError):
        resolve_music_edit(request, _source(), card, resolved_seed=0, total_input_chunks=1)


@pytest.mark.parametrize("body", [
    {"task": "add_part"},
    {"task": "repaint", "part_class": "strings", "region_end_frame": 22050},
    {"task": "continue", "target_frames": 88200, "region_start_frame": 0},
    {"task": "continue", "target_frames": 44100 * 121},
    {"task": "repaint", "region_start_frame": 12, "region_end_frame": 12},
    {"task": "generate"},
    {"task": "add_part", "part_class": "strings", "source_path": "/tmp/context.wav"},
    {"task": "add_part", "part_class": "strings", "guidance_scale": 7},
    {"task": "add_part", "part_class": "strings", "seed": True},
    {"task": "add_part", "part_class": "strings", "prompt": " \n"},
    {"task": "add_part", "part_class": "strings", "lyrics": " \n"},
])
def test_closed_public_union_refuses_cross_task_and_arbitrary_native_options(body: dict[str, object]) -> None:
    """API and generated schemas offer three explicit operations without native paths."""
    with pytest.raises(ValidationError):
        _REQUEST.validate_python({
            "model": "test/native-music", "prompt": "A warm guitar",
            "source_sample_rate": 44100,
        } | body)


def test_task_discriminator_and_forbidden_controls_are_present_in_json_schema() -> None:
    """Agents can discover the same operation-specific shape that the API validates."""
    schema = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_json(
        json.dumps(_REQUEST.json_schema())
    )
    discriminator = schema["discriminator"]
    assert isinstance(discriminator, dict) and discriminator["propertyName"] == "task"
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    for name in ("MusicAddPartRequest", "MusicRepaintRequest", "MusicContinueRequest"):
        definition = definitions[name]
        assert isinstance(definition, dict) and definition["additionalProperties"] is False


@pytest.mark.parametrize("lyric_policy", [MusicLyricRequirement.Required, MusicLyricRequirement.Unsupported])
def test_selected_card_lyric_rules_are_applied_before_dispatch(lyric_policy: MusicLyricRequirement) -> None:
    """The native endpoint must preserve each card's lyric contract."""
    card, _ = native_music_fixture()
    assert card.music is not None
    card = card.model_copy(update={"music": card.music.model_copy(update={"lyrics": lyric_policy})})
    request = MusicAddPartRequest(
        task="add_part", model=str(card.model_id), prompt="A warm strings response",
        lyrics=None if lyric_policy == MusicLyricRequirement.Required else "Sing these words",
        source_sample_rate=44100, part_class="strings",
    )
    with pytest.raises(ValueError, match="lyrics"):
        resolve_music_edit(request, _source(), card, resolved_seed=0, total_input_chunks=1)


def _result(operation: Literal["generate", "repaint", "continue"] = "repaint") -> NativeMusicResult:
    return NativeMusicResult(
        request_id="native", model_identity="b" * 64, payload_sha256="c" * 64,
        operation=operation, seed=0,
        source_sha256=None if operation == "generate" else "a" * 64,
        conditioning_sha256=None if operation == "generate" else "d" * 64,
        conditioning_gain=None if operation == "generate" else 1.0,
        execution=NativeMusicExecution(
            device="cpu", precision="float32", text_encoder_device="cpu",
            text_encoder_precision="float32", autoencoder_device="cpu", autoencoder_precision="float32",
        ),
        output=NativeMusicWav(sha256="e" * 64, size_bytes=288044, frames=48000, peak=0.2, gain=1.0),
        generation_seconds=1.0,
    )


def test_completed_manifest_binds_original_and_distinguishes_edit_from_generation() -> None:
    """A well-shaped result for other bytes or another timeline cannot be adopted."""
    source = _source()
    output = MusicEditingOutput(source=source, result=_result())
    assert MusicEditingOutput.model_validate_json(output.model_dump_json()) == output
    for result in (
        _result("generate"), _result("continue"),
        _result().model_copy(update={"source_sha256": "f" * 64}),
    ):
        with pytest.raises(ValidationError):
            MusicEditingOutput(source=source, result=result)
