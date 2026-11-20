"""Native generation/editing contracts retain exact source and policy boundaries."""

import json
import tomllib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from skulk.shared.models.model_cards import ModelCard, MusicPartClass
from skulk.shared.types.audio_separation import SeparationSource
from skulk.shared.types.native_music import (
    NativeMusicComponent,
    NativeMusicDirectoryIdentity,
    NativeMusicFile,
    NativeMusicPart,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
    NativeMusicServeRequest,
    NativeMusicWeights,
    native_music_conditioning_frames,
)


def _weights() -> NativeMusicWeights:
    path = Path(__file__).parents[2] / "models/tests/ace-native-card.toml"
    card = ModelCard.model_validate(tomllib.loads(path.read_text()))
    assert card.music is not None and card.music.native is not None
    assert card.artifact_bundle is not None
    native = card.music.native
    return NativeMusicWeights(
        base=NativeMusicComponent(
            kind="base",
            directory="/private/base",
            root=card.artifact_bundle.root,
            files=tuple(
                NativeMusicFile(
                    path=f.path, size_bytes=f.size_bytes, object_id=f.object_id or ""
                )
                for f in card.artifact_bundle.files
            ),
        ),
        text_encoder=NativeMusicComponent(
            kind="text_encoder",
            directory="/private/components",
            root=native.text_encoder.artifact_bundle.root,
            files=tuple(
                NativeMusicFile(
                    path=f.path, size_bytes=f.size_bytes, object_id=f.object_id or ""
                )
                for f in native.text_encoder.artifact_bundle.files
            ),
        ),
        autoencoder=NativeMusicComponent(
            kind="autoencoder",
            directory="/private/components",
            root=native.autoencoder.artifact_bundle.root,
            files=tuple(
                NativeMusicFile(
                    path=f.path, size_bytes=f.size_bytes, object_id=f.object_id or ""
                )
                for f in native.autoencoder.artifact_bundle.files
            ),
        ),
    )


def _source() -> SeparationSource:
    return SeparationSource(
        sha256="a" * 64,
        size_bytes=44 + 20 * 44100 * 4,
        sample_rate=44100,
        channels=2,
        frames=20 * 44100,
        encoding="pcm",
        bit_depth=16,
    )


def _request(operation: str) -> dict[str, object]:
    body: dict[str, object] = dict(
        request_id="native-request",
        operation=operation,
        prompt="A warm folk arrangement",
        seed=12025,
        target_frames=20 * 48000,
        output_directory="/private/jobs/request",
        output_directory_identity={"device": 1, "inode": 2},
    )
    if operation != "generate":
        body.update(
            source=_source().model_dump(mode="json"),
            source_path="/private/source.wav",
            region_end_frame=20 * 48000,
        )
    if operation == "add_part":
        body["part_class"] = "strings"
    if operation == "repaint":
        body["region_start_frame"] = 10 * 48000
    if operation == "continue":
        body.update(
            region_start_frame=20 * 48000,
            region_end_frame=30 * 48000,
            target_frames=30 * 48000,
        )
    return body


def test_native_component_inventory_matches_actual_card_and_closed_parts() -> None:
    """Package-bound components accept the exact registry/Core artifact selection."""
    weights = _weights()
    assert (
        sum(
            len(c.files)
            for c in (weights.base, weights.text_encoder, weights.autoencoder)
        )
        == 14
    )
    assert NativeMusicWeights.model_validate_json(weights.model_dump_json()) == weights
    assert set(get_args(NativeMusicPart)) == set(get_args(MusicPartClass))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", -1),
        ("device", 2**64),
        ("device", True),
        ("device", "1"),
        ("inode", 0),
        ("inode", -1),
        ("inode", 2**64),
        ("inode", True),
        ("inode", "2"),
    ],
)
def test_native_directory_identity_cannot_coerce_or_overflow(
    field: str, value: object
) -> None:
    """Private cleanup ownership uses strict real filesystem identity values."""
    body: dict[str, object] = {"device": 1, "inode": 2}
    body[field] = value
    with pytest.raises(ValidationError):
        NativeMusicDirectoryIdentity.model_validate_json(json.dumps(body))


def test_native_output_path_alone_cannot_establish_directory_ownership() -> None:
    """An old path-only request cannot silently bypass pre-write result custody."""
    body = _request("generate")
    del body["output_directory_identity"]
    with pytest.raises(ValidationError, match="output_directory_identity"):
        NativeMusicRequest.model_validate_json(json.dumps(body))


@pytest.mark.parametrize("role", ["base", "text_encoder", "autoencoder"])
def test_component_closure_cannot_be_shortened_or_duplicated(role: str) -> None:
    """Even a digest-bound selection must include exactly its loader's full role."""
    body = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_json(
        _weights().model_dump_json()
    )
    component = body[role]
    assert isinstance(component, dict)
    files = component["files"]
    assert isinstance(files, list)
    files[-1] = files[0]
    with pytest.raises(ValidationError, match="exact complete file set"):
        NativeMusicWeights.model_validate_json(json.dumps(body))


@pytest.mark.parametrize("operation", ["generate", "add_part", "repaint", "continue"])
def test_native_operations_bind_original_and_native_frame_geometry(
    operation: str,
) -> None:
    """The 44.1 kHz original and mapped 48 kHz region remain distinct facts."""
    request = NativeMusicRequest.model_validate_json(json.dumps(_request(operation)))
    assert NativeMusicRequest.model_validate_json(request.model_dump_json()) == request
    if request.source is not None:
        assert request.source.sample_rate == 44100
        assert request.region_end_frame == request.target_frames


@pytest.mark.parametrize(
    ("operation", "field", "value"),
    [
        ("generate", "source_path", "/private/source.wav"),
        ("generate", "part_class", "strings"),
        ("generate", "seed", -1),
        ("generate", "target_frames", 121 * 48000),
        ("generate", "native_options", {"thinking": True}),
        ("add_part", "part_class", None),
        ("add_part", "part_class", "cello"),
        ("repaint", "source_path", None),
        ("repaint", "source_path", "relative.wav"),
        ("repaint", "region_end_frame", 21 * 48000),
        ("repaint", "region_start_frame", 20 * 48000),
        ("repaint", "target_frames", 19 * 48000),
        ("continue", "target_frames", 20 * 48000),
        ("continue", "region_start_frame", 19 * 48000),
        ("continue", "region_end_frame", 31 * 48000),
        ("continue", "part_class", "drums"),
    ],
)
def test_native_request_rejects_mismatched_or_unbounded_controls(
    operation: str, field: str, value: object
) -> None:
    """No arbitrary options, ambiguous source paths or mismatched region/length pass."""
    body = _request(operation)
    body[field] = value
    with pytest.raises(ValidationError):
        NativeMusicRequest.model_validate_json(json.dumps(body))


@pytest.mark.parametrize("device", ["cpu", "mps", "rocm"])
@pytest.mark.parametrize("precision", ["bfloat16", "float16"])
def test_non_cuda_precision_must_not_inherit_checkpoint_bfloat16(
    device: str, precision: str
) -> None:
    """Admission's FP32 working set cannot be silently replaced by source dtype."""
    body = dict(
        model_identity="b" * 64,
        weights=_weights().model_dump(mode="json"),
        directory="/private/lifetime",
        directory_identity={"device": 1, "inode": 2},
        requests_identity={"device": 1, "inode": 3},
        loader_identity={"device": 1, "inode": 4},
        device=device,
        precision=precision,
    )
    with pytest.raises(ValidationError, match="requires float32"):
        NativeMusicServeConfig.model_validate_json(json.dumps(body))


@pytest.mark.parametrize("sequence", [0, -1, 2**63, True, "1"])
def test_native_control_sequence_is_strict_positive_and_bounded(
    sequence: object,
) -> None:
    """Process-local replay protection cannot accept boolean/coerced/unbounded IDs."""
    with pytest.raises(ValidationError):
        NativeMusicServeRequest.model_validate_json(
            json.dumps({"sequence": sequence, "request": _request("generate")})
        )


def _result_body(
    operation: str,
    padding: object,
    *,
    frames: int = 48000,
    conditioning_padding: object = 0,
) -> dict[str, object]:
    return dict(
        request_id="take-1",
        model_identity="a" * 64,
        payload_sha256="b" * 64,
        operation=operation,
        seed=12025,
        source_sha256="c" * 64 if operation != "generate" else None,
        conditioning_sha256="d" * 64 if operation != "generate" else None,
        conditioning_gain=1.0 if operation != "generate" else None,
        conditioning_padding_frames=conditioning_padding,
        padding_frames_removed=padding,
        execution=dict(
            device="cpu",
            precision="float32",
            text_encoder_device="cpu",
            text_encoder_precision="float32",
            autoencoder_device="cpu",
            autoencoder_precision="float32",
        ),
        output=dict(
            sha256="e" * 64,
            size_bytes=44 + frames * 6,
            frames=frames,
            peak=0.25,
            gain=1.0,
        ),
        generation_seconds=1.0,
    )


@pytest.mark.parametrize(
    ("operation", "padding", "accepted"),
    [
        ("repaint", 197760, True),
        ("add_part", 197760, True),
        ("continue", 197760, True),
        ("generate", 0, True),
        ("repaint", 0, True),
        ("repaint", 1, False),
        ("generate", 197760, False),
        ("repaint", -1, False),
        ("repaint", 197761, False),
        ("repaint", True, False),
        ("repaint", 1.0, False),
    ],
)
def test_native_result_padding_is_exact_bounded_and_edit_only(
    operation: str, padding: object, accepted: bool
) -> None:
    """A receipt cannot justify arbitrary cropping or generation length changes."""
    body = _result_body(operation, padding)
    if accepted:
        result = NativeMusicResult.model_validate_json(json.dumps(body))
        assert result.padding_frames_removed == padding
    else:
        with pytest.raises(ValidationError):
            NativeMusicResult.model_validate_json(json.dumps(body))


@pytest.mark.parametrize(
    ("frames", "conditioning_padding", "removed", "accepted"),
    [
        (288001, 1919, 1919, True),
        (1536001, 1919, 1919, True),
        (5759999, 1, 1, True),
        (48001, 1919, 197759, True),
        (288001, 1919, 0, True),
        (288001, 0, 1919, False),
        (288001, 1918, 1919, False),
        (288001, 1920, 1919, False),
        (288001, -1, 1919, False),
        (288001, True, 1919, False),
        (288001, 1919.0, 1919, False),
        (288001, 1919, 1918, False),
        (288000, 0, 1920, False),
    ],
)
def test_native_result_records_exact_codec_padding(
    frames: int, conditioning_padding: object, removed: object, accepted: bool
) -> None:
    """Recorded alignment is bounded, strict and separate from decoder removal."""
    body = _result_body(
        "repaint", removed, frames=frames, conditioning_padding=conditioning_padding
    )
    if accepted:
        result = NativeMusicResult.model_validate_json(json.dumps(body))
        assert result.conditioning_padding_frames == conditioning_padding
        assert result.padding_frames_removed == removed
    else:
        with pytest.raises(ValidationError):
            NativeMusicResult.model_validate_json(json.dumps(body))


@pytest.mark.parametrize("conditioning_padding", [1, 1919, True, 1.0])
def test_native_generation_cannot_claim_conditioning_padding(
    conditioning_padding: object,
) -> None:
    """Text-only generation cannot invent input-audio transformation lineage."""
    body = _result_body("generate", 0, conditioning_padding=conditioning_padding)
    with pytest.raises(ValidationError):
        NativeMusicResult.model_validate_json(json.dumps(body))


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (48000, 48000),
        (48001, 49920),
        (245759, 245760),
        (288001, 289920),
        (5760000, 5760000),
    ],
)
def test_native_conditioning_rounds_up_within_the_global_frame_ceiling(
    frames: int, expected: int
) -> None:
    """Codec alignment retains the native frame clock and the 120-second limit."""
    assert native_music_conditioning_frames(frames) == expected
