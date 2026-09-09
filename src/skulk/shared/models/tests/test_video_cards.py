"""Card vocabulary for audio-video generation models.

The ``[video]`` section states model truth (modes, duration and frame grid,
canvas rules, audio output, reference bounds, sampling defaults, and pinned
companions) without naming an engine. These tests pin the validation rules
that keep that section, the task list, and the immutable companion rule
telling one story, and prove the bundled MiniMax H3 fallback cards obey them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import tomllib
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from skulk.shared.models import model_cards
from skulk.shared.models.capabilities import resolve_model_capability_profile
from skulk.shared.models.model_cards import (
    LicenseCardConfig,
    ModelCard,
    ModelId,
    ModelTask,
    VideoCardConfig,
    VideoCompanionConfig,
    VideoCompanionKind,
    VideoMode,
    VideoReferenceLimits,
    card_serves_video,
)
from skulk.shared.types.memory import Memory

CARD_DIRECTORY = Path(__file__).resolve().parents[5] / "resources" / "video_model_cards"
REVISION = "a" * 40


def _video(**overrides: object) -> VideoCardConfig:
    payload: dict[str, object] = {
        "modes": ["t2va", "fl2va"],
        "frame_grid_multiple": 17,
        "frame_grid_offset": 5,
        "canvas_multiple": 32,
        "audio_output": True,
        "audio_sample_rate": 32000,
        "audio_channels": 2,
    }
    payload.update(overrides)
    return VideoCardConfig.model_validate(payload)


def _card(**overrides: object) -> ModelCard:
    payload: dict[str, object] = {
        "model_id": ModelId("example/video-model"),
        "storage_size": Memory.from_bytes(10),
        "n_layers": 50,
        "hidden_size": 5376,
        "supports_tensor": False,
        "tasks": ["TextToVideo", "ImageToVideo"],
        "video": _video(),
    }
    payload.update(overrides)
    return ModelCard.model_validate(payload)


def test_video_modes_imply_tasks() -> None:
    card = _card()
    assert card.video is not None
    assert card.video.tasks == {ModelTask.TextToVideo, ModelTask.ImageToVideo}
    assert card_serves_video(card)


def test_tasks_must_match_declared_modes() -> None:
    with pytest.raises(ValidationError, match="exactly the families"):
        _card(tasks=["TextToVideo"])
    with pytest.raises(ValidationError, match=r"require a \[video\] section"):
        _card(video=None)


def test_reference_mode_requires_limits() -> None:
    with pytest.raises(ValidationError, match="ref2va requires reference_limits"):
        _video(modes=["ref2va"])
    video = _video(
        modes=["ref2va"],
        reference_limits=VideoReferenceLimits(max_images=9, max_videos=3),
    )
    assert video.tasks == {ModelTask.ReferenceToVideo}


def test_frame_grid_alignment_matches_trained_grid() -> None:
    video = _video()
    assert video.align_frame_count(120) == 124
    assert video.align_frame_count(124) == 124
    assert video.frame_count_for_seconds(5) == 124
    assert video.frame_count_for_seconds(15) == 362
    with pytest.raises(ValueError, match="duration must lie"):
        video.frame_count_for_seconds(16)


def test_contract_bounds_are_validated() -> None:
    with pytest.raises(ValidationError, match="min_seconds cannot exceed"):
        _video(min_seconds=10, max_seconds=5)
    with pytest.raises(ValidationError, match="frame_grid_offset"):
        _video(frame_grid_offset=17)
    with pytest.raises(ValidationError, match="audio_output requires"):
        _video(audio_sample_rate=None)
    with pytest.raises(ValidationError, match="expected W:H"):
        _video(aspect_ratios=["wide"])
    with pytest.raises(ValidationError, match="at least one mode"):
        _video(modes=[])


def test_companions_are_unique_and_scoped_to_declared_modes() -> None:
    lora = {
        "kind": "lora",
        "name": "turbo",
        "path": "loras/turbo.safetensors",
        "modes": ["t2va"],
        "steps": 8,
    }
    video = _video(companions=[lora])
    assert video.companions[0].kind is VideoCompanionKind.Lora
    assert video.companions[0].modes == (VideoMode.TextToAudioVideo,)
    with pytest.raises(ValidationError, match="unique per kind and name"):
        _video(companions=[lora, lora])
    with pytest.raises(ValidationError, match="undeclared modes"):
        _video(companions=[{**lora, "modes": ["ref2va"]}])
    with pytest.raises(ValidationError, match="at most one graph template"):
        _video(
            companions=[
                {"kind": "graph_template", "name": "a", "path": "a.json", "modes": ["t2va"]},
                {"kind": "graph_template", "name": "b", "path": "b.json", "modes": ["t2va"]},
            ]
        )


def test_companion_kind_specific_fields() -> None:
    with pytest.raises(ValidationError, match="steps and sigma shifts"):
        VideoCompanionConfig(kind=VideoCompanionKind.Embedding, name="e", path="e.st", steps=4)
    with pytest.raises(ValidationError, match="must name the modes"):
        VideoCompanionConfig(kind=VideoCompanionKind.GraphTemplate, name="g", path="g.json")
    with pytest.raises(ValidationError, match="canonical and relative"):
        VideoCompanionConfig(kind=VideoCompanionKind.Lora, name="l", path="../l.st")
    with pytest.raises(ValidationError, match="immutable revision"):
        VideoCompanionConfig(
            kind=VideoCompanionKind.Lora,
            name="l",
            path="l.st",
            repo=ModelId("other/repo"),
        )


def test_external_companion_needs_revision_on_signed_cards() -> None:
    lora = VideoCompanionConfig(
        kind=VideoCompanionKind.Lora,
        name="turbo",
        path="turbo.safetensors",
        repo=ModelId("other/turbo"),
        revision=REVISION,
    )
    card = _card(video=_video(companions=[lora]))
    card.require_immutable_external_companions(context="test")
    mutable = card.model_copy(
        update={
            "video": card.video.model_copy(
                update={
                    "companions": (
                        lora.model_construct(
                            kind=lora.kind,
                            name=lora.name,
                            path=lora.path,
                            repo=lora.repo,
                            revision=None,
                            size_bytes=None,
                            modes=(),
                            steps=None,
                            strength=None,
                            video_shift=None,
                            audio_shift=None,
                        ),
                    )
                }
            )
            if card.video is not None
            else None
        }
    )
    with pytest.raises(ValueError, match="immutable"):
        mutable.require_immutable_external_companions(context="test")


def test_license_section_is_operator_facing_only() -> None:
    card = _card(
        license=LicenseCardConfig(
            name="Example Community License",
            url="https://example.invalid/LICENSE",
            display_name="Example H3",
        )
    )
    assert card.license is not None
    assert card.license.display_name == "Example H3"
    with pytest.raises(ValidationError, match="http"):
        LicenseCardConfig(name="x", url="ftp://example.invalid")
    with pytest.raises(ValidationError, match="not be empty"):
        LicenseCardConfig(name="   ")


def test_profile_projects_video_output() -> None:
    profile = resolve_model_capability_profile(
        ModelId("example/video-model"), model_card=_card()
    )
    assert profile.supports_video_output is True
    assert profile.video_modes == ("t2va", "fl2va")


def _load_bundled() -> list[tuple[Path, dict[str, object]]]:
    return [
        (path, tomllib.loads(path.read_text()))
        for path in sorted(CARD_DIRECTORY.glob("*.toml"))
    ]


def _derived_bundle_id(files: object) -> str:
    """Mirror the registry's content identity so bundled cards stay compatible."""

    assert isinstance(files, list)
    entries = cast("list[dict[str, object]]", files)
    document = [
        {
            "relative_path": item["path"],
            "size_bytes": item["size_bytes"],
            "object_id": item["object_id"],
        }
        for item in sorted(entries, key=lambda item: str(item["path"]))
    ]
    digest = hashlib.sha256(
        json.dumps(document, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).digest()
    return "bundle_" + base64.b32encode(digest).decode().rstrip("=").lower()


def test_bundled_video_cards_validate_and_pin_every_byte() -> None:
    cards = _load_bundled()
    assert len(cards) == 2
    for path, raw in cards:
        card = ModelCard.model_validate(raw)
        assert path.name == card.model_id.normalize() + ".toml"
        assert card.source_revision is not None
        assert card.artifact_bundle is not None
        assert card.video is not None
        assert card.license is not None and card.license.display_name == "MiniMax H3"
        assert card.placement.compatible_backends == frozenset({"comfy"})
        bundle_paths = {item.path for item in card.artifact_bundle.files}
        for companion in card.video.companions:
            assert companion.repo is None
            assert companion.path in bundle_paths
        for component in card.components or []:
            assert any(item.path.startswith(component.component_path) for item in card.artifact_bundle.files)
        raw_bundle = cast("dict[str, object]", raw["artifact_bundle"])
        assert card.artifact_bundle.bundle_id == _derived_bundle_id(raw_bundle["files"])
        assert card.artifact_bundle.download_size == sum(item.size_bytes for item in card.artifact_bundle.files)
        assert all(item.object_id is not None for item in card.artifact_bundle.files)
        card.require_immutable_external_companions(context="bundled video cards")


def test_video_cards_hidden_until_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _card()
    text = _card(
        model_id=ModelId("example/text"),
        tasks=["TextGeneration"],
        video=None,
    )

    async def fake_all() -> list[ModelCard]:
        return [card, text]

    monkeypatch.setattr(model_cards, "get_all_model_cards", fake_all)
    monkeypatch.setattr(model_cards, "SKULK_ENABLE_VIDEO_MODELS", False)
    import asyncio

    visible = asyncio.run(model_cards.get_model_cards())
    assert [item.model_id for item in visible] == [ModelId("example/text")]
    monkeypatch.setattr(model_cards, "SKULK_ENABLE_VIDEO_MODELS", True)
    visible = asyncio.run(model_cards.get_model_cards())
    assert {item.model_id for item in visible} == {card.model_id, text.model_id}
