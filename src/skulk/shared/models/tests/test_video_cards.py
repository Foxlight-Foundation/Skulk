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
    ArtifactBundleConfig,
    ArtifactBundleFile,
    LicenseCardConfig,
    ModelCard,
    ModelId,
    ModelTask,
    VideoCardConfig,
    VideoCompanionConfig,
    VideoCompanionKind,
    VideoMode,
    VideoPreprocessorRole,
    VideoReferenceLimits,
    card_serves_video,
)
from skulk.shared.tests.model_card_fixtures import FIXTURE_CARDS_DIR
from skulk.shared.types.memory import Memory

CARD_DIRECTORY = FIXTURE_CARDS_DIR
TEST_ENGINE_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "worker" / "runner" / "test_video"
)
REVISION = "a" * 40

PIPELINE_COMPANION_KINDS = (
    VideoCompanionKind.LatentUpscaler,
    VideoCompanionKind.DurationHead,
    VideoCompanionKind.VideoDecoder,
)


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


@pytest.mark.parametrize("feature", ["audio", "retake", "extend"])
@pytest.mark.parametrize("max_files", [None, 0, 1])
def test_source_features_require_an_available_aggregate_reference_slot(
    feature: str, max_files: int | None,
) -> None:
    """Per-media limits cannot override an aggregate prohibition on all files."""
    payload: dict[str, object] = {
        "modes": ["ref2va"],
        "audio_conditioning": feature == "audio",
        "editing": [] if feature == "audio" else [feature],
        "reference_limits": VideoReferenceLimits(
            max_audio_clips=1, max_videos=1, max_files=max_files,
        ),
    }
    if max_files == 0:
        with pytest.raises(ValidationError, match="max_files to permit at least one"):
            _video(**payload)
    else:
        video = _video(**payload)
        assert video.reference_limits is not None
        assert video.reference_limits.max_files == max_files


def test_no_source_feature_can_keep_a_zero_aggregate_reference_limit() -> None:
    """A text-capable card need not reserve input files it never requires."""
    video = _video(reference_limits=VideoReferenceLimits(max_files=0))
    assert video.reference_limits is not None and video.reference_limits.max_files == 0


def _bundled_companion_card(
    kind: VideoCompanionKind,
    *,
    path: str = "weights/pipeline.safetensors",
    companion_bytes: int | None = 13,
    repository: ModelId | None = None,
    revision: str | None = None,
    root: str | None = None,
) -> ModelCard:
    companion = VideoCompanionConfig(
        kind=kind, name="selected", path=path, size_bytes=companion_bytes,
        repo=repository, revision=revision,
    )
    bundle = ArtifactBundleConfig(
        bundle_id="bundle_" + "a" * 52, root=root,
        files=(ArtifactBundleFile(path="weights/pipeline.safetensors", size_bytes=13),),
        download_size=13,
    )
    return _card(
        source_repository=ModelId("example/upstream"), source_revision=REVISION,
        storage_size=Memory.from_bytes(13), artifact_bundle=bundle,
        video=_video(companions=[companion]),
    )


@pytest.mark.parametrize("kind", PIPELINE_COMPANION_KINDS)
@pytest.mark.parametrize("repository", [None, ModelId("example/upstream")])
def test_selected_local_pipeline_companions_must_be_in_the_bundle(
    kind: VideoCompanionKind, repository: ModelId | None,
) -> None:
    """Implicit and explicit base-repository aliases share one exact transaction."""
    with pytest.raises(ValidationError, match="must contain local video companion"):
        _bundled_companion_card(kind, path="weights/omitted.safetensors", repository=repository, revision=REVISION)


@pytest.mark.parametrize("kind", PIPELINE_COMPANION_KINDS)
def test_selected_local_pipeline_companion_sizes_must_match_manifest(
    kind: VideoCompanionKind,
) -> None:
    """A signed companion size cannot contradict the file actually installed."""
    with pytest.raises(ValidationError, match="size must match the artifact bundle"):
        _bundled_companion_card(kind, companion_bytes=14)


@pytest.mark.parametrize("kind", PIPELINE_COMPANION_KINDS)
@pytest.mark.parametrize("companion_bytes", [None, 13])
@pytest.mark.parametrize("repository", [None, ModelId("example/upstream")])
def test_complete_pipeline_companions_preserve_the_card_projection(
    kind: VideoCompanionKind, companion_bytes: int | None, repository: ModelId | None,
) -> None:
    """Exact manifest bytes suffice when optional companion size is omitted."""
    card = _bundled_companion_card(
        kind, companion_bytes=companion_bytes, root="weights", repository=repository,
        revision=REVISION if repository is not None else None,
    )
    wire = card.model_dump_json()
    assert ModelCard.model_validate_json(wire).model_dump_json() == wire
    assert card.video is not None
    assert card.video.companions[0].size_bytes == companion_bytes


@pytest.mark.parametrize("kind", PIPELINE_COMPANION_KINDS)
def test_independently_pinned_external_pipeline_weights_need_no_base_bundle_entry(
    kind: VideoCompanionKind,
) -> None:
    """Separate repositories keep their separate pinned download transactions."""
    card = _bundled_companion_card(
        kind, path="weights/external.safetensors", companion_bytes=99,
        repository=ModelId("other/pipeline"), revision="b" * 40,
    )
    assert card.external_video_companions() == (("other/pipeline", "b" * 40),)


@pytest.mark.parametrize("kind", PIPELINE_COMPANION_KINDS)
def test_repository_wide_legacy_cards_do_not_require_a_selected_bundle(
    kind: VideoCompanionKind,
) -> None:
    """Legacy whole-repository downloads keep their existing companion contract."""
    card = _card(
        source_revision=REVISION,
        video=_video(companions=[VideoCompanionConfig(kind=kind, name="legacy", path="pipeline.safetensors")]),
    )
    assert card.artifact_bundle is None


@pytest.mark.parametrize("repository", [None, ModelId("example/upstream")])
def test_local_pipeline_companion_cannot_claim_a_different_download_revision(
    repository: ModelId | None,
) -> None:
    """The base transaction cannot install another revision's local companion."""
    with pytest.raises(ValidationError, match="revision must match source_revision"):
        _bundled_companion_card(VideoCompanionKind.LatentUpscaler, repository=repository, revision="b" * 40)


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


def test_preprocessor_companions_name_their_role() -> None:
    """A preprocessor says what its weights do; no other kind may."""
    pose = {
        "kind": "preprocessor",
        "name": "pose",
        "role": "pose_estimator",
        "path": "checkpoints/pose.safetensors",
        "repo": "other/pose",
        "revision": REVISION,
        "license": "mit",
    }
    video = _video(companions=[pose])
    assert video.companions[0].role is VideoPreprocessorRole.PoseEstimator
    with pytest.raises(ValidationError, match="names its role"):
        _video(companions=[{**pose, "role": None}])
    with pytest.raises(ValidationError, match="names its role"):
        VideoCompanionConfig.model_validate(
            {"kind": "lora", "name": "l", "path": "loras/l.st", "role": "pose_estimator"}
        )
    with pytest.raises(ValidationError, match="at most once"):
        _video(companions=[pose, {**pose, "name": "pose-two"}])
    with pytest.raises(ValidationError, match="short lowercase identifier"):
        _video(companions=[{**pose, "license": "MIT License"}])
    person = {**pose, "name": "person", "role": "person_detector", "path": "diffusion_models/p.st"}
    with pytest.raises(ValidationError, match="must all pin one revision"):
        _video(companions=[pose, {**person, "revision": "b" * 40}])
    assert len(_video(companions=[pose, person]).companions) == 2


def test_external_video_companions_are_listed_once_per_repository() -> None:
    """Every path that fetches, gates or protects companions reads one list."""
    shared = {"repo": "other/pose", "revision": REVISION}
    card = _card(
        video=_video(
            companions=[
                {"kind": "preprocessor", "name": "pose", "role": "pose_estimator", "path": "checkpoints/p.st", **shared},
                {"kind": "preprocessor", "name": "person", "role": "person_detector", "path": "diffusion_models/d.st", **shared},
                {"kind": "preprocessor", "name": "depth", "role": "depth_estimator", "path": "geometry_estimation/g.st", "repo": "other/depth", "revision": "b" * 40},
                # A companion in the card's own repository is part of its artifact.
                {"kind": "lora", "name": "turbo", "path": "loras/t.st", "repo": "example/video-model", "revision": REVISION},
                {"kind": "embedding", "name": "style", "path": "embeddings/s.st"},
            ]
        )
    )
    assert card.external_video_companions() == (
        ("other/pose", REVISION),
        ("other/depth", "b" * 40),
    )
    assert _card().external_video_companions() == ()


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
    paths = sorted(CARD_DIRECTORY.glob("Comfy-Org--MiniMax-H3-*.toml")) + sorted(
        TEST_ENGINE_DIRECTORY.glob("*.toml")
    )
    return [(path, tomllib.loads(path.read_text())) for path in paths]


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
    assert len(cards) == 3
    seen_test_engine = False
    for path, raw in cards:
        card = ModelCard.model_validate(raw)
        assert path.name == card.model_id.normalize() + ".toml"
        if card.model_id == ModelId("foxlight/test-video"):
            # The test engine's card has no weights to pin: it is served by
            # the deterministic renderer and provisioned in place, and it
            # lives apart from the artifact corpus the registry imports.
            assert path.parent == TEST_ENGINE_DIRECTORY
            seen_test_engine = True
            assert card.artifact_bundle is None and card.source_revision is None
            assert card.video is not None and card.license is None
            assert {"test_video", "test_video-cpu"} <= card.placement.compatible_backends
            continue
        assert path.parent == CARD_DIRECTORY
        assert card.source_revision is not None
        assert card.artifact_bundle is not None
        assert card.video is not None
        assert card.license is not None and card.license.display_name == "MiniMax H3"
        assert card.placement.compatible_backends == frozenset({"comfy", "comfy-cuda", "comfy-rocm"})
        bundle_paths = {item.path for item in card.artifact_bundle.files}
        for companion in card.video.companions:
            if companion.repo is None:
                assert companion.path in bundle_paths
                continue
            # A companion hosted elsewhere is pinned there, sized, and names
            # the license its repository declares.
            assert companion.kind is VideoCompanionKind.Preprocessor
            assert companion.revision is not None and companion.size_bytes
            assert companion.license is not None
        roles = {
            companion.role
            for companion in card.video.companions
            if companion.role is not None
        }
        assert roles == set(VideoPreprocessorRole)
        for component in card.components or []:
            assert any(item.path.startswith(component.component_path) for item in card.artifact_bundle.files)
        raw_bundle = cast("dict[str, object]", raw["artifact_bundle"])
        assert card.artifact_bundle.bundle_id == _derived_bundle_id(raw_bundle["files"])
        assert card.artifact_bundle.download_size == sum(item.size_bytes for item in card.artifact_bundle.files)
        assert all(item.object_id is not None for item in card.artifact_bundle.files)
        card.require_immutable_external_companions(context="bundled video cards")
    assert seen_test_engine


def test_video_cards_hidden_only_when_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Video cards are listed by default; SKULK_ENABLE_VIDEO_MODELS=false hides them."""
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
