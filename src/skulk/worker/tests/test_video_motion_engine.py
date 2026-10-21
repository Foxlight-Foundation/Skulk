"""Motion Track ingress and graph behavior remain separate from Union/H3."""

import asyncio
import json

import pytest

from skulk.api.types.api import VideoCapabilitySection, VideoCreateRequest
from skulk.provisioning import comfy_extension
from skulk.provisioning.manifest import COMFY_PIN
from skulk.shared.models.model_cards import (
    ModelCard,
    VideoCompanionConfig,
    VideoCompanionKind,
)
from skulk.shared.models.video_support import video_request_supported
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.common import ModelId
from skulk.shared.types.video import VideoGenerationTaskParams, VideoReferenceSpec
from skulk.shared.types.video_motion import VideoMotionTrackData, VideoMotionTracks
from skulk.worker.runner.comfy.graph import (
    ReferenceBinding,
    build_prompt,
    plan_comfy_render,
)


@pytest.fixture
def motion_card() -> ModelCard:
    """Retain exact baseline component paths and add one selected motion adapter."""
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-distilled-comfy-int8")))
    assert card is not None and card.video is not None
    adapter = VideoCompanionConfig(kind=VideoCompanionKind.Lora, name="motion_tracks", path="loras/motion.safetensors")
    return card.model_copy(update={"video": card.video.model_copy(update={"companions": (*card.video.companions, adapter)})})


@pytest.fixture
def issued_motion(monkeypatch: pytest.MonkeyPatch) -> str:
    """Issue only a test-local build; production must wait for published bytes."""
    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.5-py3-none-any.whl", sha256="1" * 64, payload_sha256="2" * 64, size_bytes=1, comfy_revision=COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION, features=("union_control", "motion_tracks"))
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    return wheel.build_identity(f"comfy@{COMFY_PIN}/torch@2.14.0+cu130")


def _request(**changes: object) -> VideoGenerationTaskParams:
    image = VideoReferenceSpec(slot=0, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="0" * 64)
    tracks = VideoReferenceSpec(slot=1, kind="data", role="motion_tracks", media_type="application/json", size_bytes=1, sha256="1" * 64)
    return VideoGenerationTaskParams.model_validate({"model": "model", "prompt": "A fox moves right", "seconds": 1, "size": "512x384", "motion_tracks": {}, "references": (image, tracks), "reference_bytes": 2, "total_input_chunks": 2, **changes})


def test_motion_samples_full_resolution_once_and_crops_before_decode(motion_card: ModelCard, issued_motion: str) -> None:
    """The official motion recipe has no half-resolution base or refinement."""
    params = _request()
    render = plan_comfy_render(params, motion_card, engine_build=issued_motion)
    bindings = tuple(ReferenceBinding(spec=spec, input_name=f"input-{spec.slot}") for spec in params.references)
    graph = build_prompt(render, params, bindings, "motion")
    assert render.pipeline == "motion_tracks" and render.steps == 8
    assert render.engine_settings().motion_tracks == params.motion_tracks
    assert graph["ltx_video_latent"]["inputs"]["width"] == 512
    assert graph["ltx_video_latent"]["inputs"]["height"] == 384
    assert graph["ltx_base_guider"]["inputs"]["model"] == ["ltx_motion_lora", 0]
    assert graph["guide_ltx_base_0"]["inputs"]["strength"] == 0.7
    assert graph["guide_ltx_base_0"]["inputs"]["image"] == ["load_ltx_0", 0]
    assert graph["ltx_motion_guide"]["inputs"]["latent"] == ["guide_ltx_base_0", 0]
    assert graph["ltx_tracks"]["inputs"]["tracks_file"] == "input-1"
    assert graph["decode_video"]["inputs"]["samples"] == ["ltx_base_crop", 2]
    assert graph["decode_audio"]["inputs"]["samples"] == ["ltx_base_split", 1]
    assert graph["sampler_select"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert "ltx_refine_sampler" not in graph and "ltx_spatial_model" not in graph
    assert "ltx_image_0" not in graph and "ltx_union_lora" not in graph


def test_motion_requires_exact_issued_feature_without_hardware_allowlist(motion_card: ModelCard, issued_motion: str) -> None:
    """Untested compatible devices remain eligible; old builds gain no feature."""
    params = _request()
    assert video_request_supported(motion_card, params, "comfy-cuda", issued_motion)
    assert video_request_supported(motion_card, params, "comfy-rocm", issued_motion.replace("2.14.0+cu130", "2.9.1+rocm7.2"))
    assert not video_request_supported(motion_card, params, "comfy-cuda", None)
    assert not video_request_supported(motion_card, params, "mlx_video-metal", "mlx_video@sha256:" + "3" * 64)
    with pytest.raises(ValueError, match="published extension"):
        plan_comfy_render(params, motion_card)
    section = VideoCapabilitySection.from_model_card(motion_card)
    assert section is not None and section.motion_tracks
    assert section.pipeline_modes["motion_tracks"] == ["fl2va"]
    assert section.pipeline_defaults["motion_tracks"].sampling_stage_steps == [8]
    assert section.pipeline_defaults["motion_tracks"].sampler == "euler_ancestral"
    assert motion_card.video is not None and "motion_tracks" not in motion_card.video.model_dump(mode="json")


def test_motion_metadata_does_not_require_an_unused_spatial_upscaler(motion_card: ModelCard, issued_motion: str) -> None:
    """Its single-stage offer must not inherit a two-stage recipe's dependency."""
    assert motion_card.video is not None
    video = motion_card.video.model_copy(update={"companions": tuple(item for item in motion_card.video.companions if item.kind is not VideoCompanionKind.LatentUpscaler)})
    card = motion_card.model_copy(update={"video": video})
    section = VideoCapabilitySection.from_model_card(card)
    assert section is not None and section.motion_tracks
    assert section.pipelines == ["motion_tracks"]
    assert section.decoders == ["convolutional"]
    render = plan_comfy_render(_request(), card, engine_build=issued_motion)
    assert render.files.spatial_upscaler is None and render.steps == 8


def test_unissued_motion_code_does_not_advertise_a_workflow(motion_card: ModelCard, monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate source and adapter truth cannot create an executable offer."""
    from skulk.provisioning import comfy_extension

    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", tuple(wheel for wheel in comfy_extension.published_extensions() if "motion_tracks" not in wheel.features))
    section = VideoCapabilitySection.from_model_card(motion_card)
    assert section is not None and not section.motion_tracks
    assert "motion_tracks" not in section.pipelines


@pytest.mark.parametrize("changes,reason", [({"seconds": 20}, "decoded-pixel budget"), ({"seconds": 20, "size": None}, "decoded-pixel budget"), ({"size": "4096x128"}, "raster workspace"), ({"size": "128x4096"}, "raster workspace"), ({"size": "4160x128"}, "canvas grid")])
def test_motion_allocation_bounds_apply_during_family_admission(motion_card: ModelCard, changes: dict[str, object], reason: str) -> None:
    """No accelerator qualification or model load is needed to reject bad fits."""
    assert motion_card.video is not None
    with pytest.raises(ValueError, match=reason):
        _request(**changes).validate_family(motion_card.family, motion_card.video)


def test_motion_defaults_are_identical_before_and_after_media_delivery(motion_card: ModelCard, issued_motion: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening-image paths cannot change a canvas after admission checked it."""
    def refuse_image_geometry(_params: VideoGenerationTaskParams) -> tuple[int, int]:
        raise AssertionError("Motion geometry inspected worker-only image bytes")

    monkeypatch.setattr("skulk.worker.runner.video_plan.keyframe_shape", refuse_image_geometry)
    params = _request(size=None, aspect_ratio="4:3")
    render = plan_comfy_render(params, motion_card, engine_build=issued_motion)
    assert (render.plan.width, render.plan.height) == (704, 512)


@pytest.mark.parametrize("changes", [{"pipeline": "distilled_two_stage"}, {"union_control": {}}, {"duration": "auto"}, {"prompt_relay": {"segments": [{"prompt": "one", "length": 1}, {"prompt": "two", "length": 1}]}}, {"decoder": "diffusion"}, {"control_kind": "pose"}, {"temporal_upscalings": 1}])
def test_motion_refuses_ignored_or_unimplemented_combinations(motion_card: ModelCard, changes: dict[str, object]) -> None:
    """Unsupported combinations cannot silently choose another graph."""
    assert motion_card.video is not None
    with pytest.raises(ValueError):
        _request(**changes).validate_family(motion_card.family, motion_card.video)


def test_motion_requires_the_selected_adapter(motion_card: ModelCard) -> None:
    """Descriptive metadata cannot invent an adapter or its executable feature."""
    assert motion_card.video is not None
    with pytest.raises(ValueError, match="compatible motion_tracks IC-LoRA"):
        _request().validate_family(motion_card.family, motion_card.video.model_copy(update={"companions": ()}))


@pytest.mark.parametrize("changes", [{"kind": "video", "media_type": "video/mp4"}, {"role": "reference"}, {"media_type": "application/octet-stream"}, {"size_bytes": 4 * 1024 * 1024 + 1}])
def test_motion_data_has_an_honest_bounded_mime_contract(changes: dict[str, object]) -> None:
    """Trajectory JSON is never disguised as encoded video."""
    with pytest.raises(ValueError):
        VideoReferenceSpec.model_validate({"slot": 0, "kind": "data", "role": "motion_tracks", "media_type": "application/json", "size_bytes": 1, "sha256": "0" * 64, **changes})


@pytest.mark.parametrize("changes", [{"version": 2}, {"frame_count": 10}, {"tracks": []}, {"tracks": [[{"x": 0.1, "y": 0.2}]]}, {"tracks": [[{"x": 1.1, "y": 0.2}] * 9]}, {"tracks": [[{"x": True, "y": 0.2}] * 9]}])
def test_motion_trajectory_validation_rejects_malformed_and_incomplete_media(changes: dict[str, object]) -> None:
    """Every path occupies the complete grid, with finite normalized positions."""
    with pytest.raises(ValueError):
        VideoMotionTrackData.model_validate_json(json.dumps({"version": 1, "frame_count": 9, "tracks": [[{"x": 0.1, "y": 0.2}] * 9], **changes}))


def test_motion_form_controls_are_strict_and_do_not_contain_coordinates() -> None:
    """Replicated controls retain only strengths; raw paths travel as media."""
    request = VideoCreateRequest.model_validate({"model": "model", "prompt": "fox", "motion_tracks": '{"guide_strength":0.5}'})
    assert request.motion_tracks == VideoMotionTracks(guide_strength=0.5)
    with pytest.raises(ValueError):
        VideoCreateRequest.model_validate({"model": "model", "prompt": "fox", "motion_tracks": '{"tracks":[]} '})
