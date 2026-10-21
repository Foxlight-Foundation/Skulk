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
    assert motion_card.video is not None and "motion_tracks" not in motion_card.video.model_dump(mode="json")


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
