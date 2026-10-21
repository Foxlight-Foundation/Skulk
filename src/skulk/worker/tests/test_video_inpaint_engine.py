"""Keep inpainting intrinsic truth separate from the issued execution contract."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from skulk.api.types.api import VideoCapabilitySection, VideoCreateRequest
from skulk.provisioning import comfy_extension
from skulk.provisioning.manifest import COMFY_PIN
from skulk.shared.models.model_cards import (
    ModelCard,
    VideoCompanionConfig,
    VideoCompanionKind,
    VideoMode,
)
from skulk.shared.models.video_support import video_request_supported
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.common import ModelId
from skulk.shared.types.video import VideoGenerationTaskParams, VideoReferenceSpec
from skulk.worker.runner.comfy.graph import (
    ReferenceBinding,
    build_prompt,
    plan_comfy_render,
)
from skulk.worker.runner.comfy.inpaint_input import (
    InpaintInputFacts,
    inpaint_allocation_bytes,
)
from skulk.worker.runner.mlx_video.plan import plan_native_job


@pytest.fixture
def inpaint_card() -> ModelCard:
    """Retain the real base component contract and add only the selected adapter."""
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-distilled-comfy-int8")))
    assert card is not None and card.video is not None
    companion = VideoCompanionConfig(kind=VideoCompanionKind.Lora, name="inpaint", path="loras/inpaint.safetensors", strength=1.0)
    companions = tuple(item for item in card.video.companions if item.kind is not VideoCompanionKind.LatentUpscaler)
    modes = tuple(dict.fromkeys((*card.video.modes, VideoMode.ReferenceToAudioVideo)))
    return card.model_copy(update={"video": card.video.model_copy(update={"companions": (*companions, companion), "modes": modes})})


def _request(**changes: object) -> VideoGenerationTaskParams:
    source = VideoReferenceSpec(slot=0, kind="video", role="source", media_type="video/mp4", size_bytes=1, sha256="0" * 64)
    mask = VideoReferenceSpec(slot=1, kind="image", role="mask", media_type="image/png", size_bytes=1, sha256="1" * 64)
    return VideoGenerationTaskParams.model_validate({"prompt": "Restore the masked region", "model": "Lightricks/LTX-2.5-distilled-comfy-int8", "seconds": 2, "size": "512x384", "inpainting": {}, "references": (source, mask), "reference_bytes": 2, "total_input_chunks": 2, **changes})


@pytest.fixture
def issued_inpaint(monkeypatch: pytest.MonkeyPatch) -> str:
    """Exercise a test-only exact descriptor without authorizing candidate bytes."""
    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.6-py3-none-any.whl", sha256="1" * 64, payload_sha256="2" * 64, size_bytes=1, comfy_revision=COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION, features=("inpaint",))
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    return wheel.build_identity(f"comfy@{COMFY_PIN}/torch@2.14.0+cu130")


@pytest.mark.parametrize("opening,audio,video_mask", [(False, True, False), (True, True, False), (False, False, True)])
def test_inpaint_graph_keeps_the_official_pixel_refinement_and_complete_timeline(inpaint_card: ModelCard, issued_inpaint: str, opening: bool, audio: bool, video_mask: bool) -> None:
    """Catch recipe substitutions that can render successfully with the wrong model."""
    params = _request(audio=audio, inpainting={"left": 32, "top": 64})
    references = params.references
    if video_mask:
        references = (references[0], references[1].model_copy(update={"kind": "video", "media_type": "video/mp4"}))
    if opening:
        references = (*references, VideoReferenceSpec(slot=2, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="2" * 64))
    params = _request(audio=audio, inpainting={"left": 32, "top": 64}, references=references, reference_bytes=len(references), total_input_chunks=len(references))
    render = plan_comfy_render(params, inpaint_card, engine_build=issued_inpaint)
    frames = render.plan.frame_count
    mask_frames = frames if video_mask else 1
    allocation = inpaint_allocation_bytes(512, 384, frames, source_width=320, source_height=192, mask_frames=mask_frames, fps=24, opening=opening)
    render = replace(render, inpaint_inputs=InpaintInputFacts(320, 192, frames, mask_frames, allocation))
    bindings = tuple(ReferenceBinding(spec=item, input_name=f"input-{item.slot}") for item in references)
    graph = build_prompt(render, params, bindings, "inpaint-test")
    assert render.pipeline == "inpaint" and render.steps == 10
    assert render.engine_settings().inpainting == params.inpainting
    assert render.engine_settings().stage_samplers == {"base": "euler_ancestral", "refine": "euler"}
    assert graph["ltx_video_latent"]["inputs"]["length"] == frames
    assert graph["ltx_inpaint_source"]["inputs"]["left"] == 32
    assert graph["ltx_inpaint_source"]["inputs"]["top"] == 64
    assert graph["ltx_inpaint_source"]["inputs"]["maximum_bytes"] == allocation
    for stage, radius in (("base", 15), ("full", 30)):
        assert graph[f"ltx_{stage}_context"]["inputs"]["spatial_radius"] == radius
        assert graph[f"ltx_{stage}_context"]["inputs"]["mask"] == ["ltx_inpaint_source", 1]
    assert graph["ltx_half_source"]["inputs"]["upscale_method"] == "lanczos"
    assert graph["ltx_base_guide"]["inputs"]["image"] == ["ltx_base_context", 0]
    assert graph["ltx_base_crop"]["inputs"]["latent"] == ["ltx_base_split", 0]
    assert graph["ltx_base_decode"]["inputs"]["samples"] == ["ltx_base_crop", 2]
    assert graph["ltx_base_blend"]["inputs"]["context"] == ["ltx_base_context", 0]
    assert graph["ltx_base_blend"]["inputs"]["mask"] == ["ltx_base_context", 1]
    assert graph["ltx_base_blend"]["inputs"]["low_res_dilation"] == 5
    assert graph["ltx_pixel_upscale"]["inputs"]["image"] == ["ltx_base_blend", 0]
    assert graph["ltx_pixel_upscale"]["inputs"]["upscale_method"] == "lanczos"
    assert graph["ltx_refine_encoded"]["inputs"]["pixels"] == ["ltx_pixel_upscale", 0]
    assert graph["ltx_refine_encoded"]["inputs"]["temporal_size"] == 500
    assert graph["ltx_refine_encoded"]["inputs"]["temporal_overlap"] == 64
    assert graph["ltx_refine_audio_reference"]["inputs"]["positive"] == ["ltx_base_crop", 0]
    assert graph["ltx_refine_audio_reference"]["inputs"]["audio_latent"] == ["ltx_base_split", 1]
    assert graph["ltx_refine_sigmas"]["inputs"]["sigmas"] == "0.7250, 0.4219, 0.0"
    assert graph["sampler_select"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert graph["ltx_refine_sampler_select"]["inputs"]["sampler_name"] == "euler"
    assert graph["ltx_base_guider"]["inputs"]["model"] == graph["ltx_refine_guider"]["inputs"]["model"] == ["ltx_inpaint_lora", 0]
    assert graph["ltx_final_blend"]["inputs"]["context"] == ["ltx_full_context", 0]
    assert graph["ltx_final_blend"]["inputs"]["low_res_dilation"] == 6
    for identifier in ("ltx_base_decode", "decode_video"):
        assert graph[identifier]["inputs"]["tile_size"] == 512
        assert graph[identifier]["inputs"]["overlap"] == 64
        assert graph[identifier]["inputs"]["temporal_size"] == 128
        assert graph[identifier]["inputs"]["temporal_overlap"] == 32
    assert all(node["class_type"] not in ("LTXVLatentUpsampler", "LatentUpscaleModelLoader") for node in graph.values())
    assert "ltx_refine_guide" not in graph
    assert ("decode_audio" in graph) is audio
    assert graph["ltx_source_audio"]["inputs"]["regenerate_audio"] is False
    if opening:
        assert graph["ltx_base_opening_preprocess"]["inputs"]["img_compression"] == 18
        assert graph["ltx_base_opening"]["inputs"]["strength"] == 0.7
        assert graph["ltx_refine_opening"]["inputs"]["strength"] == 1.0
        assert "ltx_refine_opening_preprocess" not in graph


def test_inpaint_graph_cannot_run_before_measured_preflight(inpaint_card: ModelCard, issued_inpaint: str) -> None:
    """Even an issued feature cannot turn client declarations into source facts."""
    params = _request()
    render = plan_comfy_render(params, inpaint_card, engine_build=issued_inpaint)
    with pytest.raises(ValueError, match="verified source facts"):
        build_prompt(render, params, (), "not-verified")


def test_inpaint_intrinsic_contract_uses_pixel_refinement_without_a_latent_upscaler(inpaint_card: ModelCard) -> None:
    """The inpainting adapter does not imply the ordinary latent-upscaler recipe."""
    assert inpaint_card.video is not None and inpaint_card.video.inpainting
    params = _request()
    assert params.implied_mode() is VideoMode.ReferenceToAudioVideo
    params.validate_family(inpaint_card.family, inpaint_card.video)


@pytest.mark.parametrize("backend", ["comfy-cuda", "comfy-rocm", "mlx_video-metal"])
def test_unissued_inpainting_never_gains_execution_authority_from_a_card(inpaint_card: ModelCard, backend: str) -> None:
    """Candidate source and declared companions cannot authorize an old engine."""
    assert not video_request_supported(inpaint_card, _request(), backend, None)
    with pytest.raises(ValueError, match="published extension"):
        plan_comfy_render(_request(), inpaint_card)


@pytest.mark.parametrize("changes", [{"pipeline": "dfr"}, {"pipeline": "distilled_two_stage"}, {"duration": "auto"}, {"decoder": "diffusion"}, {"size": None}, {"aspect_ratio": "4:3"}, {"size": "4160x128"}, {"mode": "fl2va"}, {"prompt_relay": {"segments": [{"prompt": "one scene", "length": 1}]}}, {"audio_conditioning": {}}, {"editing": {"operation": "extend"}}])
def test_inpainting_refuses_workflow_substitution_and_ignored_controls(inpaint_card: ModelCard, changes: dict[str, object]) -> None:
    """Every requested setting must belong to the complete fixed source workflow."""
    assert inpaint_card.video is not None
    with pytest.raises(ValueError):
        _request(**changes).validate_family(inpaint_card.family, inpaint_card.video)


def test_inpainting_requires_source_and_mask_with_an_optional_full_canvas_opening(inpaint_card: ModelCard) -> None:
    """A source or ordinary reference cannot silently stand in for the mask."""
    assert inpaint_card.video is not None
    params = _request()
    opening = VideoReferenceSpec(slot=2, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="2" * 64)
    _request(references=(*params.references, opening), reference_bytes=3, total_input_chunks=3).validate_family(inpaint_card.family, inpaint_card.video)
    with pytest.raises(ValueError, match="mask"):
        _request(references=params.references[:1], reference_bytes=1, total_input_chunks=1)


def test_inpainting_does_not_add_a_redundant_persisted_card_flag(inpaint_card: ModelCard) -> None:
    """Historical preservation readers keep the existing companion-based schema."""
    assert inpaint_card.video is not None
    assert "inpainting" not in inpaint_card.video.model_dump()


def test_native_planner_repeats_the_actual_feature_refusal_before_reading_weights(inpaint_card: ModelCard, tmp_path: Path) -> None:
    """An unstamped direct runner call must not turn a masked edit into generation."""
    with pytest.raises(ValueError, match="does not implement masked inpainting"):
        plan_native_job(_request(), inpaint_card, tmp_path / "model", tmp_path / "output", tmp_path / "input")


def test_inpainting_form_parses_only_the_strict_source_position() -> None:
    """Multipart and JSON callers share one execution-authority boundary."""
    create = VideoCreateRequest.model_validate({"prompt": "Masked edit", "model": "model", "inpainting": '{"left":11,"top":7}'})
    assert create.inpainting is not None
    assert create.inpainting.left == 11 and create.inpainting.top == 7
    with pytest.raises(ValueError):
        VideoCreateRequest.model_validate({"prompt": "Masked edit", "model": "model", "inpainting": '{"resize":true}'})


def test_unissued_inpainting_is_not_advertised_by_the_api(inpaint_card: ModelCard, monkeypatch: pytest.MonkeyPatch) -> None:
    """A signed card and candidate source alone cannot advertise executable work."""
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", tuple(wheel for wheel in comfy_extension.published_extensions() if "inpaint" not in wheel.features))
    section = VideoCapabilitySection.from_model_card(inpaint_card)
    assert section is not None and not section.inpainting
    assert "inpaint" not in section.pipelines
    assert all("inpaint" not in features for features in section.engine_feature_builds.values())


def test_inpaint_pipeline_without_controls_cannot_select_an_ordinary_graph(inpaint_card: ModelCard) -> None:
    """A pipeline label does not bypass the explicit source-mask request."""
    assert inpaint_card.video is not None
    with pytest.raises(ValueError, match="requires explicit inpainting"):
        _request(inpainting=None, pipeline="inpaint").validate_family(inpaint_card.family, inpaint_card.video)


def test_inpainting_requires_the_exact_named_adapter(inpaint_card: ModelCard) -> None:
    """Ordinary video packs cannot gain inpainting from a caller's control object."""
    assert inpaint_card.video is not None
    ordinary = inpaint_card.video.model_copy(update={"companions": tuple(item for item in inpaint_card.video.companions if item.name != "inpaint")})
    with pytest.raises(ValueError, match="exact compatible inpaint IC-LoRA"):
        _request().validate_family(inpaint_card.family, ordinary)


def test_issued_feature_projects_inpainting_without_a_latent_upscaler_or_hardware_claim(inpaint_card: ModelCard, monkeypatch: pytest.MonkeyPatch) -> None:
    """A test-only issued descriptor exercises metadata authority without a render."""
    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.6-py3-none-any.whl", sha256="1" * 64, payload_sha256="2" * 64, size_bytes=1, comfy_revision=COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION, features=("inpaint",))
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    section = VideoCapabilitySection.from_model_card(inpaint_card)
    assert section is not None and section.inpainting
    assert section.pipeline_modes["inpaint"] == ["ref2va"]
    assert section.pipeline_defaults["inpaint"].sampling_stage_steps == [8, 2]
    for backend in ("comfy-cuda", "comfy-rocm"):
        build = wheel.build_identity(f"comfy@{COMFY_PIN}/torch@2.14.0+cu130")
        assert video_request_supported(inpaint_card, _request(), backend, build)
