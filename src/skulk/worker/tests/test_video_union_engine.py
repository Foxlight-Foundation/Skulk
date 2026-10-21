"""Union admission binds a complete card and an exact issued engine payload."""

import asyncio

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
from skulk.shared.types.video import (
    VideoGenerationTaskParams,
    VideoReferenceSpec,
    VideoUnionControl,
    derivable_guides,
    derivable_union_guides,
)
from skulk.worker.runner.comfy.graph import (
    ReferenceBinding,
    build_prompt,
    plan_comfy_render,
)


@pytest.fixture
def union_card() -> ModelCard:
    """Keep base component resolution real; add only a test-local adapter."""
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-distilled-comfy-int8")))
    assert card is not None and card.video is not None
    companion = VideoCompanionConfig(kind=VideoCompanionKind.Lora, name="union_control", path="loras/union.safetensors", strength=1.0)
    return card.model_copy(update={"video": card.video.model_copy(update={"companions": (*card.video.companions, companion)})})


@pytest.fixture
def issued_union(monkeypatch: pytest.MonkeyPatch) -> str:
    """A test-issued descriptor cannot make a real unpublished candidate eligible."""
    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.4-py3-none-any.whl", sha256="1" * 64, payload_sha256="2" * 64, size_bytes=1, comfy_revision=COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION, features=("union_control",))
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    return wheel.build_identity(f"comfy@{COMFY_PIN}/torch@2.14.0+cu130")


def _request(**changes: object) -> VideoGenerationTaskParams:
    source = VideoReferenceSpec(slot=0, kind="video", role="control", media_type="video/mp4", size_bytes=1, sha256="0" * 64)
    return VideoGenerationTaskParams.model_validate({"prompt": "A fox walks", "model": "Lightricks/LTX-2.5-distilled-comfy-int8", "seconds": 2, "size": "512x384", "union_control": {}, "references": (source,), "reference_bytes": 1, "total_input_chunks": 1, **changes})


@pytest.mark.parametrize("prepared", [False, True])
def test_union_patches_only_base_and_crops_before_upscale(union_card: ModelCard, issued_union: str, prepared: bool) -> None:
    """Never route structural pixels through an image anchor or H3 patch."""
    params = _request(union_control={"preprocessed": prepared})
    render = plan_comfy_render(params, union_card, engine_build=issued_union)
    graph = build_prompt(render, params, (ReferenceBinding(spec=params.references[0], input_name="guide.mp4"),), "union-test")
    assert graph["ltx_union_lora"]["inputs"]["model"] == ["unet", 0]
    assert graph["ltx_base_guider"]["inputs"]["model"] == ["ltx_union_lora", 0]
    assert graph["ltx_refine_guider"]["inputs"]["model"] == ["unet", 0]
    assert graph["ltx_union_guide"]["inputs"]["iclora_model"] == ["ltx_union_lora", 0]
    assert graph["ltx_base_av"]["inputs"]["video_latent"] == ["ltx_union_guide", 2]
    assert graph["ltx_spatial_upscale"]["inputs"]["samples"] == ["ltx_base_crop", 2]
    assert graph["ltx_refine_av"]["inputs"]["video_latent"] == ["ltx_spatial_upscale", 0]
    assert graph["ltx_refine_sigmas"]["inputs"]["sigmas"] == "0.909375, 0.725, 0.421875, 0.0"
    assert graph["sampler_select"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert graph["ltx_refine_sampler_select"]["inputs"]["sampler_name"] == "euler"
    assert graph["ltx_refine_sampler"]["inputs"]["sampler"] == ["ltx_refine_sampler_select", 0]
    assert ("derive_edges" in graph) is not prepared
    assert all(node["class_type"] != "MiniMaxH3FunControlNetApply" for node in graph.values())
    assert render.engine_settings().union_control == params.union_control
    assert render.engine_settings().stage_samplers == {"base": "euler_ancestral", "refine": "euler"}


def test_union_opening_image_uses_the_reference_strength_without_jpeg_preprocessing(union_card: ModelCard, issued_union: str) -> None:
    """The Union recipe differs from the ordinary I2V template in both stages."""
    source = _request().references[0]
    opening = VideoReferenceSpec(slot=1, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="1" * 64)
    params = _request(references=(source, opening), reference_bytes=2, total_input_chunks=2)
    render = plan_comfy_render(params, union_card, engine_build=issued_union)
    graph = build_prompt(render, params, (ReferenceBinding(spec=source, input_name="guide.mp4"), ReferenceBinding(spec=opening, input_name="opening.png")), "union-opening")
    for stage in ("base", "refine"):
        assert graph[f"guide_ltx_{stage}_1"]["inputs"]["strength"] == 1.0
        assert graph[f"guide_ltx_{stage}_1"]["inputs"]["image"] == ["load_ltx_1", 0]
    assert "ltx_image_1" not in graph


def test_union_routes_by_issued_feature_without_a_hardware_allowlist(union_card: ModelCard, issued_union: str) -> None:
    """The same exact payload contract works across eligible CUDA and ROCm lanes."""
    params = _request()
    assert video_request_supported(union_card, params, "comfy-cuda", issued_union)
    assert video_request_supported(union_card, params, "comfy-rocm", issued_union.replace("2.14.0+cu130", "2.9.1+rocm7.2"))
    assert not video_request_supported(union_card, params, "comfy-cuda", None)
    assert not video_request_supported(union_card, params, "mlx_video-metal", "mlx_video@sha256:" + "3" * 64)
    with pytest.raises(ValueError, match="published extension"):
        plan_comfy_render(params, union_card)


@pytest.mark.parametrize("guide", ["pose", "depth"])
def test_prepared_guides_do_not_claim_missing_preprocessor_weights(union_card: ModelCard, issued_union: str, guide: str) -> None:
    """A prepared signal is usable without pretending its annotator is installed."""
    with pytest.raises(ValueError, match="preprocessor companions"):
        plan_comfy_render(_request(union_control={"guide": guide}), union_card, engine_build=issued_union)
    plan_comfy_render(_request(union_control={"guide": guide, "preprocessed": True}), union_card, engine_build=issued_union)


@pytest.mark.parametrize("changes", [{"size": "576x384"}, {"pipeline": "dfr"}, {"duration": "auto"}, {"control_start": 0.2}, {"control_kind": "edges"}])
def test_union_refuses_incompatible_geometry_and_ignored_controls(union_card: ModelCard, changes: dict[str, object]) -> None:
    """Caller settings cannot silently fall through to another workflow."""
    assert union_card.video is not None
    with pytest.raises(ValueError):
        _request(**changes).validate_family(union_card.family, union_card.video)


def test_union_requires_the_exact_selected_adapter(union_card: ModelCard) -> None:
    """A card capability bit alone cannot invent a usable companion artifact."""
    assert union_card.video is not None
    incomplete = union_card.video.model_copy(update={"companions": ()})
    with pytest.raises(ValueError, match="compatible union_control IC-LoRA"):
        _request().validate_family(union_card.family, incomplete)


def test_union_capability_does_not_add_a_redundant_persisted_card_flag(union_card: ModelCard) -> None:
    """Existing card readers and preservation schemas retain their body contract."""
    assert union_card.video is not None and union_card.video.union_control
    assert "union_control" not in union_card.video.model_dump(mode="json")


def test_union_metadata_keeps_its_request_contract_separate(union_card: ModelCard, issued_union: str) -> None:
    """A guide offer must lead to the explicit request the engine actually accepts."""
    assert union_card.video is not None
    mode = _request().implied_mode()
    assert derivable_guides(union_card.video, mode) == ()
    assert derivable_union_guides(union_card.video, mode) == ("edges",)
    section = VideoCapabilitySection.from_model_card(union_card)
    assert section is not None and section.union_control
    assert section.guides == [] and section.default_guide is None
    assert [guide.kind for guide in section.union_guides] == ["edges"]
    plan_comfy_render(_request(), union_card, engine_build=issued_union)


def test_unissued_union_engine_does_not_offer_preprocessing(union_card: ModelCard, monkeypatch: pytest.MonkeyPatch) -> None:
    """Intrinsic adapter presence cannot invent an issued executable feature."""
    older = tuple(wheel for wheel in comfy_extension.PUBLISHED_EXTENSIONS if "union_control" not in wheel.features)
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", older)
    section = VideoCapabilitySection.from_model_card(union_card)
    assert section is not None
    assert not section.union_control and not section.union_guides


def test_published_union_does_not_upgrade_an_older_instance(union_card: ModelCard) -> None:
    """Issuing a wheel enables its exact build without changing retained instances."""
    current = next(wheel for wheel in comfy_extension.PUBLISHED_EXTENSIONS if "union_control" in wheel.features)
    older = next(wheel for wheel in comfy_extension.PUBLISHED_EXTENSIONS if "union_control" not in wheel.features)
    base = f"comfy@{COMFY_PIN}/torch@2.14.0+cu130"
    assert video_request_supported(union_card, _request(), "comfy-cuda", current.build_identity(base))
    assert not video_request_supported(union_card, _request(), "comfy-cuda", older.build_identity(base))
    assert not video_request_supported(union_card, _request(), "comfy-cuda", base + "/extension@sha256:" + "f" * 64)


def test_union_multipart_json_has_the_same_strict_typed_contract() -> None:
    """Form clients and JSON clients share validation and retained settings."""
    request = VideoCreateRequest.model_validate({"model": "model", "prompt": "fox", "union_control": '{"guide":"depth","preprocessed":true}'})
    assert request.union_control == VideoUnionControl(guide="depth", preprocessed=True)
    with pytest.raises(ValueError):
        VideoCreateRequest.model_validate({"model": "model", "prompt": "fox", "union_control": '{"preprocessed":"true"}'})
