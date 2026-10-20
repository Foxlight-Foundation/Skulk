# pyright: reportPrivateUsage=false
"""LTX graph wiring, strict controls, and broad unqualified-hardware admission."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from skulk.api.types.api import VideoCapabilitySection
from skulk.master.placement import _card_platform_backends
from skulk.shared.backends import resolve_node_backend
from skulk.shared.models import model_cards
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    VideoCompanionConfig,
    VideoCompanionKind,
    VideoMode,
)
from skulk.shared.models.registry import (
    RegistryCapabilityClaim,
    RegistryEngineSupportClaim,
)
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.video import VideoGenerationTaskParams, VideoReferenceSpec
from skulk.worker.runner.comfy.graph import (
    ReferenceBinding,
    build_prompt,
    plan_comfy_render,
    resolve_model_files,
)
from skulk.worker.runner.comfy.ltx import DISTILLED_SIGMAS, REFINEMENT_SIGMAS
from skulk.worker.runner.comfy.server import _RenderSession

MODEL = ModelId("Lightricks/LTX-2.5-distilled-comfy-int8")


def _card() -> ModelCard:
    card = asyncio.run(load_fixture_card(MODEL))
    assert card is not None
    return card


def _params(**changes: object) -> VideoGenerationTaskParams:
    payload: dict[str, object] = {
        "prompt": "A fox crosses a meadow with soft footsteps",
        "model": str(MODEL),
        "seconds": 5,
        "size": "768x512",
        "seed": 7,
    }
    payload.update(changes)
    return VideoGenerationTaskParams.model_validate(payload)


def test_distilled_two_stage_graph_keeps_audio_and_exact_tables() -> None:
    params = _params()
    render = plan_comfy_render(params, _card())
    graph = build_prompt(render, params, (), "command")
    assert render.plan.frame_count == 121
    assert render.steps == 11
    assert render.sampler_steps == {"ltx_base_sampler": 8, "ltx_refine_sampler": 3}
    assert graph["clip"]["inputs"]["type"] == "ltxv"
    assert graph["ltx_video_latent"]["inputs"]["width"] == 384
    assert graph["ltx_video_latent"]["inputs"]["height"] == 256
    assert graph["ltx_base_sigmas"]["inputs"]["sigmas"] == DISTILLED_SIGMAS
    assert graph["ltx_refine_sigmas"]["inputs"]["sigmas"] == REFINEMENT_SIGMAS
    assert graph["ltx_spatial_upscale"]["inputs"]["samples"] == ["ltx_base_split", 0]
    assert graph["ltx_refine_av"]["inputs"]["audio_latent"] == ["ltx_base_split", 1]
    assert graph["decode_audio"]["inputs"]["samples"] == ["ltx_refine_split", 1]
    assert graph["create_video"]["inputs"]["audio"] == ["decode_audio", 0]
    assert graph["save_video"]["inputs"]["filename_prefix"] == "command/output"
    assert all("MiniMax" not in node["class_type"] for node in graph.values())


def test_silent_container_still_samples_the_joint_audio_branch() -> None:
    params = _params(audio=False)
    graph = build_prompt(plan_comfy_render(params, _card()), params, (), "command")
    assert "audio_vae" in graph and "ltx_audio_latent" in graph
    assert "decode_audio" not in graph
    assert "audio" not in graph["create_video"]["inputs"]


@pytest.mark.parametrize("variant", [None, "dev"])
def test_a_dev_or_undeclared_transformer_cannot_silently_run_as_distilled(
    variant: str | None,
) -> None:
    card = _card()
    assert card.video is not None
    card = card.model_copy(
        update={"video": card.video.model_copy(update={"transformer_variant": variant})}
    )
    with pytest.raises(ValueError, match="declared distilled transformer|does not match"):
        plan_comfy_render(_params(pipeline="distilled_two_stage"), card)
    metadata = VideoCapabilitySection.from_model_card(card)
    assert metadata is not None and not metadata.samplers


def test_image_anchors_are_reapplied_and_cropped_at_each_resolution() -> None:
    references = tuple(
        VideoReferenceSpec(
            slot=slot,
            kind="image",
            role=role,
            media_type="image/png",
            size_bytes=1,
            sha256="0" * 64,
            at_seconds=3.0 if role == "keyframe" else None,
        )
        for slot, role in enumerate(("first_frame", "last_frame", "keyframe"))
    )
    params = _params(references=references, reference_bytes=3, total_input_chunks=3)
    bindings = tuple(
        ReferenceBinding(spec=item, input_name=f"frame-{item.slot}.png")
        for item in references
    )
    graph = build_prompt(
        plan_comfy_render(params, _card()), params, bindings, "command"
    )
    assert graph["guide_ltx_base_0"]["class_type"] == "LTXVImgToVideoInplace"
    assert graph["guide_ltx_base_0"]["inputs"]["strength"] == 0.7
    assert graph["guide_ltx_refine_0"]["inputs"]["strength"] == 1.0
    assert graph["guide_ltx_base_1"]["inputs"]["frame_idx"] == -1
    assert graph["guide_ltx_refine_2"]["inputs"]["frame_idx"] == 72
    assert graph["ltx_spatial_upscale"]["inputs"]["samples"] == ["ltx_base_crop", 2]
    assert graph["decode_video"]["inputs"]["samples"] == ["ltx_refine_crop", 2]


@pytest.mark.parametrize(
    "controls",
    [
        {"steps": 20},
        {"sampler": "heun"},
        {"scheduler": "simple"},
        {"video_shift": 12.0},
        {"audio_shift": 3.0},
        {"lora": "h3-turbo"},
        {"lora_strength": 0.5},
        {"size": "64x512"},
        {"seed": 2**64},
        {"styles": ("h3-style",)},
    ],
)
def test_h3_controls_are_refused_instead_of_ignored(
    controls: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="LTX-2.5"):
        plan_comfy_render(_params(**controls), _card())


def test_two_stage_geometry_never_silently_floors_explicit_output() -> None:
    with pytest.raises(ValueError, match="divisible by 64"):
        plan_comfy_render(_params(size="960x544"), _card())
    plan = plan_comfy_render(_params(size=None, aspect_ratio="21:9"), _card()).plan
    assert plan.width % 64 == plan.height % 64 == 0


def test_model_metadata_does_not_offer_h3_engine_controls() -> None:
    metadata = VideoCapabilitySection.from_model_card(_card())
    assert metadata is not None
    assert metadata.canvas_multiple == 64
    assert metadata.samplers == ["euler_ancestral", "euler"]
    assert metadata.default_sampler == "euler_ancestral"
    assert metadata.schedulers == [] and metadata.default_scheduler == "manual"
    assert metadata.shift_bounds == []
    assert metadata.sampling_stage_steps == [8, 3]


def _dev_card() -> ModelCard:
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-dev-comfy-int8")))
    assert card is not None
    return card


def test_dev_guidance_and_refinement_are_separate_models() -> None:
    params = _params(steps=20, negative_prompt="flicker", guidance={"video_cfg": 4.0, "audio_cfg": 6.0, "stg_scale": 1.0, "modality_scale": 3.0})
    render = plan_comfy_render(params, _dev_card())
    graph = build_prompt(render, params, (), "command")
    assert render.sampler_steps == {"ltx_base_sampler": 20, "ltx_refine_sampler": 3}
    assert graph["ltx_negative"]["inputs"]["text"] == "flicker"
    assert graph["ltx_base_sigmas"]["class_type"] == "LTXVScheduler"
    assert graph["ltx_base_guider"]["inputs"]["model"] == ["ltx_modality", 0]
    assert graph["ltx_base_guider"]["inputs"]["video_cfg"] == 4.0
    assert graph["ltx_base_guider"]["inputs"]["audio_cfg"] == 6.0
    assert graph["ltx_modality"]["inputs"]["model"] == ["ltx_stg", 0]
    assert graph["ltx_stg"]["inputs"]["blocks"] == "28"
    assert graph["ltx_refinement_lora"]["inputs"]["model"] == ["unet", 0]
    assert graph["ltx_refine_guider"]["inputs"]["model"] == ["ltx_refinement_lora", 0]
    assert graph["ltx_refine_guider"]["inputs"]["video_cfg"] == 1.0
    assert graph["ltx_refine_av"]["inputs"]["audio_latent"] == ["ltx_frozen_audio", 0]
    assert graph["decode_audio"]["inputs"]["samples"] == ["ltx_base_split", 1]
    assert render.engine_settings().guidance == params.guidance
    assert render.engine_settings().adapter == "distilled_refinement"
    assert render.engine_settings().adapter_strength == 1.0


def test_generated_tokens_are_removed_before_upscaling() -> None:
    params = _params(generated_keyframe_interval=24)
    render = plan_comfy_render(params, _card())
    graph = build_prompt(render, params, (), "command")
    assert graph["ltx_generated_keyframes"]["inputs"]["interval_frames"] == 24
    assert graph["ltx_base_av"]["inputs"]["video_latent"] == ["ltx_generated_keyframes", 2]
    assert graph["ltx_strip_generated"]["inputs"]["latent"] == ["ltx_base_split", 0]
    assert graph["ltx_spatial_upscale"]["inputs"]["samples"] == ["ltx_strip_generated", 2]
    assert graph["decode_video"]["inputs"]["samples"] == ["ltx_refine_split", 0]


def test_alternate_decoder_does_not_replace_conditioning_vae() -> None:
    params = _params(decoder="diffusion")
    graph = build_prompt(plan_comfy_render(params, _dev_card()), params, (), "command")
    assert graph["video_vae"]["inputs"]["vae_name"] == "ltx-2.5-video-vae-conv-bf16.safetensors"
    assert graph["ltx_spatial_upscale"]["inputs"]["vae"] == ["video_vae", 0]
    assert graph["ltx_diffusion_decoder"]["inputs"]["vae_name"] == "ltx-2.5-video-vae-bf16.safetensors"
    assert graph["decode_video"]["inputs"]["vae"] == ["ltx_diffusion_decoder", 0]


def test_duration_is_bounded_before_audio_and_video_allocation() -> None:
    params = _params(duration="auto", duration_min_seconds=2)
    render = plan_comfy_render(params, _dev_card())
    graph = build_prompt(render, params, (), "command")
    assert graph["ltx_duration"]["inputs"]["min_seconds"] == 2.0
    assert graph["ltx_duration"]["inputs"]["max_seconds"] == 5.0
    assert graph["ltx_video_latent"]["inputs"]["length"] == ["ltx_duration", 0]
    assert graph["ltx_audio_latent"]["inputs"]["frames_number"] == ["ltx_duration", 0]
    assert render.minimum_frames == 49 and render.plan.frame_count == 121


@pytest.mark.parametrize("controls", [{"decoder": "diffusion"}, {"duration": "auto"}, {"negative_prompt": "blur"}, {"guidance": {"video_cfg": 2.0}}])
def test_missing_or_inapplicable_selected_features_are_refused(controls: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="LTX-2.5"):
        plan_comfy_render(_params(**controls), _card())


def test_comfy_does_not_approximate_the_hq_sampler() -> None:
    with pytest.raises(ValueError, match="Res2s"):
        plan_comfy_render(_params(pipeline="hq_two_stage"), _dev_card())


def test_duration_metadata_and_wire_roundtrip() -> None:
    from skulk.api.types.api import VideoCreateRequest
    from skulk.shared.types.video import VideoEngineSettings

    create = VideoCreateRequest.model_validate({"model": str(MODEL), "prompt": "A fox", "guidance": '{"video_cfg": 4.0}', "duration": "auto"})
    assert create.guidance is not None and create.guidance.video_cfg == 4.0
    params = _params(duration="auto", guidance=create.guidance)
    settings = plan_comfy_render(params, _dev_card()).engine_settings()
    assert VideoEngineSettings.model_validate_json(settings.model_dump_json()) == settings
    metadata = VideoCapabilitySection.from_model_card(_dev_card())
    assert metadata is not None
    assert metadata.pipelines == ["dev_two_stage", "hq_two_stage", "retake", "extend"]
    assert metadata.decoders == ["convolutional", "diffusion"]
    assert metadata.automatic_duration and metadata.guidance is not None


@pytest.mark.parametrize("kind,controls", [
    (VideoCompanionKind.DurationHead, {"duration": "auto"}),
    (VideoCompanionKind.VideoDecoder, {"decoder": "diffusion"}),
    (VideoCompanionKind.Lora, {}),
    (VideoCompanionKind.LatentUpscaler, {}),
])
def test_scoped_companions_refuse_a_forbidden_mode_before_graph_dispatch(kind: VideoCompanionKind, controls: dict[str, object]) -> None:
    card = _dev_card()
    assert card.video is not None
    companions = tuple(item.model_copy(update={"modes": (VideoMode.FramesToAudioVideo,)}) if item.kind is kind else item for item in card.video.companions)
    card = card.model_copy(update={"video": card.video.model_copy(update={"companions": companions})})
    with pytest.raises(ValueError, match="requires"):
        plan_comfy_render(_params(**controls), card)
    first = VideoReferenceSpec(slot=0, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="0" * 64)
    params = _params(**controls, references=(first,), reference_bytes=1, total_input_chunks=1)
    render = plan_comfy_render(params, card)
    graph = build_prompt(render, params, (ReferenceBinding(spec=first, input_name="frame.png"),), "command")
    assert graph["ltx_base_sampler"]["class_type"] == "SamplerCustomAdvanced"
    metadata = VideoCapabilitySection.from_model_card(card)
    assert metadata is not None
    if kind is VideoCompanionKind.DurationHead:
        assert resolve_model_files(card, VideoMode.TextToAudioVideo).duration_head is None
        assert metadata.automatic_duration_modes == ["fl2va"]
    elif kind is VideoCompanionKind.VideoDecoder:
        assert resolve_model_files(card, VideoMode.TextToAudioVideo).diffusion_decoder is None
        assert metadata.decoder_modes["diffusion"] == ["fl2va"]
    else:
        assert metadata.pipeline_modes == {"dev_two_stage": ["fl2va"], "hq_two_stage": ["fl2va"], "retake": ["ref2va"], "extend": ["ref2va"]}


@pytest.mark.parametrize("controls", [{"duration": "auto", "duration_min_seconds": 5}, {"duration_min_seconds": 2}, {"duration": "auto", "duration_min_seconds": 6}])
def test_automatic_duration_refuses_empty_or_inverted_grid(controls: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="duration"):
        plan_comfy_render(_params(**controls), _dev_card())


@pytest.mark.parametrize("backend", ["comfy-cuda", "comfy-rocm"])
def test_no_benchmark_claim_is_not_a_hardware_execution_allowlist(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_cards, "_registry_engine_support", ())
    card = _card()
    resources = NodeResources(
        backends=frozenset({backend}),
        hardware_classes=frozenset({"new-untested-gpu"}),
        engine_builds={backend: "operator-installed-build"},
        participation="full",
    )
    compatible = _card_platform_backends(card, resources)
    assert (
        resolve_node_backend(
            compatible, card.placement.backend_preference, resources.backends
        )
        == backend
    )
    assert card.placement.min_vram_gib is None
    assert resolve_node_backend(compatible, (), frozenset({"llama_server-cpu"})) is None


def test_a_claim_for_tested_hardware_does_not_remove_broad_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publishing a hardware-specific result cannot make other GPUs unrunnable."""
    backend = "comfy-cuda"
    card_id = "card_" + "a" * 52
    build = "comfy@sha256:" + "1" * 64
    card = _card().model_copy(
        update={
            "registry_card_id": card_id,
            "registry_provenance": "foxlight",
            "registry_architecture": "ltx-2.5",
            "registry_artifact_format": "safetensors",
            "registry_capability_claims": (
                RegistryCapabilityClaim(
                    capability_id="video.generate",
                    scope="model",
                    status="observed",
                    source="upstream_structured",
                    confidence=1.0,
                ),
            ),
        }
    )
    claim = RegistryEngineSupportClaim(
        engine="comfy",
        engine_build=build,
        architecture="ltx-2.5",
        artifact_format="safetensors",
        artifact_card_id=card_id,
        quantization=card.quantization,
        capability_id="video.generate",
        status="supported",
        evidence_kind="feature_qualification",
        evidence_trust="foxlight_observed",
        source_url="https://evidence.example/ltx",
        source_sha256="2" * 64,
        rationale="Synthetic tested-hardware fixture.",
        hardware_classes=("tested-gpu",),
        claim_id="support_" + "b" * 52,
        recorded_by="test@example.com",
        created_at=datetime(2026, 10, 14, tzinfo=UTC),
    )
    monkeypatch.setattr(model_cards, "_registry_engine_support", (claim,))
    unknown = NodeResources(
        backends=frozenset({backend}),
        engine_builds={backend: build},
        hardware_classes=frozenset({"untested-gpu"}),
        participation="full",
    )
    assert (
        model_cards.registry_supported_backends_for_node(
            card,
            node_backends=unknown.backends,
            engine_builds=unknown.engine_builds,
            hardware_classes=unknown.hardware_classes,
        )
        == frozenset()
    )
    assert backend in _card_platform_backends(card, unknown)
    assert model_cards.registry_supported_backends_for_node(
        card,
        node_backends=unknown.backends,
        engine_builds=unknown.engine_builds,
        hardware_classes=frozenset({"tested-gpu"}),
    ) == frozenset({backend})


def test_multistage_progress_counts_each_sampler_once_and_never_rewinds() -> None:
    fractions: list[float] = []
    session = _RenderSession(
        "prompt",
        lambda _stage, _step, _total, fraction: fractions.append(fraction),
        11,
        {"ltx_base_sampler": 8, "ltx_refine_sampler": 3},
    )
    base = {"value": 8, "max": 8}
    session.handle(
        {
            "type": "progress_state",
            "data": {"prompt_id": "prompt", "nodes": {"ltx_base_sampler": base}},
        }
    )
    session.handle(
        {
            "type": "executing",
            "data": {"prompt_id": "prompt", "node": "ltx_spatial_upscale"},
        }
    )
    session.handle(
        {
            "type": "progress_state",
            "data": {
                "prompt_id": "prompt",
                "nodes": {
                    "ltx_base_sampler": base,
                    "ltx_refine_sampler": {"value": 1, "max": 3},
                },
            },
        }
    )
    assert session.steps_seen == 9
    session.handle(
        {
            "type": "progress_state",
            "data": {
                "prompt_id": "prompt",
                "nodes": {
                    "ltx_base_sampler": base,
                    "ltx_refine_sampler": {"value": 3, "max": 3},
                },
            },
        }
    )
    assert session.steps_seen == 11
    assert fractions == sorted(fractions)
    assert fractions[-1] <= 1.0


def _dfr_card(*, temporal: bool = True) -> ModelCard:
    card = _card()
    assert card.video is not None
    companions = (*card.video.companions, VideoCompanionConfig(kind=VideoCompanionKind.Lora, name="dfr_spatial", path="loras/dfr.safetensors"))
    if temporal:
        companions = (*companions, VideoCompanionConfig(kind=VideoCompanionKind.LatentUpscaler, name="temporal_x2", path="latent_upscale_models/temporal.safetensors"))
    return card.model_copy(update={"video": card.video.model_copy(update={"companions": companions})})


def _issued_dfr_build(monkeypatch: pytest.MonkeyPatch) -> str:
    """Bind candidate tests to an explicit issued inventory without publishing it."""
    from skulk.provisioning import comfy_extension

    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.2-py3-none-any.whl", sha256="e" * 64, payload_sha256="f" * 64, size_bytes=1000, features=("res2s", "retake", "extend", "dfr"), comfy_revision=comfy_extension.COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION)
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    return wheel.build_identity(f"comfy@{wheel.comfy_revision}/torch@2.14.0+cu130")


@pytest.mark.parametrize("backend", ["comfy-cuda", "comfy-rocm"])
def test_text_only_dfr_requires_issued_mask_code_but_old_image_conditioning_remains_usable(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    from skulk.shared.models.video_support import video_request_supported

    card, text = _dfr_card(), _params(pipeline="dfr")
    image = VideoReferenceSpec(slot=0, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="0" * 64)
    conditioned = _params(pipeline="dfr", references=(image,), reference_bytes=1, total_input_chunks=1)
    for previous in (None, "unknown"):
        assert not video_request_supported(card, text, backend, previous)
        with pytest.raises(ValueError, match="exact published detailing extension"):
            plan_comfy_render(text, card, engine_build=previous)
        assert video_request_supported(card, conditioned, backend, previous)
        render = plan_comfy_render(conditioned, card, engine_build=previous)
        graph = build_prompt(render, conditioned, (ReferenceBinding(spec=image, input_name="image.png"),), "old")
        assert graph["ltx_refine_keyframes"]["class_type"] == "LTXVAddGeneratedKeyframes"
    issued = _issued_dfr_build(monkeypatch)
    if backend == "comfy-rocm":
        issued = issued.replace("cu130", "rocm7.4")
    assert video_request_supported(card, text, backend, issued)
    metadata = VideoCapabilitySection.from_model_card(card)
    assert metadata is not None and any("dfr" in features for features in metadata.engine_feature_builds.values())
    for unknown in (issued.split("/extension@")[0], issued.replace("f" * 64, "a" * 64)):
        assert not video_request_supported(card, text, backend, unknown)


@pytest.mark.parametrize("temporal", [0, 1])
def test_dfr_carries_detailing_tokens_and_preserves_the_timeline(monkeypatch: pytest.MonkeyPatch, temporal: int) -> None:
    params = _params(pipeline="dfr", temporal_upscalings=temporal)
    render = plan_comfy_render(params, _dfr_card(), engine_build=_issued_dfr_build(monkeypatch))
    graph = build_prompt(render, params, (), "command")
    assert graph["ltx_generated_keyframes"]["class_type"] == "SkulkLTXAddGeneratedKeyframes"
    assert graph["ltx_refine_keyframes"]["class_type"] == "SkulkLTXAddGeneratedKeyframes"
    assert graph["ltx_refine_keyframes"]["inputs"]["latent"] == ["ltx_spatial_upscale", 0]
    assert render.sampler == "euler"
    assert graph["sampler_select"]["inputs"]["sampler_name"] == "euler"
    assert render.plan.fps == 24 * (2 ** temporal)
    assert render.plan.frame_count == 120 * (2 ** temporal) + 1
    assert (render.plan.frame_count - 1) / render.plan.fps == 5
    assert graph["condition"]["inputs"]["frame_rate"] == 24
    assert graph["ltx_video_latent"]["inputs"]["length"] == 121
    assert graph["ltx_audio_latent"]["inputs"]["frames_number"] == 121
    assert graph["ltx_upscale_keyframes"]["inputs"]["samples"] == ["ltx_strip_generated", 3]
    assert graph["ltx_refine_keyframes"]["inputs"]["keyframes"] == ["ltx_upscale_keyframes", 0]
    assert graph["ltx_dfr_detail_guide"]["inputs"]["guiding_latent"] == ["ltx_strip_generated", 2]
    assert graph["ltx_dfr_lora"]["inputs"]["lora_name"] == "dfr.safetensors"
    assert graph["ltx_refine_guider"]["inputs"]["model"] == ["ltx_dfr_lora", 0]
    assert graph["ltx_refine_sigmas"]["inputs"]["sigmas"] == "0.909375, 0.725, 0.421875, 0.0"
    assert graph["decode_audio"]["inputs"]["samples"] == ["ltx_refine_split", 1]
    assert graph["create_video"]["inputs"]["fps"] == float(render.plan.fps)
    assert render.steps == 11 + 4 * temporal
    assert render.engine_settings().temporal_upscalings == temporal
    if temporal:
        assert graph["ltx_temporal_keyframes"]["class_type"] == "SkulkLTXAddGeneratedKeyframes"
        assert graph["ltx_temporal_upscale"]["inputs"]["samples"] == ["ltx_refine_crop", 2]
        assert graph["ltx_temporal_guides"]["inputs"]["keyframes"] == ["ltx_refine_strip_generated", 3]
        assert graph["ltx_temporal_condition"]["inputs"]["frame_rate"] == 48
        assert graph["ltx_temporal_frozen_audio"]["inputs"]["latent"] == ["ltx_refine_split", 1]
        assert graph["decode_video"]["inputs"]["samples"] == ["ltx_temporal_crop", 2]
        assert render.sampler_steps["ltx_temporal_sampler"] == 4
    else:
        assert "ltx_temporal_sampler" not in graph
        assert graph["decode_video"]["inputs"]["samples"] == ["ltx_refine_crop", 2]


@pytest.mark.parametrize("controls,card,error", [
    ({"pipeline": "dfr"}, _card, "dfr_spatial"),
    ({"pipeline": "dfr", "temporal_upscalings": 1}, lambda: _dfr_card(temporal=False), "temporal_x2"),
    ({"temporal_upscalings": 1}, _card, "requires.*DFR"),
    ({"pipeline": "dfr", "duration": "auto"}, _dfr_card, "fixed duration"),
])
def test_dfr_requires_its_selected_assets_and_explicit_controls(controls: dict[str, object], card: Callable[[], ModelCard], error: str) -> None:
    # Each case constructs an independent card; no qualification allowlist is involved.
    with pytest.raises(ValueError, match=error):
        plan_comfy_render(_params(**controls), card())


def test_dfr_metadata_requires_mode_matching_companions() -> None:
    metadata = VideoCapabilitySection.from_model_card(_card())
    assert metadata is not None and "dfr" not in metadata.pipelines and not metadata.temporal_upscaling
    metadata = VideoCapabilitySection.from_model_card(_dfr_card())
    assert metadata is not None and "dfr" in metadata.pipelines and metadata.temporal_upscaling
    metadata = VideoCapabilitySection.from_model_card(_dfr_card(temporal=False))
    assert metadata is not None and "dfr" in metadata.pipelines and not metadata.temporal_upscaling


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_replicated_temporal_control_is_a_strict_integer(value: object) -> None:
    with pytest.raises(ValueError, match="temporal_upscalings"):
        _params(temporal_upscalings=value)


def test_temporal_control_accepts_multipart_integer_and_roundtrips(monkeypatch: pytest.MonkeyPatch) -> None:
    from skulk.api.types.api import VideoCreateRequest
    from skulk.shared.types.video import VideoEngineSettings
    create = VideoCreateRequest.model_validate({"model": str(MODEL), "prompt": "A fox", "pipeline": "dfr", "temporal_upscalings": "1"})
    assert create.temporal_upscalings == 1
    settings = plan_comfy_render(_params(pipeline="dfr", temporal_upscalings=create.temporal_upscalings), _dfr_card(), engine_build=_issued_dfr_build(monkeypatch)).engine_settings()
    assert VideoEngineSettings.model_validate_json(settings.model_dump_json()) == settings



def test_temporal_metadata_does_not_expand_a_companions_modes() -> None:
    card = _dfr_card()
    assert card.video is not None
    companions = tuple(item.model_copy(update={"modes": (VideoMode.FramesToAudioVideo,)}) if item.name == "temporal_x2" else item for item in card.video.companions)
    card = card.model_copy(update={"video": card.video.model_copy(update={"companions": companions})})
    metadata = VideoCapabilitySection.from_model_card(card)
    assert metadata is not None and metadata.temporal_upscaling
    assert metadata.temporal_upscaling_modes == ["fl2va"]
    with pytest.raises(ValueError, match="temporal_x2"):
        plan_comfy_render(_params(pipeline="dfr", temporal_upscalings=1), card)



def test_dfr_fixed_sampler_cannot_be_replaced_by_distilled_ancestral(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="fixed euler sampler"):
        plan_comfy_render(_params(pipeline="dfr", sampler="euler_ancestral"), _dfr_card())
    render = plan_comfy_render(_params(pipeline="dfr", sampler="euler"), _dfr_card(), engine_build=_issued_dfr_build(monkeypatch))
    assert render.engine_settings().sampler == "euler"
    metadata = VideoCapabilitySection.from_model_card(_dfr_card())
    assert metadata is not None and metadata.samplers == ["euler_ancestral", "euler"]
    assert metadata.default_sampler == "euler_ancestral"


def _issued_res2s_build(monkeypatch: pytest.MonkeyPatch) -> str:
    """Represent issued-code inventory without publishing any test candidate."""
    from skulk.provisioning import comfy_extension

    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.0-py3-none-any.whl", sha256="a" * 64, payload_sha256="b" * 64, size_bytes=1000, features=("res2s",), comfy_revision=comfy_extension.COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION)
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    return wheel.build_identity(f"comfy@{wheel.comfy_revision}/torch@2.14.0+cu130")


@pytest.mark.parametrize("backend", ["comfy-cuda", "comfy-rocm"])
def test_hq_requires_the_exact_issued_instance_build(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    from skulk.shared.models.video_support import video_request_supported

    params, card = _params(pipeline="hq_two_stage"), _dev_card()
    build = _issued_res2s_build(monkeypatch)
    if backend == "comfy-rocm":
        build = build.replace("cu130", "rocm7.4")
    assert video_request_supported(card, params, backend, build)
    for old in (None, "old-build", build.split("/extension@")[0], build.replace("b" * 64, "c" * 64)):
        assert not video_request_supported(card, params, backend, old)
        with pytest.raises(ValueError, match="published Res2s"):
            plan_comfy_render(params, card, engine_build=old)


def test_hq_uses_reference_stages_guidance_and_joint_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _params(pipeline="hq_two_stage")
    render = plan_comfy_render(params, _dev_card(), engine_build=_issued_res2s_build(monkeypatch))
    graph = build_prompt(render, params, (), "hq")
    assert render.sampler_steps == {"ltx_base_sampler": 15, "ltx_refine_sampler": 3}
    assert render.sampler == "res2s" and render.steps == 18
    assert graph["sampler_select"]["class_type"] == "SkulkLTXRes2sSampler"
    assert graph["ltx_hq_base_lora"]["inputs"]["strength_model"] == 0.25
    assert graph["ltx_refinement_lora"]["inputs"]["strength_model"] == 0.5
    assert graph["ltx_hq_base_lora"]["inputs"]["model"] == ["unet", 0]
    assert graph["ltx_refinement_lora"]["inputs"]["model"] == ["unet", 0]
    assert graph["ltx_base_guider"]["inputs"]["model"] == ["ltx_hq_rescale", 0]
    assert graph["ltx_base_guider"]["inputs"]["video_cfg"] == 3.0
    assert graph["ltx_base_guider"]["inputs"]["audio_cfg"] == 7.0
    assert graph["ltx_modality"]["inputs"]["modality_scale"] == 3.0
    assert graph["ltx_modality"]["inputs"]["model"] == ["ltx_hq_base_lora", 0]
    assert graph["ltx_hq_rescale"]["inputs"]["video_scale"] == 0.45
    assert graph["ltx_hq_rescale"]["inputs"]["audio_scale"] == 1.0
    assert graph["ltx_base_sigmas"]["inputs"]["latent"] == ["ltx_video_latent", 0]
    assert graph["ltx_refine_sigmas"]["inputs"]["sigmas"] == "0.909375, 0.725, 0.421875, 0.0"
    assert "ltx_frozen_audio" not in graph
    assert graph["ltx_refine_av"]["inputs"]["audio_latent"] == ["ltx_base_split", 1]
    assert graph["decode_audio"]["inputs"]["samples"] == ["ltx_base_split", 1]
    assert graph["ltx_refine_guider"]["inputs"]["video_cfg"] == 1.0
    settings = render.engine_settings()
    assert settings.stage_lora_strengths == {"base": 0.25, "refine": 0.5}
    assert (settings.guidance_video_rescale, settings.guidance_audio_rescale) == (0.45, 1.0)


def test_hq_standard_decode_strips_generated_slots_before_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _params(pipeline="hq_two_stage", generated_keyframe_interval=24)
    card = _dev_card()
    assert card.video is not None
    card = card.model_copy(update={"video": card.video.model_copy(update={"generated_keyframes": True})})
    graph = build_prompt(plan_comfy_render(params, card, engine_build=_issued_res2s_build(monkeypatch)), params, (), "hq")
    assert graph["ltx_spatial_upscale"]["inputs"]["samples"] == ["ltx_strip_generated", 2]
    assert graph["ltx_refine_av"]["inputs"]["video_latent"] == ["ltx_spatial_upscale", 0]
    assert graph["decode_video"]["inputs"]["samples"] == ["ltx_refine_split", 0]
    assert "ltx_refine_keyframes" not in graph and "ltx_dfr_detail_guide" not in graph


def test_hq_metadata_binds_issued_builds_and_its_own_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from skulk.shared.models.video_support import video_feature_build_key

    build = _issued_res2s_build(monkeypatch)
    metadata = VideoCapabilitySection.from_model_card(_dev_card())
    assert metadata is not None and "hq_two_stage" in metadata.pipelines
    defaults = metadata.pipeline_defaults["hq_two_stage"]
    assert defaults.base_steps == 15 and defaults.sampler == "res2s"
    assert defaults.sampling_stage_steps == [15, 3]
    assert defaults.guidance is not None and defaults.guidance.modality_scale == 3.0
    assert metadata.engine_feature_builds[video_feature_build_key(build) or ""] == ["hq_two_stage"]
    assert video_feature_build_key(build + "/extra") is None
    assert video_feature_build_key(build.replace("/torch@", "/unknown@")) is None
    explicit = plan_comfy_render(_params(pipeline="hq_two_stage", steps=7), _dev_card(), engine_build=build)
    assert explicit.sampler_steps == {"ltx_base_sampler": 7, "ltx_refine_sampler": 3}


@pytest.mark.parametrize("sampler", ["euler", "euler_ancestral"])
def test_hq_never_accepts_another_sampler_override(sampler: str) -> None:
    with pytest.raises(ValueError, match="fixed Res2s"):
        plan_comfy_render(_params(pipeline="hq_two_stage", sampler=sampler), _dev_card())


def _issued_editing_build(monkeypatch: pytest.MonkeyPatch) -> str:
    """Represent a distinct issued editing payload, with no hardware qualification."""
    from skulk.provisioning import comfy_extension

    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.0-py3-none-any.whl", sha256="c" * 64, payload_sha256="d" * 64, size_bytes=1000, features=("res2s", "retake", "extend"), comfy_revision=comfy_extension.COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION)
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    return wheel.build_identity(f"comfy@{wheel.comfy_revision}/torch@2.14.0+cu130")


def _editing_params(operation: str, **changes: object) -> VideoGenerationTaskParams:
    source = VideoReferenceSpec(slot=0, kind="video", role="source", media_type="video/mp4", size_bytes=1, sha256="0" * 64)
    editing = {"operation": operation, **({"start_seconds": 0.5, "end_seconds": 1.0} if operation == "retake" else {})}
    return _params(seconds=2, editing=editing, references=(source,), reference_bytes=1, total_input_chunks=1, **changes)


@pytest.mark.parametrize("backend,torch", [("comfy-cuda", "2.14.0+cu130"), ("comfy-rocm", "2.14.0+rocm7.4")])
def test_published_engine_enables_only_its_issued_workflows(backend: str, torch: str) -> None:
    """Real issued code supports GPU lanes without an empirical device allowlist."""
    from skulk.provisioning.comfy_extension import published_extensions
    from skulk.shared.models.video_support import (
        video_engine_feature_builds,
        video_feature_build_key,
        video_request_supported,
    )

    wheels = published_extensions()
    builds = tuple(wheel.build_identity(f"comfy@{wheel.comfy_revision}/torch@{torch}") for wheel in wheels)
    assert video_engine_feature_builds({"comfy"}) == {video_feature_build_key(build): ["hq_two_stage", "retake", "extend", *(["dfr"] if "dfr" in wheel.features else [])] for wheel, build in zip(wheels, builds, strict=True)}
    for wheel, build in zip(wheels, builds, strict=True):
        text_dfr = _params(pipeline="dfr")
        assert video_request_supported(_dfr_card(), text_dfr, backend, build) == ("dfr" in wheel.features)
        if "dfr" in wheel.features:
            graph = build_prompt(plan_comfy_render(text_dfr, _dfr_card(), engine_build=build), text_dfr, (), "issued-dfr")
            assert graph["ltx_refine_keyframes"]["class_type"] == "SkulkLTXAddGeneratedKeyframes"
        assert video_request_supported(_dev_card(), _params(pipeline="hq_two_stage"), backend, build)
        for operation in ("retake", "extend"):
            assert video_request_supported(_card(), _editing_params(operation), backend, build)
        assert not video_request_supported(_dev_card(), _params(pipeline="hq_two_stage"), backend, build.split("/extension@")[0])


@pytest.mark.parametrize("operation", ["retake", "extend"])
@pytest.mark.parametrize("backend", ["comfy-cuda", "comfy-rocm"])
def test_temporal_edits_require_the_issued_instance_payload(monkeypatch: pytest.MonkeyPatch, operation: str, backend: str) -> None:
    from skulk.shared.models.video_support import video_request_supported

    params, card = _editing_params(operation), _card()
    build = _issued_editing_build(monkeypatch)
    if backend == "comfy-rocm":
        build = build.replace("cu130", "rocm7.4")
    assert video_request_supported(card, params, backend, build)
    for unknown in (None, build.split("/extension@")[0], build.replace("d" * 64, "e" * 64)):
        assert not video_request_supported(card, params, backend, unknown)
        with pytest.raises(ValueError, match="published extension"):
            plan_comfy_render(params, card, engine_build=unknown)
    sampler_only = _issued_res2s_build(monkeypatch)
    assert not video_request_supported(card, params, backend, sampler_only)


@pytest.mark.parametrize("dev", [False, True])
def test_retake_uses_one_complete_source_pass_and_preserves_decoded_regions(monkeypatch: pytest.MonkeyPatch, dev: bool) -> None:
    params, card = _editing_params("retake"), _dev_card() if dev else _card()
    render = plan_comfy_render(params, card, engine_build=_issued_editing_build(monkeypatch))
    assert render.steps == (30 if dev else 8)
    assert render.sampler_steps == {"ltx_base_sampler": 30 if dev else 8}
    assert render.sampler == "euler"
    with pytest.raises(ValueError, match="verified source facts"):
        build_prompt(render, params, (), "edit")
    render = replace(render, source_frames=49)
    graph = build_prompt(render, params, (ReferenceBinding(spec=params.references[0], input_name="verified.mp4"),), "edit")
    assert graph["ltx_video_latent"]["inputs"]["width"] == 768
    assert graph["ltx_source_video"]["inputs"]["file"] == "verified.mp4"
    assert graph["ltx_edit_audio"]["inputs"]["source_frames"] == 49
    assert "ltx_refine_sampler" not in graph and "ltx_spatial_upscale" not in graph
    assert graph["create_video"]["inputs"]["images"] == ["ltx_preserve_video", 0]
    assert graph["thumbnail_frame"]["inputs"]["image"] == ["ltx_preserve_video", 0]
    assert graph["create_video"]["inputs"]["audio"] == ["ltx_preserve_audio", 0]
    assert graph["ltx_preserve_audio"]["inputs"]["source"] == ["ltx_edit_audio", 1]
    assert graph["ltx_base_guider"]["inputs"]["video_cfg"] == (3.0 if dev else 1.0)
    if dev:
        assert graph["ltx_base_sigmas"]["class_type"] == "LTXVScheduler"
        assert graph["ltx_edit_stg"]["inputs"]["scale"] == 1.0
        assert graph["ltx_base_guider"]["inputs"]["audio_cfg"] == 7.0
        assert render.engine_settings().guidance_video_rescale == 0.7
    else:
        assert graph["ltx_base_sigmas"]["inputs"]["sigmas"] == DISTILLED_SIGMAS


@pytest.mark.parametrize("dev", [False, True])
def test_extension_restores_full_resolution_source_context_before_refinement(monkeypatch: pytest.MonkeyPatch, dev: bool) -> None:
    params = _editing_params("extend")
    render = replace(plan_comfy_render(params, _dev_card() if dev else _card(), engine_build=_issued_editing_build(monkeypatch)), source_frames=9)
    graph = build_prompt(render, params, (ReferenceBinding(spec=params.references[0], input_name="source.mp4"),), "extend")
    assert graph["ltx_edit_base_video"]["inputs"]["operation"] == "extend"
    assert graph["ltx_preserve_video"]["inputs"]["source"] == ["ltx_source_components", 0]
    if dev:
        assert render.steps == 30 and "ltx_refine_sampler" not in graph
        assert graph["ltx_video_latent"]["inputs"]["width"] == 768
        assert render.engine_settings().adapter is None
    else:
        assert render.steps == 11 and render.sampler == "euler_ancestral"
        assert graph["ltx_video_latent"]["inputs"]["width"] == 384
        assert graph["ltx_edit_refine_video"]["inputs"]["source"] == ["ltx_source_full_encoded", 0]
        assert graph["ltx_edit_refine_video"]["inputs"]["target"] == ["ltx_spatial_upscale", 0]
        assert graph["ltx_refine_av"]["inputs"]["video_latent"] == ["ltx_edit_refine_video", 0]
        assert graph["ltx_refine_av"]["inputs"]["audio_latent"] == ["ltx_base_split", 1]
        assert graph["decode_audio"]["inputs"]["samples"] == ["ltx_refine_split", 1]


@pytest.mark.parametrize("silent", [False, True])
def test_retake_preserved_audio_bypasses_the_lossy_audio_vae(monkeypatch: pytest.MonkeyPatch, silent: bool) -> None:
    params = _editing_params("retake", audio=not silent)
    assert params.editing is not None
    params = params.model_copy(update={"editing": params.editing.model_copy(update={"regenerate_audio": False})})
    render = replace(plan_comfy_render(params, _card(), engine_build=_issued_editing_build(monkeypatch)), source_frames=49)
    graph = build_prompt(render, params, (ReferenceBinding(spec=params.references[0], input_name="source.mp4"),), "hold")
    assert "decode_audio" not in graph and "ltx_audio_latent" in graph
    assert graph["ltx_edit_audio"]["inputs"]["regenerate_audio"] is False
    if silent:
        assert "audio" not in graph["create_video"]["inputs"]
    else:
        assert graph["create_video"]["inputs"]["audio"] == ["ltx_edit_audio", 1]


def test_editing_metadata_exposes_own_defaults_and_scoped_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    _issued_editing_build(monkeypatch)
    distilled = VideoCapabilitySection.from_model_card(_card())
    dev = VideoCapabilitySection.from_model_card(_dev_card())
    assert distilled is not None and dev is not None
    assert distilled.editing == ["retake", "extend"]
    assert distilled.pipeline_defaults["retake"].sampling_stage_steps == [8]
    assert distilled.pipeline_defaults["retake"].sampler == "euler"
    assert "euler" in distilled.samplers
    assert distilled.pipeline_defaults["extend"].sampling_stage_steps == [8, 3]
    assert dev.pipeline_defaults["extend"].sampling_stage_steps == [30]
    assert "ref2va" in dev.decoder_modes["diffusion"]
    assert dev.pipeline_defaults["retake"].guidance is None


@pytest.mark.parametrize("operation,frames", [("retake", 41), ("extend", 49), ("extend", 57)])
def test_editing_refuses_incomplete_or_oversized_source_facts(monkeypatch: pytest.MonkeyPatch, operation: str, frames: int) -> None:
    params = _editing_params(operation)
    render = replace(plan_comfy_render(params, _card(), engine_build=_issued_editing_build(monkeypatch)), source_frames=frames)
    with pytest.raises(ValueError, match="complete output|exceed the complete"):
        build_prompt(render, params, (ReferenceBinding(spec=params.references[0], input_name="source.mp4"),), "invalid")
