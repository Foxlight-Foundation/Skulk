# pyright: reportPrivateUsage=false
"""LTX graph wiring, strict controls, and broad unqualified-hardware admission."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from skulk.api.types.api import VideoCapabilitySection
from skulk.master.placement import _card_platform_backends
from skulk.shared.backends import resolve_node_backend
from skulk.shared.models import model_cards
from skulk.shared.models.model_cards import ModelCard, ModelId
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
    assert metadata.samplers == ["euler_ancestral"]
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


@pytest.mark.parametrize("pipeline", ["hq_two_stage", "dfr"])
def test_comfy_does_not_approximate_an_unimplemented_pipeline(pipeline: str) -> None:
    with pytest.raises(ValueError, match="Res2s|DFR temporal"):
        plan_comfy_render(_params(pipeline=pipeline), _card() if pipeline == "dfr" else _dev_card())


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
    assert metadata.pipelines == ["dev_two_stage"]
    assert metadata.decoders == ["convolutional", "diffusion"]
    assert metadata.automatic_duration and metadata.guidance is not None


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
