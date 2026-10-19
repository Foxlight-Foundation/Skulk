"""LTX-2.5 two-stage graphs for the pinned core-node ComfyUI runtime.

The fixed tables and stage wiring follow workflow templates 0.11.57. Audio
latents participate in both stages even when the caller asks for a silent
container: muting the mux is different from changing the joint model. DFR
stage wiring follows Comfy-Org/workflow_templates at immutable revision
86ce86e92ec77e3c3c6d4a754e69cfce37886542, video_ltx2_5_dfr_i2v.json.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.video import VideoGenerationTaskParams

if TYPE_CHECKING:
    from skulk.worker.runner.comfy.graph import (
        ComfyPrompt,
        ComfyRenderPlan,
        ReferenceBinding,
    )

DISTILLED_SIGMAS: Final = (
    "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
)
DFR_REFINEMENT_SIGMAS: Final = "0.909375, 0.725, 0.421875, 0.0"
DFR_TEMPORAL_SIGMAS: Final = "0.975, 0.909375, 0.725, 0.421875, 0.0"
REFINEMENT_SIGMAS: Final = "0.85, 0.7250, 0.4219, 0.0"
"""The pinned Comfy template's refinement schedule, not a guessed step count."""


def validate_ltx_request(params: VideoGenerationTaskParams, card: ModelCard) -> None:
    """Refuse controls this distilled graph cannot honor before sampling.

    Args:
        params: Validated caller controls and conditioning references.
        card: Exact staged LTX artifact and its video contract.

    Raises:
        ValueError: The artifact or requested controls cannot run this graph.
    """
    video = card.video
    if video is None:
        raise ValueError("LTX-2.5 requires a video contract")
    params.validate_family(card.family, video)
    if not video.audio_output:
        raise ValueError("LTX-2.5 requires its audio VAE even for silent output")


def _node(class_type: str, title: str, **inputs: object) -> dict[str, object]:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _link(node: str, output: int = 0) -> list[str | int]:
    return [node, output]


def _condition_images(
    prompt: ComfyPrompt,
    references: tuple[ReferenceBinding, ...],
    *,
    stage: str,
    positive: list[str | int],
    negative: list[str | int],
    latent: list[str | int],
    fps: int,
    refinement: bool,
) -> tuple[list[str | int], list[str | int], list[str | int], bool]:
    guided = False
    for binding in references:
        reference = binding.spec
        if reference.kind != "image":
            continue
        load = f"load_ltx_{reference.slot}"
        preprocess = f"ltx_image_{reference.slot}"
        prompt[load] = _node("LoadImage", "Load keyframe", image=binding.input_name)
        prompt[preprocess] = _node(
            "LTXVPreprocess",
            "LTX image compression",
            image=_link(load),
            img_compression=18,
        )
        anchor = f"guide_ltx_{stage}_{reference.slot}"
        if reference.role == "first_frame":
            # The official I2V template pins its first image in place, so no
            # extra guide token survives into spatial upsampling.
            prompt[anchor] = _node(
                "LTXVImgToVideoInplace",
                "Pin first frame",
                vae=_link("video_vae"),
                image=_link(preprocess),
                latent=latent,
                strength=1.0 if refinement else 0.7,
                bypass=False,
            )
            latent = _link(anchor)
            continue
        frame = (
            -1
            if reference.role == "last_frame"
            else round((reference.at_seconds or 0.0) * fps)
        )
        prompt[anchor] = _node(
            "LTXVAddGuide",
            "Anchor keyframe",
            positive=positive,
            negative=negative,
            vae=_link("video_vae"),
            latent=latent,
            image=_link(preprocess),
            frame_idx=frame,
            strength=0.7,
        )
        positive, negative, latent = (
            _link(anchor, 0),
            _link(anchor, 1),
            _link(anchor, 2),
        )
        guided = True
    return positive, negative, latent, guided


def _sampling_stage(
    prompt: ComfyPrompt,
    name: str,
    *,
    seed: int,
    sigmas: str,
    positive: list[str | int],
    negative: list[str | int],
    video: list[str | int],
    audio: list[str | int],
    render: ComfyRenderPlan,
    model: list[str | int],
) -> list[str | int]:
    prompt[f"ltx_{name}_av"] = _node(
        "LTXVConcatAVLatent",
        "Join audio/video latents",
        video_latent=video,
        audio_latent=audio,
    )
    prompt[f"ltx_{name}_noise"] = _node("RandomNoise", "Noise", noise_seed=seed)
    if name == "base" and render.pipeline == "dev_two_stage":
        prompt[f"ltx_{name}_sigmas"] = _node(
            "LTXVScheduler", "Guided base schedule", steps=render.plan.steps,
            max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1,
            latent=video,
        )
    else:
        prompt[f"ltx_{name}_sigmas"] = _node(
            "ManualSigmas", "Distilled schedule", sigmas=sigmas
        )
    guidance = render.guidance if name == "base" else None
    prompt[f"ltx_{name}_guider"] = _node(
        "LTXVDualCFGGuider",
        "Distilled guidance",
        model=model,
        positive=positive,
        negative=negative,
        video_cfg=1.0 if guidance is None else guidance.video_cfg,
        audio_cfg=1.0 if guidance is None else guidance.audio_cfg,
    )
    sampler = f"ltx_{name}_sampler"
    prompt[sampler] = _node(
        "SamplerCustomAdvanced",
        "Base sampling" if name == "base" else "Refinement sampling",
        noise=_link(f"ltx_{name}_noise"),
        guider=_link(f"ltx_{name}_guider"),
        sampler=_link("sampler_select"),
        sigmas=_link(f"ltx_{name}_sigmas"),
        latent_image=_link(f"ltx_{name}_av"),
    )
    return _link(sampler)


def _crop_guides(
    prompt: ComfyPrompt,
    name: str,
    positive: list[str | int],
    negative: list[str | int],
    latent: list[str | int],
) -> list[str | int]:
    node = f"ltx_{name}_crop"
    prompt[node] = _node(
        "LTXVCropGuides",
        "Remove guide tokens",
        positive=positive,
        negative=negative,
        latent=latent,
    )
    return _link(node, 2)



def _temporal_refinement(
    prompt: ComfyPrompt,
    render: ComfyRenderPlan,
    references: tuple[ReferenceBinding, ...],
    video: list[str | int],
    model: list[str | int],
) -> list[str | int]:
    """Densify the refined timeline while preserving its audio and keyframes."""
    if render.files.temporal_upscaler is None:
        raise ValueError("DFR temporal refinement requires the selected temporal_x2 upscaler")
    prompt["ltx_temporal_model"] = _node("LatentUpscaleModelLoader", "Load temporal x2 upscaler", model_name=render.files.temporal_upscaler)
    prompt["ltx_temporal_upscale"] = _node("LTXVLatentUpsampler", "Double temporal density", samples=video, upscale_model=_link("ltx_temporal_model"), vae=_link("video_vae"))
    positive, negative, video, _ = _condition_images(prompt, references, stage="temporal", positive=_link("condition", 0), negative=_link("condition", 1), latent=_link("ltx_temporal_upscale"), fps=render.plan.fps, refinement=True)
    prompt["ltx_temporal_guides"] = _node("LTXVGeneratedKeyframesToGuides", "Pin previous detailing frames", positive=positive, negative=negative, vae=_link("video_vae"), latent=video, keyframes=_link("ltx_refine_strip_generated", 3), strength=0.95)
    prompt["ltx_temporal_keyframes"] = _node("LTXVAddGeneratedKeyframes", "Generate intermediate detailing frames", positive=_link("ltx_temporal_guides", 0), negative=_link("ltx_temporal_guides", 1), vae=_link("video_vae"), latent=_link("ltx_temporal_guides", 2), interval_frames=render.generated_keyframe_interval, keyframes=_link("ltx_temporal_upscale"))
    prompt["ltx_temporal_condition"] = _node("LTXVConditioning", "Temporal refinement frame rate", positive=_link("ltx_temporal_keyframes", 0), negative=_link("ltx_temporal_keyframes", 1), frame_rate=render.plan.fps)
    prompt["ltx_temporal_frozen_audio"] = _node("LTXVFreezeLatent", "Preserve refined audio", latent=_link("ltx_refine_split", 1))
    sampled = _sampling_stage(prompt, "temporal", seed=(render.plan.seed + 2) % (2**64), sigmas=DFR_TEMPORAL_SIGMAS, positive=_link("ltx_temporal_condition", 0), negative=_link("ltx_temporal_condition", 1), video=_link("ltx_temporal_keyframes", 2), audio=_link("ltx_temporal_frozen_audio"), render=render, model=model)
    prompt["ltx_temporal_split"] = _node("LTXVSeparateAVLatent", "Separate temporal latents", av_latent=sampled)
    prompt["ltx_temporal_strip_generated"] = _node("LTXVSeparateGeneratedKeyframes", "Remove intermediate detailing tokens", positive=_link("ltx_temporal_keyframes", 0), negative=_link("ltx_temporal_keyframes", 1), latent=_link("ltx_temporal_split", 0), keyframes_to_batch=False)
    return _crop_guides(prompt, "temporal", _link("ltx_temporal_strip_generated", 0), _link("ltx_temporal_strip_generated", 1), _link("ltx_temporal_strip_generated", 2))


def build_ltx_prompt(
    render: ComfyRenderPlan,
    params: VideoGenerationTaskParams,
    references: tuple[ReferenceBinding, ...],
    output_subdir: str,
) -> ComfyPrompt:
    """Build a selected two-stage audio-video graph using pinned core nodes.

    Args:
        render: Resolved output geometry, seed and exact model filenames.
        params: Prompt and already validated caller controls.
        references: Verified image files relative to the Comfy input directory.
        output_subdir: Command-owned directory beneath the output root.

    Returns:
        ComfyUI API prompt with container and thumbnail save nodes.

    Raises:
        ValueError: A required audio decoder or spatial upscaler is absent.
    """
    plan, files = render.plan, render.files
    # Temporal refinement doubles frame density and RoPE FPS together. The base
    # model still samples the original timeline at the card's trained rate.
    factor = 2 ** render.temporal_upscalings
    base_fps = plan.fps // factor
    base_frames = (plan.frame_count - 1) // factor + 1
    if files.audio_vae is None or files.spatial_upscaler is None:
        raise ValueError(
            "LTX-2.5 two-stage rendering requires audio VAE and spatial_x2 upscaler"
        )
    prompt: ComfyPrompt = {
        "unet": _node(
            "UNETLoader",
            "Load LTX-2.5",
            unet_name=files.diffusion_model,
            weight_dtype="default",
        ),
        "clip": _node(
            "CLIPLoader", "Load LTX encoder", clip_name=files.text_encoder, type="ltxv"
        ),
        "video_vae": _node("VAELoader", "Load video VAE", vae_name=files.video_vae),
        "audio_vae": _node("VAELoader", "Load audio VAE", vae_name=files.audio_vae),
        "ltx_positive": _node(
            "CLIPTextEncode", "Encode prompt", text=params.prompt, clip=_link("clip")
        ),
        "ltx_negative": _node(
            "CLIPTextEncode", "Negative conditioning", text=params.negative_prompt or "", clip=_link("clip")
        ),
        "condition": _node(
            "LTXVConditioning",
            "Frame-rate conditioning",
            positive=_link("ltx_positive"),
            negative=_link("ltx_negative"),
            frame_rate=base_fps,
        ),
        "ltx_video_latent": _node(
            "EmptyLTXVLatentVideo",
            "Half-resolution video latent",
            width=plan.width // 2,
            height=plan.height // 2,
            length=base_frames,
            batch_size=1,
        ),
        "ltx_audio_latent": _node(
            "LTXVEmptyLatentAudio",
            "Audio latent",
            frames_number=base_frames,
            frame_rate=base_fps,
            batch_size=1,
            audio_vae=_link("audio_vae"),
        ),
        "sampler_select": _node(
            "KSamplerSelect", "LTX sampler", sampler_name=render.sampler
        ),
        "ltx_spatial_model": _node(
            "LatentUpscaleModelLoader",
            "Load spatial upscaler",
            model_name=files.spatial_upscaler,
        ),
    }
    frame_count: int | list[str | int] = base_frames
    if render.duration == "auto":
        if files.duration_head is None:
            raise ValueError("LTX-2.5 automatic duration requires a selected duration head")
        prompt["ltx_duration_head"] = _node("ModelPatchLoader", "Duration head", name=files.duration_head)
        prompt["ltx_duration"] = _node(
            "LTXVDurationPredictor", "Bounded shot duration", model=_link("unet"),
            positive=_link("ltx_positive"), duration_head=_link("ltx_duration_head"),
            frame_rate=float(plan.fps), min_seconds=float(render.duration_min_seconds or 1),
            max_seconds=float(render.duration_max_seconds or params.seconds),
        )
        frame_count = _link("ltx_duration")
    prompt["ltx_video_latent"]["inputs"]["length"] = frame_count
    prompt["ltx_audio_latent"]["inputs"]["frames_number"] = frame_count
    audio_latent = _link("ltx_audio_latent")
    if params.audio_conditioning is not None:
        source = next(binding for binding in references if binding.spec.kind == "audio")
        prompt["ltx_source_audio"] = _node("LoadAudio", "Load prepared source audio", audio=source.input_name)
        prompt["ltx_encoded_source_audio"] = _node("LTXVAudioVAEEncode", "Encode source audio", audio=_link("ltx_source_audio"), audio_vae=_link("audio_vae"))
        prompt["ltx_frozen_source_audio"] = _node("LTXVFreezeLatent", "Freeze source audio", latent=_link("ltx_encoded_source_audio"))
        audio_latent = _link("ltx_frozen_source_audio")
    base_model = _link("unet")
    if render.guidance is not None:
        if render.guidance.stg_scale:
            prompt["ltx_stg"] = _node("LTXVSpatioTemporalGuidance", "Spatiotemporal guidance", model=base_model, scale=render.guidance.stg_scale, blocks="28", start_percent=0.0, end_percent=1.0)
            base_model = _link("ltx_stg")
        if render.guidance.modality_scale != 1.0:
            prompt["ltx_modality"] = _node("LTXVModalityGuidance", "Audio/video guidance", model=base_model, modality_scale=render.guidance.modality_scale, start_percent=0.0, end_percent=1.0)
            base_model = _link("ltx_modality")
    positive, negative, video, guided = _condition_images(
        prompt,
        references,
        stage="base",
        positive=_link("condition", 0),
        negative=_link("condition", 1),
        latent=_link("ltx_video_latent"),
        fps=base_fps,
        refinement=False,
    )
    if render.generated_keyframe_interval is not None:
        prompt["ltx_generated_keyframes"] = _node(
            "LTXVAddGeneratedKeyframes", "Internal detailing keyframes",
            positive=positive, negative=negative, vae=_link("video_vae"),
            latent=video, interval_frames=render.generated_keyframe_interval,
        )
        positive, negative, video = (_link("ltx_generated_keyframes", 0), _link("ltx_generated_keyframes", 1), _link("ltx_generated_keyframes", 2))
    first = _sampling_stage(
        prompt,
        "base",
        seed=plan.seed,
        sigmas=DISTILLED_SIGMAS,
        positive=positive,
        negative=negative,
        video=video,
        audio=audio_latent,
        render=render,
        model=base_model,
    )
    prompt["ltx_base_split"] = _node(
        "LTXVSeparateAVLatent", "Separate base latents", av_latent=first
    )
    base_video = _link("ltx_base_split", 0)
    if render.generated_keyframe_interval is not None:
        # Internal detailing tokens are not output frames. Strip them before
        # image-guide cropping and upscaling, retaining the ordinary guides.
        prompt["ltx_strip_generated"] = _node("LTXVSeparateGeneratedKeyframes", "Strip internal keyframes", positive=positive, negative=negative, latent=base_video, keyframes_to_batch=False)
        positive, negative, base_video = (_link("ltx_strip_generated", 0), _link("ltx_strip_generated", 1), _link("ltx_strip_generated", 2))
    if guided:
        # Appended image guides must be removed before upscaling, then added
        # again at the new resolution; otherwise they become output frames.
        base_video = _crop_guides(prompt, "base", positive, negative, base_video)
    prompt["ltx_spatial_upscale"] = _node(
        "LTXVLatentUpsampler",
        "Spatial refinement",
        samples=base_video,
        upscale_model=_link("ltx_spatial_model"),
        vae=_link("video_vae"),
    )
    positive, negative, video, guided = _condition_images(
        prompt,
        references,
        stage="refine",
        positive=_link("condition", 0),
        negative=_link("condition", 1),
        latent=_link("ltx_spatial_upscale"),
        fps=base_fps,
        refinement=True,
    )
    refinement_model = _link("unet")
    refinement_audio = audio_latent if params.audio_conditioning is not None else _link("ltx_base_split", 1)
    if render.pipeline == "dfr":
        if files.dfr_spatial_lora is None:
            raise ValueError("LTX-2.5 DFR requires its selected pixel-spatial IC-LoRA")
        prompt["ltx_dfr_lora"] = _node("LoraLoaderModelOnly", "DFR spatial detail model", model=refinement_model, lora_name=files.dfr_spatial_lora, strength_model=1.0)
        refinement_model = _link("ltx_dfr_lora")
        prompt["ltx_upscale_keyframes"] = _node("LTXVLatentUpsampler", "Upscale detailing keyframes", samples=_link("ltx_strip_generated", 3), upscale_model=_link("ltx_spatial_model"), vae=_link("video_vae"))
        prompt["ltx_refine_keyframes"] = _node("LTXVAddGeneratedKeyframes", "Carry detailing keyframes", positive=positive, negative=negative, vae=_link("video_vae"), latent=video, interval_frames=render.generated_keyframe_interval, keyframes=_link("ltx_upscale_keyframes"))
        prompt["ltx_dfr_detail_guide"] = _node("LTXVAddLatentGuide", "Half-resolution detail reference", positive=_link("ltx_refine_keyframes", 0), negative=_link("ltx_refine_keyframes", 1), vae=_link("video_vae"), latent=_link("ltx_refine_keyframes", 2), guiding_latent=base_video, latent_idx=0, strength=1.0)
        positive, negative, video = (_link("ltx_dfr_detail_guide", 0), _link("ltx_dfr_detail_guide", 1), _link("ltx_dfr_detail_guide", 2))
        guided = True
    if render.pipeline == "dev_two_stage":
        if files.refinement_lora is None:
            raise ValueError("LTX-2.5 guided refinement requires its exact distilled LoRA")
        prompt["ltx_refinement_lora"] = _node("LoraLoaderModelOnly", "Distilled refinement model", model=refinement_model, lora_name=files.refinement_lora, strength_model=files.refinement_strength)
        refinement_model = _link("ltx_refinement_lora")
        # Guided upstream refinement keeps stage-one audio clean; it is
        # cross-attention context rather than a second audio generation.
        prompt["ltx_frozen_audio"] = _node("LTXVFreezeLatent", "Keep base audio", latent=refinement_audio)
        refinement_audio = _link("ltx_frozen_audio")
    second = _sampling_stage(
        prompt,
        "refine",
        seed=(plan.seed + 1) % (2**64),
        sigmas=REFINEMENT_SIGMAS if render.pipeline == "distilled_two_stage" else DFR_REFINEMENT_SIGMAS,
        positive=positive,
        negative=negative,
        video=video,
        audio=refinement_audio,
        render=render,
        model=refinement_model,
    )
    prompt["ltx_refine_split"] = _node(
        "LTXVSeparateAVLatent", "Separate refined latents", av_latent=second
    )
    final_video = _link("ltx_refine_split", 0)
    if render.pipeline == "dfr":
        prompt["ltx_refine_strip_generated"] = _node("LTXVSeparateGeneratedKeyframes", "Separate refined detailing keyframes", positive=positive, negative=negative, latent=final_video, keyframes_to_batch=False)
        positive, negative, final_video = (_link("ltx_refine_strip_generated", 0), _link("ltx_refine_strip_generated", 1), _link("ltx_refine_strip_generated", 2))
    if guided:
        final_video = _crop_guides(prompt, "refine", positive, negative, final_video)
    if render.temporal_upscalings:
        final_video = _temporal_refinement(prompt, render, references, final_video, refinement_model)
    decoder = _link("video_vae")
    if render.decoder == "diffusion":
        if files.diffusion_decoder is None:
            raise ValueError("LTX-2.5 diffusion decoding requires an exact selected decoder")
        prompt["ltx_diffusion_decoder"] = _node("VAELoader", "Diffusion decoder", vae_name=files.diffusion_decoder)
        decoder = _link("ltx_diffusion_decoder")
    prompt["decode_video"] = _node(
        "VAEDecodeTiled",
        "Decode video",
        samples=final_video,
        vae=decoder,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=16,
    )
    create: dict[str, object] = {
        "images": _link("decode_video"),
        "fps": float(plan.fps),
    }
    if plan.audio and params.audio_conditioning is not None:
        # Preserve the normalized source waveform, rather than round-tripping
        # it through the VAE decoder after using it as frozen context.
        create["audio"] = _link("ltx_source_audio")
    elif plan.audio:
        prompt["decode_audio"] = _node(
            "LTXVAudioVAEDecode",
            "Decode audio",
            samples=_link("ltx_refine_split", 1) if render.pipeline in ("distilled_two_stage", "dfr") else _link("ltx_base_split", 1),
            audio_vae=_link("audio_vae"),
        )
        create["audio"] = _link("decode_audio")
    prompt["create_video"] = _node("CreateVideo", "Mux", **create)
    prompt["save_video"] = _node(
        "SaveVideo",
        "Save clip",
        video=_link("create_video"),
        filename_prefix=f"{output_subdir}/output",
        format="mp4",
        **{"format.codec": render.codec},
    )
    prompt["thumbnail_frame"] = _node(
        "ImageFromBatch",
        "First frame",
        image=_link("decode_video"),
        batch_index=0,
        length=1,
    )
    prompt["save_thumbnail"] = _node(
        "SaveImage",
        "Save thumbnail",
        images=_link("thumbnail_frame"),
        filename_prefix=f"{output_subdir}/thumbnail",
    )
    return prompt
