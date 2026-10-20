"""Source-bound temporal LTX graphs, separate from generated-keyframe workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skulk.shared.types.video import VideoGenerationTaskParams
from skulk.worker.runner.comfy.ltx import (
    DISTILLED_SIGMAS,
    REFINEMENT_SIGMAS,
    sampling_stage,
)

if TYPE_CHECKING:
    from skulk.worker.runner.comfy.graph import (
        ComfyPrompt,
        ComfyRenderPlan,
        ReferenceBinding,
    )


def _node(kind: str, title: str, **inputs: object) -> dict[str, object]:
    return {"class_type": kind, "inputs": inputs, "_meta": {"title": title}}


def _link(identifier: str, output: int = 0) -> list[str | int]:
    return [identifier, output]


def _encode_video(prompt: ComfyPrompt, identifier: str, images: list[str | int]) -> list[str | int]:
    prompt[identifier] = _node("VAEEncodeTiled", "Encode source context", pixels=images, vae=_link("video_vae"), tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    return _link(identifier)


def build_editing_prompt(render: ComfyRenderPlan, params: VideoGenerationTaskParams, references: tuple[ReferenceBinding, ...], output_subdir: str) -> ComfyPrompt:
    """Build a verified source retake or extension with decoded source preservation.

    Worker preflight supplies the complete source frame count. Single-pass retake
    and Dev extension operate at final resolution. Distilled extension uses its
    fixed two-stage schedule and restores full-resolution latent context before
    refinement. Original frames and normalized PCM are restored before saving.
    """
    plan, files, editing = render.plan, render.files, render.editing
    if editing is None or render.source_frames is None or files.audio_vae is None:
        raise ValueError("temporal editing requires verified source facts and the selected audio VAE")
    source = next((binding for binding in references if binding.spec.kind == "video" and binding.spec.role == "source"), None)
    if source is None:
        raise ValueError("temporal editing requires its verified source video")
    if editing.operation == "retake" and render.source_frames != plan.frame_count:
        raise ValueError("retake source frame count must equal its complete output")
    if editing.operation == "extend" and render.source_frames >= plan.frame_count:
        raise ValueError("extension output must exceed the complete source video")
    two_stage = editing.operation == "extend" and render.transformer_variant == "distilled"
    operation: dict[str, object] = {"operation": editing.operation, "fps": plan.fps, "start_seconds": editing.start_seconds or 0.0, "end_seconds": editing.end_seconds or 0.0}
    factor = 2 if two_stage else 1
    prompt: ComfyPrompt = {
        "unet": _node("UNETLoader", "Load selected LTX transformer", unet_name=files.diffusion_model, weight_dtype="default"),
        "clip": _node("CLIPLoader", "Load selected LTX encoder", clip_name=files.text_encoder, type="ltxv"),
        "video_vae": _node("VAELoader", "Load video VAE", vae_name=files.video_vae),
        "audio_vae": _node("VAELoader", "Load audio VAE", vae_name=files.audio_vae),
        "ltx_positive": _node("CLIPTextEncode", "Encode edit prompt", clip=_link("clip"), text=params.prompt),
        "ltx_negative": _node("CLIPTextEncode", "Encode negative prompt", clip=_link("clip"), text=params.negative_prompt or ""),
        "condition": _node("LTXVConditioning", "Frame-rate conditioning", positive=_link("ltx_positive"), negative=_link("ltx_negative"), frame_rate=plan.fps),
        "ltx_source_video": _node("LoadVideo", "Load verified source video", file=source.input_name),
        "ltx_source_components": _node("GetVideoComponents", "Decode source frames and PCM", video=_link("ltx_source_video")),
        "ltx_video_latent": _node("EmptyLTXVLatentVideo", "Target video canvas", width=plan.width // factor, height=plan.height // factor, length=plan.frame_count, batch_size=1),
        "ltx_audio_latent": _node("LTXVEmptyLatentAudio", "Target audio canvas", frames_number=plan.frame_count, frame_rate=plan.fps, batch_size=1, audio_vae=_link("audio_vae")),
        "sampler_select": _node("KSamplerSelect", "Fixed edit sampler", sampler_name=render.sampler),
    }
    source_images = _link("ltx_source_components")
    base_images = source_images
    if two_stage:
        prompt["ltx_half_source"] = _node("ImageScale", "Half-resolution source context", image=source_images, upscale_method="bicubic", width=plan.width // 2, height=plan.height // 2, crop="disabled")
        base_images = _link("ltx_half_source")
    encoded = _encode_video(prompt, "ltx_source_base_encoded", base_images)
    prompt["ltx_edit_base_video"] = _node("SkulkLTXEditVideoLatent", "Temporal video context", source=encoded, target=_link("ltx_video_latent"), **operation)
    prompt["ltx_edit_audio"] = _node("SkulkLTXEditAudioLatent", "Temporal audio context", audio=_link("ltx_source_components", 1), target=_link("ltx_audio_latent"), audio_vae=_link("audio_vae"), source_frames=render.source_frames, sample_rate=plan.sample_rate, regenerate_audio=editing.regenerate_audio, **operation)
    model = _link("unet")
    if render.transformer_variant == "dev":
        # The fixed editing preset follows the detected non-HQ reference profile.
        # It is separate from the caller-adjustable two-stage generation controls.
        prompt["ltx_edit_stg"] = _node("LTXVSpatioTemporalGuidance", "Fixed edit STG", model=model, scale=1.0, blocks="28", start_percent=0.0, end_percent=1.0)
        prompt["ltx_edit_modality"] = _node("LTXVModalityGuidance", "Fixed edit AV guidance", model=_link("ltx_edit_stg"), modality_scale=3.0, start_percent=0.0, end_percent=1.0)
        prompt["ltx_edit_rescale"] = _node("SkulkLTXGuidanceRescale", "Fixed edit guidance rescaling", model=_link("ltx_edit_modality"), video_scale=0.7, audio_scale=0.7)
        model = _link("ltx_edit_rescale")
    sampled = sampling_stage(prompt, "base", seed=plan.seed, sigmas=DISTILLED_SIGMAS, positive=_link("condition", 0), negative=_link("condition", 1), video=_link("ltx_edit_base_video"), audio=_link("ltx_edit_audio"), render=render, model=model)
    prompt["ltx_base_split"] = _node("LTXVSeparateAVLatent", "Separate edited latents", av_latent=sampled)
    final_video, final_audio = _link("ltx_base_split", 0), _link("ltx_base_split", 1)
    if two_stage:
        if files.spatial_upscaler is None:
            raise ValueError("distilled extension requires its selected spatial upscaler")
        prompt["ltx_spatial_model"] = _node("LatentUpscaleModelLoader", "Load selected spatial upscaler", model_name=files.spatial_upscaler)
        prompt["ltx_spatial_upscale"] = _node("LTXVLatentUpsampler", "Spatial extension refinement", samples=final_video, upscale_model=_link("ltx_spatial_model"), vae=_link("video_vae"))
        full_source = _encode_video(prompt, "ltx_source_full_encoded", source_images)
        prompt["ltx_edit_refine_video"] = _node("SkulkLTXEditVideoLatent", "Restore full-resolution source context", source=full_source, target=_link("ltx_spatial_upscale"), **operation)
        # SeparateAVLatent retains the exact prefix mask on the base audio.
        # Reuse it at refinement; do not re-noise the preserved source prefix.
        refined = sampling_stage(prompt, "refine", seed=(plan.seed + 1) % (2**64), sigmas=REFINEMENT_SIGMAS, positive=_link("condition", 0), negative=_link("condition", 1), video=_link("ltx_edit_refine_video"), audio=final_audio, render=render, model=model)
        prompt["ltx_refine_split"] = _node("LTXVSeparateAVLatent", "Separate refined extension", av_latent=refined)
        final_video, final_audio = _link("ltx_refine_split", 0), _link("ltx_refine_split", 1)
    decoder = _link("video_vae")
    if render.decoder == "diffusion":
        if files.diffusion_decoder is None:
            raise ValueError("editing diffusion decode requires its selected decoder")
        prompt["ltx_diffusion_decoder"] = _node("VAELoader", "Load selected diffusion decoder", vae_name=files.diffusion_decoder)
        decoder = _link("ltx_diffusion_decoder")
    prompt["decode_video"] = _node("VAEDecodeTiled", "Decode edited video", samples=final_video, vae=decoder, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16)
    prompt["ltx_preserve_video"] = _node("SkulkLTXPreserveVideo", "Preserve unchanged source frames", source=source_images, generated=_link("decode_video"), **operation)
    create: dict[str, object] = {"images": _link("ltx_preserve_video"), "fps": float(plan.fps)}
    if plan.audio:
        if editing.regenerate_audio:
            prompt["decode_audio"] = _node("LTXVAudioVAEDecode", "Decode edited audio", samples=final_audio, audio_vae=_link("audio_vae"))
            prompt["ltx_preserve_audio"] = _node("SkulkLTXPreserveAudio", "Preserve unchanged source PCM", source=_link("ltx_edit_audio", 1), generated=_link("decode_audio"), output_frames=plan.frame_count, regenerate_audio=True, **operation)
            create["audio"] = _link("ltx_preserve_audio")
        else:
            create["audio"] = _link("ltx_edit_audio", 1)
    prompt["create_video"] = _node("CreateVideo", "Mux edited clip", **create)
    prompt["save_video"] = _node("SaveVideo", "Save edited clip", video=_link("create_video"), filename_prefix=f"{output_subdir}/output", format="mp4", **{"format.codec": render.codec})
    prompt["thumbnail_frame"] = _node("ImageFromBatch", "First preserved frame", image=_link("ltx_preserve_video"), batch_index=0, length=1)
    prompt["save_thumbnail"] = _node("SaveImage", "Save thumbnail", images=_link("thumbnail_frame"), filename_prefix=f"{output_subdir}/thumbnail")
    return prompt
