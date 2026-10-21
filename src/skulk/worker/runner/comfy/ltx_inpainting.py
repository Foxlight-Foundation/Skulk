"""The pinned LTX inpainting pixel-refinement recipe, with complete timelines.

Topology follows Lightricks/ComfyUI-LTXVideo at immutable revision
44c8a36b7c31ee00b4271daed11dc2a4aa481140, the 2.5 distilled inpaint
workflow. The source guide is appended only at base resolution, then removed
before decoding. Refinement re-encodes blended pixels and uses the same LoRA;
it does not append another full-resolution guide or use a latent upscaler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from skulk.shared.types.video import VideoGenerationTaskParams
from skulk.worker.runner.comfy.ltx import DISTILLED_SIGMAS, sampling_stage

if TYPE_CHECKING:
    from skulk.worker.runner.comfy.graph import (
        ComfyPrompt,
        ComfyRenderPlan,
        ReferenceBinding,
    )

INPAINT_REFINEMENT_SIGMAS: Final = "0.7250, 0.4219, 0.0"


def _node(kind: str, title: str, **inputs: object) -> dict[str, object]:
    return {"class_type": kind, "inputs": inputs, "_meta": {"title": title}}


def _link(identifier: str, output: int = 0) -> list[str | int]:
    return [identifier, output]


def _opening(prompt: ComfyPrompt, binding: ReferenceBinding | None, *, latent: list[str | int], width: int, height: int, stage: str) -> list[str | int]:
    if binding is None:
        return latent
    prompt["load_ltx_inpaint_opening"] = _node("LoadImage", "Load full-canvas opening image", image=binding.input_name)
    identifier = f"ltx_{stage}_opening"
    prompt[f"{identifier}_size"] = _node("ImageScale", "Opening at actual stage resolution", image=_link("load_ltx_inpaint_opening"), upscale_method="lanczos", width=width, height=height, crop="disabled")
    image = _link(f"{identifier}_size")
    if stage == "base":
        prompt[f"{identifier}_preprocess"] = _node("LTXVPreprocess", "Native opening compression", image=image, img_compression=18)
        image = _link(f"{identifier}_preprocess")
    prompt[identifier] = _node("LTXVImgToVideoInplace", "Pin opening image", vae=_link("video_vae"), image=image, latent=latent, strength=0.7 if stage == "base" else 1.0, bypass=False)
    return _link(identifier)


def build_inpainting_prompt(render: ComfyRenderPlan, params: VideoGenerationTaskParams, references: tuple[ReferenceBinding, ...], output_subdir: str) -> ComfyPrompt:
    """Build two bounded pixel-blended stages after complete worker preflight.

    Args:
        render: Exact issued engine plan with measured source/mask facts.
        params: Validated source-position, output and prompt controls.
        references: Worker-verified source, red-channel mask and optional opening.
        output_subdir: Command-owned output and thumbnail destination.

    Returns:
        Normal Comfy prompt using managed loaders, cancellation and output nodes.

    Raises:
        ValueError: Required measured inputs or selected components are absent.
    """
    from skulk.worker.runner.comfy.graph import load_reference_frames

    plan, files, control, facts = render.plan, render.files, render.inpainting, render.inpaint_inputs
    if control is None or facts is None or files.inpaint_lora is None or files.audio_vae is None:
        raise ValueError("Inpainting requires verified source facts and its selected adapter/audio VAE")
    control.validate_source_canvas(plan.width, plan.height, source_width=facts.source_width, source_height=facts.source_height)
    if facts.frames != plan.frame_count or facts.mask_frames not in (1, facts.frames) or facts.allocation_bytes < 1:
        raise ValueError("Inpainting graph requires its complete preflight timeline and allocation reservation")
    source = next(binding for binding in references if binding.spec.role == "source")
    mask = next(binding for binding in references if binding.spec.role == "mask")
    opening = next((binding for binding in references if binding.spec.role == "first_frame"), None)
    budget = facts.allocation_bytes
    prompt: ComfyPrompt = {
        "unet": _node("UNETLoader", "Load selected LTX transformer", unet_name=files.diffusion_model, weight_dtype="default"),
        "clip": _node("CLIPLoader", "Load selected LTX encoder", clip_name=files.text_encoder, type="ltxv"),
        "video_vae": _node("VAELoader", "Load video VAE", vae_name=files.video_vae),
        "audio_vae": _node("VAELoader", "Load audio VAE", vae_name=files.audio_vae),
        "ltx_inpaint_lora": _node("LoraLoaderModelOnly", "Inpainting adapter for both stages", model=_link("unet"), lora_name=files.inpaint_lora, strength_model=1.0),
        "ltx_positive": _node("CLIPTextEncode", "Encode inpainting scene prompt", clip=_link("clip"), text=params.prompt),
        "ltx_negative": _node("CLIPTextEncode", "Encode negative prompt", clip=_link("clip"), text=params.negative_prompt or ""),
        "condition": _node("LTXVConditioning", "Source frame-rate conditioning", positive=_link("ltx_positive"), negative=_link("ltx_negative"), frame_rate=plan.fps),
        "ltx_source_video": _node("LoadVideo", "Load complete verified source", file=source.input_name),
        "ltx_source_components": _node("GetVideoComponents", "Decode complete source frames and audio", video=_link("ltx_source_video")),
        "ltx_video_latent": _node("EmptyLTXVLatentVideo", "Half-resolution base canvas", width=plan.width // 2, height=plan.height // 2, length=plan.frame_count, batch_size=1),
        "ltx_audio_latent": _node("LTXVEmptyLatentAudio", "Native audio clock", frames_number=plan.frame_count, frame_rate=plan.fps, batch_size=1, audio_vae=_link("audio_vae")),
        "sampler_select": _node("KSamplerSelect", "Ancestral base sampler", sampler_name="euler_ancestral"),
    }
    mask_images = load_reference_frames(prompt, "ltx_inpaint_mask", "Load complete verified red-channel mask", mask)
    prompt["ltx_inpaint_source"] = _node("SkulkLTXInpaintSource", "Place complete source and mask", source=_link("ltx_source_components"), mask=mask_images, width=plan.width, height=plan.height, left=control.left, top=control.top, expected_frames=plan.frame_count, maximum_bytes=budget)
    prompt["ltx_half_source"] = _node("ImageScale", "Native half-resolution source", image=_link("ltx_inpaint_source"), upscale_method="lanczos", width=plan.width // 2, height=plan.height // 2, crop="disabled")
    prompt["ltx_base_context"] = _node("SkulkLTXInpaintStage", "Half-size area mask, dilation 15 and green context", images=_link("ltx_half_source"), mask=_link("ltx_inpaint_source", 1), spatial_radius=15, maximum_bytes=budget)
    prompt["ltx_full_context"] = _node("SkulkLTXInpaintStage", "Full-size area mask, dilation 30 and green context", images=_link("ltx_inpaint_source"), mask=_link("ltx_inpaint_source", 1), spatial_radius=30, maximum_bytes=budget)
    base_video = _opening(prompt, opening, latent=_link("ltx_video_latent"), width=plan.width // 2, height=plan.height // 2, stage="base")
    prompt["ltx_base_guide"] = _node("SkulkLTXInpaintGuide", "Native factor-one base source guide", positive=_link("condition", 0), negative=_link("condition", 1), vae=_link("video_vae"), latent=base_video, image=_link("ltx_base_context"), iclora_model=_link("ltx_inpaint_lora"), expected_frames=plan.frame_count)
    # Reuse the qualified causal source-audio encoder. A non-regenerating full
    # retake keeps the encoder's actual tail, including its final causal token.
    prompt["ltx_source_audio"] = _node("SkulkLTXEditAudioLatent", "Normalize and encode complete source audio", audio=_link("ltx_source_components", 1), target=_link("ltx_audio_latent"), audio_vae=_link("audio_vae"), source_frames=plan.frame_count, sample_rate=plan.sample_rate, regenerate_audio=False, operation="retake", fps=plan.fps, start_seconds=0.0, end_seconds=plan.seconds)
    prompt["ltx_base_audio_reference"] = _node("SkulkLTXInpaintAudioReference", "Freeze complete source audio and attach reference tokens", positive=_link("ltx_base_guide", 0), negative=_link("ltx_base_guide", 1), audio_latent=_link("ltx_source_audio"), maximum_bytes=budget)
    _ = sampling_stage(prompt, "base", seed=plan.seed, sigmas=DISTILLED_SIGMAS, positive=_link("ltx_base_audio_reference", 0), negative=_link("ltx_base_audio_reference", 1), video=_link("ltx_base_guide", 2), audio=_link("ltx_base_audio_reference", 2), render=render, model=_link("ltx_inpaint_lora"))
    prompt["ltx_base_split"] = _node("LTXVSeparateAVLatent", "Separate denoised base latents", av_latent=_link("ltx_base_sampler", 1))
    prompt["ltx_base_crop"] = _node("LTXVCropGuides", "Remove base source guide before decode", positive=_link("ltx_base_audio_reference", 0), negative=_link("ltx_base_audio_reference", 1), latent=_link("ltx_base_split", 0))
    prompt["ltx_base_decode"] = _node("VAEDecodeTiled", "Decode base pixels", samples=_link("ltx_base_crop", 2), vae=_link("video_vae"), tile_size=512, overlap=64, temporal_size=128, temporal_overlap=32)
    prompt["ltx_base_blend"] = _node("SkulkLTXInpaintBlend", "Native base pyramid blend", generated=_link("ltx_base_decode"), context=_link("ltx_base_context"), mask=_link("ltx_base_context", 1), low_res_dilation=5, maximum_bytes=budget)
    prompt["ltx_pixel_upscale"] = _node("ImageScale", "Lanczos pixel refinement", image=_link("ltx_base_blend"), upscale_method="lanczos", width=plan.width, height=plan.height, crop="disabled")
    prompt["ltx_refine_encoded"] = _node("VAEEncodeTiled", "Re-encode blended full-size pixels", pixels=_link("ltx_pixel_upscale"), vae=_link("video_vae"), tile_size=512, overlap=64, temporal_size=500, temporal_overlap=64)
    refine_video = _opening(prompt, opening, latent=_link("ltx_refine_encoded"), width=plan.width, height=plan.height, stage="refine")
    # The pinned recipe carries the cropped conditioning into refinement.
    # Appending a second full-resolution source guide would change the model.
    prompt["ltx_refine_audio_reference"] = _node("SkulkLTXInpaintAudioReference", "Retain complete frozen audio at refinement", positive=_link("ltx_base_crop", 0), negative=_link("ltx_base_crop", 1), audio_latent=_link("ltx_base_split", 1), maximum_bytes=budget)
    _ = sampling_stage(prompt, "refine", seed=(plan.seed + 1) % (2**64), sigmas=INPAINT_REFINEMENT_SIGMAS, positive=_link("ltx_refine_audio_reference", 0), negative=_link("ltx_refine_audio_reference", 1), video=refine_video, audio=_link("ltx_refine_audio_reference", 2), render=render, model=_link("ltx_inpaint_lora"))
    prompt["ltx_refine_split"] = _node("LTXVSeparateAVLatent", "Separate refined latents", av_latent=_link("ltx_refine_sampler", 1))
    prompt["decode_video"] = _node("VAEDecodeTiled", "Decode full-size inpainting", samples=_link("ltx_refine_split", 0), vae=_link("video_vae"), tile_size=512, overlap=64, temporal_size=128, temporal_overlap=32)
    prompt["ltx_final_blend"] = _node("SkulkLTXInpaintBlend", "Native final pyramid blend", generated=_link("decode_video"), context=_link("ltx_full_context"), mask=_link("ltx_full_context", 1), low_res_dilation=6, maximum_bytes=budget)
    create: dict[str, object] = {"images": _link("ltx_final_blend"), "fps": float(plan.fps)}
    if plan.audio:
        prompt["decode_audio"] = _node("LTXVAudioVAEDecode", "Decode frozen source-audio reconstruction", samples=_link("ltx_refine_split", 1), audio_vae=_link("audio_vae"))
        create["audio"] = _link("decode_audio")
    prompt["create_video"] = _node("CreateVideo", "Mux complete inpainted take", **create)
    prompt["save_video"] = _node("SaveVideo", "Save inpainted take", video=_link("create_video"), filename_prefix=f"{output_subdir}/output", format="mp4", **{"format.codec": render.codec})
    prompt["thumbnail_frame"] = _node("ImageFromBatch", "First final blended frame", image=_link("ltx_final_blend"), batch_index=0, length=1)
    prompt["save_thumbnail"] = _node("SaveImage", "Save inpainting thumbnail", images=_link("thumbnail_frame"), filename_prefix=f"{output_subdir}/thumbnail")
    return prompt
