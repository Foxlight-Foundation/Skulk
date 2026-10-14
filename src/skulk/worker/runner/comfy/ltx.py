"""LTX-2.5 distilled graphs for the pinned core-node ComfyUI runtime.

The fixed tables and stage wiring follow workflow templates 0.11.57. Audio
latents participate in both stages even when the caller asks for a silent
container: muting the mux is different from changing the joint model.
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
) -> list[str | int]:
    prompt[f"ltx_{name}_av"] = _node(
        "LTXVConcatAVLatent",
        "Join audio/video latents",
        video_latent=video,
        audio_latent=audio,
    )
    prompt[f"ltx_{name}_noise"] = _node("RandomNoise", "Noise", noise_seed=seed)
    prompt[f"ltx_{name}_sigmas"] = _node(
        "ManualSigmas", "Distilled schedule", sigmas=sigmas
    )
    prompt[f"ltx_{name}_guider"] = _node(
        "LTXVDualCFGGuider",
        "Distilled guidance",
        model=_link("unet"),
        positive=positive,
        negative=negative,
        video_cfg=1.0,
        audio_cfg=1.0,
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


def build_ltx_prompt(
    render: ComfyRenderPlan,
    params: VideoGenerationTaskParams,
    references: tuple[ReferenceBinding, ...],
    output_subdir: str,
) -> ComfyPrompt:
    """Build one distilled two-stage audio-video graph using core nodes only.

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
            "CLIPTextEncode", "Empty negative", text="", clip=_link("clip")
        ),
        "condition": _node(
            "LTXVConditioning",
            "Frame-rate conditioning",
            positive=_link("ltx_positive"),
            negative=_link("ltx_negative"),
            frame_rate=plan.fps,
        ),
        "ltx_video_latent": _node(
            "EmptyLTXVLatentVideo",
            "Half-resolution video latent",
            width=plan.width // 2,
            height=plan.height // 2,
            length=plan.frame_count,
            batch_size=1,
        ),
        "ltx_audio_latent": _node(
            "LTXVEmptyLatentAudio",
            "Audio latent",
            frames_number=plan.frame_count,
            frame_rate=plan.fps,
            batch_size=1,
            audio_vae=_link("audio_vae"),
        ),
        "sampler_select": _node(
            "KSamplerSelect", "Ancestral Euler", sampler_name="euler_ancestral"
        ),
        "ltx_spatial_model": _node(
            "LatentUpscaleModelLoader",
            "Load spatial upscaler",
            model_name=files.spatial_upscaler,
        ),
    }
    positive, negative, video, guided = _condition_images(
        prompt,
        references,
        stage="base",
        positive=_link("condition", 0),
        negative=_link("condition", 1),
        latent=_link("ltx_video_latent"),
        fps=plan.fps,
        refinement=False,
    )
    first = _sampling_stage(
        prompt,
        "base",
        seed=plan.seed,
        sigmas=DISTILLED_SIGMAS,
        positive=positive,
        negative=negative,
        video=video,
        audio=_link("ltx_audio_latent"),
    )
    prompt["ltx_base_split"] = _node(
        "LTXVSeparateAVLatent", "Separate base latents", av_latent=first
    )
    base_video = _link("ltx_base_split", 0)
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
        fps=plan.fps,
        refinement=True,
    )
    second = _sampling_stage(
        prompt,
        "refine",
        seed=(plan.seed + 1) % (2**64),
        sigmas=REFINEMENT_SIGMAS,
        positive=positive,
        negative=negative,
        video=video,
        audio=_link("ltx_base_split", 1),
    )
    prompt["ltx_refine_split"] = _node(
        "LTXVSeparateAVLatent", "Separate refined latents", av_latent=second
    )
    final_video = _link("ltx_refine_split", 0)
    if guided:
        final_video = _crop_guides(prompt, "refine", positive, negative, final_video)
    prompt["decode_video"] = _node(
        "VAEDecodeTiled",
        "Decode video",
        samples=final_video,
        vae=_link("video_vae"),
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=16,
    )
    create: dict[str, object] = {
        "images": _link("decode_video"),
        "fps": float(plan.fps),
    }
    if plan.audio:
        prompt["decode_audio"] = _node(
            "LTXVAudioVAEDecode",
            "Decode audio",
            samples=_link("ltx_refine_split", 1),
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
