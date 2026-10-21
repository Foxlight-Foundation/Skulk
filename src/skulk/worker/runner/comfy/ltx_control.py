"""LTX Union IC-LoRA graph wiring, independent of H3 ControlNet placement."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skulk.worker.runner.comfy.graph import (
        ComfyPrompt,
        ComfyRenderPlan,
        ReferenceBinding,
    )


def add_union_control(
    prompt: ComfyPrompt,
    render: ComfyRenderPlan,
    references: tuple[ReferenceBinding, ...],
    positive: list[str | int],
    negative: list[str | int],
    latent: list[str | int],
    model: list[str | int],
) -> tuple[list[str | int], list[str | int], list[str | int], list[str | int]]:
    """Append one aligned guide to the base stage and return its patched model.

    The selected adapter's metadata must declare its trained reference scale.
    Prepared pixels bypass preprocessing; RGB inputs use the exact selected
    native Comfy guide chain. The caller crops guide tokens before upscaling.
    """
    from skulk.worker.runner.comfy.graph import derive_guide, load_reference_frames

    def node(class_type: str, title: str, **inputs: object) -> dict[str, object]:
        return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}

    def link(name: str, output: int = 0) -> list[str | int]:
        return [name, output]

    control = render.union_control
    filename = render.files.union_control_lora
    if control is None or filename is None:
        raise ValueError("Union graph requires its selected IC-LoRA and explicit controls")
    source = next(binding for binding in references if binding.spec.role == "control")
    images = load_reference_frames(prompt, "ltx_union_video", "Union reference clip", source)
    if not control.preprocessed:
        if render.union_guide is None:
            raise ValueError("Union RGB input requires its selected guide preprocessor")
        images = derive_guide(prompt, render.union_guide, images)
    prompt["ltx_union_resize"] = node("ImageScale", "Fit guide to base canvas", image=images, upscale_method="bilinear", width=render.plan.width // 2, height=render.plan.height // 2, crop="disabled")
    prompt["ltx_union_lora"] = node("LoraLoaderModelOnly", "Union base model", model=model, lora_name=filename, strength_model=control.adapter_strength)
    prompt["ltx_union_guide"] = node("SkulkLTXUnionGuide", "Half-grid Union reference", positive=positive, negative=negative, vae=link("video_vae"), latent=latent, image=link("ltx_union_resize"), iclora_model=link("ltx_union_lora"), strength=control.guide_strength, expected_frames=render.plan.frame_count)
    return link("ltx_union_guide", 0), link("ltx_union_guide", 1), link("ltx_union_guide", 2), link("ltx_union_lora")
