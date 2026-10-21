"""Native Comfy bridges for bounded complete-source inpainting primitives."""

from collections.abc import Mapping
from typing import Protocol, final

import comfy.model_management
import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_lt import LTXVAddGuide
from torch import Tensor

from skulk_comfy_ltx.inpaint_audio import inpaint_audio_context
from skulk_comfy_ltx.inpaint_blend import blend_inpaint_images
from skulk_comfy_ltx.inpaint_control import validate_inpaint_guide
from skulk_comfy_ltx.inpaint_pixels import place_inpaint_source, prepare_inpaint_stage


class _LoadedAdapter(Protocol):
    def get_attachment(self, key: str) -> object: ...


def _budget_input() -> io.Int.Input:
    return io.Int.Input("maximum_bytes", min=1, max=2**63 - 1)


@final
class SkulkLTXInpaintSource(io.ComfyNode):
    """Place all verified source frames and explicitly expand red-channel masks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Require private worker geometry and budget, without arbitrary paths."""
        return io.Schema(node_id="SkulkLTXInpaintSource", display_name="Skulk LTX inpainting source", category="video", inputs=[io.Image.Input("source"), io.Image.Input("mask"), io.Int.Input("width", min=128, max=4096), io.Int.Input("height", min=128, max=4096), io.Int.Input("left", min=0, max=4095), io.Int.Input("top", min=0, max=4095), io.Int.Input("expected_frames", min=1, max=2881), _budget_input()], outputs=[io.Image.Output(), io.Mask.Output()])

    @classmethod
    def execute(cls, source: Tensor, mask: Tensor, width: int, height: int, left: int, top: int, expected_frames: int, maximum_bytes: int) -> io.NodeOutput:
        """Retain the original source pixels and mark all added canvas as white."""
        if source.ndim != 4 or source.shape[0] != expected_frames or mask.ndim != 4 or mask.shape[-1] != 3:
            raise ValueError("Inpainting source and RGB mask must cover the verified timeline")
        images, masks = place_inpaint_source(source, mask[..., 0], width=width, height=height, left=left, top=top, maximum_bytes=maximum_bytes, output_device=comfy.model_management.intermediate_device(), check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted)
        return io.NodeOutput(images, masks)


@final
class SkulkLTXInpaintStage(io.ComfyNode):
    """Prepare native stage masks and trained green context without truncation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Keep stage resolution on the image input and mask dilation explicit."""
        return io.Schema(node_id="SkulkLTXInpaintStage", display_name="Skulk LTX inpainting context", category="video", inputs=[io.Image.Input("images"), io.Mask.Input("mask"), io.Int.Input("spatial_radius", min=0, max=300), _budget_input()], outputs=[io.Image.Output(), io.Mask.Output()])

    @classmethod
    def execute(cls, images: Tensor, mask: Tensor, spatial_radius: int, maximum_bytes: int) -> io.NodeOutput:
        """Area-resize before dilation and keep the SDR key in both stages."""
        green, masks = prepare_inpaint_stage(images, mask, spatial_radius, maximum_bytes=maximum_bytes, output_device=comfy.model_management.intermediate_device(), check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted)
        return io.NodeOutput(green, masks)


@final
class SkulkLTXInpaintBlend(io.ComfyNode):
    """Blend decoded output with its actual stage's green-prepared source."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Require matching complete tensors, rather than a shortest-sequence option."""
        return io.Schema(node_id="SkulkLTXInpaintBlend", display_name="Skulk LTX inpainting blend", category="video", inputs=[io.Image.Input("generated"), io.Image.Input("context"), io.Mask.Input("mask"), io.Int.Input("low_res_dilation", min=0, max=15), _budget_input()], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, generated: Tensor, context: Tensor, mask: Tensor, low_res_dilation: int, maximum_bytes: int) -> io.NodeOutput:
        """Use one native pyramid frame at a time with the normal interrupt flag."""
        return io.NodeOutput(blend_inpaint_images(generated, context, mask, low_res_dilation, maximum_bytes=maximum_bytes, device=comfy.model_management.get_torch_device(), output_device=comfy.model_management.intermediate_device(), check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted))


@final
class SkulkLTXInpaintAudioReference(io.ComfyNode):
    """Attach native source-audio tokens and freeze the complete encoded context."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Reuse the selected causal audio VAE output without another encoding."""
        return io.Schema(node_id="SkulkLTXInpaintAudioReference", display_name="Skulk LTX inpainting audio context", category="model/conditioning/ltxv", inputs=[io.Conditioning.Input("positive"), io.Conditioning.Input("negative"), io.Latent.Input("audio_latent"), _budget_input()], outputs=[io.Conditioning.Output(), io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls, positive: list[list[object]], negative: list[list[object]], audio_latent: Mapping[str, object], maximum_bytes: int) -> io.NodeOutput:
        """Preserve both conditioning branches and the latent's original dtype/layout."""
        samples = audio_latent.get("samples")
        if not isinstance(samples, Tensor):
            raise ValueError("Inpainting requires the selected VAE's encoded source audio")
        tokens, mask = inpaint_audio_context(samples, maximum_bytes=maximum_bytes, check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted)
        values: dict[str, object] = {"ref_audio": {"tokens": tokens}}
        return io.NodeOutput(node_helpers.conditioning_set_values(positive, values), node_helpers.conditioning_set_values(negative, values), {**audio_latent, "noise_mask": mask})


@final
class SkulkLTXInpaintGuide(io.ComfyNode):
    """Delegate native guide creation only for the actual loaded factor-one adapter."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Keep source guide creation separate from Union's factor-two contract."""
        return io.Schema(node_id="SkulkLTXInpaintGuide", display_name="Skulk LTX inpainting guide", category="model/conditioning/ltxv", inputs=[io.Conditioning.Input("positive"), io.Conditioning.Input("negative"), io.Vae.Input("vae"), io.Latent.Input("latent"), io.Image.Input("image"), io.Model.Input("iclora_model"), io.Int.Input("expected_frames", min=1, max=2881)], outputs=[io.Conditioning.Output(), io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls, positive: list[list[object]], negative: list[list[object]], vae: object, latent: Mapping[str, object], image: Tensor, iclora_model: _LoadedAdapter, expected_frames: int) -> io.NodeOutput:
        """Retain pinned Comfy guide tokens, masks, positions and attention metadata."""
        validate_inpaint_guide(iclora_model.get_attachment("lora_metadata"), latent, image, expected_frames, check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted)
        return LTXVAddGuide.execute(positive, negative, vae, latent, image, frame_idx=0, strength=1.0, iclora_parameters={"reference_downscale_factor": 1})
