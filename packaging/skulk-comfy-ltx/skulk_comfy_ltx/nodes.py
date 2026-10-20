"""A bounded, versioned node set loaded from the verified Foxlight wheel."""

from typing import final

import comfy.samplers
from comfy_api.latest import ComfyExtension, io

from skulk_comfy_ltx.comfy_sampler import sample_ltx_res2s
from skulk_comfy_ltx.guidance import ModelPatcher, patch_guidance_rescale


@final
class SkulkLTXRes2sSampler(io.ComfyNode):
    """Select the joint AV midpoint solver; graph geometry remains caller-bound."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Describe the fixed sampler output; no arbitrary options or source paths."""
        return io.Schema(node_id="SkulkLTXRes2sSampler", display_name="Skulk LTX Res2s", category="sampling/custom_sampling/samplers", inputs=[], outputs=[io.Sampler.Output()])

    @classmethod
    def execute(cls) -> io.NodeOutput:
        """Return the sampler through Comfy's normal cancellation/progress path."""
        return io.NodeOutput(comfy.samplers.KSAMPLER(sample_ltx_res2s))


@final
class SkulkLTXGuidanceRescale(io.ComfyNode):
    """Rescale joint guidance independently on the real audio and video tensors."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Describe the bounded model patch used by the exact HQ graph."""
        return io.Schema(node_id="SkulkLTXGuidanceRescale", display_name="Skulk LTX guidance rescale", category="advanced/guidance", inputs=[io.Model.Input("model"), io.Float.Input("video_scale", default=0.45, min=0.0, max=1.0, step=0.01), io.Float.Input("audio_scale", default=1.0, min=0.0, max=1.0, step=0.01)], outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model: ModelPatcher, video_scale: float, audio_scale: float) -> io.NodeOutput:
        """Append rescaling after the graph's CFG/STG/modality calculations."""
        return io.NodeOutput(patch_guidance_rescale(model, video_scale=video_scale, audio_scale=audio_scale))


@final
class SkulkLTXExtension(ComfyExtension):
    """Expose only the node implementations carried by this exact wheel."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return the complete allowlisted node set without installing plugins."""
        return [SkulkLTXRes2sSampler, SkulkLTXGuidanceRescale]
