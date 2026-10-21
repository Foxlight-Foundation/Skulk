"""Bind validated Union references to the exact pinned native Comfy guide API."""

from collections.abc import Mapping
from typing import Protocol, final

from comfy_api.latest import io
from comfy_extras.nodes_lt import LTXVAddGuide
from torch import Tensor

from skulk_comfy_ltx.union_control import validate_union_inputs


class LoadedAdapter(Protocol):
    """Metadata attachment supplied by the native ModelPatcher clone."""

    def get_attachment(self, key: str) -> object:
        """Return the attachment from the model actually used by the sampler."""
        ...


@final
class SkulkLTXUnionGuide(io.ComfyNode):
    """Delegate guide creation only after actual adapter and timeline validation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose bounded guide inputs without arbitrary paths or source code."""
        return io.Schema(node_id="SkulkLTXUnionGuide", display_name="Skulk LTX Union guide", category="model/conditioning/ltxv", inputs=[io.Conditioning.Input("positive"), io.Conditioning.Input("negative"), io.Vae.Input("vae"), io.Latent.Input("latent"), io.Image.Input("image"), io.Model.Input("iclora_model"), io.Float.Input("strength", default=1.0, min=0.0, max=1.0), io.Int.Input("expected_frames", default=49, min=1, max=2881)], outputs=[io.Conditioning.Output(), io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls, positive: list[list[object]], negative: list[list[object]], vae: object, latent: Mapping[str, object], image: Tensor, iclora_model: LoadedAdapter, strength: float, expected_frames: int) -> io.NodeOutput:
        """Preserve native dilation, guide masks, positions and attention entries."""
        validate_union_inputs(iclora_model.get_attachment("lora_metadata"), latent, image, expected_frames)
        return LTXVAddGuide.execute(positive, negative, vae, latent, image, frame_idx=0, strength=strength, iclora_parameters={"reference_downscale_factor": 2})
