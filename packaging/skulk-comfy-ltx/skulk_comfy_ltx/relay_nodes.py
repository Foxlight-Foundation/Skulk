"""Bounded scene-text encoding and clone-scoped routing for managed LTX graphs."""

from typing import final

from comfy_api.latest import io

from skulk_comfy_ltx.relay_binding import RelayModelPatcher, patch_prompt_relay
from skulk_comfy_ltx.relay_encoding import RelayClip, encode_prompt_relay


@final
class SkulkLTXPromptRelayEncode(io.ComfyNode):
    """Encode one combined global-and-local prompt with measured scene columns."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Accept the placed encoder and a bounded ordered scene specification."""
        return io.Schema(node_id="SkulkLTXPromptRelayEncode", display_name="Skulk LTX timed scene prompts", category="model/conditioning/ltxv", inputs=[io.Clip.Input("clip"), io.String.Input("global_prompt", default="", multiline=True), io.String.Input("segments_json", default="[]", multiline=True)], outputs=[io.Conditioning.Output()])

    @classmethod
    def execute(cls, clip: RelayClip, global_prompt: str, segments_json: str) -> io.NodeOutput:
        """Encode once and return positive conditioning; audio keeps the full text."""
        return io.NodeOutput(encode_prompt_relay(clip, global_prompt, segments_json))


@final
class SkulkLTXApplyPromptRelay(io.ComfyNode):
    """Apply restorable temporal routing only to the model's video text attention."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose a model clone without arbitrary attention options or code paths."""
        return io.Schema(node_id="SkulkLTXApplyPromptRelay", display_name="Skulk LTX prompt relay", category="model/conditioning/ltxv", inputs=[io.Model.Input("model")], outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model: RelayModelPatcher) -> io.NodeOutput:
        """Return a clone managed by Comfy's normal installation/restoration path."""
        return io.NodeOutput(patch_prompt_relay(model))
