"""Delegate DFR execution to pinned Comfy nodes with explicit sampling masks."""

from collections.abc import Mapping
from typing import Protocol, final

from comfy_api.latest import io
from comfy_extras.nodes_lt import get_keyframe_idxs
from comfy_extras.nodes_lt_keyframes import (
    LTXVAddGeneratedKeyframes,
    detailing_positions,
    get_generated_keyframes,
    occupied_pixel_frames,
    pixel_frames_from_keyframe_idxs,
)
from torch import Tensor

from skulk_comfy_ltx.dfr import detailing_frame_override, prepare_dfr_latent


class VideoVae(Protocol):
    """Only the pinned video VAE's causal scale factors are read by this bridge."""

    downscale_index_formula: tuple[int, int, int]


@final
class SkulkLTXAddGeneratedKeyframes(io.ComfyNode):
    """Prepare sampleable dense video and preserve real image/guide occupancy."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose the stock detailing inputs without arbitrary indices or source paths."""
        return io.Schema(node_id="SkulkLTXAddGeneratedKeyframes", display_name="Skulk LTX detailing keyframes", category="model/conditioning/ltxv", inputs=[io.Conditioning.Input("positive"), io.Conditioning.Input("negative"), io.Vae.Input("vae"), io.Latent.Input("latent"), io.Int.Input("interval_frames", default=24, min=1, max=1024), io.Latent.Input("keyframes", optional=True)], outputs=[io.Conditioning.Output(), io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls, positive: list[list[object]], negative: list[list[object]], vae: VideoVae, latent: Mapping[str, object], interval_frames: int = 24, keyframes: Mapping[str, object] | None = None) -> io.NodeOutput:
        """Delegate token creation, initialization, conditioning and masks unchanged."""
        prepared = prepare_dfr_latent(latent)
        samples = prepared["samples"]
        assert isinstance(samples, Tensor)
        scale = vae.downscale_index_formula[0]
        if scale < 1 or not 1 <= interval_frames <= 1024:
            raise ValueError("DFR requires valid temporal scale and detailing interval")
        guide_indices, guide_frames = get_keyframe_idxs(positive, samples.shape)
        video_frames = samples.shape[2] - guide_frames
        num_frames = (video_frames - 1) * scale + 1
        occupied = occupied_pixel_frames(prepared, scale, num_frames, video_latent_frames=video_frames)
        occupied |= pixel_frames_from_keyframe_idxs(guide_indices)
        existing = get_generated_keyframes(positive)
        if existing is not None:
            occupied |= set(existing["frame_indices"])
        indices = detailing_frame_override(num_frames, occupied, detailing_positions(num_frames, float(interval_frames)))
        return LTXVAddGeneratedKeyframes.execute(positive, negative, vae, prepared, interval_frames=interval_frames, keyframes=keyframes, frame_indices=indices)
