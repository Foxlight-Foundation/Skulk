"""Bind scene routing to the pinned model's actual surviving video coordinates."""

from dataclasses import dataclass

import torch
from torch import Tensor

from skulk_comfy_ltx.relay import RelaySegment, relay_segments
from skulk_comfy_ltx.relay_encoding import RelayLayout


@dataclass(frozen=True, slots=True)
class RelayGeometry:
    """Forward-scoped coordinates and ranges; never stored on a model clone."""

    query_frames: Tensor
    real_queries: Tensor
    segments: tuple[RelaySegment, ...]


def prepare_relay_geometry(video: Tensor, pixel_coords: Tensor, keyframe_indices: Tensor | None, guide_tokens: int, *, scale_factors: tuple[int, int, int], patch_size: tuple[int, int, int], causal: bool, layout: RelayLayout) -> RelayGeometry:
    """Invert the real causal grid after Comfy's guide filtering and replacement.

    The original latent shape and pre-filter keyframe count define the output
    timeline; actual surviving coordinates define each attention query. Using
    query index instead would shift every later scene when a grid row is removed.
    Surviving guide rows attend freely, matching the native video relay profile.
    """
    if video.ndim != 5 or pixel_coords.ndim != 4 or pixel_coords.shape[1] != 3 or pixel_coords.shape[3] != 2 or pixel_coords.device != video.device:
        raise ValueError("relay requires the pinned five-dimensional video and start/end coordinate grid")
    batch, _, latent_frames, height, width = video.shape
    if len(scale_factors) != 3 or any(type(scale) is not int or scale < 1 for scale in scale_factors) or len(patch_size) != 3 or patch_size[0] != 1 or any(type(size) is not int or size < 1 for size in patch_size):
        raise ValueError("relay requires valid pinned temporal scale and single-frame patches")
    if height % patch_size[1] or width % patch_size[2]:
        raise ValueError("relay latent dimensions must match the actual patch grid")
    tokens_per_frame = (height // patch_size[1]) * (width // patch_size[2])
    if tokens_per_frame < 1:
        raise ValueError("relay requires a nonempty spatial latent grid")
    pre_filter_guides = 0
    if keyframe_indices is not None:
        if keyframe_indices.ndim != 4 or keyframe_indices.shape[1] != 3 or keyframe_indices.shape[3] != 2 or keyframe_indices.shape[2] % tokens_per_frame:
            raise ValueError("relay guide indices do not match whole latent frames")
        pre_filter_guides = keyframe_indices.shape[2] // tokens_per_frame
    timeline_frames = latent_frames - pre_filter_guides
    queries = pixel_coords.shape[2]
    if batch < 1 or pixel_coords.shape[0] not in (1, batch) or queries < 1 or type(guide_tokens) is not int or not 0 <= guide_tokens <= queries or guide_tokens > pre_filter_guides * tokens_per_frame:
        raise ValueError("relay surviving guide metadata does not match its actual query grid")
    starts = pixel_coords[:, 0, :, 0].to(dtype=torch.float32).expand(batch, -1)
    if not bool(torch.isfinite(starts).all()):
        raise ValueError("relay query coordinates must be finite")
    frames = starts / scale_factors[0]
    if causal:
        # Pinned causal starts are 0, 1, 9, ... for an eight-frame VAE scale.
        # Ceil is the exact inverse on this grid, including filtered queries.
        frames = frames.ceil()
    real = torch.ones((batch, queries), device=video.device, dtype=torch.bool)
    if guide_tokens:
        real[:, -guide_tokens:] = False
    if bool((real & ((starts < 0) | (frames < 0) | (frames >= timeline_frames))).any()):
        raise ValueError("relay video queries exceed the original output timeline")
    return RelayGeometry(frames, real, relay_segments(layout.token_ranges, layout.weights, timeline_frames))
