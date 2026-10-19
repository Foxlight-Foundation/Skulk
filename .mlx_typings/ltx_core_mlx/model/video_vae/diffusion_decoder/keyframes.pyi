"""DecodeKeyframes at ltx-2-mlx revision 9e4a1a54; values are only forwarded."""

from dataclasses import dataclass
import mlx.core as mx

@dataclass(frozen=True)
class DecodeKeyframes:
    """Immutable per-plane latents and global frame indices for diffusion decode."""
    latents: mx.array
    pixel_frame_indices: tuple[int, ...]
    clip_start_frame: int = 0

    @property
    def num_planes(self) -> int:
        """Return the number of keyframe planes."""
        ...
