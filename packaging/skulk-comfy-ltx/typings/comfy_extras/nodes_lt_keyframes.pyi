from collections.abc import Mapping
from typing import Protocol, TypedDict

from comfy_api.latest import io
from torch import Tensor

class _VideoVae(Protocol):
    downscale_index_formula: tuple[int, int, int]

class _GeneratedKeyframes(TypedDict):
    frame_indices: list[int]

def get_generated_keyframes(conditioning: list[list[object]]) -> _GeneratedKeyframes | None: ...
def occupied_pixel_frames(latent: Mapping[str, object], temporal_scale: int, num_frames: int, video_latent_frames: int) -> set[int]: ...
def pixel_frames_from_keyframe_idxs(keyframe_idxs: Tensor | None) -> set[int]: ...
def detailing_positions(num_frames: int, interval_frames: float) -> list[int]: ...

class LTXVAddGeneratedKeyframes:
    @classmethod
    def execute(cls, positive: list[list[object]], negative: list[list[object]], vae: _VideoVae, latent: Mapping[str, object], *, interval_frames: int, keyframes: Mapping[str, object] | None, frame_indices: str) -> io.NodeOutput: ...
