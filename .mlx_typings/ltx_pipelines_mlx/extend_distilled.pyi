from collections.abc import Callable
import mlx.core as mx
from ltx_core_mlx.utils.ffmpeg import VideoInfo
from .distilled import DistilledPipeline

probe_video_info: Callable[[str], VideoInfo]

class ExtendDistilledPipeline(DistilledPipeline):
    def extend_from_video(self, prompt: str, video_path: str, extend_frames: int, *, seed: int) -> tuple[mx.array, mx.array]: ...
