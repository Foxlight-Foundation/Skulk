from collections.abc import Callable
import mlx.core as mx
from ltx_core_mlx.utils.ffmpeg import VideoInfo
from ._base import BasePipeline

probe_video_info: Callable[[str], VideoInfo]

class RetakePipeline(BasePipeline):
    def __init__(self, model_dir: str, gemma_model_id: str, low_memory: bool, low_ram_streaming: bool, distilled: bool) -> None: ...
    def retake_from_video(self, prompt: str, video_path: str, start_frame: int, end_frame: int, *, seed: int, num_steps: int, regenerate_audio: bool, negative_prompt: str | None) -> tuple[mx.array, mx.array]: ...
    def extend_from_video(self, prompt: str, video_path: str, extend_frames: int, *, seed: int, num_steps: int, negative_prompt: str | None) -> tuple[mx.array, mx.array]: ...
