from collections.abc import Callable, Sequence
from typing import Protocol
import mlx.core as mx
from ltx_core_mlx.components.guiders import MultiModalGuiderParams
from .utils.args import ImageConditioningInput
from .utils.types import AutoDuration

OnStep = Callable[[int, int, mx.array, float], None]

class StepwiseHook(Protocol):
    def __call__(self, latent_frames: int, latent_height: int, latent_width: int, *, stage: int | None = None) -> OnStep: ...

class DurationResolver(Protocol):
    def __call__(self, num_frames: int | AutoDuration, *, video_encoding: mx.array, audio_encoding: mx.array, frame_rate: float) -> int: ...

class BasePipeline:
    verbose: bool
    generate_audio: bool
    video_decoder: str
    _stepwise_hook: StepwiseHook
    _resolve_num_frames: DurationResolver
    def generate_two_stage(self, prompt: str, height: int, width: int, num_frames: int | AutoDuration, *, frame_rate: float, seed: int, images: Sequence[ImageConditioningInput] | None = None, generated_keyframes: int | Sequence[int] = 0, enable_teacache: bool = False, stage1_steps: int | None = None, negative_prompt: str | None = None, video_guider_params: MultiModalGuiderParams | None = None, audio_guider_params: MultiModalGuiderParams | None = None) -> tuple[mx.array, mx.array]: ...
    def _decode_and_save_video(self, video_latent: mx.array, audio_latent: mx.array, output_path: str, *, frame_rate: float, seed: int = 0) -> str: ...
