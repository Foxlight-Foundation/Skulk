from collections.abc import Callable, Sequence
from typing import Protocol
import mlx.core as mx
from ltx_core_mlx.components.guiders import MultiModalGuiderParams
from ltx_core_mlx.conditioning.prompt_relay import PromptRelayInput
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import DecodeKeyframes
from .utils.args import ImageConditioningInput
from .utils.types import AutoDuration

OnStep = Callable[[int, int, mx.array, float], None]

class StepwiseHook(Protocol):
    def __call__(self, latent_frames: int, latent_height: int, latent_width: int, *, stage: int | None = None) -> OnStep: ...

class DurationResolver(Protocol):
    def __call__(self, num_frames: int | AutoDuration, *, video_encoding: mx.array, audio_encoding: mx.array, frame_rate: float) -> int: ...

class DecoderProfiler(Protocol):
    """Verified verbose flag on the pinned composition-based decoder block."""
    verbose: bool

class BasePipeline:
    def _load_decoders(self) -> None: ...
    verbose: bool
    generate_audio: bool
    video_decoder: str
    video_decoder_block: DecoderProfiler
    _stepwise_hook: StepwiseHook
    _resolve_num_frames: DurationResolver
    def generate_two_stage(self, prompt: str, height: int, width: int, num_frames: int | AutoDuration, *, frame_rate: float, seed: int, images: Sequence[ImageConditioningInput] | None = None, generated_keyframes: int | Sequence[int] = 0, enable_teacache: bool = False, stage1_steps: int | None = None, negative_prompt: str | None = None, video_guider_params: MultiModalGuiderParams | None = None, audio_guider_params: MultiModalGuiderParams | None = None, prompt_relay: PromptRelayInput | None = None) -> tuple[mx.array, mx.array]: ...
    def _decode_and_save_video(self, video_latent: mx.array, audio_latent: mx.array, output_path: str, *, frame_rate: float, seed: int = 0, keyframes: DecodeKeyframes | None = None) -> str: ...
