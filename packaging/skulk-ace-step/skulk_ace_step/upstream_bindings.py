"""Audited signatures for the pinned native ACE inference boundary.

Only the isolated Python runtime imports this module. The structural interfaces
describe the inspected upstream API without permitting caller-selected modules,
model repository Python, arbitrary keywords or a language-model planner.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast, final

import numpy as np
import numpy.typing as npt
import torch


class NativeHandler(Protocol):
    """The instruction formatter used by the closed Skulk operation adapter."""

    def generate_instruction(
        self,
        task_type: str,
        track_name: str | None = None,
        complete_track_classes: list[str] | None = None,
    ) -> str:
        """Format the pinned task instruction, including the selected part name."""
        ...


class SourceProcessor(Protocol):
    """Pinned source entry point, observed without replacing its computation."""

    def __call__(self, audio_file: str | None) -> torch.Tensor | None:
        """Decode and normalize the actual source supplied to native inference."""
        ...


class SourceEncoder(Protocol):
    """Pinned overlap-discard encoder, with sample-domain chunk arguments."""

    def __call__(
        self,
        audio: torch.Tensor,
        chunk_size: int | None = None,
        overlap: int | None = None,
        offload_latent_to_cpu: bool = True,
    ) -> torch.Tensor:
        """Encode the complete source while retaining its native latent clock."""
        ...


class ModelPrecheck(Protocol):
    """Offline guard replacing the broader upstream default-model download path."""

    def __call__(
        self,
        *,
        checkpoint_path: Path,
        config_path: str,
        prefer_source: str | None,
        vae_variant: str | None = None,
    ) -> tuple[str, bool] | None:
        """Permit only the complete already verified local base/encoder/VAE view."""
        ...


class LoadedHandler(NativeHandler, Protocol):
    """Inspected initialization and observation surface of one native model."""

    model: torch.nn.Module | None
    text_encoder: torch.nn.Module | None
    vae: torch.nn.Module | None
    device: str
    dtype: torch.dtype
    sample_rate: int
    use_mlx_dit: bool
    use_mlx_vae: bool
    process_src_audio: SourceProcessor
    tiled_encode: SourceEncoder

    def initialize_service(
        self,
        *,
        project_root: str,
        config_path: str,
        device: str,
        use_flash_attention: Literal[False],
        compile_model: Literal[False],
        offload_to_cpu: Literal[False],
        offload_dit_to_cpu: Literal[False],
        quantization: None,
        use_mlx_dit: Literal[False],
        vae_checkpoint: Literal["official"],
    ) -> tuple[str, bool]:
        """Load explicit local artifacts without compilation, offload or planner."""
        ...


class HandlerFactory(Protocol):
    """Fixed runtime handler constructor, never a model-repository entry point."""

    def __call__(self) -> LoadedHandler:
        """Construct the pinned native handler; no model is loaded yet."""
        ...


_MODEL_ACQUISITION_HOOK: Final = "_ensure_models_present"
_MAXIMUM_ENCODING_FRAMES: Final = 15 * 48000
_ENCODING_OVERLAP_FRAMES: Final = 2 * 48000


@final
@dataclass(frozen=True)
class _BoundedSourceEncoder:
    original: SourceEncoder

    def __call__(
        self,
        audio: torch.Tensor,
        chunk_size: int | None = None,
        overlap: int | None = None,
        offload_latent_to_cpu: bool = True,
    ) -> torch.Tensor:
        # Upstream selects 30s from total GPU capacity, independently of this
        # instance's admitted workspace. Its existing 15s overlap-discard path
        # bounds each encode without replacing codec math or dropping context.
        bounded_chunk = (
            _MAXIMUM_ENCODING_FRAMES
            if chunk_size is None
            else min(chunk_size, _MAXIMUM_ENCODING_FRAMES)
        )
        return self.original(
            audio,
            chunk_size=bounded_chunk,
            overlap=_ENCODING_OVERLAP_FRAMES if overlap is None else overlap,
            offload_latent_to_cpu=offload_latent_to_cpu,
        )


def install_bounded_source_encoding(handler: LoadedHandler) -> None:
    """Bound this loaded handler's encoding to 15s with native 2s overlap.

    Shorter explicit chunks and explicit overlap/offload arguments survive.
    The original pinned callable still owns codec execution and geometry. This
    changes one resident handler only; no upstream class or module is patched.
    Decode and the admitted live resource guard retain their separate policies.
    """
    if handler.sample_rate != 48000:
        raise ValueError("native music encoding requires the pinned 48kHz clock")
    handler.tiled_encode = _BoundedSourceEncoder(handler.tiled_encode)


def install_offline_precheck(handler: LoadedHandler, precheck: ModelPrecheck) -> None:
    """Install the one audited acquisition hook with its exact callable signature.

    The upstream private attribute has no public setter. Keep its fixed name and
    typed effect at this binding boundary, as in the accepted native prototype;
    callers cannot select an attribute or supply an untyped native option.
    """
    setattr(handler, _MODEL_ACQUISITION_HOOK, precheck)


class ParametersFactory(Protocol):
    """Exact constructor fields implemented by this base-model adapter."""

    def __call__(
        self,
        *,
        task_type: str,
        src_audio: str | None,
        instruction: str,
        caption: str,
        lyrics: str,
        instrumental: bool,
        duration: float,
        inference_steps: int,
        guidance_scale: float,
        seed: int,
        repainting_start: float,
        repainting_end: float,
        chunk_mask_mode: Literal["explicit"],
        thinking: Literal[False],
        use_cot_metas: Literal[False],
        use_cot_caption: Literal[False],
        use_cot_language: Literal[False],
        dcw_enabled: Literal[False],
        enable_normalization: Literal[False],
        repaint_strength: float,
    ) -> object:
        """Build native parameters; retain all other defaults from pinned source."""
        ...


class ConfigurationFactory(Protocol):
    """One explicit seeded result, without language-model batch processing."""

    def __call__(
        self,
        *,
        batch_size: Literal[1],
        use_random_seed: Literal[False],
        seeds: list[int],
        audio_format: Literal["wav32"],
    ) -> object:
        """Build the pinned generation configuration; never accept native options."""
        ...


class NativeResult(Protocol):
    """Only the success flag and one native CPU tensor are consumed."""

    success: bool
    error: str | None
    audios: list[dict[str, object]]


class NativeGenerate(Protocol):
    """Run native diffusion without a planner or native output-file management."""

    def __call__(
        self,
        dit_handler: LoadedHandler,
        llm_handler: None,
        params: object,
        config: object,
        *,
        save_dir: None,
    ) -> NativeResult:
        """Return the one measured CPU float32 output tensor, not a local path."""
        ...


class TensorArray(Protocol):
    """The actual bounded CPU float32 Tensor.numpy result, absent from its stub."""

    def numpy(self) -> npt.NDArray[np.float32]:
        """Share already checked detached CPU float32 sample storage."""
        ...


@dataclass(frozen=True)
class NativeConstructors:
    """Fixed pinned constructors, imported after offline runtime setup."""

    parameters: ParametersFactory
    configuration: ConfigurationFactory


@dataclass(frozen=True)
class NativeAPI:
    """Closed pinned native entry points used by one resident process."""

    handler: HandlerFactory
    constructors: NativeConstructors
    generate: NativeGenerate


def native_api() -> NativeAPI:
    """Import the exact sealed source after private offline environment setup."""
    handlers = importlib.import_module("acestep.handler")
    inference = importlib.import_module("acestep.inference")
    return NativeAPI(
        handler=cast(HandlerFactory, cast(object, handlers.AceStepHandler)),
        constructors=native_constructors(),
        generate=cast(NativeGenerate, cast(object, inference.generate_music)),
    )


def native_constructors() -> NativeConstructors:
    """Import the two pinned dataclasses from the already verified runtime library."""
    module = importlib.import_module("acestep.inference")
    return NativeConstructors(
        parameters=cast(ParametersFactory, cast(object, module.GenerationParams)),
        configuration=cast(ConfigurationFactory, cast(object, module.GenerationConfig)),
    )


def tensor_samples(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """Expose a detached CPU float32 tensor after geometry/device checks upstream."""
    return cast(TensorArray, cast(object, tensor.detach())).numpy()
