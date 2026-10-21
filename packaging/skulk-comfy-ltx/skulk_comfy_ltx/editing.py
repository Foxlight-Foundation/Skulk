"""Causal temporal masks and source preservation for the owned LTX nodes.

Sampling masks operate on each VAE's own clock. Decoded frames and normalized
PCM are restored separately because a frozen latent still has a lossy decoder.
All helpers return new tensors and leave Comfy's cached inputs untouched.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import copy
from typing import Literal, Protocol, TypedDict

import torch
from torch import Tensor

Modality = Literal["video", "audio"]
Operation = Literal["retake", "extend"]
AudioResampler = Callable[[Tensor, int, int], Tensor]


class AudioWave(TypedDict):
    """Comfy's decoded waveform, with an explicit sample clock."""

    waveform: Tensor
    sample_rate: int


class AudioClock(Protocol):
    """Time-scale properties supplied by the selected LTX audio VAE."""

    sample_rate: int
    mel_hop_length: int
    latents_per_second: float


class AudioVae(Protocol):
    """The selected Comfy audio VAE's clock, without importing its implementation."""

    first_stage_model: AudioClock
    crop_input: bool


def uncropped_audio_vae[AudioVaeType: AudioVae](audio_vae: AudioVaeType) -> AudioVaeType:
    """Return an encoder view preserving the complete normalized source waveform.

    Share the selected weights and placement, while disabling the generic
    spatial crop on this view. The cached VAE's configuration stays unchanged.
    """
    # The pinned Comfy VAE applies its 4096-sample spatial crop to audio's
    # time axis, dropping context at both ends before the causal encoder.
    # A shallow view retains model management without mutating cached inputs.
    encoder = copy(audio_vae)
    encoder.crop_input = False
    return encoder


def _validate_tensor(value: Tensor, dimensions: int) -> None:
    if value.is_nested or value.ndim != dimensions or min(value.shape) < 1:
        raise ValueError("editing requires the explicit nonempty modality layout")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("editing requires finite floating-point tensors")


def _region(start_seconds: float, end_seconds: float) -> None:
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds) or not 0 <= start_seconds < end_seconds <= 121:
        raise ValueError("retake requires a finite increasing time interval")


def temporal_mask(
    samples: Tensor, *, modality: Modality, start_seconds: float,
    end_seconds: float, fps: int, sample_rate: int = 16000,
    hop_length: int = 160, downsample_factor: int = 4,
) -> Tensor:
    """Select causal VAE intervals overlapping a retake in real seconds.

    Video uses B,C,T,H,W with a one-frame first token followed by eight-frame
    intervals. Audio uses B,C,T,F and its selected VAE's causal hop clock. The
    returned mask broadcasts across channels and space without altering input.
    """
    dimensions = 5 if modality == "video" else 4
    _validate_tensor(samples, dimensions)
    _region(start_seconds, end_seconds)
    if min(fps, sample_rate, hop_length, downsample_factor) < 1:
        raise ValueError("editing time scales must be positive")
    indices = torch.arange(samples.shape[2], device=samples.device, dtype=torch.float32)
    scale = 8 if modality == "video" else downsample_factor
    # Match the reference patchifiers' FP32 coordinate construction and
    # operation order, including comparisons exactly on a causal boundary.
    starts = (indices * scale + 1 - scale).clamp(min=0)
    ends = ((indices + 1) * scale + 1 - scale).clamp(min=0)
    if modality == "video":
        starts, ends = starts / fps, ends / fps
    else:
        starts, ends = starts * hop_length / sample_rate, ends * hop_length / sample_rate
    selected = (ends > start_seconds) & (starts < end_seconds)
    return selected.to(samples.dtype).reshape(1, 1, samples.shape[2], *([1] * (dimensions - 3)))


def edit_latent(
    source: Tensor, target: Tensor, *, modality: Modality, operation: Operation,
    start_seconds: float, end_seconds: float, fps: int,
    regenerate_audio: bool = True, sample_rate: int = 16000,
    hop_length: int = 160, downsample_factor: int = 4,
) -> tuple[Tensor, Tensor]:
    """Copy source context and return the exact sampling mask for one modality.

    Retake changes only overlapping temporal tokens. Extension freezes the entire
    encoded source prefix, including at spatial refinement. Audio may differ by
    one final token because waveform rounding and the VAE canvas use different
    grids. Keep the encoded source's clock rather than dropping its causal tail;
    any larger retake mismatch is refused rather than silently retimed.
    """
    dimensions = 5 if modality == "video" else 4
    _validate_tensor(source, dimensions)
    _validate_tensor(target, dimensions)
    if source.shape[:2] != target.shape[:2] or source.shape[3:] != target.shape[3:]:
        raise ValueError("source and target latent channels and spatial geometry must match")
    source = source.to(device=target.device, dtype=target.dtype)
    shape = (1, 1, target.shape[2], *([1] * (dimensions - 3)))
    if operation == "extend":
        if source.shape[2] >= target.shape[2]:
            raise ValueError("extension must exceed its complete source prefix")
        output = target.clone()
        output[:, :, :source.shape[2]] = source
        mask = torch.ones(shape, device=target.device, dtype=target.dtype)
        mask[:, :, :source.shape[2]] = 0
        return output, mask
    if source.shape[2] != target.shape[2] and (modality != "audio" or abs(source.shape[2] - target.shape[2]) > 1):
        raise ValueError("retake source must cover the complete target timeline")
    # A retake uses the source encoder's audio length, as the reference
    # pipeline does. Forcing it onto the empty generation canvas can drop
    # the last causal token and prevent a retake near the clip's end.
    output = source.clone()
    mask = temporal_mask(output, modality=modality, start_seconds=start_seconds, end_seconds=end_seconds, fps=fps, sample_rate=sample_rate, hop_length=hop_length, downsample_factor=downsample_factor)
    if modality == "audio" and not regenerate_audio:
        mask = torch.zeros_like(mask)
    return output, mask


def normalize_source_audio(
    source: AudioWave | None, *, frames: int, fps: int, sample_rate: int,
    resample: AudioResampler,
) -> AudioWave:
    """Normalize bounded source PCM to the declared output clock and stereo.

    A missing track supplies silence as source context. At most 50 ms of missing
    codec tail is padded; a longer shortfall is a malformed source. Extra codec
    padding is cropped to the verified source-video frame count.
    """
    if min(frames, fps) < 1 or fps > 1000 or not 8000 <= sample_rate <= 192000 or frames / fps > 121:
        raise ValueError("source audio needs a bounded positive video timeline")
    needed = math.ceil(frames * sample_rate / fps)
    if source is None:
        return AudioWave(waveform=torch.zeros((1, 2, needed)), sample_rate=sample_rate)
    waveform = source["waveform"]
    _validate_tensor(waveform, 3)
    if waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or not 8000 <= source["sample_rate"] <= 192000:
        raise ValueError("editing source audio requires mono or stereo at 8-192 kHz")
    if source["sample_rate"] != sample_rate:
        waveform = resample(waveform, source["sample_rate"], sample_rate)
    _validate_tensor(waveform, 3)
    if waveform.shape[1] == 1:
        waveform = waveform.expand(-1, 2, -1)
    if waveform.shape[-1] < needed - math.ceil(0.05 * sample_rate):
        raise ValueError("source audio does not cover the verified video timeline")
    if waveform.shape[-1] < needed:
        waveform = torch.nn.functional.pad(waveform, (0, needed - waveform.shape[-1]))
    return AudioWave(waveform=waveform[..., :needed].clone(), sample_rate=sample_rate)


def preserve_video(
    source: Tensor, generated: Tensor, *, operation: Operation, fps: int,
    start_seconds: float, end_seconds: float,
) -> Tensor:
    """Restore unchanged decoded source frames before lossy container encoding."""
    _validate_tensor(source, 4)
    _validate_tensor(generated, 4)
    if source.shape[1:] != generated.shape[1:] or fps < 1:
        raise ValueError("source and generated frame geometry must match")
    output = generated.clone()
    source = source.to(device=output.device, dtype=output.dtype)
    if operation == "extend":
        if source.shape[0] >= output.shape[0]:
            raise ValueError("extension must exceed its source frames")
        output[:source.shape[0]] = source
    else:
        _region(start_seconds, end_seconds)
        if source.shape[0] != output.shape[0] or end_seconds > source.shape[0] / fps:
            raise ValueError("retake source and interval must match the target duration")
        beginning, end = math.floor(start_seconds * fps), math.ceil(end_seconds * fps)
        output[:beginning] = source[:beginning]
        output[end:] = source[end:]
    return output


def preserve_audio(
    source: AudioWave, generated: AudioWave, *, operation: Operation,
    output_frames: int, fps: int, start_seconds: float, end_seconds: float,
    regenerate_audio: bool,
) -> AudioWave:
    """Restore normalized source PCM outside a retake or through an extension prefix."""
    waveform, original = generated["waveform"], source["waveform"]
    _validate_tensor(waveform, 3)
    _validate_tensor(original, 3)
    rate = generated["sample_rate"]
    if rate != source["sample_rate"] or rate < 1 or fps < 1 or output_frames < 1 or waveform.shape[:2] != original.shape[:2]:
        raise ValueError("source and generated audio clocks and channels must match")
    needed = math.ceil(output_frames * rate / fps)
    if waveform.shape[-1] < needed - math.ceil(0.05 * rate):
        raise ValueError("generated audio does not cover the output timeline")
    output = torch.nn.functional.pad(waveform, (0, max(0, needed - waveform.shape[-1])))[..., :needed].clone()
    original = original.to(device=output.device, dtype=output.dtype)
    if operation == "extend":
        if original.shape[-1] >= needed:
            raise ValueError("extension audio must exceed its source prefix")
        output[..., :original.shape[-1]] = original
    else:
        _region(start_seconds, end_seconds)
        if original.shape[-1] != needed or end_seconds > output_frames / fps:
            raise ValueError("retake PCM and interval must match the output timeline")
        if not regenerate_audio:
            return AudioWave(waveform=original.clone(), sample_rate=rate)
        beginning, end = math.floor(start_seconds * rate), math.ceil(end_seconds * rate)
        output[..., :beginning] = original[..., :beginning]
        output[..., end:] = original[..., end:]
    return AudioWave(waveform=output, sample_rate=rate)
