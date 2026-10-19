"""Bound native jobs to their placed reservation using selected weight headers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final, final

import psutil
from pydantic import Field, NonNegativeInt, TypeAdapter, ValidationError

from skulk.shared.models.memory_estimate import (
    MEMORY_OVERHEAD_FACTOR,
    MEMORY_OVERHEAD_FLOOR,
    estimate_shard_footprint,
    gpu_working_set_ceiling,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.profiling import MemoryUsage
from skulk.utils.pydantic_ext import FrozenModel
from skulk.worker.runner.image_dimensions import image_dimensions
from skulk.worker.runner.mlx_video.plan import NativeVideoJob


@final
class _TensorHeader(FrozenModel):
    dtype: str = Field(description="Stored safetensors scalar representation.")
    shape: tuple[NonNegativeInt, ...] = Field(description="Declared tensor axes, without materializing their values.")
    data_offsets: tuple[NonNegativeInt, NonNegativeInt] = Field(description="Half-open byte range inside the weight payload.")


_RAW_HEADERS: Final = TypeAdapter(dict[str, object])
_HEADERS: Final = TypeAdapter(dict[str, _TensorHeader])
_MAXIMUM_HEADER_BYTES: Final = 16 * 1024**2


def _tensor_sizes(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"selected weight header is truncated: {path.name}")
        header_size = int.from_bytes(prefix, byteorder="little")
        if not 0 < header_size <= _MAXIMUM_HEADER_BYTES:
            raise ValueError(f"selected weight header exceeds its metadata bound: {path.name}")
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"selected weight header is truncated: {path.name}")
    payload_size = path.stat().st_size - 8 - header_size
    try:
        raw = _RAW_HEADERS.validate_json(header)
        # Converted packs can use null or other JSON values for this unused
        # member. Only tensor records participate in allocation accounting.
        raw.pop("__metadata__", None)
        if len(raw) > 100_000:
            raise ValueError("selected weights contain too many tensor declarations")
        tensors = _HEADERS.validate_json(json.dumps(raw))
    except ValidationError:
        # Do not turn thousands of invalid tensor declarations into an
        # unbounded task-error payload or replicated event-log entry.
        raise ValueError(f"selected weight header has invalid tensor metadata: {path.name}") from None
    sizes: dict[str, int] = {}
    for name, tensor in tensors.items():
        start, end = tensor.data_offsets
        if not 0 <= start <= end <= payload_size:
            raise ValueError(f"selected weight tensor exceeds its file: {path.name}")
        sizes[name] = end - start
    if not sizes:
        raise ValueError(f"selected weight file has no tensors: {path.name}")
    return sizes


@final
class NativeWeightProfile(FrozenModel):
    """Conservative resident weight groups derived without loading tensor payloads."""

    encoder_bytes: int = Field(ge=1, description="Complete text encoder and connector tensor bytes.")
    sampling_bytes: int = Field(ge=1, description="Non-block transformer, two streamed blocks and all other selected weights.")
    decoder_bytes: int = Field(ge=1, description="Selected video and audio decoder tensor bytes.")


def read_native_weight_profile(job: NativeVideoJob) -> NativeWeightProfile:
    """Read bounded selected-pack headers and conservatively group live weights.

    This opens local safetensors metadata only. It neither maps tensor payloads
    into MLX nor downloads components. Unknown block layouts fail before load.
    """
    root = Path(job.model_dir)
    if job.pipeline in {"dev_two_stage", "hq_two_stage"} and abs(job.refinement_strength - 1.0) <= 1e-6 and not (root / "transformer-distilled.safetensors").is_file():
        # The pinned low-RAM strength-one path swaps its streamer to this
        # pre-fused checkpoint. A LoRA alone cannot satisfy stage two.
        raise ValueError("the selected native two-stage pack is missing transformer-distilled.safetensors for strength-one refinement")
    transformer_name = "transformer-dev.safetensors" if job.transformer_variant == "dev" or job.pipeline in {"dev_two_stage", "hq_two_stage"} else "transformer-distilled.safetensors"
    transformer = _tensor_sizes(root / transformer_name)
    blocks: dict[int, int] = {}
    non_block_bytes = 0
    for name, size in transformer.items():
        match = re.search(r"(?:^|\.)transformer_blocks\.(\d+)\.", name)
        if match is None:
            non_block_bytes += size
        else:
            index = int(match.group(1))
            blocks[index] = blocks.get(index, 0) + size
    if not blocks or non_block_bytes <= 0:
        raise ValueError("cannot establish the selected native transformer's streamed weight layout")
    text = sum(_tensor_sizes(root / "text_encoder.safetensors").values())
    connector = sum(_tensor_sizes(root / "connector.safetensors").values())
    video_decoder_name = "vae_decoder_conv.safetensors" if job.decoder == "convolutional" else "vae_decoder_av.safetensors"
    video_decoder = sum(_tensor_sizes(root / video_decoder_name).values())
    audio = sum(_tensor_sizes(root / "audio_vae.safetensors").values()) + sum(_tensor_sizes(root / "vocoder.safetensors").values())
    # Charging all other selected weights also covers upscalers, source encoders
    # and unfused LoRAs without relying on a benchmark's particular workflow.
    remaining = sum(
        sum(_tensor_sizes(path).values())
        for path in root.glob("*.safetensors")
        if path.name not in {transformer_name, "text_encoder.safetensors", "connector.safetensors"}
    )
    return NativeWeightProfile(
        encoder_bytes=text + connector,
        sampling_bytes=non_block_bytes + 2 * max(blocks.values()) + remaining,
        decoder_bytes=video_decoder + (audio if job.audio else 0),
    )


@final
class NativeMemoryPlan(FrozenModel):
    """One request's checked working set and controlled decoder activation budget."""

    reservation_bytes: int = Field(ge=1, description="Memory charged consistently by placement and the worker.")
    estimated_peak_bytes: int = Field(ge=1, description="Conservative weights, geometry, attention and input workspace estimate.")
    decode_budget_bytes: int = Field(ge=1, description="Owned value supplied to the pinned upstream decoder's budget control.")


def read_native_image_pixels(job: NativeVideoJob) -> tuple[int, ...]:
    """Read bounded source-image headers without decoding pixel buffers.

    Returns each selected anchor's actual pixel count. An unreadable/unknown
    shape fails before model loading instead of assuming output-sized inputs.
    """
    pixels: list[int] = []
    for anchor in job.images:
        shape = image_dimensions(Path(anchor.path))
        if shape is None:
            raise ValueError("cannot determine the source image dimensions safely; convert the image and retry")
        pixels.append(shape[0] * shape[1])
    return tuple(pixels)


def _stage_peak(weights: int, workspace: int) -> int:
    # The common 30% multiplier already reserves workspace. Charge a larger
    # explicit request workspace instead of adding the same overhead twice.
    return max(int(weights * MEMORY_OVERHEAD_FACTOR), weights + workspace) + MEMORY_OVERHEAD_FLOOR.in_bytes


def plan_native_memory(
    job: NativeVideoJob, card: ModelCard, weights: NativeWeightProfile, memory: MemoryUsage,
    *, image_pixels: tuple[int, ...] = (),
) -> NativeMemoryPlan:
    """Check a request against the placement reservation and current local capacity.

    Canvas/duration changes may need a larger explicit card runtime reservation;
    they never become intrinsic model limits or hardware-qualification gates.
    The automatic-duration frame count is already the requested allocation bound.
    """
    reservation = estimate_shard_footprint(card, 1.0, resolved_backend="mlx_video-metal").in_bytes
    usable = min(memory.ram_available, gpu_working_set_ceiling(memory.ram_total)).in_bytes
    if reservation > usable:
        raise ValueError(f"Native video needs its {reservation / 1024**3:.2f} GiB reservation but only {usable / 1024**3:.2f} GiB is currently usable; free memory and retry")
    pixels = job.width * job.height
    pixel_frames = pixels * job.frame_count
    source_frames = job.source_media.frames if job.source_media is not None else 0
    source_workspace = source_frames * pixels * 3 * 4 * 2
    if len(image_pixels) != len(job.images) or any(value <= 0 for value in image_pixels):
        raise ValueError("native video memory admission requires verified source image dimensions")
    image_workspace = sum(max(value, pixels) for value in image_pixels) * 3 * 4 * 2
    generated_planes = (job.frame_count - 2) // job.generated_keyframe_interval if job.generated_keyframe_interval is not None else 0
    latent_planes = (job.frame_count - 1) // 8 + 1 + len(job.images) + generated_planes
    video_tokens = latent_planes * (job.height // 32) * (job.width // 32)
    # A dense attention upper bound avoids assuming every GPU kernel uses the
    # same fused attention path. Audio frequency channels are already flattened
    # into each 128-channel token by the pinned pipeline's audio_shape.
    # Muting output skips the audio decoder, not joint audio-latent sampling.
    audio_tokens = (job.frame_count * 25 + job.fps - 1) // job.fps
    tokens = video_tokens + audio_tokens
    # Reserve fp32 scores/probabilities/workspace, plus cross-attention to the
    # pinned 1024-token text context. Inherited context tuning is stripped.
    attention_workspace = tokens * tokens * 32 * 4 * 3 + tokens * 1024 * 32 * 4 * 2
    linear_workspace = max(pixel_frames * 128, tokens * card.hidden_size * 2 * 128)
    guidance_passes = 3 if job.pipeline in {"dev_two_stage", "hq_two_stage"} or job.transformer_variant == "dev" else 1
    workspace = guidance_passes * (attention_workspace + linear_workspace) + source_workspace + image_workspace
    estimated = max(
        _stage_peak(weights.encoder_bytes, source_workspace + image_workspace),
        _stage_peak(weights.sampling_bytes, workspace),
    )
    if estimated > reservation:
        raise ValueError(f"This native video request needs an estimated {estimated / 1024**3:.2f} GiB but its placement reserves {reservation / 1024**3:.2f} GiB; reduce duration/resolution or increase the video memory reservation before placing again")
    remaining = reservation - _stage_peak(weights.decoder_bytes, source_workspace + image_workspace)
    decode_budget = min(memory.ram_total.in_bytes // 2, int(remaining / MEMORY_OVERHEAD_FACTOR))
    if decode_budget <= 0:
        raise ValueError("the native placement has no remaining video decode workspace")
    if job.decoder == "convolutional":
        # The pinned conv tiler otherwise warns and runs its smallest rung even
        # when no rung fits. Refuse that unsafe path before loading any weights.
        latent_height, latent_width = job.height // 32, job.width // 32
        long_side = max(latent_height, latent_width)
        tile_height = min(latent_height, max(2, round(8 * latent_height / long_side))) * 32
        tile_width = min(latent_width, max(2, round(8 * latent_width / long_side))) * 32
        tile_frames = min(job.frame_count, 16)
        minimum_decode = tile_frames * (750 * tile_height * tile_width + pixels * 48)
        if minimum_decode > decode_budget:
            raise ValueError("the smallest native video decode tile exceeds this placement's memory reservation; reduce resolution or increase the video memory reservation")
    return NativeMemoryPlan(reservation_bytes=reservation, estimated_peak_bytes=estimated, decode_budget_bytes=decode_budget)


def native_memory_pressure_error(process_id: int, reservation_bytes: int) -> str | None:
    """Return a live host/RSS guard refusal, without adding overlapping counters.

    This is a pressure backstop, not a hard Metal allocator quota. It must be
    paired with preflight workspace admission and controlled decoder tiling.
    """
    try:
        resident = psutil.Process(process_id).memory_info().rss
    except psutil.NoSuchProcess:
        return None
    if resident > reservation_bytes:
        return "native video process exceeded its reserved resident memory"
    memory = MemoryUsage.from_local_gpu_wireable()
    if memory.ram_available.in_bytes < memory.ram_total.in_bytes * 0.08:
        return "native video stopped because usable system memory fell below 8 percent"
    return None
