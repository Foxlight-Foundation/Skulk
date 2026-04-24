import contextlib
import functools
import itertools
import math
import os
import sys
import threading
import time
import traceback
import types
from copy import deepcopy
from importlib import import_module, metadata
from typing import Callable, Generator, Protocol, cast, get_args

import mlx.core as mx
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models.cache import ArraysCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from packaging.version import InvalidVersion, Version

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.constants import preferred_env_value
from exo.shared.models.model_cards import ModelCard
from exo.shared.tracing import TraceAttrValue, record_trace_marker, trace
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    flush_prefill_sends,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_CACHE_BACKEND,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    log_request_shape,
    mx_barrier,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    get_inner_model,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.diagnostics import record_runner_phase, runner_phase

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5
_MIN_CANCEL_CHECK_INTERVAL = 10


class _FrameLookup(Protocol):
    """Typed callable surface for CPython's private frame lookup helper."""

    def __call__(self) -> dict[int, types.FrameType]: ...


class _DetokenizerProtocol(Protocol):
    """Minimal detokenizer surface used by native-vision generation."""

    last_segment: str

    def reset(self) -> None: ...

    def add_token(self, token: int) -> None: ...

    def finalize(self) -> None: ...


class _VlmGenerateStep(Protocol):
    """Typed callable surface for mlx-vlm's multimodal generate step."""

    def __call__(
        self,
        *,
        input_ids: mx.array,
        model: Model,
        pixel_values: mx.array | list[mx.array],
        mask: object,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
        prefill_step_size: int,
        kv_group_size: int,
        kv_bits: int | None,
    ) -> Generator[tuple[mx.array, mx.array], None, None]: ...


def _current_frames() -> dict[int, types.FrameType]:
    """Return the current Python frames when the interpreter exposes them."""
    lookup = cast(_FrameLookup | None, getattr(sys, "_current_frames", None))
    return lookup() if lookup is not None else {}


def _mlx_hang_debug_enabled() -> bool:
    """Return whether verbose warmup/prefill hang diagnostics are enabled."""
    value = preferred_env_value("SKULK_MLX_HANG_DEBUG", "EXO_MLX_HANG_DEBUG")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _mlx_hang_debug_interval_seconds() -> float:
    """Return the periodic interval for hang-debug watchdog logs."""
    raw = preferred_env_value(
        "SKULK_MLX_HANG_DEBUG_INTERVAL_SECONDS",
        "EXO_MLX_HANG_DEBUG_INTERVAL_SECONDS",
    )
    if raw is None:
        return 30.0
    with contextlib.suppress(ValueError):
        return max(float(raw), 1.0)
    return 30.0


def _warmup_repeat_count() -> int:
    """Return the neutral warmup token repeat count used for debugging."""
    raw = preferred_env_value(
        "SKULK_DEBUG_WARMUP_REPEAT_COUNT",
        "EXO_DEBUG_WARMUP_REPEAT_COUNT",
    )
    if raw is None:
        return 1
    with contextlib.suppress(ValueError):
        return max(int(raw), 1)
    return 1


def _is_distributed_warmup(group: mx.distributed.Group | None) -> bool:
    """Return whether warmup is running with a multi-node distributed group."""
    return group is not None and group.size() > 1


def _warmup_user_content(group: mx.distributed.Group | None) -> str:
    """Return the synthetic warmup user content.

    Distributed pipeline warmup intentionally stays minimal because richer
    synthetic prompts have been observed to wedge stream_generate prefill.
    Single-node debugging may still scale the neutral content via environment.
    """
    if _is_distributed_warmup(group):
        return "hello"
    return " ".join(["hello"] * _warmup_repeat_count())


def _warmup_instructions(group: mx.distributed.Group | None) -> str | None:
    """Return optional warmup instructions for prompt-shape debugging."""
    if _is_distributed_warmup(group):
        return None
    raw = preferred_env_value(
        "SKULK_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS",
        "EXO_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS",
    )
    if raw is None:
        return None
    include = raw.strip().lower() not in {"", "0", "false", "no", "off"}
    if not include:
        return None
    return "You are a helpful assistant. Answer the user in one short sentence."


@contextlib.contextmanager
def _hang_debug_watch(label: str) -> Generator[None]:
    """Emit periodic stack-rich logs while the current thread is stuck in one phase."""
    if not _mlx_hang_debug_enabled():
        yield
        return

    interval_seconds = _mlx_hang_debug_interval_seconds()
    started_at = time.monotonic()
    monitored_thread_id = threading.get_ident()
    finished = threading.Event()

    logger.info(
        f"[hang-debug] Entering {label} (watchdog interval={interval_seconds:.0f}s)"
    )

    def watchdog() -> None:
        while not finished.wait(timeout=interval_seconds):
            elapsed = time.monotonic() - started_at
            frame = _current_frames().get(monitored_thread_id)
            if frame is None:
                stack_text = "<no Python frame available>"
            else:
                stack_text = "".join(traceback.format_stack(frame))
            logger.warning(
                f"[hang-debug] Still in {label} after {elapsed:.1f}s\n{stack_text}"
            )

    watchdog_thread = threading.Thread(
        target=watchdog,
        name=f"hang-debug:{label}",
        daemon=True,
    )
    watchdog_thread.start()

    try:
        yield
    finally:
        finished.set()
        elapsed = time.monotonic() - started_at
        logger.info(f"[hang-debug] Leaving {label} after {elapsed:.1f}s")


def _should_use_native_vision_reference_path() -> bool:
    """Return whether native vision should force MLX-VLM's reference decode path.

    The reference path was needed to work around older upstream MLX Gemma 4
    vision behavior, but it bypasses Skulk's faster pipeline-aware generation
    path. Once the runtime is on the fixed upstream stack, we prefer the legacy
    Skulk path again for throughput.
    """
    override = os.environ.get("EXO_NATIVE_VISION_REFERENCE_PATH")
    if override is not None:
        normalized = override.strip().lower()
        return normalized not in {"0", "false", "no", "off"}

    try:
        mlx_version = Version(metadata.version("mlx"))
        mlx_vlm_version = Version(metadata.version("mlx-vlm"))
    except (metadata.PackageNotFoundError, InvalidVersion):
        # Be conservative if metadata is unavailable.
        return True

    return not (
        mlx_version >= Version("0.31.1") and mlx_vlm_version >= Version("0.4.4")
    )


def _native_pixel_values_debug_state(
    pixel_values: mx.array | list[mx.array] | None,
) -> str:
    """Return a compact string describing native-vision pixel injection state."""
    if pixel_values is None:
        return "fully_cached"
    if isinstance(pixel_values, list):
        return f"list[{len(pixel_values)}]"
    return f"array{tuple(pixel_values.shape)}"


def _native_pixel_values_attrs(
    pixel_values: mx.array | list[mx.array] | None,
) -> dict[str, object]:
    """Return bounded JSON-safe diagnostic attrs for native pixel tensors."""

    if pixel_values is None:
        return {"pixel_values": "none", "pixel_value_count": 0}
    if isinstance(pixel_values, list):
        return {
            "pixel_values": "list",
            "pixel_value_count": len(pixel_values),
            "pixel_value_shapes": [str(tuple(value.shape)) for value in pixel_values],
        }
    return {
        "pixel_values": "array",
        "pixel_value_count": int(pixel_values.shape[0]) if pixel_values.ndim else 1,
        "pixel_value_shapes": [str(tuple(pixel_values.shape))],
    }


def _media_region_attrs(media_regions: list[MediaRegion]) -> dict[str, object]:
    """Return compact diagnostic attrs for prompt media regions."""

    return {
        "media_region_count": len(media_regions),
        "media_region_lengths": [
            str(region.end_pos - region.start_pos) for region in media_regions
        ],
        "media_region_hashes": [
            region.content_hash[:12] for region in media_regions
        ],
    }


def _vision_trace_attrs(vision: VisionResult | None) -> dict[str, TraceAttrValue]:
    """Return compact trace attrs for a prepared multimodal request."""
    if vision is None or vision.debug_info is None:
        return {"vision_used": vision is not None}
    return {
        "vision_used": True,
        **vision.debug_info.attrs(),
    }


def _native_pixel_values_trace_attrs(
    pixel_values: mx.array | list[mx.array] | None,
) -> dict[str, TraceAttrValue]:
    """Return native pixel-value attrs typed for saved trace markers."""
    return cast(dict[str, TraceAttrValue], _native_pixel_values_attrs(pixel_values))


def _decode_debug_context(
    *,
    task: TextGenerationTaskParams,
    group: mx.distributed.Group | None,
    trace_task_id: str | None,
    total_prompt_tokens: int,
    uncached_prompt_tokens: int,
    prefix_hit_length: int,
    media_region_count: int,
    is_native_vision: bool,
    native_pixel_values: mx.array | list[mx.array] | None,
) -> str:
    """Summarize the request shape around the decode handoff for hang debugging."""
    rank = group.rank() if group is not None else 0
    group_size = group.size() if group is not None else 1
    return (
        f"task_id={trace_task_id or '<unknown>'}, "
        f"model={task.model}, "
        f"rank={rank}, "
        f"group_size={group_size}, "
        f"total_prompt_tokens={total_prompt_tokens}, "
        f"uncached_prompt_tokens={uncached_prompt_tokens}, "
        f"prefix_hit_length={prefix_hit_length}, "
        f"media_regions={media_region_count}, "
        f"native_vision={is_native_vision}, "
        f"native_pixel_values={_native_pixel_values_debug_state(native_pixel_values)}"
    )


def _should_force_native_vision_full_prefill_for_request(
    *,
    group: mx.distributed.Group | None,
    is_native_vision: bool,
    prefix_hit_length: int,
    media_region_count: int,
    native_pixel_values: mx.array | list[mx.array] | None,
) -> bool:
    """Return whether a distributed native-vision request should skip prefix cache.

    Distributed Gemma 4 follow-up turns with cached image prefixes can wedge
    during decode after the pipeline-aware path trims native pixel values to the
    uncached suffix. The upstream MLX-VLM reference path also crashes for this
    shape because its prompt cache contains uninitialized entries, so the stable
    fallback is to keep Skulk's distributed path but re-prefill the full prompt
    with every image tensor.
    """
    has_cached_multimodal_prefix = prefix_hit_length > 0 and media_region_count > 1
    has_fully_cached_native_images = prefix_hit_length > 0 and native_pixel_values is None
    return (
        is_native_vision
        and group is not None
        and group.size() > 1
        and (has_cached_multimodal_prefix or has_fully_cached_native_images)
    )


def _slice_native_pixel_values_for_uncached_suffix(
    pixel_values: mx.array | list[mx.array],
    media_regions: list[MediaRegion],
    prefix_hit_length: int,
) -> mx.array | list[mx.array] | None:
    """Keep only native vision pixel values still referenced by uncached tokens.

    Prefix-cache hits can reuse earlier image regions from the KV cache. Native
    vision models consume raw pixel values in prompt order, so after a prefix
    hit we must drop any already-cached images from ``pixel_values`` or the
    first stale image will be paired with the next uncached image token span.
    """
    if prefix_hit_length <= 0 or not media_regions:
        return pixel_values

    available_images = (
        len(pixel_values) if isinstance(pixel_values, list) else int(pixel_values.shape[0])
    )
    remaining_indices = [
        idx
        for idx, region in enumerate(media_regions)
        if idx < available_images and region.end_pos > prefix_hit_length
    ]
    if not remaining_indices:
        logger.info(
            "Native vision prefix cache hit reused all image regions; "
            "skipping pixel-value injection for cached images"
        )
        return None

    if available_images != len(media_regions):
        logger.warning(
            "Native vision pixel_values/media_regions length mismatch: "
            f"{available_images} image tensor(s) for {len(media_regions)} media region(s)"
        )

    if remaining_indices == list(range(available_images)):
        return pixel_values

    logger.info(
        "Native vision prefix cache hit trimmed pixel_values from "
        f"{available_images} to {len(remaining_indices)} image(s) "
        f"(restore_pos={prefix_hit_length})"
    )

    if isinstance(pixel_values, list):
        return [pixel_values[idx] for idx in remaining_indices]

    first_idx = remaining_indices[0]
    expected_suffix = list(range(first_idx, available_images))
    if remaining_indices == expected_suffix:
        return pixel_values[first_idx:]

    return mx.stack([pixel_values[idx] for idx in remaining_indices], axis=0)


def _native_media_regions_for_uncached_suffix(
    pixel_values: mx.array | list[mx.array],
    media_regions: list[MediaRegion],
    prefix_hit_length: int,
) -> list[MediaRegion]:
    """Return media regions aligned with cache-trimmed native pixel values."""
    if prefix_hit_length <= 0 or not media_regions:
        return media_regions

    available_images = (
        len(pixel_values) if isinstance(pixel_values, list) else int(pixel_values.shape[0])
    )
    return [
        region
        for idx, region in enumerate(media_regions)
        if idx < available_images and region.end_pos > prefix_hit_length
    ]


def slice_native_pixel_values_for_uncached_suffix(
    pixel_values: mx.array | list[mx.array],
    media_regions: list[MediaRegion],
    prefix_hit_length: int,
) -> mx.array | list[mx.array] | None:
    """Public wrapper for native-vision pixel-value trimming.

    Batch and single-request generators both need the same cache-aware image
    slicing behavior, so this helper is intentionally shared across modules.
    """
    return _slice_native_pixel_values_for_uncached_suffix(
        pixel_values,
        media_regions,
        prefix_hit_length,
    )


def _slice_native_pixel_values_by_indices(
    pixel_values: mx.array | list[mx.array],
    indices: list[int],
) -> mx.array | list[mx.array] | None:
    """Select native pixel values by media-region order."""
    if not indices:
        return None

    available_images = (
        len(pixel_values) if isinstance(pixel_values, list) else int(pixel_values.shape[0])
    )
    valid_indices = [idx for idx in indices if idx < available_images]
    if not valid_indices:
        return None

    if isinstance(pixel_values, list):
        return [pixel_values[idx] for idx in valid_indices]

    if valid_indices == list(range(valid_indices[0], valid_indices[-1] + 1)):
        return pixel_values[valid_indices[0] : valid_indices[-1] + 1]

    return mx.stack([pixel_values[idx] for idx in valid_indices], axis=0)


def _native_pixel_values_for_media_range(
    pixel_values: mx.array | list[mx.array],
    media_regions: list[MediaRegion],
    start_pos: int,
    end_pos: int,
) -> mx.array | list[mx.array] | None:
    """Return pixel values whose media spans overlap a prompt token range."""
    region_indices = [
        idx
        for idx, region in enumerate(media_regions)
        if region.end_pos > start_pos and region.start_pos < end_pos
    ]
    return _slice_native_pixel_values_by_indices(pixel_values, region_indices)


def _vision_safe_prefill_chunk_sizes(
    total_tokens: int,
    max_chunk_size: int,
    media_regions: list[MediaRegion] | None = None,
    prompt_token_offset: int = 0,
) -> list[int]:
    """Split prefill tokens without cutting through native image spans."""
    real_prefill_tokens = max(total_tokens - 1, 0)
    if real_prefill_tokens == 0:
        return []

    if max_chunk_size <= 0:
        return [real_prefill_tokens]

    sorted_regions = sorted(media_regions or [], key=lambda region: region.start_pos)
    chunk_sizes: list[int] = []
    start = 0
    while start < real_prefill_tokens:
        end = min(start + max_chunk_size, real_prefill_tokens)
        full_start = prompt_token_offset + start
        full_end = prompt_token_offset + end

        for region in sorted_regions:
            if region.end_pos <= full_start:
                continue
            if region.start_pos >= full_end:
                break
            if region.start_pos < full_end < region.end_pos:
                end = min(region.end_pos - prompt_token_offset, real_prefill_tokens)
                full_end = prompt_token_offset + end
                break

        if end <= start:
            end = min(start + max_chunk_size, real_prefill_tokens)
        chunk_sizes.append(end - start)
        start = end

    return chunk_sizes


def _set_native_pixel_values(
    model: Model,
    pixel_values: mx.array | list[mx.array] | None,
) -> None:
    """Set native vision tensors on wrappers that inject them into prefill calls."""
    if hasattr(model, "set_pixel_values"):
        cast(
            Callable[[mx.array | list[mx.array] | None], None],
            object.__getattribute__(model, "set_pixel_values"),
        )(pixel_values)
    else:
        object.__setattr__(model, "_pixel_values", pixel_values)


@contextlib.contextmanager
def patch_embed_tokens(
    model: Model, embeddings: mx.array, start_offset: int = 0, token_count: int = 0
) -> Generator[None]:
    inner = get_inner_model(model)  # type: ignore
    original_embed = inner.embed_tokens  # type: ignore
    end_offset = start_offset + token_count
    offset = [start_offset]

    def _inject(input_ids: mx.array) -> mx.array:
        start = offset[0]
        if start >= end_offset:
            return original_embed(input_ids)  # type: ignore
        chunk_len = input_ids.shape[-1]
        end = min(start + chunk_len, end_offset)
        offset[0] = start + chunk_len
        vision_len = end - start
        if vision_len == chunk_len:
            return embeddings[:, start:end, :]
        # Partial overlap: splice vision embeddings for the covered portion
        # and fall back to text embeddings for the remainder, so image tokens
        # at chunk boundaries still get correct vision features.
        text_embeds: mx.array = original_embed(input_ids)  # type: ignore
        vision_slice = embeddings[:, start:end, :]
        text_embeds[:, :vision_len, :] = vision_slice
        return text_embeds

    for attr in dir(original_embed):  # type: ignore
        if not attr.startswith("_") and not hasattr(_inject, attr):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(_inject, attr, getattr(original_embed, attr))  # type: ignore

    inner.embed_tokens = _inject
    try:
        yield
    finally:
        inner.embed_tokens = original_embed


class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


def _noop_quantize_cache(_cache: KVCacheType) -> None:
    return None


def _has_pipeline_communication_layer(model: Model):
    for layer in model.layers:
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            return True
    return False


def pipeline_parallel_prefill(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
    distributed_prompt_progress_callback: Callable[[], None] | None,
    group: mx.distributed.Group,
    native_pixel_values: mx.array | list[mx.array] | None = None,
    native_media_regions: list[MediaRegion] | None = None,
    prompt_token_offset: int = 0,
) -> None:
    """Prefill the KV cache for pipeline parallel with overlapping stages.

    Each rank processes the full prompt through its real cache, offset by leading
    and trailing dummy iterations.

    Total iterations per rank = N_real_chunks + world_size - 1:
      - rank r leading dummies  (skip_pipeline_io, throwaway cache)
      - N_real_chunks real      (pipeline IO active, real cache)
      - (world_size-1-r) trailing dummies (skip_pipeline_io, throwaway cache)

    e.g.
    Timeline (2 ranks, 3 chunks of 10240 tokens @ step=4096):
        iter 0: R0 real[0:4096]     R1 dummy
        iter 1: R0 real[4096:8192]  R1 real[0:4096]
        iter 2: R0 real[8192:10240] R1 real[4096:8192]
        iter 3: R0 dummy            R1 real[8192:10240]

    This function is designed to match mlx_lm's stream_generate exactly in terms of
    side effects (given the same prefill step size)
    """
    prefill_step_size = prefill_step_size // min(4, group.size())

    quantize_cache_fn: Callable[[KVCacheType], None]
    if KV_CACHE_BACKEND == "mlx_quantized":
        quantize_cache_fn = cast(
            Callable[[KVCacheType], None],
            functools.partial(
                maybe_quantize_kv_cache,
                quantized_kv_start=0,
                kv_group_size=kv_group_size,
                kv_bits=kv_bits,
            ),
        )
    else:
        quantize_cache_fn = _noop_quantize_cache

    _prompt_cache: KVCacheType = prompt_cache
    rank = group.rank()
    world_size = group.size()

    total = len(prompt)
    real_chunk_sizes = _vision_safe_prefill_chunk_sizes(
        total,
        prefill_step_size,
        native_media_regions if native_pixel_values is not None else None,
        prompt_token_offset,
    )
    n_real = len(real_chunk_sizes)

    # Each rank does: [rank leading dummies] [N real chunks] [world_size-1-rank trailing dummies]
    n_leading = rank
    n_trailing = world_size - 1 - rank
    n_total = n_leading + n_real + n_trailing

    t_start = time.perf_counter()
    processed = 0
    logger.info(
        f"[R{rank}] Pipeline prefill: {n_real} real + {n_leading} leading + {n_trailing} trailing = {n_total} iterations"
    )
    record_runner_phase(
        "prefill_pipeline",
        event="pipeline_prefill_start",
        attrs={
            "rank": rank,
            "world_size": world_size,
            "real_chunks": n_real,
            "leading_dummies": n_leading,
            "trailing_dummies": n_trailing,
            "prompt_tokens": total,
        },
        include_memory=True,
    )
    clear_prefill_sends()

    # Initial callback matching generate_step
    prompt_progress_callback(0, total)

    try:
        with mx.stream(generation_stream):
            for _ in range(n_leading):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                chunk_start = processed
                chunk_end = processed + chunk_size
                if i == 0 or i == n_real - 1 or i % 4 == 0:
                    record_runner_phase(
                        "prefill_pipeline",
                        event="pipeline_prefill_chunk",
                        attrs={
                            "rank": rank,
                            "chunk_index": i,
                            "chunk_size": chunk_size,
                            "processed": processed,
                            "total": total,
                        },
                    )
                if native_pixel_values is not None and native_media_regions is not None:
                    # Gemma 4 scatters image features from the start of the
                    # provided tensor on every forward call. Pipeline chunks
                    # therefore need chunk-local pixel values, otherwise a
                    # later image span can be paired with image 0 again.
                    chunk_pixel_values = _native_pixel_values_for_media_range(
                        native_pixel_values,
                        native_media_regions,
                        prompt_token_offset + chunk_start,
                        prompt_token_offset + chunk_end,
                    )
                    _set_native_pixel_values(model, chunk_pixel_values)
                    record_runner_phase(
                        "prefill_pipeline",
                        event="pipeline_prefill_native_pixel_values",
                        attrs={
                            "rank": rank,
                            "chunk_index": i,
                            "chunk_start": prompt_token_offset + chunk_start,
                            "chunk_end": prompt_token_offset + chunk_end,
                            **_native_pixel_values_attrs(chunk_pixel_values),
                        },
                    )
                model(
                    prompt[processed : processed + chunk_size][None],
                    cache=_prompt_cache,
                )
                quantize_cache_fn(_prompt_cache)
                processed += chunk_size

                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

                flush_prefill_sends()

                prompt_progress_callback(processed, total)

            for _ in range(n_trailing):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

    finally:
        clear_prefill_sends()

    # Post-loop: process remaining 1 token + add +1 entry to match stream_generate.
    if native_pixel_values is not None:
        _set_native_pixel_values(model, None)
    for _ in range(2):
        with mx.stream(generation_stream):
            model(prompt[-1:][None], cache=_prompt_cache)
            quantize_cache_fn(_prompt_cache)
        flush_prefill_sends()

    assert _prompt_cache is not None
    with mx.stream(generation_stream):
        mx.eval([c.state for c in _prompt_cache])  # type: ignore

    # Final callback matching generate_step
    prompt_progress_callback(total, total)

    logger.info(
        f"[R{rank}] Prefill: {n_real} real + {n_leading}+{n_trailing} dummy iterations, "
        f"Processed {processed} tokens in {(time.perf_counter() - t_start) * 1000:.1f}ms"
    )
    record_runner_phase(
        "prefill_pipeline",
        event="pipeline_prefill_complete",
        attrs={
            "rank": rank,
            "processed": processed,
            "total": total,
            "elapsed_ms": (time.perf_counter() - t_start) * 1000,
        },
        include_memory=True,
    )


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
    native_pixel_values: mx.array | list[mx.array] | None = None,
    native_media_regions: list[MediaRegion] | None = None,
    prompt_token_offset: int = 0,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        (tokens_per_sec, num_tokens, snapshots)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    rank = group.rank() if group is not None else 0
    group_size = group.size() if group is not None else 1

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            snapshots.append(snapshot_ssm_states(cache))

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    is_pipeline = _has_pipeline_communication_layer(model)

    prefill_step_size = 4096
    effective_prefill_step_size = (
        prefill_step_size // min(4, group_size) if is_pipeline else prefill_step_size
    )

    # Mirror pipeline_parallel_prefill's chunking math here by reserving the
    # final token for the post-loop pass before computing the real chunk count.
    # This value is primarily diagnostic now: all pipeline prompts use the
    # explicit pipeline prefill path, but short one-chunk prompts intentionally
    # suppress the distributed progress callback because there is no useful
    # cancellation window inside a millisecond-scale prefill.
    pipeline_chunk_sizes = (
        _vision_safe_prefill_chunk_sizes(
            num_tokens,
            effective_prefill_step_size,
            native_media_regions if native_pixel_values is not None else None,
            prompt_token_offset,
        )
        if is_pipeline
        else []
    )
    pipeline_chunks = len(pipeline_chunk_sizes)
    use_pipeline_prefill = is_pipeline
    logger.info(
        "Prefill path selected: "
        f"{'pipeline_parallel_prefill' if use_pipeline_prefill else 'stream_generate'} "
        f"(rank={rank}, prompt_tokens={num_tokens}, is_pipeline={is_pipeline}, "
        f"prefill_step_size_input={prefill_step_size}, "
        f"prefill_step_size_effective={effective_prefill_step_size}, "
        f"pipeline_chunks={pipeline_chunks})"
    )
    # Pipeline models must run in prefill mode during any prefill forward
    # pass. With is_prefill=False, pipeline wrappers can queue collectives
    # that are never consumed during prefill and later deadlock the ranks.
    set_pipeline_prefill(model, is_prefill=is_pipeline)

    with runner_phase(
        "prefill_barrier",
        detail="prefill_mx_barrier",
        attrs={"rank": rank, "group_size": group_size, "prompt_tokens": num_tokens},
        include_memory=True,
    ), _hang_debug_watch(f"prefill barrier rank={rank} group_size={group_size}"):
        mx_barrier(group)
    logger.info(
        f"Starting prefill (rank={rank}, group_size={group_size}, prompt_tokens={num_tokens})"
    )

    try:
        if use_pipeline_prefill:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            pipeline_distributed_callback = (
                distributed_prompt_progress_callback if pipeline_chunks >= 2 else None
            )
            with runner_phase(
                "prefill_pipeline",
                detail="pipeline_parallel_prefill",
                attrs={
                    "rank": rank,
                    "group_size": group_size,
                    "pipeline_chunks": pipeline_chunks,
                    "prompt_tokens": num_tokens,
                },
                include_memory=True,
            ), _hang_debug_watch(
                f"pipeline_parallel_prefill rank={rank} group_size={group_size}"
            ):
                pipeline_parallel_prefill(
                    model=model,
                    prompt=prompt_tokens,
                    prompt_cache=cache,
                    prefill_step_size=prefill_step_size,
                    kv_group_size=KV_GROUP_SIZE,
                    kv_bits=KV_BITS,
                    prompt_progress_callback=progress_callback,
                    distributed_prompt_progress_callback=pipeline_distributed_callback,
                    group=group,
                    native_pixel_values=native_pixel_values,
                    native_media_regions=native_media_regions,
                    prompt_token_offset=prompt_token_offset,
                )
        else:
            # Non-pipeline models can safely use upstream stream_generate for
            # prefill. Pipeline models avoid it entirely: mlx-lm can prefetch
            # a decode step before yielding, leaving hidden sends/recvs in
            # flight and wedging the next collective.
            # Use max_tokens=1 because max_tokens=0 does not work.
            # We just throw away the generated token - we only care about filling the cache
            with runner_phase(
                "prefill_stream",
                detail="stream_generate_prefill",
                attrs={
                    "rank": rank,
                    "group_size": group_size,
                    "prompt_tokens": num_tokens,
                },
                include_memory=True,
            ), _hang_debug_watch(
                f"stream_generate prefill rank={rank} group_size={group_size}"
            ):
                for _ in stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_tokens,
                    max_tokens=1,
                    sampler=sampler,
                    prompt_cache=cache,
                    prefill_step_size=prefill_step_size,
                    kv_group_size=KV_GROUP_SIZE,
                    kv_bits=KV_BITS,
                    prompt_progress_callback=progress_callback,
                ):
                    logger.info(
                        f"Prefill stream_generate yielded first token (rank={rank})"
                    )
                    break  # Stop after first iteration - cache is now filled
    except PrefillCancelled:
        raise
    finally:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)

    # stream_generate added 1 extra generated token to the cache, so we should trim it.
    # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
    pre_gen = deepcopy(snapshots[-2]) if has_ssm else None
    for i, c in enumerate(cache):
        if has_ssm and isinstance(c, (ArraysCache, RotatingKVCache)):
            assert pre_gen is not None
            if pre_gen.states[i] is not None:
                cache[i] = deepcopy(pre_gen.states[i])  # type: ignore
        else:
            assert not isinstance(c, (ArraysCache, RotatingKVCache))
            trim_fn = getattr(c, "trim", None)
            assert callable(trim_fn)
            trim_fn(2)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Exclude the last snapshot
    return tokens_per_sec, num_tokens, snapshots[:-1] if snapshots else []


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
    model_card: ModelCard | None = None,
) -> int:
    """Run a conservative synthetic warmup request to validate the MLX path.

    Distributed pipeline warmup intentionally stays on the minimal sanity-check
    prompt shape. Richer synthetic prompt templates have been observed to hang
    warmup prefill in that distributed path, so prompt-shaping overrides remain
    reserved for single-node investigation.
    """
    logger.info(f"warming up inference for instance: {model_id}")
    record_runner_phase(
        "warmup",
        event="warmup_start",
        attrs={
            "model_id": str(model_id),
            "group_size": group.size() if group is not None else 1,
        },
        include_memory=True,
    )

    if _is_distributed_warmup(group) and (
        _warmup_repeat_count() != 1 or _warmup_instructions(None) is not None
    ):
        logger.info(
            "Distributed pipeline warmup is forcing the minimal sanity-check "
            "prompt and ignoring warmup shaping overrides"
        )

    is_distributed = _is_distributed_warmup(group)
    warmup_task_params = TextGenerationTaskParams(
        model=model_id,
        # Distributed pipeline warmup always uses the minimal sanity-check
        # shape; single-node debugging can still opt into prompt shaping.
        instructions=_warmup_instructions(group),
        input=[
            InputMessage(
                role="user",
                content=_warmup_user_content(group),
            )
        ],
        max_output_tokens=1,
        enable_thinking=False,
        temperature=0.0 if is_distributed else 1.0,
        top_p=1.0 if is_distributed else 0.95,
        top_k=0 if is_distributed else 64,
    )

    with _hang_debug_watch(f"warmup apply_chat_template model={model_id}"):
        warmup_prompt = apply_chat_template(
            tokenizer=tokenizer,
            task_params=warmup_task_params,
            model_card=model_card,
            suppress_empty_gemma4_thought_channel=True,
        )
    logger.info(
        "Warmup prompt prepared "
        f"(model={model_id}, prompt_chars={len(warmup_prompt)}, group_size={group.size() if group is not None else 1})"
    )
    log_request_shape(
        "warmup",
        warmup_task_params,
        warmup_prompt,
        extra={
            "group_size": group.size() if group is not None else 1,
            "model_id": str(model_id),
        },
    )

    tokens_generated = 0

    with runner_phase(
        "prefill_barrier",
        detail="warmup_pre_generation_barrier",
        attrs={"model_id": str(model_id)},
        include_memory=True,
    ), _hang_debug_watch(f"warmup pre-generation barrier model={model_id}"):
        mx_barrier(group)

    logger.info("Generating warmup tokens")

    t = time.monotonic()

    with runner_phase(
        "warmup",
        detail="warmup_mlx_generate",
        attrs={"model_id": str(model_id)},
        include_memory=True,
    ), _hang_debug_watch(f"warmup mlx_generate model={model_id}"):
        for _r in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=warmup_task_params,
            prompt=warmup_prompt,
            kv_prefix_cache=None,
            group=group,
        ):
            tokens_generated += 1

    # The single-token warmup path intentionally samples only the first decode
    # step, so cold-start compile/prefill latency can dominate the elapsed
    # measurement here. Keep cancellation checks reasonably spaced even when
    # that first-token latency is slow, while still capping the interval for
    # fast models.
    if tokens_generated == 0:
        check_for_cancel_every = 0
    else:
        check_for_cancel_every = max(
            _MIN_CANCEL_CHECK_INTERVAL,
            min(math.ceil(tokens_generated / max(time.monotonic() - t, 0.001)), 100),
        )

    with runner_phase(
        "decode_barrier",
        detail="warmup_final_barrier",
        attrs={"model_id": str(model_id), "tokens_generated": tokens_generated},
        include_memory=True,
    ), _hang_debug_watch(
        f"warmup final barrier model={model_id} tokens_generated={tokens_generated}"
    ):
        mx_barrier(group)

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    if group is not None:
        world_size = group.size()
        rank = group.rank()
        slots = [0] * world_size
        slots[rank] = check_for_cancel_every
        cpu_stream = mx.default_stream(mx.Device(mx.cpu))
        merged = mx.distributed.all_sum(
            mx.array(slots, dtype=mx.int32), group=group, stream=cpu_stream
        )
        check_for_cancel_every = int(mx.max(merged).item())

    logger.info(
        f"runner checking for cancellation every {check_for_cancel_every} tokens"
    )

    return check_for_cancel_every


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def extract_top_logprobs(
    logprobs: mx.array,
    tokenizer: TokenizerWrapper,
    top_logprobs: int,
    selected_token: int,
    precomputed_indices: list[int] | None = None,
    precomputed_values: list[float] | None = None,
    precomputed_selected: float | None = None,
) -> tuple[float, list[TopLogprobItem]]:
    if (
        precomputed_indices is not None
        and precomputed_values is not None
        and precomputed_selected is not None
    ):
        top_indices_list: list[int] = precomputed_indices[:top_logprobs]
        top_values_list: list[float] = precomputed_values[:top_logprobs]
        selected_logprob = precomputed_selected
    else:
        selected_logprob_arr = logprobs[selected_token]
        top_logprobs = min(top_logprobs, logprobs.shape[0] - 1)
        top_indices = mx.argpartition(-logprobs, top_logprobs)[:top_logprobs]
        top_values = logprobs[top_indices]
        sort_order = mx.argsort(-top_values)
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]
        mx.eval(selected_logprob_arr, top_indices, top_values)
        selected_logprob = float(selected_logprob_arr.item())
        top_indices_list = top_indices.tolist()  # type: ignore
        top_values_list = top_values.tolist()  # type: ignore

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for token_id, token_logprob in zip(top_indices_list, top_values_list, strict=True):
        if math.isnan(token_logprob):
            continue

        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=list(token_str.encode("utf-8")),
            )
        )

    return selected_logprob, top_logprob_items


def _mlx_generate_native_vision(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    all_prompt_tokens: mx.array,
    vision: VisionResult,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    on_prefill_progress: Callable[[int, int], None] | None,
    on_generation_token: Callable[[], None] | None,
    group: mx.distributed.Group | None,
    trace_task_id: str | None = None,
) -> Generator[GenerationResponse]:
    """Generate for native-vision models via MLX-VLM's multimodal path.

    Gemma 4's reference MLX-VLM generation path computes multimodal input
    embeddings up front through ``model.get_input_embeddings(...)`` and then
    drives ``model.language_model`` directly. Running native-vision models
    through generic ``mlx_lm.stream_generate`` can diverge from that path even
    when prompt tokens and image preprocessing are correct, so we follow the
    reference execution strategy here.
    """
    vlm_generate_step = cast(
        _VlmGenerateStep,
        cast(dict[str, object], vars(import_module("mlx_vlm.generate")))[
            "generate_step"
        ],
    )

    if vision.pixel_values is None:
        raise ValueError("Native vision generation requires pixel values")

    prompt_token_count = len(all_prompt_tokens)
    max_tokens = task.max_output_tokens or MAX_TOKENS
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    eos_token_ids = set(eos_ids_from_tokenizer(tokenizer))

    detokenizer = cast(_DetokenizerProtocol, cast(object, tokenizer.detokenizer))
    detokenizer.reset()

    if on_prefill_progress is not None:
        on_prefill_progress(0, prompt_token_count)

    prompt_start_time = time.perf_counter()
    generation_start_time = prompt_start_time
    prompt_tps = 0.0
    first_token_seen = False
    accumulated_text = ""
    completion_tokens = 0
    final_finish_reason: FinishReason = "length"
    final_usage: Usage | None = None
    final_stats: GenerationStats | None = None
    decode_context = _decode_debug_context(
        task=task,
        group=group,
        trace_task_id=trace_task_id,
        total_prompt_tokens=prompt_token_count,
        uncached_prompt_tokens=prompt_token_count,
        prefix_hit_length=0,
        media_region_count=len(vision.media_regions),
        is_native_vision=True,
        native_pixel_values=vision.pixel_values,
    )

    logger.info(f"Native decode context: {decode_context}")
    logger.info("Starting native mlx-vlm multimodal decode")
    with runner_phase(
        "decode_barrier",
        detail="native_decode_barrier",
        attrs={
            "prompt_tokens": prompt_token_count,
            "media_regions": len(vision.media_regions),
            **_native_pixel_values_attrs(vision.pixel_values),
        },
        task_id=trace_task_id,
        include_memory=True,
    ), _hang_debug_watch(f"native decode barrier ({decode_context})"):
        mx_barrier(group)

    with runner_phase(
        "prefill_pipeline",
        detail="native_vlm_generate_step",
        attrs={
            "prompt_tokens": prompt_token_count,
            "max_tokens": max_tokens,
            **_native_pixel_values_attrs(vision.pixel_values),
        },
        task_id=trace_task_id,
        include_memory=True,
    ):
        token_generator = vlm_generate_step(
            input_ids=all_prompt_tokens[None],
            model=model,
            pixel_values=vision.pixel_values,
            mask=None,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=4096,
            kv_group_size=KV_GROUP_SIZE or 32,
            kv_bits=KV_BITS,
        )

    first_token_wait_started = time.perf_counter()
    with runner_phase(
        "decode_wait_first_token",
        detail="native_decode_first_token",
        attrs={"prompt_tokens": prompt_token_count, "max_tokens": max_tokens},
        task_id=trace_task_id,
        include_memory=True,
    ), _hang_debug_watch(f"native decode first token ({decode_context})"):
        try:
            first_token, first_logprobs = next(token_generator)
        except StopIteration:
            logger.warning(
                "Native decode generator ended before yielding a response "
                f"({decode_context})"
            )
            return
    logger.info(
        "Native decode produced the first response after "
        f"{time.perf_counter() - first_token_wait_started:.2f}s ({decode_context})"
    )
    record_runner_phase(
        "decode_stream",
        event="native_first_token",
        task_id=trace_task_id,
        attrs={"wait_seconds": time.perf_counter() - first_token_wait_started},
        include_memory=True,
    )

    for token, logprobs in itertools.chain(
        [(first_token, first_logprobs)],
        token_generator,
    ):
        token_id = int(token)

        if not first_token_seen:
            prompt_elapsed = time.perf_counter() - prompt_start_time
            prompt_tps = (
                prompt_token_count / prompt_elapsed if prompt_elapsed > 0 else 0.0
            )
            generation_start_time = time.perf_counter()
            first_token_seen = True
            if on_prefill_progress is not None:
                on_prefill_progress(prompt_token_count, prompt_token_count)

        if token_id in eos_token_ids:
            final_finish_reason = "stop"
            break

        completion_tokens += 1
        detokenizer.add_token(token_id)
        text = detokenizer.last_segment
        accumulated_text += text

        stop_matched = False
        if stop_sequences:
            for stop_sequence in stop_sequences:
                if stop_sequence in accumulated_text:
                    stop_index = accumulated_text.find(stop_sequence)
                    text_before_stop = accumulated_text[:stop_index]
                    chunk_start = len(accumulated_text) - len(text)
                    text = text_before_stop[chunk_start:]
                    accumulated_text = text_before_stop
                    final_finish_reason = "stop"
                    stop_matched = True
                    break

        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if task.logprobs:
            with mx.stream(generation_stream):
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=logprobs,
                    tokenizer=tokenizer,
                    top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=token_id,
                )

        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(
            text=text,
            token=token_id,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=None,
            stats=None,
            usage=None,
        )

        if stop_matched:
            break
    else:
        final_finish_reason = "length"

    detokenizer.finalize()
    final_text = detokenizer.last_segment

    if stop_sequences and final_text:
        for stop_sequence in stop_sequences:
            combined_text = accumulated_text + final_text
            if stop_sequence in combined_text:
                stop_index = combined_text.find(stop_sequence)
                final_text = combined_text[len(accumulated_text) : stop_index]
                accumulated_text = combined_text[:stop_index]
                final_finish_reason = "stop"
                break
    else:
        accumulated_text += final_text

    generation_elapsed = time.perf_counter() - generation_start_time
    generation_tps = completion_tokens / generation_elapsed if generation_elapsed > 0 else 0.0

    final_stats = GenerationStats(
        prompt_tps=float(prompt_tps),
        generation_tps=float(generation_tps),
        prompt_tokens=prompt_token_count,
        generation_tokens=completion_tokens,
        peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
    )
    final_usage = Usage(
        prompt_tokens=prompt_token_count,
        completion_tokens=completion_tokens,
        total_tokens=prompt_token_count + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
    )

    if on_generation_token is not None:
        on_generation_token()

    yield GenerationResponse(
        text=final_text,
        token=0,
        finish_reason=final_finish_reason,
        stats=final_stats,
        usage=final_usage,
    )

    with runner_phase(
        "decode_barrier",
        detail="native_decode_final_barrier",
        task_id=trace_task_id,
        include_memory=True,
    ), _hang_debug_watch(f"native decode final barrier ({decode_context})"):
        mx_barrier(group)


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
    vision_processor: VisionProcessor | None = None,
    trace_task_id: str | None = None,
    trace_rank: int = 0,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    record_runner_phase(
        "prompt_build",
        event="mlx_generate_start",
        attrs={
            "model": str(task.model),
            "input_images": len(task.images),
            "max_output_tokens": task.max_output_tokens or MAX_TOKENS,
        },
        task_id=trace_task_id,
        include_memory=True,
    )
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    with runner_phase(
        "prompt_build",
        detail="prompt_tokenization",
        attrs={"prompt_chars": len(prompt), "seed": seed},
        task_id=trace_task_id,
    ):
        all_prompt_tokens = encode_prompt(tokenizer, prompt)
        all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)
    record_runner_phase(
        "prompt_build",
        event="prompt_tokenized",
        attrs={"prompt_tokens": len(all_prompt_tokens)},
        task_id=trace_task_id,
    )
    min_prefix_hit_length = max(1000, system_prompt_token_count(task, tokenizer))

    vision: VisionResult | None = None
    if vision_processor is not None:
        try:
            with runner_phase(
                "vision_preprocess",
                detail="prepare_vision",
                attrs={
                    "image_count": len(task.images),
                    "chat_template_messages": len(task.chat_template_messages or []),
                },
                task_id=trace_task_id,
                include_memory=True,
            ), trace(
                "native_vision_preprocess",
                trace_rank,
                "vision",
                task_id=trace_task_id,
            ):
                vision = prepare_vision(
                    images=task.images,
                    chat_template_messages=task.chat_template_messages,
                    vision_processor=vision_processor,
                    tokenizer=tokenizer,
                    model=model,
                    task_id=trace_task_id,
                )
        except Exception:
            logger.opt(exception=True).warning(
                "Vision processing failed, falling back to text-only"
            )
            record_runner_phase(
                "vision_preprocess",
                event="prepare_vision_failed",
                detail="falling back to text-only",
                task_id=trace_task_id,
                include_memory=True,
            )
    if vision is not None:
        all_prompt_tokens = vision.prompt_tokens
    media_regions: list[MediaRegion] = vision.media_regions if vision else []
    record_runner_phase(
        "vision_preprocess",
        event="vision_ready" if vision is not None else "vision_not_used",
        attrs={
            "prompt_tokens": len(all_prompt_tokens),
            **_media_region_attrs(media_regions),
            **_native_pixel_values_attrs(vision.pixel_values if vision else None),
            **_vision_trace_attrs(vision),
        },
        task_id=trace_task_id,
        include_memory=True,
    )
    record_trace_marker(
        "vision_ready" if vision is not None else "vision_not_used",
        trace_rank,
        "vision",
        task_id=trace_task_id,
        attrs={
            "prompt_tokens": len(all_prompt_tokens),
            **_vision_trace_attrs(vision),
        },
    )

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    with runner_phase(
        "kv_cache_lookup",
        detail="prefix_cache_lookup",
        attrs={
            "cache_enabled": kv_prefix_cache is not None,
            "prompt_tokens": len(all_prompt_tokens),
            **_media_region_attrs(media_regions),
        },
        task_id=trace_task_id,
        include_memory=True,
    ):
        if kv_prefix_cache is None:
            caches = make_kv_cache(model=model)
            prompt_tokens = all_prompt_tokens
        else:
            caches, prompt_tokens, matched_index = kv_prefix_cache.get_kv_cache(
                model, all_prompt_tokens, media_regions=media_regions
            )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )
        else:
            logger.info(
                f"KV cache miss: 0/{len(all_prompt_tokens)} tokens cached "
                f"(media_regions={len(media_regions)})"
            )
    record_runner_phase(
        "kv_cache_lookup",
        event="kv_cache_result",
        attrs={
            "prefix_hit_length": prefix_hit_length,
            "uncached_prompt_tokens": len(prompt_tokens),
            "total_prompt_tokens": len(all_prompt_tokens),
            "matched_index": matched_index if matched_index is not None else -1,
        },
        task_id=trace_task_id,
    )
    record_trace_marker(
        "kv_cache_result",
        trace_rank,
        "kv_cache",
        task_id=trace_task_id,
        attrs={
            "prefix_hit_length": prefix_hit_length,
            "uncached_prompt_tokens": len(prompt_tokens),
            "total_prompt_tokens": len(all_prompt_tokens),
            "matched_index": matched_index if matched_index is not None else -1,
            "media_region_count": len(media_regions),
        },
    )

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
        make_logits_processors(
            repetition_penalty=task.repetition_penalty,
            repetition_context_size=task.repetition_context_size,
        )
    )
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)] + logits_processors

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        min_p=task.min_p if task.min_p is not None else 0.05,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    if vision is not None and vision.pixel_values is not None:
        if _should_use_native_vision_reference_path():
            if kv_prefix_cache is not None:
                logger.info(
                    "Disabling KV prefix cache for native vision generation to follow "
                    "the mlx-vlm reference execution path"
                )
            record_runner_phase(
                "vision_preprocess",
                event="native_reference_path_selected",
                attrs={
                    **_media_region_attrs(media_regions),
                    **_native_pixel_values_attrs(vision.pixel_values),
                },
                task_id=trace_task_id,
                include_memory=True,
            )
            record_trace_marker(
                "native_reference_path_selected",
                trace_rank,
                "vision",
                task_id=trace_task_id,
                attrs={
                    **_vision_trace_attrs(vision),
                    "media_region_count": len(media_regions),
                },
            )
            yield from _mlx_generate_native_vision(
                model=model,
                tokenizer=tokenizer,
                task=task,
                all_prompt_tokens=all_prompt_tokens,
                vision=vision,
                sampler=sampler,
                logits_processors=logits_processors,
                on_prefill_progress=on_prefill_progress,
                on_generation_token=on_generation_token,
                group=group,
                trace_task_id=trace_task_id,
            )
            return

        logger.info(
            "Using pipeline-aware native vision generation path on fixed "
            "mlx/mlx-vlm stack"
        )
        record_runner_phase(
            "vision_preprocess",
            event="pipeline_native_vision_path_selected",
            attrs={
                **_media_region_attrs(media_regions),
                **_native_pixel_values_attrs(vision.pixel_values),
            },
            task_id=trace_task_id,
        )
        record_trace_marker(
            "pipeline_native_vision_path_selected",
            trace_rank,
            "vision",
            task_id=trace_task_id,
            attrs={
                **_vision_trace_attrs(vision),
                "media_region_count": len(media_regions),
            },
        )

    is_native_vision = vision is not None and vision.pixel_values is not None
    native_pixel_values: mx.array | list[mx.array] | None = None
    native_media_regions: list[MediaRegion] | None = None
    if is_native_vision:
        assert vision is not None
        if vision.pixel_values is None:
            raise ValueError("Native vision generation requires pixel values")
        native_pixel_values = _slice_native_pixel_values_for_uncached_suffix(
            vision.pixel_values,
            media_regions,
            prefix_hit_length,
        )
        native_media_regions = _native_media_regions_for_uncached_suffix(
            vision.pixel_values,
            media_regions,
            prefix_hit_length,
        )
    decode_context = _decode_debug_context(
        task=task,
        group=group,
        trace_task_id=trace_task_id,
        total_prompt_tokens=len(all_prompt_tokens),
        uncached_prompt_tokens=len(prompt_tokens),
        prefix_hit_length=prefix_hit_length,
        media_region_count=len(media_regions),
        is_native_vision=is_native_vision,
        native_pixel_values=native_pixel_values,
    )

    if _should_force_native_vision_full_prefill_for_request(
        group=group,
        is_native_vision=is_native_vision,
        prefix_hit_length=prefix_hit_length,
        media_region_count=len(media_regions),
        native_pixel_values=native_pixel_values,
    ):
        logger.warning(
            "Disabling KV prefix cache for a distributed native-vision follow-up "
            "request with cached image regions to avoid known Gemma 4 decode "
            f"wedge/reference-cache crashes ({decode_context})"
        )
        record_runner_phase(
            "kv_cache_lookup",
            event="forced_full_prefill_fallback",
            detail="distributed native-vision follow-up",
            attrs={
                "prefix_hit_length": prefix_hit_length,
                **_media_region_attrs(media_regions),
                **_native_pixel_values_attrs(native_pixel_values),
            },
            task_id=trace_task_id,
            include_memory=True,
        )
        record_trace_marker(
            "forced_full_prefill_fallback",
            trace_rank,
            "kv_cache",
            task_id=trace_task_id,
            attrs={
                "prefix_hit_length": prefix_hit_length,
                "media_region_count": len(media_regions),
                **_native_pixel_values_trace_attrs(native_pixel_values),
            },
        )
        assert vision is not None
        caches = make_kv_cache(model=model)
        prompt_tokens = all_prompt_tokens
        prefix_hit_length = 0
        matched_index = None
        native_pixel_values = vision.pixel_values
        native_media_regions = media_regions
        decode_context = _decode_debug_context(
            task=task,
            group=group,
            trace_task_id=trace_task_id,
            total_prompt_tokens=len(all_prompt_tokens),
            uncached_prompt_tokens=len(prompt_tokens),
            prefix_hit_length=prefix_hit_length,
            media_region_count=len(media_regions),
            is_native_vision=is_native_vision,
            native_pixel_values=native_pixel_values,
        )

    if native_pixel_values is not None:
        with runner_phase(
            "vision_preprocess",
            detail="set_pixel_values",
            attrs=_native_pixel_values_attrs(native_pixel_values),
            task_id=trace_task_id,
            include_memory=True,
        ):
            _set_native_pixel_values(model, native_pixel_values)
        maybe_vision_ctx = contextlib.nullcontext()
    elif vision is not None and not is_native_vision:
        maybe_vision_ctx = patch_embed_tokens(
            model, vision.embeddings, prefix_hit_length, len(prompt_tokens) - 1
        )
    else:
        maybe_vision_ctx = contextlib.nullcontext()
    try:
        with maybe_vision_ctx, trace(
            "prefill",
            trace_rank,
            "prefill",
            task_id=trace_task_id,
        ):
            prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
                model,
                tokenizer,
                sampler,
                prompt_tokens[:-1],
                caches,
                group,
                on_prefill_progress,
                distributed_prompt_progress_callback,
                native_pixel_values=native_pixel_values,
                native_media_regions=native_media_regions,
                prompt_token_offset=prefix_hit_length,
            )
    finally:
        record_runner_phase(
            "vision_preprocess",
            event="clear_pixel_values_begin",
            task_id=trace_task_id,
            include_memory=True,
        )
        if hasattr(model, "set_pixel_values") or hasattr(model, "_pixel_values"):
            _set_native_pixel_values(model, None)
        record_runner_phase(
            "vision_preprocess",
            event="clear_pixel_values_complete",
            task_id=trace_task_id,
            include_memory=True,
        )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    # stream_generate starts from the last token
    last_token = prompt_tokens[-2:]

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    in_thinking = False
    reasoning_tokens = 0
    think_start = tokenizer.think_start
    think_end = tokenizer.think_end

    logger.info(f"Decode context: {decode_context}")
    logger.info("Starting decode")
    with runner_phase(
        "decode_barrier",
        detail="decode_mx_barrier",
        attrs={
            "total_prompt_tokens": len(all_prompt_tokens),
            "uncached_prompt_tokens": len(prompt_tokens),
            "prefix_hit_length": prefix_hit_length,
            **_media_region_attrs(media_regions),
            **_native_pixel_values_attrs(native_pixel_values),
        },
        task_id=trace_task_id,
        include_memory=True,
    ), _hang_debug_watch(f"decode barrier ({decode_context})"):
        mx_barrier(group)

    with runner_phase(
        "decode_stream",
        detail="stream_generate_setup",
        attrs={"max_tokens": max_tokens, "last_token_count": len(last_token)},
        task_id=trace_task_id,
        include_memory=True,
    ):
        token_generator = stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=last_token,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=caches,
            prefill_step_size=1,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
        )
    first_token_wait_started = time.perf_counter()
    with runner_phase(
        "decode_wait_first_token",
        detail="stream_generate_first_token",
        attrs={"max_tokens": max_tokens},
        task_id=trace_task_id,
        include_memory=True,
    ), _hang_debug_watch(f"decode first token ({decode_context})"):
        try:
            first_out = next(token_generator)
        except StopIteration:
            logger.warning(
                "Decode generator ended before yielding a response "
                f"({decode_context})"
            )
            return
    logger.info(
        "Decode produced the first response after "
        f"{time.perf_counter() - first_token_wait_started:.2f}s ({decode_context})"
    )
    record_runner_phase(
        "decode_stream",
        event="first_token",
        attrs={"wait_seconds": time.perf_counter() - first_token_wait_started},
        task_id=trace_task_id,
        include_memory=True,
    )

    for completion_tokens, out in enumerate(
        itertools.chain([first_out], token_generator),
        start=1,
    ):
        if completion_tokens == 1 or completion_tokens % 25 == 0:
            record_runner_phase(
                "decode_stream",
                event="decode_token_progress",
                attrs={"completion_tokens": completion_tokens},
                task_id=trace_task_id,
                include_memory=completion_tokens == 1,
            )
        generated_text_parts.append(out.text)
        accumulated_text += out.text

        if think_start is not None and out.text == think_start:
            in_thinking = True
        elif think_end is not None and out.text == think_end:
            in_thinking = False
        if in_thinking:
            reasoning_tokens += 1

        # Check for stop sequences
        text = out.text
        finish_reason: FinishReason | None = cast(
            FinishReason | None, out.finish_reason
        )
        stop_matched = False

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in accumulated_text:
                    # Trim text to just before the stop sequence
                    stop_index = accumulated_text.find(stop_seq)
                    text_before_stop = accumulated_text[:stop_index]
                    chunk_start = len(accumulated_text) - len(out.text)
                    text = text_before_stop[chunk_start:]
                    finish_reason = "stop"
                    stop_matched = True
                    break

        is_done = finish_reason is not None

        stats: GenerationStats | None = None
        if is_done:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )
            if not stop_matched and out.finish_reason not in get_args(FinishReason):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

            total_prompt_tokens = len(all_prompt_tokens)
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
            )

        # Extract logprobs from the full vocabulary logprobs array
        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if task.logprobs:
            with mx.stream(generation_stream):
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=out.logprobs,
                    tokenizer=tokenizer,
                    top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=out.token,
                )

        if is_done:
            # Log generation stats
            generation_elapsed = time.perf_counter() - generation_start_time
            generated_tokens = len(generated_text_parts)
            generation_tps = (
                generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )
            logger.debug(
                f"Generation complete: prefill {prompt_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )
            if kv_prefix_cache is not None:
                generated_tokens_array = mx.array(
                    tokenizer.encode(
                        "".join(generated_text_parts), add_special_tokens=False
                    )
                )
                full_prompt_tokens = mx.concatenate(
                    [all_prompt_tokens, generated_tokens_array]
                )
                hit_ratio = (
                    prefix_hit_length / len(all_prompt_tokens)
                    if len(all_prompt_tokens) > 0
                    else 0.0
                )
                if matched_index is not None and (
                    prefix_hit_length >= min_prefix_hit_length
                    and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
                ):
                    record_runner_phase(
                        "kv_cache_lookup",
                        event="kv_cache_update",
                        attrs={
                            "matched_index": matched_index,
                            "prefix_hit_length": prefix_hit_length,
                            "hit_ratio": hit_ratio,
                        },
                        task_id=trace_task_id,
                        include_memory=True,
                    )
                    kv_prefix_cache.update_kv_cache(
                        matched_index,
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        restore_pos=prefix_hit_length,
                        media_regions=media_regions,
                    )
                else:
                    record_runner_phase(
                        "kv_cache_lookup",
                        event="kv_cache_add",
                        attrs={
                            "prefix_hit_length": prefix_hit_length,
                            "hit_ratio": hit_ratio,
                        },
                        task_id=trace_task_id,
                        include_memory=True,
                    )
                    kv_prefix_cache.add_kv_cache(
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        media_regions=media_regions,
                    )

        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(
            text=text,
            token=out.token,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
        )

        if is_done:
            record_runner_phase(
                "completion",
                event="generation_complete",
                attrs={
                    "completion_tokens": completion_tokens,
                    "finish_reason": str(finish_reason),
                },
                task_id=trace_task_id,
                include_memory=True,
            )
            with runner_phase(
                "decode_barrier",
                detail="decode_final_barrier",
                task_id=trace_task_id,
                include_memory=True,
            ), _hang_debug_watch(f"decode final barrier ({decode_context})"):
                mx_barrier(group)
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
