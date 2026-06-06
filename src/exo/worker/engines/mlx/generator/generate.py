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
    GenerationResponse as MlxGenerationResponse,
)
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models import base as _mlx_lm_base
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
    eval_with_timeout,
    flush_prefill_sends,
    pipeline_eval_timeout_seconds,
    pipeline_timeout_callback,
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
    trim_cache,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_CACHE_BACKEND,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.drafters import Drafter, build_drafter
from exo.worker.engines.mlx.generator.speculative_sampling import (
    SamplingParams,
    ratio_accept,
    residual_sample,
    warp_to_probs,
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
# Ceiling on deferred-replay tokens riding along in an MTP verify forward.
# Verify width is near-free while decode stays memory-bound; the cap only
# exists so a pathological reject streak cannot grow the window unboundedly.
_MTP_MAX_PENDING_REPLAY = 8


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

    Multi-image Gemma 4 follow-up turns with cached image prefixes can pair the
    next uncached image span with stale pixel values. Keep those conservative.
    Single-image fully cached follow-ups should reuse the prefix cache instead:
    forcing a full native-vision re-prefill has reproduced first-decode stalls
    even for modest prompts.
    """
    has_cached_multi_image_prefix = prefix_hit_length > 0 and media_region_count > 1
    return (
        is_native_vision
        and group is not None
        and group.size() > 1
        and has_cached_multi_image_prefix
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


def _has_pipeline_communication_layer(model: Model) -> bool:
    layers = cast(list[object], getattr(model, "layers", []))
    for layer in layers:
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
    # The explicit pipeline prefill path only buys us overlap once there are at
    # least two real chunks. Single-chunk prompts keep the PR90-era
    # stream_generate path because they do not benefit from the custom pipeline
    # scheduler and have repeatedly been the sharpest wedge shape in Gemma 4
    # cluster testing.
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
    use_pipeline_prefill = is_pipeline and pipeline_chunks >= 2
    logger.info(
        "Prefill path selected: "
        f"{'pipeline_parallel_prefill' if use_pipeline_prefill else 'stream_generate'} "
        f"(rank={rank}, prompt_tokens={num_tokens}, is_pipeline={is_pipeline}, "
        f"prefill_step_size_input={prefill_step_size}, "
        f"prefill_step_size_effective={effective_prefill_step_size}, "
        f"pipeline_chunks={pipeline_chunks}, pipeline_min_chunks=2)"
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


def _stream_generate_without_lookahead(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    prompt_cache: KVCacheType,
    kv_group_size: int | None,
    kv_bits: int | None,
) -> Generator[MlxGenerationResponse, None, None]:
    """Generate tokens sequentially without scheduling a decode lookahead.

    ``mlx_lm.stream_generate`` evaluates the next token before yielding the
    current one. That is fine for local models, but the extra in-flight decode
    forward can leave Skulk's pipeline send/recv wrappers waiting on different
    collectives across ranks. Pipeline decode favors boring one-step-at-a-time
    execution over speculative throughput.
    """
    if len(prompt) == 0:
        raise ValueError("Decode requires at least one prompt token")

    detokenizer = cast(_DetokenizerProtocol, cast(object, tokenizer.detokenizer))
    detokenizer.reset()

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    token_history: mx.array | None = None
    prompt_start = time.perf_counter()
    if len(prompt) > 1:
        with mx.stream(generation_stream):
            model(prompt[:-1][None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            mx.eval([cache_entry.state for cache_entry in prompt_cache])  # type: ignore

    prompt_tps = len(prompt) / max(time.perf_counter() - prompt_start, 1e-9)
    generation_start = time.perf_counter()
    input_tokens = prompt[-1:]
    eos_token_ids = {int(token_id) for token_id in eos_ids_from_tokenizer(tokenizer)}

    last_token = 0
    last_logprobs = mx.zeros((1,), dtype=mx.float32)
    finish_reason = "length"
    generated_count = 0

    for token_index in range(max_tokens):
        with mx.stream(generation_stream):
            logits = model(input_tokens[None], cache=prompt_cache)
            logits = logits[:, -1, :]
            if logits_processors:
                token_history = (
                    mx.concat([token_history, input_tokens])
                    if token_history is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(token_history, logits)
            quantize_cache_fn(prompt_cache)
            logprobs = logits.astype(mx.float32) - mx.logsumexp(
                logits.astype(mx.float32), keepdims=True
            )
            sampled = sampler(logprobs)
            mx.eval(sampled, logprobs)

        last_token = int(sampled.item())
        last_logprobs = logprobs.squeeze(0)
        generated_count = token_index + 1

        if last_token in eos_token_ids:
            finish_reason = "stop"
            break

        detokenizer.add_token(last_token)
        if generated_count == max_tokens:
            finish_reason = "length"
            break

        yield MlxGenerationResponse(
            text=detokenizer.last_segment,
            token=last_token,
            logprobs=last_logprobs,
            from_draft=False,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=generated_count,
            generation_tps=generated_count
            / max(time.perf_counter() - generation_start, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )
        input_tokens = sampled

    detokenizer.finalize()
    yield MlxGenerationResponse(
        text=detokenizer.last_segment,
        token=last_token,
        logprobs=last_logprobs,
        from_draft=False,
        prompt_tokens=prompt.size,
        prompt_tps=prompt_tps,
        generation_tokens=generated_count,
        generation_tps=generated_count
        / max(time.perf_counter() - generation_start, 1e-9),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason=finish_reason,
    )


def _make_prenorm_trunk_fn(trunk: object) -> Callable[..., mx.array] | None:
    """Wrap a TextModel trunk to return PRE-final-norm hidden states.

    MTP heads consume the trunk's hidden state *before* the final RMSNorm —
    that is what they were trained against, and feeding post-norm hiddens
    measured 0% draft acceptance live while the offline pre-norm validation
    measured 72.4% (issue #192). This mirrors ``Qwen3_5TextModel.__call__``
    (embed → per-layer-type masks → layers) minus the trailing ``norm``.

    Returns ``None`` when the trunk does not expose the expected structure;
    callers fall back to the post-norm trunk (and should not enable
    block-based drafting in that case).
    """
    embed: object | None = getattr(trunk, "embed_tokens", None)
    layers: object | None = getattr(trunk, "layers", None)
    if embed is None or not callable(embed) or not isinstance(layers, list):
        return None
    # Gate to qwen-shaped trunks: the manual mask construction below mirrors
    # Qwen3.5's ssm/full-attention split. Other families (gemma4's
    # sliding/full split) would get WRONG masks — and the gemma4 assistant
    # drafter wants the post-norm fallback path anyway.
    if not hasattr(trunk, "fa_idx") and not any(
        hasattr(layer, "is_linear") for layer in cast("list[object]", layers)
    ):
        return None
    layer_list = cast("list[Callable[..., mx.array]]", layers)
    # Hybrid (GDN/SSM) trunks key their masks off specific layer indices;
    # pure-attention trunks use the first cache entry.
    fa_idx = int(cast(int, getattr(trunk, "fa_idx", 0)))
    ssm_idx_attr: object | None = getattr(trunk, "ssm_idx", None)
    ssm_idx: int | None = int(cast(int, ssm_idx_attr)) if ssm_idx_attr is not None else None
    make_attention_mask = cast(
        "Callable[[mx.array, object], object]",
        _mlx_lm_base.create_attention_mask,
    )
    make_ssm_mask = cast(
        "Callable[[mx.array, object], object]",
        _mlx_lm_base.create_ssm_mask,
    )
    embed_fn = cast(Callable[..., mx.array], embed)

    def trunk_fn(tokens: mx.array, cache: KVCacheType | None = None) -> mx.array:
        h = embed_fn(tokens)
        layer_caches: list[object | None] = (
            list(cache) if cache is not None else [None] * len(layer_list)
        )
        fa_mask = make_attention_mask(h, layer_caches[fa_idx])
        ssm_mask = (
            make_ssm_mask(h, layer_caches[ssm_idx]) if ssm_idx is not None else None
        )
        for layer, layer_cache in zip(layer_list, layer_caches, strict=True):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
        return h

    return trunk_fn


def _get_trunk_and_head(
    model: Model,
) -> tuple[Callable[..., mx.array], Callable[..., mx.array]] | None:
    """Return (trunk_fn, head_fn) for hidden-state access, or None if unsupported.

    trunk_fn(tokens, cache) → (1, seq, hidden_size) hidden states. For
    Qwen3.5-style models these are PRE-final-norm (what MTP heads consume);
    head_fn applies the final norm itself, so main-path logits are unchanged.
    head_fn(hidden) → (1, seq, vocab_size) logits.
    """
    # Qwen3.5-style: model.language_model wraps a TextModel with .model and .lm_head
    lm: object | None = getattr(model, "language_model", None)
    if lm is not None:
        trunk: object | None = getattr(lm, "model", None)
        if trunk is not None and callable(trunk):
            head: object | None = getattr(lm, "lm_head", None)
            if head is None or not callable(head):
                # Tied word embeddings: embed_tokens.as_linear acts as the head
                embed: object | None = getattr(trunk, "embed_tokens", None)
                as_linear: object | None = getattr(embed, "as_linear", None)
                head = as_linear if as_linear is not None and callable(as_linear) else None
            if head is not None:
                head_fn = cast(Callable[..., mx.array], head)
                norm: object | None = getattr(trunk, "norm", None)
                prenorm_trunk_fn = _make_prenorm_trunk_fn(trunk)
                if prenorm_trunk_fn is not None and norm is not None and callable(norm):
                    norm_fn = cast(Callable[..., mx.array], norm)

                    def normed_head_fn(hidden: mx.array) -> mx.array:
                        return head_fn(norm_fn(hidden))

                    return (prenorm_trunk_fn, normed_head_fn)
                # Post-norm fallback. For qwen-family sidecar heads this
                # degrades acceptance (they train on pre-norm hiddens); for
                # the gemma4 assistant it is exactly the convention the
                # drafter consumes.
                logger.info(
                    "MTP: using post-norm trunk hiddens (pre-norm wrapper "
                    "unavailable for this trunk shape)"
                )
                return (cast(Callable[..., mx.array], trunk), head_fn)
    # DeepSeek-style: trunk = model.model, head = model.lm_head
    ds_trunk: object | None = getattr(model, "model", None)
    if ds_trunk is not None and callable(ds_trunk):
        ds_head: object | None = getattr(model, "lm_head", None)
        if ds_head is not None and callable(ds_head):
            return (
                cast(Callable[..., mx.array], ds_trunk),
                cast(Callable[..., mx.array], ds_head),
            )
    return None


def _broadcast_via_all_sum(
    group: "mx.distributed.Group", payload: mx.array, *, detail: str
) -> mx.array:
    """One-to-all broadcast built from ``all_sum`` (non-senders contribute
    zeros), on the CPU stream with the shared pipeline watchdog."""
    summed = mx.distributed.all_sum(
        payload, group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    eval_with_timeout(
        summed,
        timeout_seconds=pipeline_eval_timeout_seconds(),
        on_timeout=pipeline_timeout_callback(
            detail, {"group_size": group.size()}, is_prefill=False
        ),
    )
    return summed


def _exchange_drafts(
    *,
    draft_group: "mx.distributed.Group",
    drafter: Drafter | None,
    hidden: mx.array,
    bonus: int,
    depth: int,
    sampling: SamplingParams,
    vocab_size: int,
    draft_key: int,
) -> tuple[list[int], mx.array | None]:
    """Distributed-draft round: drafting rank drafts, every rank receives.

    Payload layout (fp32; token ids are exact in fp32 up to 2^24, far above
    any vocab): ``[chain_len, tok_0..tok_{D-1} (zero-padded)]``, plus the
    drafter's full effective distribution row appended under sampling so
    ratio-acceptance and residual resampling run identically on every rank.
    The payload shape is fixed per request, so the collective schedule stays
    symmetric regardless of which rank drafts.

    Returns ``(draft_tokens, draft_probs_or_None)``. Drafting-rank failures
    propagate (fail-loud is mandatory in distributed mode); the peers'
    watchdogs surface the abandoned collective.
    """
    slots = max(depth, 1)
    payload_len = 1 + slots + (0 if sampling.is_greedy else vocab_size)
    payload = mx.zeros((payload_len,), dtype=mx.float32)
    if drafter is not None:
        chain_logits = drafter.draft(hidden, bonus, depth=depth).astype(mx.float32)
        if sampling.is_greedy:
            toks = mx.argmax(chain_logits, axis=-1)
            tok_list = [int(t) for t in cast("list[int]", toks.tolist())]
            payload = mx.concatenate(
                [
                    mx.array([float(len(tok_list))]),
                    toks.astype(mx.float32),
                    mx.zeros((slots - len(tok_list),), dtype=mx.float32),
                ]
            )
        else:
            chain_lp = chain_logits - mx.logsumexp(
                chain_logits, axis=-1, keepdims=True
            )
            q_row = warp_to_probs(chain_lp[0], sampling)
            # Explicit per-round key: only this rank draws here, and using
            # the global stream would desync it from the peers' aligned
            # streams (they don't draft). The key folds in a per-request
            # value drawn from the SEEDED stream (symmetrically, on every
            # rank — see the loop init), so the proposal remains a true
            # seed-dependent sample from q and the Leviathan-Chen
            # acceptance below stays distribution-preserving.
            key = mx.random.key(draft_key)
            tok = mx.random.categorical(mx.log(q_row + 1e-12), key=key)
            payload = mx.concatenate(
                [
                    mx.array([1.0]),
                    tok.astype(mx.float32)[None],
                    mx.zeros((slots - 1,), dtype=mx.float32),
                    q_row.astype(mx.float32),
                ]
            )
        mx.eval(payload)
    summed = _broadcast_via_all_sum(
        draft_group, payload, detail="mtp_draft_exchange"
    )
    chain_len = int(summed[0].item())
    draft_toks = [int(t) for t in cast("list[int]", summed[1 : 1 + chain_len].tolist())]
    draft_probs = None if sampling.is_greedy else summed[1 + slots :]
    return draft_toks, draft_probs


def _stream_generate_with_mtp(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    drafter: Drafter | None,
    trunk_fn: Callable[..., mx.array],
    head_fn: Callable[..., mx.array],
    prompt: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    prompt_cache: KVCacheType,
    kv_group_size: int | None,
    kv_bits: int | None,
    depth: int = 1,
    sampling: SamplingParams | None = None,
    fail_loud_on_drafter_error: bool = False,
    draft_group: "mx.distributed.Group | None" = None,
) -> Generator[MlxGenerationResponse, None, None]:
    """Speculative decode loop: bonus-driven rounds via a Drafter.

    Round structure (matching the reference implementations this was
    validated against): the loop carries a *bonus* token ``b`` — emitted but
    not yet forwarded — and the hidden state ``h`` at the position whose
    logits produced it. Each round drafts up to ``depth`` candidates from
    ``(h, b)``, verifies ``[b, d0..dK-1]`` in a single K+1-token forward
    (the round's ONLY target forward), commits the longest matching draft
    prefix, and samples the next bonus from the first non-matching row —
    which is the correction on a partial reject and the free next token on
    a full accept. Crucially, the very next round drafts from the
    correction position: those post-correction drafts are statistically the
    easiest, and a cadence that skips them (as this loop's previous shape
    did) measurably forfeits ~25pp of acceptance on identical inputs.

    At temperature > 0 (``sampling.is_greedy`` false) acceptance switches
    to Leviathan-Chen probability-ratio rejection sampling over the
    effective distributions (see :mod:`.speculative_sampling`), with the
    residual resample supplying the correction; depth is forced to 1.

    The loop owns verification, accept/reject, and cache reconciliation —
    preferring the language model's own ``rollback_speculative_cache`` when
    it exists (gemma4; no snapshots, no replay), falling back to
    snapshot/restore with *deferred replay* for hybrid-SSM caches (qwen
    GDN) — committed-but-restored tokens ride at the front of the next
    verify forward instead of paying a dedicated replay pass — and plain
    trim for pure-KV models. Drafters stay behind the
    :class:`~exo.worker.engines.mlx.drafters.protocol.Drafter` protocol,
    and the pair-stream contract holds: ``draft()`` consumes the
    ``(h, b)`` pair, and the round's accepted drafts are observed as
    ``(v_h[:, :p], drafts[:p])`` so stateful drafters keep gapless
    positional history.

    This path is rank-symmetric and used for single-node, tensor-parallel,
    and pipeline inference alike — distributed ranks run it in validated
    lockstep (#201 Tracks 1-2a): TP collectives and pipeline's decode-mode
    all_gather both give every rank identical logits, and per-request RNG
    seeding aligns every sampled accept/reject decision. Sidecar drafters
    are rank-local (replicated embed/head, zero collectives).

    Assistant drafters (gemma4) cross-attend the target's KV cache, which a
    pipeline shard only holds for its own layers — under pipeline they run
    on the LAST rank only (#201 Track 2b: the last full-attention and
    sliding KV layers live there by construction, and the post-norm hidden
    is already all-gathered to every rank). Pass ``draft_group`` to enable
    the distributed-draft exchange: the drafting rank (non-None *drafter*)
    drafts locally and every rank joins one fixed-shape ``all_sum`` per
    round that lands the draft tokens (and, under sampling, the drafter's
    effective distribution) everywhere — keeping the collective schedule
    symmetric while only one rank pays the draft. Drafting-rank sampled
    draws use an explicit per-round key so the shared global RNG stream
    stays aligned across ranks.
    """
    if len(prompt) == 0:
        raise ValueError("MTP decode requires at least one prompt token")
    if drafter is None and draft_group is None:
        raise ValueError("drafter may only be None in distributed-draft mode")

    detokenizer = cast(_DetokenizerProtocol, cast(object, tokenizer.detokenizer))
    detokenizer.reset()
    if drafter is not None:
        drafter.begin_request(prompt_cache)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    eos_token_ids = {int(t) for t in eos_ids_from_tokenizer(tokenizer)}
    token_history: mx.array | None = None
    if sampling is None:
        sampling = SamplingParams(temperature=0.0)
    # Sampled decoding forces depth 1: the drafter chains greedily, which
    # does not represent the sampled trajectory beyond the first draft.
    depth = max(depth, 1) if sampling.is_greedy else 1

    # Model-native speculative rollback (gemma4): no snapshots, no replay.
    _lm: object | None = getattr(model, "language_model", None)
    native_rollback = cast(
        "Callable[..., None] | None",
        getattr(_lm, "rollback_speculative_cache", None) if _lm is not None else None,
    )
    # Hybrid models (e.g. Qwen3.5 GDN) carry recurrent SSM state that cannot
    # be trimmed positionally — and trim_cache without a snapshot ZEROES it.
    # Rejects must restore a pre-verify snapshot instead (unless the model
    # provides native rollback, which handles its own cache types).
    mtp_has_ssm = has_non_kv_caches(prompt_cache) and not callable(native_rollback)

    # Deferred replay (snapshot path only): committed tokens whose cache
    # entries were lost to a reject-restore. Instead of paying a dedicated
    # replay forward per reject (~a full vanilla step), they ride along at
    # the front of the next verify — extra verify *width* is effectively
    # free on memory-bound decode (measured 46.6ms 2-wide vs 47.8ms 1-wide
    # on Qwen3.5-9B). Capped so pathological reject streaks cannot grow the
    # verify window without bound.
    pending_replay: list[int] = []
    # Cross-attending drafters read the TARGET's cache when drafting, so
    # deferral starves them of the newest committed tokens. Derived
    # rank-invariantly: distributed-draft mode is assistant-only, and on
    # local/TP placements every rank builds the same drafter object.
    drafter_reads_target_cache = draft_group is not None or bool(
        getattr(drafter, "reads_target_cache", False)
    )
    # Per-request seed component for the drafting rank's explicit draw
    # keys. Drawn from the SEEDED global stream on EVERY rank (one
    # symmetric draw, so the shared stream stays aligned): without it the
    # distributed sampled proposal would be a fixed function of the round
    # index, ignoring task.seed and breaking the over-seeds output
    # distribution Leviathan-Chen acceptance guarantees.
    draft_key_base = 0
    if draft_group is not None and not sampling.is_greedy:
        draft_key_base = int(mx.random.randint(0, 2**31 - 1, shape=()).item())

    generated_count = 0
    accepted = 0
    attempted_drafts = 0
    finish_reason = "length"
    speculation_disabled = False

    def _response(
        token_int: int,
        logprobs_row: mx.array,
        *,
        from_draft: bool,
        finish: str | None,
    ) -> MlxGenerationResponse:
        """Build a response for *token_int* using the enclosing loop state.

        Terminal responses (``finish`` set) finalize the detokenizer first
        so buffered partial byte/BPE sequences flush into the final segment
        — sentencepiece-backed tokenizers buffer tail bytes until
        ``finalize()`` (#180 item 4; tiktoken-backed targets flush eagerly,
        so this is latent for current models).
        """
        if finish is not None:
            detokenizer.finalize()
        return MlxGenerationResponse(
            text=detokenizer.last_segment,
            token=token_int,
            logprobs=logprobs_row.squeeze(0)
            if logprobs_row.ndim > 1
            else logprobs_row,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=generated_count,
            generation_tps=generated_count
            / max(time.perf_counter() - generation_start, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None if finish is None else finish,
        )

    def _sample_row(lp_row: mx.array) -> tuple[int, mx.array]:
        """Sample one token from a normalized logprob row, with processors."""
        nonlocal token_history
        row = lp_row
        if logits_processors and token_history is not None:
            for proc in logits_processors:
                row = proc(token_history, row[None])[0]
            row = row.astype(mx.float32) - mx.logsumexp(
                row.astype(mx.float32), keepdims=True
            )
        tok = sampler(row)
        return int(tok.item()), row

    def _record_history(tokens: list[int]) -> None:
        nonlocal token_history
        if not logits_processors or not tokens:
            return
        addition = mx.array(tokens)
        token_history = (
            mx.concat([token_history, addition])
            if token_history is not None
            else addition
        )

    def _flush_pending_replay() -> None:
        """Forward deferred-replay tokens so the prompt cache catches up.

        Called when the next forward cannot carry them (plain-decode
        fallback), when the pending window hits its cap, and at stream end
        so a persisted prefix cache stays consistent with the emitted text.
        """
        nonlocal pending_replay
        if not pending_replay:
            return
        with mx.stream(generation_stream):
            trunk_fn(mx.array(pending_replay)[None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
        pending_replay = []

    # ---- prefill (loop-internal tail; the heavy prefill ran upstream) ----
    prompt_start = time.perf_counter()
    if len(prompt) > 1:
        with mx.stream(generation_stream):
            prefill_hidden = trunk_fn(prompt[:-1][None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            # Bulk-ingest the prompt pairs so stateful drafters start decode
            # with positional history (positions 0..P-2 pair with tokens
            # 1..P-1).
            if drafter is not None:
                drafter.observe(prefill_hidden[0], prompt[1:])
            mx.eval([c.state for c in prompt_cache])  # type: ignore[attr-defined]
    _record_history([int(t) for t in cast("list[int]", prompt.tolist())])

    # First bonus: forward the last prompt token, sample from its logits.
    with mx.stream(generation_stream):
        h_seq = trunk_fn(prompt[-1:][None], cache=prompt_cache)
        quantize_cache_fn(prompt_cache)
        hidden = h_seq[0, -1, :]
        first_logits = head_fn(hidden[None, None])[0, 0, :].astype(mx.float32)
        first_lp = first_logits - mx.logsumexp(first_logits, keepdims=True)
        mx.eval(hidden, first_lp)

    prompt_tps = len(prompt) / max(time.perf_counter() - prompt_start, 1e-9)
    generation_start = time.perf_counter()
    # Distributed-draft payloads carry a full-vocab row under sampling; the
    # non-drafting ranks size their zero contribution from here.
    vocab_size = int(first_lp.shape[0])

    bonus, bonus_lp = _sample_row(first_lp)
    # EOS never enters the detokenizer (matching the non-MTP decode paths
    # and the accepted-draft emit below) — its decoded special-token text
    # must not leak into the terminal segment.
    if bonus not in eos_token_ids:
        detokenizer.add_token(bonus)
    generated_count = 1
    _record_history([bonus])
    if bonus in eos_token_ids or generated_count >= max_tokens:
        finish_reason = "stop" if bonus in eos_token_ids else "length"
        yield _response(bonus, bonus_lp, from_draft=False, finish=finish_reason)
        return
    yield _response(bonus, bonus_lp, from_draft=False, finish=None)

    while generated_count < max_tokens:
        if speculation_disabled:
            # Plain decode: forward the bonus, sample the next one.
            _flush_pending_replay()
            with mx.stream(generation_stream):
                h_seq = trunk_fn(mx.array([[bonus]]), cache=prompt_cache)
                quantize_cache_fn(prompt_cache)
                hidden = h_seq[0, -1, :]
                lp = head_fn(hidden[None, None])[0, 0, :].astype(mx.float32)
                lp = lp - mx.logsumexp(lp, keepdims=True)
                mx.eval(hidden, lp)
            bonus, bonus_lp = _sample_row(lp)
            # EOS never enters the detokenizer — see the first-bonus emit.
            if bonus not in eos_token_ids:
                detokenizer.add_token(bonus)
            generated_count += 1
            _record_history([bonus])
            finish = (
                "stop"
                if bonus in eos_token_ids
                else finish_reason
                if generated_count >= max_tokens
                else None
            )
            yield _response(bonus, bonus_lp, from_draft=False, finish=finish)
            if finish is not None:
                if finish == "stop":
                    finish_reason = "stop"
                break
            continue

        # ---- DRAFT from (hidden, bonus) ----
        draft_probs: mx.array | None = None
        if draft_group is not None:
            # Distributed drafts (#201 Track 2b): only the drafting rank
            # holds a drafter; the exchange lands identical tokens (and the
            # effective draft distribution, under sampling) on every rank.
            # No try/except: distributed drafter failures are always loud.
            with mx.stream(generation_stream):
                draft_toks, draft_probs = _exchange_drafts(
                    draft_group=draft_group,
                    drafter=drafter,
                    hidden=hidden,
                    bonus=bonus,
                    depth=depth,
                    sampling=sampling,
                    vocab_size=vocab_size,
                    draft_key=draft_key_base + generated_count,
                )
        else:
            assert drafter is not None  # checked at entry
            try:
                with mx.stream(generation_stream):
                    chain_logits = drafter.draft(hidden, bonus, depth=depth).astype(
                        mx.float32
                    )  # (K, V), K <= depth
                    if sampling.is_greedy:
                        # Log-softmax is rank-preserving: argmax reads raw
                        # logits.
                        chain_sampled = sampler(chain_logits)  # (K,)
                    else:
                        chain_lp = chain_logits - mx.logsumexp(
                            chain_logits, axis=-1, keepdims=True
                        )
                        draft_probs = warp_to_probs(chain_lp[0], sampling)
                        chain_sampled = sampler(chain_lp)  # (1,)
                    mx.eval(chain_sampled)
            except Exception as draft_error:  # noqa: BLE001 — best-effort
                if fail_loud_on_drafter_error:
                    # Multi-rank placements: a rank-local fallback to plain
                    # decode would silently fork the collective schedule and
                    # corrupt/wedge the peers — abort the request loudly
                    # instead (#201 Track 2a, design seam 3). Deterministic
                    # drafter failures hit every rank identically before
                    # this point; what reaches here is resource-class (OOM).
                    raise
                logger.warning(
                    f"Drafter failed ({draft_error}); disabling speculation "
                    "for the remainder of this request"
                )
                speculation_disabled = True
                continue

            draft_toks = [int(t) for t in cast("list[int]", chain_sampled.tolist())]
        # Truncate after the first EOS draft (keep the EOS itself: the
        # verifier may legitimately accept it and end the stream).
        for eos_index, candidate in enumerate(draft_toks):
            if candidate in eos_token_ids:
                draft_toks = draft_toks[: eos_index + 1]
                break

        # ---- VERIFY: [pending..., bonus, d0..dK-1] — the round's only ----
        # ---- target forward. Deferred-replay tokens ride at the front; ----
        # ---- their rows are discarded (they were scored last round).   ----
        chain_len = len(draft_toks)
        replay_len = len(pending_replay)
        pre_verify_snapshot = snapshot_ssm_states(prompt_cache) if mtp_has_ssm else None
        verify_input = mx.array([*pending_replay, bonus, *draft_toks])
        with mx.stream(generation_stream):
            full_h = trunk_fn(verify_input[None], cache=prompt_cache)  # (1,R+K+1,H)
            quantize_cache_fn(prompt_cache)
            v_h = full_h[:, replay_len:, :]  # (1, K+1, H)
            v_logits = head_fn(v_h)  # (1, K+1, V)
            v_lp = v_logits[0].astype(mx.float32)
            v_lp = v_lp - mx.logsumexp(v_lp, axis=-1, keepdims=True)

        if sampling.is_greedy:
            verify_sampled = sampler(v_lp)  # (K+1,)
            mx.eval(v_h, verify_sampled)
            target_ints = [int(t) for t in cast("list[int]", verify_sampled.tolist())]
            prefix_len = 0
            while (
                prefix_len < chain_len
                and draft_toks[prefix_len] == target_ints[prefix_len]
            ):
                prefix_len += 1
            # Row prefix_len yields the next bonus: the correction on a
            # partial reject, the free next token on a full accept. (With
            # processors active the row is re-sampled through them below.)
            raw_bonus_next = target_ints[prefix_len] if not logits_processors else None
        else:
            assert draft_probs is not None and chain_len == 1
            verify_probs = warp_to_probs(v_lp[0], sampling)
            mx.eval(v_h, verify_probs)
            if ratio_accept(draft_toks[0], draft_probs, verify_probs):
                prefix_len = 1
                raw_bonus_next = None  # sampled from row 1 below
            else:
                prefix_len = 0
                raw_bonus_next = residual_sample(draft_probs, verify_probs)
        attempted_drafts += chain_len
        accepted += prefix_len
        # Consumers usually abandon this generator mid-stream and the public
        # GenerationResponse carries no from_draft field — this periodic
        # line is the production acceptance signal (issue #192 false alarm).
        if attempted_drafts // 32 != (attempted_drafts - chain_len) // 32:
            logger.info(
                f"MTP acceptance so far: {accepted}/{attempted_drafts} "
                f"({accepted / attempted_drafts:.0%})"
            )

        full_accept = prefix_len == chain_len

        # ---- cache reconciliation BEFORE emitting (emits may break out) ----
        # Verify forwarded R+K+1 positions: [pending, bonus, drafts].
        # Committed: pending + bonus + accepted prefix; the next bonus is
        # NOT forwarded (next round's verify carries it).
        if not full_accept:
            rejected = chain_len - prefix_len
            if callable(native_rollback):
                # Native rollback never coexists with deferred replay
                # (pending accrues on the snapshot path only).
                with mx.stream(generation_stream):
                    native_rollback(prompt_cache, None, prefix_len, chain_len + 1)
            elif pre_verify_snapshot is not None:
                # Hybrid model: SSM state cannot be trimmed positionally, so
                # restore the pre-verify snapshot and DEFER the committed
                # prefix to the next verify instead of paying a dedicated
                # replay forward now (a reject used to cost a full extra
                # trunk pass; riding along in the next verify is free).
                trim_cache(prompt_cache, replay_len + chain_len + 1, pre_verify_snapshot)
                pending_replay = [*pending_replay, bonus, *draft_toks[:prefix_len]]
                if (
                    drafter_reads_target_cache
                    or len(pending_replay) >= _MTP_MAX_PENDING_REPLAY
                ):
                    # Cross-attending drafters (gemma4 assistants) read the
                    # TARGET's KV cache when drafting — deferred tokens are
                    # invisible to them, so every post-reject draft would
                    # run against a stale cache and crater acceptance
                    # (measured 74% -> 28% on a 31B target whose loader
                    # lacks native rollback). Pay the replay immediately
                    # for those drafters; sidecars keep the free deferral.
                    _flush_pending_replay()
            else:
                trim_cache(prompt_cache, rejected)
        else:
            # Full accept: the verify forward itself committed the pending
            # tokens (and everything after them).
            pending_replay = []

        # Pair-stream: draft() consumed the (hidden, bonus) pair; the
        # accepted drafts commit as next-tokens for positions bonus..d_{p-2},
        # whose hiddens are v_h rows 0..p-1. The next bonus pair is consumed
        # by the next draft() from v_h[:, p].
        if prefix_len > 0 and drafter is not None:
            drafter.observe(
                v_h[0, :prefix_len, :], mx.array(draft_toks[:prefix_len])
            )

        # ---- emit accepted drafts ----
        stream_done = False
        committed_this_round: list[int] = []
        for i in range(prefix_len):
            token_int = draft_toks[i]
            committed_this_round.append(token_int)
            if token_int in eos_token_ids:
                generated_count += 1
                finish_reason = "stop"
                yield _response(token_int, v_lp[i], from_draft=True, finish="stop")
                stream_done = True
                break
            detokenizer.add_token(token_int)
            generated_count += 1
            if generated_count >= max_tokens:
                yield _response(
                    token_int, v_lp[i], from_draft=True, finish=finish_reason
                )
                stream_done = True
                break
            yield _response(token_int, v_lp[i], from_draft=True, finish=None)
        if stream_done:
            break
        _record_history(committed_this_round)

        # ---- next bonus from row prefix_len ----
        if raw_bonus_next is not None:
            bonus = raw_bonus_next
            bonus_lp = v_lp[prefix_len]
        else:
            bonus, bonus_lp = _sample_row(v_lp[prefix_len])
        # EOS never enters the detokenizer — see the first-bonus emit.
        if bonus not in eos_token_ids:
            detokenizer.add_token(bonus)
        generated_count += 1
        _record_history([bonus])
        finish = (
            "stop"
            if bonus in eos_token_ids
            else finish_reason
            if generated_count >= max_tokens
            else None
        )
        if finish == "stop":
            finish_reason = "stop"
            generated_count = generated_count  # emitted below with finish
        yield _response(bonus, bonus_lp, from_draft=False, finish=finish)
        if finish is not None:
            break

        # Hidden at the last committed position seeds the next round.
        hidden = v_h[0, prefix_len, :]

    # Catch the cache up on any still-deferred tokens so a persisted prefix
    # cache stays consistent with the emitted text. (Abandoned generators
    # skip this, but they hold a private deepcopy from get_kv_cache — never
    # the stored entry — so nothing shared can go stale.)
    _flush_pending_replay()

    # No finalize here: every terminal yield above finalizes inside
    # _response — finalization happens exactly once, structurally.
    acceptance_rate = accepted / attempted_drafts if attempted_drafts > 0 else 0.0
    logger.debug(
        f"MTP decode complete: {generated_count} tokens, "
        f"acceptance={acceptance_rate:.1%} ({accepted}/{attempted_drafts})"
    )
    # Yield the final state accumulated in the detokenizer
    # (tokens already yielded above via the break path)


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
    mtp_weights: "dict[str, mx.array] | None" = None,
    assistant_model: object | None = None,
    model_card: ModelCard | None = None,
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
    uses_pipeline_decode = group is not None and _has_pipeline_communication_layer(model)
    if uses_pipeline_decode:
        prefill_prompt_tokens = (
            prompt_tokens if len(prompt_tokens) > 1 else prompt_tokens[:0]
        )
    else:
        prefill_prompt_tokens = prompt_tokens[:-1]

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
                prefill_prompt_tokens,
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
        with runner_phase(
            "decode_stream",
            detail="clear_prefill_mlx_cache",
            task_id=trace_task_id,
            include_memory=True,
        ):
            mx.clear_cache()
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    # Pipeline prefill now advances the cache through the penultimate prompt
    # token, so pipeline decode can begin with a single token and avoid an
    # extra decode-mode prompt bridge collective before the first generated
    # token.
    last_token = prompt_tokens[-1:] if uses_pipeline_decode else prompt_tokens[-2:]

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

    # Resolve a drafter for single-node speculative decoding.
    # Greedy uses argmax-prefix acceptance; temperature > 0 uses
    # Leviathan-Chen probability-ratio acceptance (distribution-preserving,
    # issue #180) with the effective sampler distributions mirrored by
    # SamplingParams — which covers temp/top_p/min_p/top_k but not
    # arbitrary samplers, so the params below must stay in lockstep with
    # the make_sampler call above.
    _mtp_sampling = SamplingParams(
        temperature=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        min_p=task.min_p if task.min_p is not None else 0.05,
        top_k=task.top_k if task.top_k is not None else 0,
    )
    _drafter: Drafter | None = None
    _trunk_fn: Callable[..., mx.array] | None = None
    _head_fn: Callable[..., mx.array] | None = None
    _speculation_assets = mtp_weights is not None or assistant_model is not None
    # Assistant drafters on pipeline placements draft on the LAST rank only
    # (#201 Track 2b): assets load there alone, but every rank must resolve
    # trunk/head and join the distributed-draft exchange — so the mode is
    # derived from the rank-invariant card declaration, not local assets.
    _pipeline_assistant_mode = (
        model_card is not None
        and model_card.runtime is not None
        and model_card.runtime.assistant_model_repo is not None
        and group is not None
        and group.size() > 1
        and _has_pipeline_communication_layer(model)
    )
    if _speculation_assets or _pipeline_assistant_mode:
        if logits_processors:
            # Accepted draft tokens are committed from RAW verifier logits —
            # logits processors (repetition penalty, bench EOS ban) are only
            # applied on the main-forward sampling path, so a constrained
            # token could slip through via an accepted draft. Until the
            # verify pass applies processors, MTP and processors are
            # mutually exclusive.
            logger.info(
                "MTP speculative decoding is incompatible with logits "
                f"processors ({len(logits_processors)} active); skipping MTP"
            )
        else:
            trunk_head = _get_trunk_and_head(model)
            if trunk_head is not None:
                _trunk_fn, _head_fn = trunk_head
                if _speculation_assets:
                    _drafter = build_drafter(
                        model,
                        mtp_weights,
                        assistant_model=assistant_model,
                        runtime=model_card.runtime
                        if model_card is not None
                        else None,
                    )
                if _drafter is not None:
                    _card_draft_depth = (
                        model_card.runtime.mtp_max_depth
                        if model_card is not None
                        and model_card.runtime is not None
                        and model_card.runtime.mtp_max_depth is not None
                        else 1
                    )
                    logger.info(
                        "MTP speculative decoding enabled "
                        f"(D={_card_draft_depth})"
                    )

    # Gate the agreement on the model CARD declaring speculation — the card
    # is identical on every rank, so the collective count stays symmetric;
    # what varies per rank (a missing sidecar download, a failed drafter
    # build) is exactly what the collective settles.
    _distributed_drafts_ok = False
    _card_declares_speculation = (
        model_card is not None
        and model_card.runtime is not None
        and (
            model_card.runtime.mtp_sidecar_repo is not None
            or model_card.runtime.assistant_model_repo is not None
        )
    )
    if _card_declares_speculation and group is not None and group.size() > 1:
        # Distributed lockstep requires every rank to make the same
        # speculate-or-not choice: a rank whose sidecar download is missing
        # or whose drafter build failed would otherwise run plain decode
        # while its peers run speculative rounds — desynchronizing the
        # collective schedule. One tiny all_sum per request settles it
        # (#201 Track 2a, design seam 3).
        drafter_ready = mx.distributed.all_sum(
            mx.array(1.0 if _drafter is not None else 0.0),
            group=group,
            stream=mx.default_stream(mx.Device(mx.cpu)),
        )
        eval_with_timeout(
            drafter_ready,
            timeout_seconds=pipeline_eval_timeout_seconds(),
            on_timeout=pipeline_timeout_callback(
                "mtp_drafter_agreement",
                {"group_size": group.size()},
                is_prefill=False,
            ),
        )
        ready_count = int(drafter_ready.item())
        if _pipeline_assistant_mode:
            # Last-rank drafting: exactly ONE rank should hold the
            # assistant; every rank joins the per-round draft exchange.
            _distributed_drafts_ok = ready_count == 1 and _trunk_fn is not None
            if not _distributed_drafts_ok:
                if _drafter is not None or ready_count > 0:
                    logger.warning(
                        "Assistant speculation disabled for this request: "
                        f"{ready_count} rank(s) hold a drafter (expected "
                        "exactly 1 on the last pipeline rank)"
                    )
                _drafter = None
        elif _drafter is not None and ready_count != group.size():
            logger.warning(
                f"Speculation disabled for this request: only {ready_count}/"
                f"{group.size()} ranks have a working drafter"
            )
            _drafter = None

    with runner_phase(
        "decode_stream",
        detail="stream_generate_setup",
        attrs={
            "max_tokens": max_tokens,
            "last_token_count": len(last_token),
            "mtp": _drafter is not None,
        },
        task_id=trace_task_id,
        include_memory=True,
    ):
        if _drafter is not None or _distributed_drafts_ok:
            # The MTP loop is rank-symmetric and runs on single-node, TP,
            # and pipeline placements alike (#201 Tracks 1-2a): pipeline
            # decode all-gathers the final hidden to every rank and
            # embed/norm/head are replicated, so every rank sees identical
            # logits and the per-request seed aligns sampled draws.
            # Assistant drafters on pipeline draft on the last rank only
            # and fan drafts out via the per-round exchange (Track 2b).
            assert _trunk_fn is not None and _head_fn is not None
            token_generator = _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=_drafter,
                trunk_fn=_trunk_fn,
                head_fn=_head_fn,
                prompt=last_token,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=caches,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                depth=(
                    model_card.runtime.mtp_max_depth
                    if model_card is not None
                    and model_card.runtime is not None
                    and model_card.runtime.mtp_max_depth is not None
                    else 1
                ),
                sampling=_mtp_sampling,
                # A rank-local drafter failure mid-request would silently
                # fork the collective schedule; on multi-rank placements
                # the loop aborts loud instead (peers surface via the
                # pipeline eval watchdogs).
                fail_loud_on_drafter_error=group is not None and group.size() > 1,
                draft_group=group if _distributed_drafts_ok else None,
            )
        elif group is not None and _has_pipeline_communication_layer(model):
            logger.info(
                "Using sequential pipeline decode without stream_generate lookahead"
            )
            token_generator = _stream_generate_without_lookahead(
                model=model,
                tokenizer=tokenizer,
                prompt=last_token,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=caches,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
            )
        else:
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
