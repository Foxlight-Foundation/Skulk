import contextlib
import functools
import math
import os
import sys
import threading
import time
import traceback
from copy import deepcopy
from importlib import metadata
from typing import Callable, Generator, cast, get_args

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
from exo.shared.models.model_cards import ModelCard
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

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5
_MIN_CANCEL_CHECK_INTERVAL = 10


def _preferred_env_value(skulk_key: str, exo_key: str) -> str | None:
    """Return the SKULK env value when present, else the legacy EXO value."""
    if skulk_key in os.environ:
        return os.environ[skulk_key]
    return os.environ.get(exo_key)


def _mlx_hang_debug_enabled() -> bool:
    """Return whether verbose warmup/prefill hang diagnostics are enabled."""
    value = _preferred_env_value("SKULK_MLX_HANG_DEBUG", "EXO_MLX_HANG_DEBUG")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _mlx_hang_debug_interval_seconds() -> float:
    """Return the periodic interval for hang-debug watchdog logs."""
    raw = _preferred_env_value(
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
    raw = _preferred_env_value(
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
    raw = _preferred_env_value(
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
            frame = sys._current_frames().get(monitored_thread_id)
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

    # Build list of real prompt chunk sizes
    total = len(prompt)
    real_chunk_sizes: list[int] = []
    remaining = total - 1
    while remaining:
        n = min(prefill_step_size, remaining)
        real_chunk_sizes.append(n)
        remaining -= n
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


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
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

    def combined_progress_callback(processed: int, total: int) -> None:
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    is_pipeline = _has_pipeline_communication_layer(model)

    prefill_step_size = 4096
    effective_prefill_step_size = (
        prefill_step_size // min(4, group_size) if is_pipeline else prefill_step_size
    )

    # Pipeline-parallel prefill can wedge when the prompt fits inside a single
    # effective per-rank chunk (i.e. ``n_real == 1`` in the prefill loop).
    # Mirror pipeline_parallel_prefill's chunking math here by reserving the
    # final token for the post-loop pass before computing the real chunk count.
    # That keeps multi-chunk prompts on pipeline_parallel_prefill while routing
    # single-chunk short / warmup prompts through stream_generate.
    real_prefill_tokens = max(num_tokens - 1, 0)
    pipeline_chunks = (
        (real_prefill_tokens + effective_prefill_step_size - 1)
        // effective_prefill_step_size
        if effective_prefill_step_size > 0
        else 0
    )
    use_pipeline_prefill = is_pipeline and pipeline_chunks >= 2
    logger.info(
        "Prefill path selected: "
        f"{'pipeline_parallel_prefill' if use_pipeline_prefill else 'stream_generate'} "
        f"(rank={rank}, prompt_tokens={num_tokens}, is_pipeline={is_pipeline}, "
        f"prefill_step_size_input={prefill_step_size}, "
        f"prefill_step_size_effective={effective_prefill_step_size}, "
        f"pipeline_chunks={pipeline_chunks})"
    )
    # Only enable pipeline-prefill mode for the true pipeline prefill path.
    # Short prompts routed through stream_generate must behave like normal
    # generation; leaving pipeline wrappers in prefill mode there can strand
    # non-zero ranks in the pipeline send/recv path without the coordinated
    # queue/flush behavior used by pipeline_parallel_prefill.
    set_pipeline_prefill(model, is_prefill=use_pipeline_prefill)

    with _hang_debug_watch(f"prefill barrier rank={rank} group_size={group_size}"):
        mx_barrier(group)
    logger.info(
        f"Starting prefill (rank={rank}, group_size={group_size}, prompt_tokens={num_tokens})"
    )

    try:
        if use_pipeline_prefill:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            with _hang_debug_watch(
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
                    distributed_prompt_progress_callback=distributed_prompt_progress_callback,
                    group=group,
                )
        else:
            # Use max_tokens=1 because max_tokens=0 does not work.
            # We just throw away the generated token - we only care about filling the cache
            with _hang_debug_watch(
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
                    prompt_progress_callback=combined_progress_callback,
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

    if _is_distributed_warmup(group) and (
        _warmup_repeat_count() != 1 or _warmup_instructions(None) is not None
    ):
        logger.info(
            "Distributed pipeline warmup is forcing the minimal sanity-check "
            "prompt and ignoring warmup shaping overrides"
        )

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
        max_output_tokens=1024,
        enable_thinking=False,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

    with _hang_debug_watch(f"warmup apply_chat_template model={model_id}"):
        warmup_prompt = apply_chat_template(
            tokenizer=tokenizer,
            task_params=warmup_task_params,
            model_card=model_card,
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

    with _hang_debug_watch(f"warmup pre-generation barrier model={model_id}"):
        mx_barrier(group)

    logger.info("Generating warmup tokens")

    t = time.monotonic()

    with _hang_debug_watch(f"warmup mlx_generate model={model_id}"):
        for _r in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=warmup_task_params,
            prompt=warmup_prompt,
            kv_prefix_cache=None,
            group=group,
        ):
            tokens_generated += 1
            # Warmup is only meant to validate prefill plus the first decode
            # step. Stopping after the first token keeps runner readiness
            # bounded even if the model would otherwise continue sampling.
            break

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

    with _hang_debug_watch(
        f"warmup final barrier model={model_id} tokens_generated={tokens_generated}"
    ):
        mx_barrier(group)

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    if group is not None:
        check_for_cancel_every = int(
            mx.max(
                mx.distributed.all_gather(
                    mx.array([check_for_cancel_every]),
                    group=group,
                )
            ).item()
        )

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
) -> Generator[GenerationResponse]:
    """Generate for native-vision models via MLX-VLM's multimodal path.

    Gemma 4's reference MLX-VLM generation path computes multimodal input
    embeddings up front through ``model.get_input_embeddings(...)`` and then
    drives ``model.language_model`` directly. Running native-vision models
    through generic ``mlx_lm.stream_generate`` can diverge from that path even
    when prompt tokens and image preprocessing are correct, so we follow the
    reference execution strategy here.
    """
    from mlx_vlm.generate import generate_step as vlm_generate_step

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

    detokenizer = tokenizer.detokenizer
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

    logger.info("Starting native mlx-vlm multimodal decode")
    mx_barrier(group)

    for token, logprobs in vlm_generate_step(
        input_ids=all_prompt_tokens[None],
        model=model,
        pixel_values=vision.pixel_values,
        mask=None,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prefill_step_size=4096,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
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
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)
    min_prefix_hit_length = max(1000, system_prompt_token_count(task, tokenizer))

    vision: VisionResult | None = None
    if vision_processor is not None:
        try:
            vision = prepare_vision(
                images=task.images,
                chat_template_messages=task.chat_template_messages,
                vision_processor=vision_processor,
                tokenizer=tokenizer,
                model=model,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Vision processing failed, falling back to text-only"
            )
    if vision is not None:
        all_prompt_tokens = vision.prompt_tokens
    media_regions: list[MediaRegion] = vision.media_regions if vision else []

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
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
            )
            return

        logger.info(
            "Using pipeline-aware native vision generation path on fixed "
            "mlx/mlx-vlm stack"
        )

    is_native_vision = vision is not None and vision.pixel_values is not None
    native_pixel_values: mx.array | list[mx.array] | None = None
    if is_native_vision:
        assert vision is not None
        native_pixel_values = _slice_native_pixel_values_for_uncached_suffix(
            vision.pixel_values,
            media_regions,
            prefix_hit_length,
        )

    if native_pixel_values is not None:
        if hasattr(model, "set_pixel_values"):
            model.set_pixel_values(native_pixel_values)  # type: ignore[attr-defined]
        else:
            model._pixel_values = native_pixel_values  # type: ignore[attr-defined]
        maybe_vision_ctx = contextlib.nullcontext()
    elif vision is not None and not is_native_vision:
        maybe_vision_ctx = patch_embed_tokens(
            model, vision.embeddings, prefix_hit_length, len(prompt_tokens) - 1
        )
    else:
        maybe_vision_ctx = contextlib.nullcontext()
    try:
        with maybe_vision_ctx:
            prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
                model,
                tokenizer,
                sampler,
                prompt_tokens[:-1],
                caches,
                group,
                on_prefill_progress,
                distributed_prompt_progress_callback,
            )
    finally:
        if hasattr(model, "set_pixel_values"):
            model.set_pixel_values(None)  # type: ignore[attr-defined]
        elif hasattr(model, "_pixel_values"):
            model._pixel_values = None  # type: ignore[attr-defined]
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

    logger.info("Starting decode")
    mx_barrier(group)

    for completion_tokens, out in enumerate(
        stream_generate(
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
        ),
        start=1,
    ):
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
                    kv_prefix_cache.update_kv_cache(
                        matched_index,
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        restore_pos=prefix_hit_length,
                        media_regions=media_regions,
                    )
                else:
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
            mx_barrier(group)
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
