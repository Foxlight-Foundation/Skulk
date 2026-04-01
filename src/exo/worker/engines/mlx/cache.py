import os
from copy import deepcopy
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    DEFAULT_KV_CACHE_BACKEND,
    DEFAULT_TURBOQUANT_K_BITS,
    DEFAULT_TURBOQUANT_V_BITS,
    KV_CACHE_BACKEND,
    KV_CACHE_BITS,
    OPTIQ_BITS,
    OPTIQ_FP16_LAYERS,
    TURBOQUANT_FP16_LAYERS,
    TURBOQUANT_K_BITS,
    TURBOQUANT_V_BITS,
    KVCacheBackend,
)
from exo.worker.engines.mlx.turboquant import (
    make_turboquant_adaptive_cache,
    make_turboquant_cache_from_template,
)
from exo.worker.engines.mlx.turboquant.cache import ensure_standard_attention
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self, states: list[RotatingKVCache | ArraysCache | None], token_count: int
    ):
        self.states = states
        self.token_count = token_count


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[ArraysCache | RotatingKVCache | None] = []
    for c in cache:
        if isinstance(c, (ArraysCache, RotatingKVCache)):
            states.append(deepcopy(c))
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(isinstance(c, (ArraysCache, RotatingKVCache)) for c in cache)


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self._access_counter: int = 0
        self._group = group

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        logger.info(f"KV cache added: {len(prompt_tokens)} tokens")

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = deepcopy(cache)
        self._snapshots[index] = merged or None
        self._media_regions[index] = media_regions or []
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        logger.info(f"KV cache updated (index {index}): {len(prompt_tokens)} tokens")

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache match
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        if best_index is None:
            return make_kv_cache(model), prompt_tokens, None

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        target = (max_length - 1) if is_exact and not has_ssm else best_length
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model), prompt_tokens, None

        prompt_cache = deepcopy(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset = restore_pos  # pyright: ignore[reportAttributeAccessIssue]

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

        # Evict LRU entries until below threshold
        while (
            len(self.caches) > 0
            and self.get_memory_used_percentage() > _MEMORY_THRESHOLD
        ):
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._snapshots.pop(lru_index)
            self._media_regions.pop(lru_index)
            self._last_used.pop(lru_index)
            logger.info(
                f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory usage"
            )

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        # .item() evals.
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        if isinstance(c, (ArraysCache, RotatingKVCache)):
            if snapshot is not None and snapshot.states[i] is not None:
                cache[i] = deepcopy(snapshot.states[i])  # type: ignore
            else:
                c.state = [None] * len(c.state)
        else:
            trim_fn = getattr(c, "trim", None)
            if callable(trim_fn):
                trim_fn(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(c: object) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return int(c.offset)  # type: ignore[attr-defined]
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max(_entry_length(c) for c in cache)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    backend = get_kv_cache_backend()
    if max_kv_size is not None:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]

    if backend == "mlx_quantized":
        if KV_CACHE_BITS is None:
            raise ValueError(
                "EXO_KV_CACHE_BACKEND=mlx_quantized requires EXO_KV_CACHE_BITS to be set"
            )
        logger.info(
            f"Using MLX quantized KV cache with bits={KV_CACHE_BITS} group_size={CACHE_GROUP_SIZE}"
        )
        if hasattr(model, "make_cache"):
            template_cache = cast(
                list[object],
                model.make_cache(),  # type: ignore[reportUnknownMemberType]
            )
            caches: list[object] = []
            for entry in template_cache:
                if isinstance(entry, KVCache):
                    caches.append(
                        QuantizedKVCache(
                            group_size=CACHE_GROUP_SIZE,
                            bits=KV_CACHE_BITS,
                        )
                    )
                else:
                    caches.append(entry)
            return caches
        return [
            QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
            for _ in model.layers
        ]

    if backend in ("turboquant", "turboquant_adaptive"):
        ensure_standard_attention(model)
        key_bits = TURBOQUANT_K_BITS or DEFAULT_TURBOQUANT_K_BITS
        value_bits = TURBOQUANT_V_BITS or DEFAULT_TURBOQUANT_V_BITS
        if backend == "turboquant_adaptive":
            logger.info(
                "Using TurboQuant adaptive KV cache "
                f"with key_bits={key_bits} value_bits={value_bits} fp16_layers={TURBOQUANT_FP16_LAYERS}"
            )
            return make_turboquant_adaptive_cache(
                model,
                key_bits=key_bits,
                value_bits=value_bits,
                fp16_layers=TURBOQUANT_FP16_LAYERS,
            )
        logger.info(
            f"Using TurboQuant KV cache with key_bits={key_bits} value_bits={value_bits}"
        )
        return make_turboquant_cache_from_template(
            model,
            key_bits=key_bits,
            value_bits=value_bits,
        )

    if backend == "optiq":
        try:
            from optiq.core.turbo_kv_cache import (  # type: ignore[import-untyped]
                TurboQuantKVCache as OptiqKVCache,
            )
            from optiq.core.turbo_kv_cache import (
                patch_attention,
            )
        except ImportError as exc:
            raise RuntimeError(
                "EXO_KV_CACHE_BACKEND=optiq requires mlx-optiq to be installed. "
                "Install with: pip install mlx-optiq"
            ) from exc

        def _is_power_of_two(n: int) -> bool:
            return n > 0 and (n & (n - 1)) == 0

        # Check model compatibility before patching attention.
        # optiq requires power-of-two head_dim and currently produces
        # incorrect output with GQA models (num_kv_heads != num_q_heads).
        if len(model.layers) > 0:
            sample_layer = model.layers[0]
            sample_attn = getattr(sample_layer, "self_attn", sample_layer)
            sample_head_dim = getattr(sample_attn, "head_dim", 128)
            n_heads = getattr(sample_attn, "n_heads", 0)
            n_kv_heads = getattr(sample_attn, "n_kv_heads", n_heads)

            incompatible = False
            if not _is_power_of_two(sample_head_dim):
                logger.warning(
                    f"mlx-optiq requires power-of-two head_dim but model has head_dim={sample_head_dim}; "
                    f"falling back to default KV cache"
                )
                incompatible = True
            elif n_kv_heads != 0 and n_kv_heads != n_heads:
                logger.warning(
                    f"mlx-optiq does not yet support GQA (n_heads={n_heads}, n_kv_heads={n_kv_heads}); "
                    f"falling back to default KV cache"
                )
                incompatible = True

            if incompatible:
                if hasattr(model, "make_cache"):
                    return model.make_cache()  # type: ignore
                return [KVCache() for _ in model.layers]

        patch_attention()
        logger.info(f"Using mlx-optiq KV cache with bits={OPTIQ_BITS}")

        if hasattr(model, "make_cache"):
            template_cache = cast(
                list[object],
                model.make_cache(),  # type: ignore[reportUnknownMemberType]
            )
            num_kv = sum(1 for entry in template_cache if isinstance(entry, KVCache))
            kv_pos = -1
            caches: list[object] = []
            for layer_i, entry in enumerate(template_cache):
                if isinstance(entry, KVCache):
                    kv_pos += 1
                    is_edge = kv_pos < OPTIQ_FP16_LAYERS or kv_pos >= max(
                        num_kv - OPTIQ_FP16_LAYERS, 0
                    )
                    if is_edge:
                        caches.append(KVCache())
                    else:
                        if 0 <= layer_i < len(model.layers):
                            attn = getattr(
                                model.layers[layer_i],
                                "self_attn",
                                model.layers[layer_i],
                            )
                            head_dim = getattr(attn, "head_dim", 128)
                        else:
                            head_dim = 128
                        caches.append(
                            OptiqKVCache(
                                head_dim=head_dim, bits=OPTIQ_BITS, seed=42 + kv_pos
                            )
                        )
                else:
                    caches.append(entry)
            return caches

        if len(model.layers) > 0:
            first_layer = model.layers[0]
            attn = getattr(first_layer, "self_attn", first_layer)
            head_dim = getattr(attn, "head_dim", 128)
        else:
            head_dim = 128
        n_layers = len(model.layers)
        return [
            KVCache()
            if (i < OPTIQ_FP16_LAYERS or i >= n_layers - OPTIQ_FP16_LAYERS)
            else OptiqKVCache(head_dim=head_dim, bits=OPTIQ_BITS, seed=42 + i)
            for i, _ in enumerate(model.layers)
        ]

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    logger.info("Using default KV cache")
    return [KVCache() for _ in model.layers]


def get_kv_cache_backend() -> KVCacheBackend:
    backend = KV_CACHE_BACKEND
    valid_backends: tuple[KVCacheBackend, ...] = (
        "default",
        "mlx_quantized",
        "turboquant",
        "turboquant_adaptive",
        "optiq",
    )
    if backend not in valid_backends:
        logger.warning(
            f"Unknown EXO_KV_CACHE_BACKEND={backend!r}; falling back to {DEFAULT_KV_CACHE_BACKEND!r}"
        )
        return DEFAULT_KV_CACHE_BACKEND
    return backend
