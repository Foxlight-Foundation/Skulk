"""Shared memory-footprint estimation for placement and the worker OOM guard.

Single source of truth so the master's placement fit-check
(``master/placement_utils.py``) and the worker's local pre-spawn guard
(``worker/main.py``) agree on what a shard will cost to load and serve.
Disagreement would let placement admit a shard the worker then refuses, or let
the worker abort on a load placement believed safe — both produce the
GLM-4.7-Flash failure class (oversized load -> uncaught Metal OOM -> SIGABRT ->
wired GPU memory leaked until reboot, 2026-06-08).

The estimate is intentionally a planning approximation, not a measurement:
weights are known exactly, but KV cache is reserved for an assumed
``KV_CONTEXT_BUDGET_TOKENS`` (models advertise a max context far larger than
typical serving use) and runtime overhead is a multiplicative factor measured
on real loads.
"""

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.memory import Memory

MEMORY_OVERHEAD_FACTOR: float = 1.30
"""Multiplier on a shard's *weight* bytes covering runtime overhead that scales
with weight size but is not the KV cache (estimated separately): activation
workspace, the MLX buffer cache, and the Python/MLX runtime. Measured on
Qwen3.5-9B-MLX-4bit at ~6.4 GB resident for 5.2 GB of weights (1.23x) during
warmup with negligible KV; 1.30 adds margin. Raised from a historical 1.05
after the 16 GB GLM-4.7-Flash incident (2026-06-08)."""

MEMORY_OVERHEAD_FLOOR: Memory = Memory.from_mb(256)
"""Flat per-shard overhead (Python interpreter, MLX runtime, IPC buffers) that
the multiplicative factor under-counts for small shards."""

GPU_WORKING_SET_FRACTION: float = 0.75
"""Fraction of a node's *total* RAM usable as the Metal GPU working set on
Apple Silicon. ``mx.device_info()["max_recommended_working_set_size"]`` measured
11.84 GB on a 16 GB M-series box (0.74); 0.75 tracks it. Both the master
placement check and the worker's local guard derive the ceiling from
``ram_total`` via ``gpu_working_set_ceiling`` (placement cannot gossip the exact
``max_recommended_working_set_size`` under ``extra=forbid``, and keeping the
worker on the same heuristic makes the two checks agree); the worker's advantage
is using *current local* ``ram_available`` rather than the gossiped value."""

KV_CONTEXT_BUDGET_TOKENS: int = 8192
"""Per-sequence context length reserved for KV cache during the fit check.
Reserving a model's advertised max (e.g. GLM-4.7-Flash: 131072) would over-
refuse by tens of GB. Planning assumption only; exposing it as an operator/UI
knob is tracked follow-up work."""

KV_HEAD_DIM_FALLBACK: int = 128
"""Attention head dimension assumed when a model card omits it (cards do not
persist ``head_dim``). 128 dominates current MLX families (Llama/Qwen/GLM)."""

KV_DTYPE_BYTES: int = 2
"""Bytes per KV-cache element. MLX keeps the KV cache in fp16 even for 4-bit
weights unless quantized-KV is explicitly enabled, which Skulk does not."""


def estimate_kv_cache_bytes(
    model_card: ModelCard, n_layers: int, context_tokens: int
) -> Memory:
    """Estimate KV-cache bytes for ``n_layers`` layers at ``context_tokens``.

    The cache holds a key and a value vector per token, per layer, each sized
    ``num_key_value_heads * head_dim``::

        bytes = 2 (K+V) * n_layers * context_tokens
                * num_key_value_heads * head_dim * KV_DTYPE_BYTES

    Returns zero when the card lacks ``num_key_value_heads`` or an argument is
    non-positive — the weight-overhead factor must absorb the slack then.
    ``head_dim`` falls back to ``KV_HEAD_DIM_FALLBACK`` (cards omit it).
    """
    kv_heads = model_card.num_key_value_heads
    if kv_heads is None or context_tokens <= 0 or n_layers <= 0:
        return Memory()
    kv_bytes = (
        2 * n_layers * context_tokens * kv_heads * KV_HEAD_DIM_FALLBACK * KV_DTYPE_BYTES
    )
    return Memory.from_bytes(kv_bytes)


def estimate_shard_footprint(
    model_card: ModelCard,
    shard_fraction: float,
    context_budget: int = KV_CONTEXT_BUDGET_TOKENS,
) -> Memory:
    """Estimate resident memory for a shard holding ``shard_fraction`` of a model.

    ``weights_share * MEMORY_OVERHEAD_FACTOR + kv_share + MEMORY_OVERHEAD_FLOOR``
    where weights and KV both scale by ``shard_fraction``. That single fraction
    works for every sharding because both quantities are linear in it:

    * Pipeline: ``shard_fraction = layers_held / n_layers`` (a node holds a
      contiguous layer range; weights and KV scale with the layer count).
    * Tensor: ``shard_fraction = 1 / world_size`` (a node holds all layers but
      ``1/world_size`` of each weight matrix and of the KV heads).

    ``shard_fraction == 1.0`` gives the whole-model footprint (single node).
    """
    if shard_fraction <= 0.0:
        return Memory()
    weights_share = model_card.storage_size * shard_fraction
    full_kv = estimate_kv_cache_bytes(model_card, model_card.n_layers, context_budget)
    kv_share = full_kv * shard_fraction
    return weights_share * MEMORY_OVERHEAD_FACTOR + kv_share + MEMORY_OVERHEAD_FLOOR


def gpu_working_set_ceiling(ram_total: Memory) -> Memory:
    """Metal GPU working-set ceiling derived from total RAM (placement path)."""
    return ram_total * GPU_WORKING_SET_FRACTION
