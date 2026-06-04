# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
# pyright: reportUnusedImport=false
"""Test B=1 vs B=2 equivalence for batch generation.

Verifies that running two requests concurrently in a batch (B=2) produces
the same logits as running them sequentially (B=1), within a numerical
tolerance. Uses random weights — no model download required.

The comparison is teacher-forced: the B=1 pass defines the canonical token
sequence and the B=2 pass is fed those same tokens, so per-step logits are
compared like-for-like. The tests originally asserted bit-exactness
(max diff < 0.002), but that premise is hardware-dependent: on M5-class
GPUs, MLX routes float32 GEMM (B>=2) through the Neural Accelerators at
TF32-style reduced precision by default, while GEMV (B=1) keeps full fp32
(ml-explore/mlx#3534, closed as expected behavior; opt out with
``MLX_ENABLE_TF32=0``, under which these paths are bit-exact again). On an
M5 the default-precision step-0 divergence is ~0.15 — far over the
bit-exact threshold before any Skulk code runs. Without teacher forcing,
that wobble eventually flips an argmax on near-flat random-weight logits
and the self-fed sequences cascade apart, producing meaningless 100+
"diffs". The tolerance is asserted under default precision deliberately —
that is what production runs. Real corruption (e.g. wrong batched
attention masking) still blows past it even when teacher-forced.
"""

from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest
from mlx_lm.generate import _merge_caches
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

# Import batch_generate to activate the right-padding BatchKVCache patch
import exo.worker.engines.mlx.generator.batch_generate  # noqa: F401
from exo.shared.types.mlx import Model
from exo.worker.engines.mlx.cache import encode_prompt, make_kv_cache
from exo.worker.engines.mlx.generator.generate import prefill

NUM_STEPS = 20

# Per-step relative logit tolerance for B=1 vs B=2 under teacher forcing.
# Empirical envelope on an M5 / mlx 0.31.2: steady-state ~0.002 with isolated
# spikes to ~0.056 (batched decode dispatches a different kernel/reduction
# order than B=1). Real batched-attention corruption (wrong masking/layout)
# produces relative diffs of O(1) at every step, so 0.25 keeps ~4x headroom
# in both directions.
MAX_RELATIVE_LOGIT_DIFF = 0.25


def _init_random(model: nn.Module) -> None:
    """Initialize all model parameters with random values."""
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())


def _run_b1_vs_b2(
    model: Model,
    tokenizer: TokenizerWrapper,
    tokens_a: mx.array,
    tokens_b: mx.array,
) -> tuple[float, int]:
    """Run B=1 sequential and teacher-forced B=2 batched decode.

    The B=1 pass picks tokens greedily and defines the canonical
    continuation; the B=2 pass is fed those exact tokens so per-step logits
    compare like-for-like (no argmax-flip cascade).

    Returns:
        (max_relative_logit_diff, argmax_disagreements) where the relative
        diff is per-step ``max|b1 - b2| / max|b1|`` — scale-robust across
        models and hardware. Disagreements is informational: on near-flat
        random-weight logits a sub-tolerance wobble can legitimately flip
        an argmax.
    """
    sampler = make_sampler(temp=0.0)

    # B=1 sequential
    cache_a1 = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_a[:-1], cache_a1, None, None, None)
    merged_a1 = _merge_caches([[c for c in cache_a1]])
    for c in merged_a1:
        c.prepare(lengths=[1], right_padding=[0])
    model(mx.array([[tokens_a[-2].item()]]), cache=merged_a1)
    mx.eval([c.state for c in merged_a1])
    for c in merged_a1:
        c.finalize()

    cache_b1 = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_b[:-1], cache_b1, None, None, None)
    merged_b1 = _merge_caches([[c for c in cache_b1]])
    for c in merged_b1:
        c.prepare(lengths=[1], right_padding=[0])
    model(mx.array([[tokens_b[-2].item()]]), cache=merged_b1)
    mx.eval([c.state for c in merged_b1])
    for c in merged_b1:
        c.finalize()

    # The B=1 greedy pass defines the canonical token sequence; the B=2 pass
    # below is teacher-forced with these same inputs.
    b1_logits_a: list[mx.array] = []
    b1_logits_b: list[mx.array] = []
    forced_a: list[int] = [int(tokens_a[-1].item())]
    forced_b: list[int] = [int(tokens_b[-1].item())]
    for _ in range(NUM_STEPS):
        la = model(mx.array([[forced_a[-1]]]), cache=merged_a1)
        mx.eval(la)
        b1_logits_a.append(la[0, -1])
        forced_a.append(int(mx.argmax(la[0, -1]).item()))
        lb = model(mx.array([[forced_b[-1]]]), cache=merged_b1)
        mx.eval(lb)
        b1_logits_b.append(lb[0, -1])
        forced_b.append(int(mx.argmax(lb[0, -1]).item()))

    # B=2 batched
    cache_a2 = make_kv_cache(model)
    cache_b2 = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_a[:-1], cache_a2, None, None, None)
    prefill(model, tokenizer, sampler, tokens_b[:-1], cache_b2, None, None, None)
    merged_b2 = _merge_caches([list(cache_a2), list(cache_b2)])
    for c in merged_b2:
        c.prepare(lengths=[1, 1], right_padding=[0, 0])
    model(
        mx.array([[tokens_a[-2].item()], [tokens_b[-2].item()]]),
        cache=merged_b2,
    )
    mx.eval([c.state for c in merged_b2])
    for c in merged_b2:
        c.finalize()

    b2_logits_a: list[mx.array] = []
    b2_logits_b: list[mx.array] = []
    for step in range(NUM_STEPS):
        l2 = model(
            mx.array([[forced_a[step]], [forced_b[step]]]), cache=merged_b2
        )
        mx.eval(l2)
        b2_logits_a.append(l2[0, -1])
        b2_logits_b.append(l2[1, -1])

    # Compare per-step, relative to that step's B=1 logit scale.
    max_rel_diff = 0.0
    mismatches = 0
    for step in range(NUM_STEPS):
        for b1_logits, b2_logits in (
            (b1_logits_a[step], b2_logits_a[step]),
            (b1_logits_b[step], b2_logits_b[step]),
        ):
            b1_f32 = b1_logits.astype(mx.float32)
            b2_f32 = b2_logits.astype(mx.float32)
            diff = float(mx.max(mx.abs(b1_f32 - b2_f32)).item())
            scale = float(mx.max(mx.abs(b1_f32)).item())
            max_rel_diff = max(max_rel_diff, diff / max(scale, 1e-6))
            if int(mx.argmax(b1_logits).item()) != int(mx.argmax(b2_logits).item()):
                mismatches += 1

    return max_rel_diff, mismatches


def _make_tokenizer() -> TokenizerWrapper:
    """Load the Qwen tokenizer (tiny download, shared across Qwen models)."""
    from huggingface_hub import snapshot_download

    model_path = Path(
        snapshot_download(
            "mlx-community/Qwen3.5-35B-A3B-4bit",
            allow_patterns=["tokenizer*", "*.jinja"],
        )
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return TokenizerWrapper(hf_tokenizer)


@pytest.mark.slow
def test_batch_b2_llama() -> None:
    """Llama-style model (KVCache only): B=2 logits must match B=1 within tolerance.

    Right-padded BatchKVCache keeps data at position 0 for all sequences, so
    batched attention sees the same data layout as B=1. Output is compared
    teacher-forced within MAX_RELATIVE_LOGIT_DIFF (bit-exactness is
    hardware-dependent — see module docstring).
    """
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaModel(args)
    _init_random(model)

    tokenizer = _make_tokenizer()
    tokens_a = encode_prompt(tokenizer, "Write a short essay about AI.")
    tokens_b = encode_prompt(tokenizer, "Explain evolution briefly.")

    max_rel_diff, mismatches = _run_b1_vs_b2(
        cast(Model, model), tokenizer, tokens_a, tokens_b
    )
    assert max_rel_diff < MAX_RELATIVE_LOGIT_DIFF, (
        f"Llama B=2 max relative logit diff: {max_rel_diff} "
        f"(argmax disagreements: {mismatches}/{NUM_STEPS * 2})"
    )


@pytest.mark.slow
def test_batch_b2_qwen35_moe() -> None:
    """Qwen3.5 MoE (hybrid SSM+attention+MoE): B=2 logits must match B=1 within tolerance.

    Right-padded BatchKVCache keeps data at position 0 for all sequences, so
    batched attention sees the same data layout as B=1. Output is compared
    teacher-forced within MAX_RELATIVE_LOGIT_DIFF (bit-exactness is
    hardware-dependent — see module docstring).
    """
    from mlx_lm.models.qwen3_5_moe import Model as Qwen35MoeModel
    from mlx_lm.models.qwen3_5_moe import ModelArgs

    mx.random.seed(42)
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "num_hidden_layers": 8,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 248320,
            "head_dim": 64,
            "max_position_embeddings": 4096,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 64,
            "mamba_ssm_dtype": "float32",
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
            },
            "attention_bias": False,
            "attn_output_gate": True,
        },
    }
    args = ModelArgs.from_dict(config)
    model = Qwen35MoeModel(args)
    _init_random(model)

    tokenizer = _make_tokenizer()
    tokens_a = encode_prompt(tokenizer, "Write a short essay about AI.")
    tokens_b = encode_prompt(tokenizer, "Explain evolution briefly.")

    max_rel_diff, mismatches = _run_b1_vs_b2(
        cast(Model, model), tokenizer, tokens_a, tokens_b
    )
    assert max_rel_diff < MAX_RELATIVE_LOGIT_DIFF, (
        f"Qwen3.5 MoE B=2 max relative logit diff: {max_rel_diff} "
        f"(argmax disagreements: {mismatches}/{NUM_STEPS * 2})"
    )
