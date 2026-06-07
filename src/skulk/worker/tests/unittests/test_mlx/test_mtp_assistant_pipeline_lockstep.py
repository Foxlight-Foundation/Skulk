# pyright: reportAny=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
"""Distributed assistant-drafting regression tests (#201 Track 2b).

Two local ranks pipeline-shard a real gemma4 target; the assistant drafter
is built ONLY on the last rank, and every rank joins the per-round draft
exchange. Three invariants are pinned:

1. **Lockstep**: identical token/decision traces across ranks (the
   exchange keeps the collective schedule symmetric).
2. **No starvation**: the assistant cross-attends the TARGET's cache, so
   the loop must keep that cache fully committed before every draft
   (`reads_target_cache`) — broken, acceptance craters to ~28%.
3. **No decay**: the drafter must hold the LIVE cache list — a copied
   list freezes its cross-attention view at the first reject-restore
   (measured 56% -> 26% decay over 150 tokens).

The acceptance floor assertion (50%) trips on either regression while
sitting far below the healthy 80%+ this artifact measures.

Requires gemma-4-12B-it-4bit and its assistant in the model store;
skipped otherwise.
"""

import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from skulk.shared.constants import SKULK_MODELS_DIR

TARGET_DIR = SKULK_MODELS_DIR / "mlx-community--gemma-4-12B-it-4bit"
ASSISTANT_DIR = SKULK_MODELS_DIR / "mlx-community--gemma-4-12B-it-assistant-bf16"
WORLD_SIZE = 2
BASE_PORT = 29750
N_LAYERS = 48

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (TARGET_DIR.exists() and ASSISTANT_DIR.exists()),
        reason=f"gemma4 assistant test artifacts not found ({TARGET_DIR})",
    ),
]


def _assistant_rank(
    rank: int,
    hostfile_path: str,
    temperature: float,
    max_tokens: int,
    result_queue: Any,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from skulk.shared.models.model_cards import ModelCard, ModelTask
        from skulk.shared.types.common import ModelId
        from skulk.shared.types.memory import Memory
        from skulk.shared.types.worker.shards import PipelineShardMetadata
        from skulk.worker.engines.mlx.drafters.gemma4_assistant import (
            build_gemma4_assistant_drafter,
            load_assistant_model,
        )
        from skulk.worker.engines.mlx.generator.generate import (
            _get_trunk_and_head,
            _stream_generate_with_mtp,
        )
        from skulk.worker.engines.mlx.generator.speculative_sampling import (
            SamplingParams,
        )
        from skulk.worker.engines.mlx.utils_mlx import shard_and_load

        group = mx.distributed.init(backend="ring", strict=True)

        per = N_LAYERS // WORLD_SIZE
        shard_meta = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId("mlx-community/gemma-4-12B-it-4bit"),
                storage_size=Memory.from_bytes(1),
                n_layers=N_LAYERS,
                hidden_size=3840,
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=rank,
            world_size=WORLD_SIZE,
            start_layer=rank * per,
            end_layer=(rank + 1) * per if rank < WORLD_SIZE - 1 else N_LAYERS,
            n_layers=N_LAYERS,
        )
        from typing import cast

        from skulk.shared.types.mlx import Model

        loaded, tokenizer = shard_and_load(
            shard_meta, group, on_timeout=None, on_layer_loaded=None
        )
        model = cast(Model, loaded)

        drafter = None
        if rank == WORLD_SIZE - 1:
            assistant = load_assistant_model(ASSISTANT_DIR)
            assert assistant is not None
            drafter = build_gemma4_assistant_drafter(model, assistant)
            assert drafter is not None

        trunk_head = _get_trunk_and_head(model)
        assert trunk_head is not None
        trunk_fn, head_fn = trunk_head

        # Production seeds per request AFTER model load (mlx_generate);
        # model init consumes RNG draws and the drafting rank loads an
        # extra model, so seeding earlier would desync streams.
        mx.random.seed(42)

        toks = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Briefly explain what a mutex is."}],
            add_generation_prompt=True,
        )
        prompt = mx.array(list(toks))
        prompt_cache = make_prompt_cache(model)

        tokens: list[int] = []
        decisions: list[bool] = []
        for response in _stream_generate_with_mtp(
            model=model,
            tokenizer=tokenizer,
            drafter=drafter,
            trunk_fn=trunk_fn,
            head_fn=head_fn,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            logits_processors=[],
            prompt_cache=prompt_cache,
            kv_group_size=None,
            kv_bits=None,
            depth=2,
            sampling=SamplingParams(temperature=temperature),
            fail_loud_on_drafter_error=True,
            draft_group=group,
        ):
            tokens.append(response.token)
            decisions.append(response.from_draft)
            if response.finish_reason is not None:
                break

        result_queue.put((rank, True, (tokens, decisions)))
    except Exception as error:  # noqa: BLE001 — report to the parent
        result_queue.put((rank, False, repr(error)))


def _run(temperature: float, max_tokens: int, port_offset: int) -> None:
    hosts = [f"127.0.0.1:{BASE_PORT + port_offset + i}" for i in range(WORLD_SIZE)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    procs = [
        ctx.Process(
            target=_assistant_rank,
            args=(rank, hostfile_path, temperature, max_tokens, queue),
        )
        for rank in range(WORLD_SIZE)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=600)
        timed_out = any(p.is_alive() for p in procs)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        results: dict[int, tuple[bool, Any]] = {}
        while not queue.empty():
            rank, ok, value = queue.get()
            results[rank] = (ok, value)
    finally:
        Path(hostfile_path).unlink(missing_ok=True)

    assert not timed_out, "rank deadlocked — exchange schedule divergence"
    assert len(results) == WORLD_SIZE, f"missing ranks: {sorted(results)}"
    for rank in range(WORLD_SIZE):
        ok, value = results[rank]
        assert ok, f"rank {rank} failed: {value}"
    assert results[0][1] == results[1][1], "ranks diverged"

    tokens, decisions = results[0][1]
    accepted = sum(decisions)
    rounds = len(tokens) - accepted
    acceptance = accepted / max(accepted + rounds * 2, 1)
    # Healthy 12B measures 80%+; the starvation and frozen-cache
    # regressions both land under 30%. 0.4 of emitted tokens from drafts
    # is a robust floor that only those regressions cross.
    assert accepted / max(len(tokens), 1) > 0.4, (
        f"assistant acceptance collapsed: {accepted}/{len(tokens)} drafted "
        f"tokens (~{acceptance:.0%} per-attempt) — cross-attention cache "
        "starvation or frozen-view regression"
    )


def test_assistant_pipeline_greedy_lockstep() -> None:
    """Greedy last-rank drafting: lockstep + healthy acceptance."""
    _run(temperature=0.0, max_tokens=60, port_offset=0)


def test_assistant_pipeline_sampled_lockstep() -> None:
    """Sampled last-rank drafting: the exchange carries the draft
    distribution; explicit per-round keys keep global RNG streams aligned."""
    _run(temperature=0.7, max_tokens=60, port_offset=4)
