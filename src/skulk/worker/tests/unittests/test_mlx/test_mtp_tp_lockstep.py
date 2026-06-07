# pyright: reportAny=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
"""Distributed lockstep regression tests for MTP (#201 Tracks 1-2a).

Spawns two local ranks over the ring backend, shards a real MTP model
(tensor-parallel or pipeline), and drives the production speculative loop
on both ranks. The lockstep invariant under test: distributed placements
give every rank identical logits (TP via collectives; pipeline via the
decode-mode all_gather plus replicated embed/norm/head) and per-request
RNG seeding aligns sampled draws, so accept/reject decisions — and
therefore token traces and cache lengths — are identical on every rank.
A divergence either fails the trace comparison or deadlocks a collective
(caught by the join timeout).

Requires Qwen3.5-2B-4bit and its sidecar in the model store; skipped
otherwise (same convention as test_distributed_fix.py).
"""

import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from skulk.shared.constants import SKULK_MODELS_DIR

MTP_TEST_MODEL_ID = "mlx-community/Qwen3.5-2B-4bit"
MTP_TEST_MODEL_DIR = SKULK_MODELS_DIR / "mlx-community--Qwen3.5-2B-4bit"
MTP_TEST_SIDECAR = (
    SKULK_MODELS_DIR / "FoxlightAI--qwen3-5-2b-base-mtp" / "mtp.safetensors"
)
WORLD_SIZE = 2
BASE_PORT = 29720

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MTP_TEST_MODEL_DIR.exists() and MTP_TEST_SIDECAR.exists()),
        reason=(
            f"MTP lockstep test model not found ({MTP_TEST_MODEL_DIR} + "
            f"{MTP_TEST_SIDECAR})"
        ),
    ),
]


def _lockstep_rank(
    rank: int,
    hostfile_path: str,
    temperature: float,
    max_tokens: int,
    shard_kind: str,
    result_queue: Any,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        from typing import cast

        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from skulk.shared.models.model_cards import ModelCard, ModelTask
        from skulk.shared.types.common import ModelId
        from skulk.shared.types.memory import Memory
        from skulk.shared.types.worker.shards import (
            PipelineShardMetadata,
            TensorShardMetadata,
        )
        from skulk.worker.engines.mlx.cache import cache_length
        from skulk.worker.engines.mlx.drafters import build_drafter
        from skulk.worker.engines.mlx.generator.generate import (
            _get_trunk_and_head,
            _stream_generate_with_mtp,
        )
        from skulk.worker.engines.mlx.generator.speculative_sampling import (
            SamplingParams,
        )
        from skulk.worker.engines.mlx.utils_mlx import shard_and_load

        group = mx.distributed.init(backend="ring", strict=True)
        # The production loop seeds per request (task.seed or 42) in
        # mlx_generate — replicate that contract here.
        mx.random.seed(42)

        model_card = ModelCard(
            model_id=ModelId(MTP_TEST_MODEL_ID),
            storage_size=Memory.from_bytes(1755848704),
            n_layers=24,
            hidden_size=2048,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        )
        if shard_kind == "tensor":
            shard_meta: TensorShardMetadata | PipelineShardMetadata = (
                TensorShardMetadata(
                    model_card=model_card,
                    device_rank=rank,
                    world_size=WORLD_SIZE,
                    start_layer=0,
                    end_layer=24,
                    n_layers=24,
                )
            )
        else:
            # Pipeline: split the 24 layers across the two ranks.
            start_layer, end_layer = [(0, 12), (12, 24)][rank]
            shard_meta = PipelineShardMetadata(
                model_card=model_card,
                device_rank=rank,
                world_size=WORLD_SIZE,
                start_layer=start_layer,
                end_layer=end_layer,
                n_layers=24,
            )
        from skulk.shared.types.mlx import Model

        loaded_model, tokenizer = shard_and_load(
            shard_meta, group, on_timeout=None, on_layer_loaded=None
        )
        model = cast(Model, loaded_model)

        weights = cast("dict[str, mx.array]", mx.load(str(MTP_TEST_SIDECAR)))
        drafter = build_drafter(model, weights)
        assert drafter is not None
        trunk_head = _get_trunk_and_head(model)
        assert trunk_head is not None
        trunk_fn, head_fn = trunk_head

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
            depth=1,
            sampling=SamplingParams(temperature=temperature),
        ):
            tokens.append(response.token)
            decisions.append(response.from_draft)
            if response.finish_reason is not None:
                break

        result_queue.put(
            (rank, True, (tokens, decisions, cache_length(prompt_cache)))
        )
    except Exception as error:  # noqa: BLE001 — report to the parent, don't hang
        result_queue.put((rank, False, repr(error)))


def _run_lockstep(
    temperature: float,
    max_tokens: int,
    port_offset: int,
    shard_kind: str = "tensor",
) -> None:
    hosts = [f"127.0.0.1:{BASE_PORT + port_offset + i}" for i in range(WORLD_SIZE)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    procs = [
        ctx.Process(
            target=_lockstep_rank,
            args=(rank, hostfile_path, temperature, max_tokens, shard_kind, queue),
        )
        for rank in range(WORLD_SIZE)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=300)
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

    assert not timed_out, "rank deadlocked — lockstep divergence signature"
    assert len(results) == WORLD_SIZE, f"missing rank results: {sorted(results)}"
    for rank in range(WORLD_SIZE):
        ok, value = results[rank]
        assert ok, f"rank {rank} failed: {value}"
    assert results[0][1] == results[1][1], (
        "ranks diverged: tokens/decisions/cache_len differ"
    )


def test_greedy_tp_lockstep() -> None:
    """Greedy accept/reject is a pure function of logits — byte parity."""
    _run_lockstep(temperature=0.0, max_tokens=60, port_offset=0)


def test_sampled_tp_lockstep() -> None:
    """Seeded sampled decoding: aligned RNG streams + data-dependent draw
    counts stay in lockstep (the #201 Track 1 open question)."""
    _run_lockstep(temperature=0.7, max_tokens=60, port_offset=4)


def test_greedy_pipeline_lockstep() -> None:
    """Pipeline shards run the same rank-symmetric loop (#201 Track 2a):
    decode-mode all_gather hands every rank the final hidden, and
    embed/norm/head are replicated — byte parity across the layer split."""
    _run_lockstep(temperature=0.0, max_tokens=60, port_offset=8, shard_kind="pipeline")


def test_sampled_pipeline_lockstep() -> None:
    """Seeded sampled decoding under a pipeline split stays in lockstep."""
    _run_lockstep(temperature=0.7, max_tokens=60, port_offset=12, shard_kind="pipeline")
