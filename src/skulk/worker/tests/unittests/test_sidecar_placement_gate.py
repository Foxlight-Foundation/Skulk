"""Tests for the sidecar load-eligibility gate (#254 / #263).

Pins the placement rule that decides which ranks may hold MTP sidecar
weights: single-node always; pipeline decider (last rank) only on
multi-node pipelines; and — the #263 regression fix — EVERY rank on a
multi-rank Tensor placement, because a lone TP decider's "local" draft
deadlocks on the TP-sharded lm_head's all-rank collectives, while
rank-symmetric drafting keeps the collective schedule identical (the
measured +31% TP configuration that #254's decider-only load regressed).
"""

from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    TensorShardMetadata,
)
from skulk.worker.engines.mlx.utils_mlx import sidecar_load_eligible

_CARD = ModelCard(
    model_id=ModelId("test-org/test-model"),
    storage_size=Memory.from_bytes(1_000_000),
    n_layers=24,
    hidden_size=2048,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
)


def _pipeline(rank: int, world: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_CARD,
        device_rank=rank,
        world_size=world,
        start_layer=rank * 12,
        end_layer=(rank + 1) * 12,
        n_layers=24,
    )


def _tensor(rank: int, world: int) -> TensorShardMetadata:
    return TensorShardMetadata(
        model_card=_CARD,
        device_rank=rank,
        world_size=world,
        start_layer=0,
        end_layer=24,
        n_layers=24,
    )


def test_single_node_always_loads():
    assert sidecar_load_eligible(
        _pipeline(0, 1), single_node=True, speculation_blocked=False
    )
    assert sidecar_load_eligible(
        _tensor(0, 1), single_node=True, speculation_blocked=False
    )


def test_pipeline_decider_rank_loads():
    assert sidecar_load_eligible(
        _pipeline(1, 2), single_node=False, speculation_blocked=False
    )


def test_pipeline_receiver_ranks_skip():
    assert not sidecar_load_eligible(
        _pipeline(0, 2), single_node=False, speculation_blocked=False
    )
    assert not sidecar_load_eligible(
        _pipeline(1, 3), single_node=False, speculation_blocked=False
    )


def test_tensor_multi_rank_loads_everywhere():
    # The #263 rule: every TP rank loads and drafts rank-symmetrically. A
    # decider-only TP drafter blocks on sharded-lm_head collectives the idle
    # receivers never join (GPU-timeout SIGABRT); all-rank drafting keeps
    # the collective schedule identical, and the drafter agreement in the
    # generator requires ready_count == group.size() on this path.
    for rank in (0, 1):
        assert sidecar_load_eligible(
            _tensor(rank, 2), single_node=False, speculation_blocked=False
        )


def test_card_block_wins_everywhere():
    assert not sidecar_load_eligible(
        _pipeline(1, 2), single_node=False, speculation_blocked=True
    )
    assert not sidecar_load_eligible(
        _tensor(1, 2), single_node=False, speculation_blocked=True
    )
