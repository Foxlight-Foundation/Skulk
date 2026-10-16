# pyright: reportPrivateUsage=false
"""Settings drift must be refused before any local memory or model allocation."""

import pytest

from skulk.shared.models.engine_memory_settings import EngineMemorySettings
from skulk.shared.models.tests.test_engine_memory_settings import memory_card
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.main import Worker


def test_worker_refuses_changed_cache_policy_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compressed placement cannot launch with a newly uncompressed cache."""
    monkeypatch.delenv("SKULK_KV_CACHE_BACKEND", raising=False)
    shard = PipelineShardMetadata(
        model_card=memory_card(),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
        resolved_backend="mlx-metal",
        memory_settings=EngineMemorySettings(
            cache_backend="mlx_quantized", key_bits=4, value_bits=4
        ),
    )
    worker = object.__new__(Worker)
    # An uninitialized worker has no telemetry/allocation hooks: a drift result
    # proves the guard returns before trying to read or allocate model memory.
    assert "changed since placement" in (worker._local_shard_fit_error(shard) or "")
