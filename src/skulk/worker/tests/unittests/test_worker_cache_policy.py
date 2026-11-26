# pyright: reportPrivateUsage=false
"""Settings drift must be refused before any local memory or model allocation."""

import pytest

from skulk import facts as facts_module
from skulk.facts.testing import make_facts
from skulk.shared.models.engine_memory_settings import (
    EngineMemorySettings,
    backend_engine_memory_settings,
    engine_package_memory_settings,
    node_engine_memory_settings,
)
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


_PACKAGE_TAG = "tensorfold-rocm"


def _package_shard(overhead_bytes: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=memory_card(),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
        resolved_backend=_PACKAGE_TAG,
        memory_settings=engine_package_memory_settings(overhead_bytes),
    )


@pytest.mark.parametrize(
    "held",
    [
        pytest.param({_PACKAGE_TAG: 3 * 1024**3}, id="another build's figure"),
        pytest.param({}, id="no package on this node"),
        pytest.param(None, id="facts not gathered yet"),
    ],
)
def test_worker_refuses_a_package_whose_declared_memory_is_not_the_admitted_one(
    monkeypatch: pytest.MonkeyPatch, held: dict[str, int] | None
) -> None:
    """The launch was admitted on one figure; the node must still state it."""
    monkeypatch.setattr(
        facts_module,
        "_facts",
        None
        if held is None
        else make_facts(platform="linux").model_copy(
            update={"engine_package_overhead_bytes": held}
        ),
    )
    worker = object.__new__(Worker)

    refusal = worker._local_shard_fit_error(_package_shard(2 * 1024**3))

    assert "changed since placement" in (refusal or "")


def test_the_worker_compares_the_figure_the_node_advertised() -> None:
    """Advertisement and load guard read one function, so they cannot disagree."""
    overhead = 2 * 1024**3
    advertised = node_engine_memory_settings(
        [_PACKAGE_TAG, "llama_cpp-vulkan"], {_PACKAGE_TAG: overhead}, {}
    )

    assert advertised[_PACKAGE_TAG] == _package_shard(overhead).memory_settings
    assert (
        backend_engine_memory_settings(_PACKAGE_TAG, {_PACKAGE_TAG: overhead}, {})
        == advertised[_PACKAGE_TAG]
    )
    # A built-in engine is untouched by a package's figure.
    assert advertised["llama_cpp"] == EngineMemorySettings()
