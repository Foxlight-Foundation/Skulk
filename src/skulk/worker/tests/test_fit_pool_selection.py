"""The pre-load guard checks a shard against the pool its stamped backend uses."""

import pytest

import skulk.worker.main as worker_main
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.main import Worker


def _shard(backend: str | None) -> PipelineShardMetadata:
    card = ModelCard(
        model_id=ModelId("org/served"),
        storage_size=Memory.from_gb(16),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        num_key_value_heads=8,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model-Q4_K_M.gguf",
        context_length=1048576,
    )
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
        resolved_backend=backend,
    )


@pytest.mark.parametrize(
    ("backend", "refused"),
    [
        # A CPU-resolved shard lives in system RAM beside the small GPU.
        ("llama_server-cpu", False),
        # A GPU-offload shard is checked against the discrete card.
        ("llama_server-cuda", True),
        # An unresolved shard on a GPU host keeps the VRAM pool placement used.
        (None, True),
    ],
)
def test_guard_pool_follows_the_stamped_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str | None, refused: bool
) -> None:
    # 8 GB of VRAM beside 128 GB of RAM; the stamped window fits RAM only.
    monkeypatch.setattr(worker_main, "_local_usable_vram", lambda: Memory.from_gb(8))
    def wireable(_cls: type[MemoryUsage]) -> MemoryUsage:
        return MemoryUsage.from_bytes(
            ram_total=Memory.from_gb(128).in_bytes,
            ram_available=Memory.from_gb(100).in_bytes,
            swap_total=0,
            swap_available=0,
        )

    monkeypatch.setattr(MemoryUsage, "from_local_gpu_wireable", classmethod(wireable))
    worker = object.__new__(Worker)
    worker.node_id = NodeId("worker-under-test")

    error = worker._local_shard_fit_error(_shard(backend), context_token_limit=65536)  # pyright: ignore[reportPrivateUsage]

    assert (error is not None) is refused


@pytest.mark.parametrize(
    ("unified", "host_available_gb", "refused"),
    [
        # A discrete card: the combined pool alone decides.
        (False, 20, False),
        # A unified-memory APU with host RAM that still holds the window.
        (True, 100, False),
        # The same APU after host RAM fell: the carve-out keeps the combined
        # pool high, but the fixed window no longer fits host memory.
        (True, 20, True),
    ],
)
def test_guard_checks_a_fixed_window_against_host_ram_on_unified_memory(
    monkeypatch: pytest.MonkeyPatch,
    unified: bool,
    host_available_gb: int,
    refused: bool,
) -> None:
    monkeypatch.setattr(worker_main, "_local_usable_vram", lambda: Memory.from_gb(100))
    monkeypatch.setattr(worker_main, "_local_unified_memory_gpu", lambda: unified)

    def wireable(_cls: type[MemoryUsage]) -> MemoryUsage:
        return MemoryUsage.from_bytes(
            ram_total=Memory.from_gb(128).in_bytes,
            ram_available=Memory.from_gb(host_available_gb).in_bytes,
            swap_total=0,
            swap_available=0,
        )

    monkeypatch.setattr(MemoryUsage, "from_local_gpu_wireable", classmethod(wireable))
    worker = object.__new__(Worker)
    worker.node_id = NodeId("worker-under-test")

    error = worker._local_shard_fit_error(  # pyright: ignore[reportPrivateUsage]
        _shard("llama_server-rocm"), context_token_limit=131072
    )

    assert (error is not None) is refused
