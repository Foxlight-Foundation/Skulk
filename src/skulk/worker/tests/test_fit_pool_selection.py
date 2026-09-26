"""The pre-load guard checks a shard against the pool its stamped backend uses."""

import pytest

import skulk.shared.backends as backends
import skulk.worker.main as worker_main
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import AcceleratorMetrics, MemoryUsage
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.info_gatherer import linux_gpu, nvidia_gpu
from skulk.worker.main import Worker


@pytest.mark.parametrize(
    ("advertised", "uses_vram"),
    [
        (frozenset({"audio_cpp-cuda"}), True),
        (frozenset({"audio_cpp-vulkan"}), True),
        (frozenset({"comfy-cuda"}), True),
        (frozenset({"audio_cpp-cpu"}), False),
    ],
)
def test_nvidia_vram_probe_uses_the_placement_offload_classes(
    monkeypatch: pytest.MonkeyPatch,
    advertised: frozenset[str],
    uses_vram: bool,
) -> None:
    """The worker accepts NVIDIA VRAM for each standalone offload lane."""

    monkeypatch.setattr(worker_main.sys, "platform", "linux")
    monkeypatch.setattr(linux_gpu, "find_amd_gpu_device", lambda: None)
    monkeypatch.setattr(nvidia_gpu, "prefer_nvidia_telemetry", lambda: True)
    monkeypatch.setattr(backends, "probe_node_backends", lambda: advertised)
    monkeypatch.setattr(nvidia_gpu, "load_nvml", lambda: object())

    def has_nvidia_gpu(_nvml: object) -> bool:
        return True

    def accelerator_metrics(_nvml: object) -> AcceleratorMetrics:
        return AcceleratorMetrics(
            vendor="nvidia",
            name="NVIDIA test GPU",
            vram_total_bytes=Memory.from_gb(16).in_bytes,
            vram_used_bytes=Memory.from_gb(4).in_bytes,
        )

    monkeypatch.setattr(nvidia_gpu, "has_nvidia_gpu", has_nvidia_gpu)
    monkeypatch.setattr(
        nvidia_gpu, "read_accelerator_metrics", accelerator_metrics,
    )

    usable = worker_main._local_usable_vram()  # pyright: ignore[reportPrivateUsage]

    assert (usable is not None) is uses_vram


def test_gb10_local_fit_uses_shared_cuda_and_host_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker applies the same GB10 admission bound as the master."""
    total = Memory.from_gb(128).in_bytes
    monkeypatch.setattr(worker_main.sys, "platform", "linux")
    monkeypatch.setattr(linux_gpu, "find_amd_gpu_device", lambda: None)
    monkeypatch.setattr(nvidia_gpu, "prefer_nvidia_telemetry", lambda: True)
    monkeypatch.setattr(backends, "probe_node_backends", lambda: frozenset({"audio_cpp-cuda"}))
    monkeypatch.setattr(nvidia_gpu, "load_nvml", lambda: object())
    def has_nvidia_gpu(_nvml: object) -> bool:
        return True

    def gb10_metrics(_nvml: object) -> AcceleratorMetrics:
        return AcceleratorMetrics(
            vendor="nvidia",
            name="NVIDIA GB10",
            compute_capability="12.1",
            vram_total_bytes=total,
            vram_used_bytes=Memory.from_gb(48).in_bytes,
        )

    monkeypatch.setattr(nvidia_gpu, "has_nvidia_gpu", has_nvidia_gpu)
    monkeypatch.setattr(
        nvidia_gpu,
        "read_accelerator_metrics",
        gb10_metrics,
    )

    def local_memory(_cls: type[MemoryUsage]) -> MemoryUsage:
        return MemoryUsage.from_bytes(
            ram_total=total,
            ram_available=Memory.from_gb(80).in_bytes,
            swap_total=0,
            swap_available=0,
        )

    monkeypatch.setattr(MemoryUsage, "from_local_gpu_wireable", classmethod(local_memory))
    usable = worker_main._local_usable_vram()  # pyright: ignore[reportPrivateUsage]
    assert usable is not None
    assert usable.in_bytes == Memory.from_gb(64).in_bytes
    assert worker_main._local_unified_memory_gpu()  # pyright: ignore[reportPrivateUsage]


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
