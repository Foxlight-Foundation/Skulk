"""The live guard repeats native policy/complete-workload admission before load."""

import tomllib
from pathlib import Path

import pytest

import skulk.worker.main as worker_main
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.separation_runtime import (
    estimate_demucs_peak,
    resolve_demucs_policy,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.main import Worker


def shard() -> PipelineShardMetadata:
    """Use exact native policy without any hardware qualification records."""
    card = ModelCard.model_validate(tomllib.loads(
        (Path(__file__).parents[2] / "shared/models/tests/htdemucs-card.toml").read_text()
    ))
    return PipelineShardMetadata(
        model_card=card, device_rank=0, world_size=1, start_layer=0,
        end_layer=card.n_layers, n_layers=card.n_layers,
        resolved_backend="demucs-cpu", resolved_engine_build="demucs@sha256:" + "a" * 64,
        separation_policy=resolve_demucs_policy(card, "demucs-cpu"),
    )


@pytest.mark.parametrize("available", [512 * 1024**2, 3 * 1024**3])
def test_cpu_guard_accounts_for_full_native_workload_not_an_unrelated_gpu(
    monkeypatch: pytest.MonkeyPatch, available: int,
) -> None:
    """The same CPU reservation fits/refuses both master and worker entry points."""
    def local(_cls: type[MemoryUsage]) -> MemoryUsage:
        return MemoryUsage.from_bytes(ram_total=4 * 1024**3, ram_available=available,
                                      swap_total=0, swap_available=0)

    def gpu() -> None:
        pytest.fail("the explicit native CPU workload cannot use a GPU pool")

    monkeypatch.setattr(MemoryUsage, "from_local_gpu_wireable", classmethod(local))
    monkeypatch.setattr(worker_main, "_local_usable_vram", gpu)
    worker = object.__new__(Worker)
    worker.node_id = NodeId("native-worker")
    selected = shard()
    error = worker._local_shard_fit_error(selected)  # pyright: ignore[reportPrivateUsage]
    assert (error is None) is (available >= estimate_demucs_peak(selected.model_card, backend="demucs-cpu").in_bytes)


@pytest.mark.parametrize("change", ["missing", "threads", "device"])
def test_worker_refuses_absent_or_changed_execution_policy(change: str) -> None:
    """A saved placement cannot silently adopt newer precision/device/budget settings."""
    selected = shard()
    policy = selected.separation_policy
    assert policy is not None
    altered = None if change == "missing" else policy.model_copy(update={
        "threads": 8,
    } if change == "threads" else {"device": "mps"})
    selected = selected.model_copy(update={"separation_policy": altered})
    worker = object.__new__(Worker)
    worker.node_id = NodeId("native-worker")
    error = worker._local_shard_fit_error(selected)  # pyright: ignore[reportPrivateUsage]
    assert error is not None and "policy differs" in error
