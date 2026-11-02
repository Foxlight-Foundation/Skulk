from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from queue import Empty
from typing import cast

import mlx.core as mx
import mlx.nn as mlx_nn
import pytest
from mlx_lm.models.cache import ArraysCache

from skulk.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
    patch_pipeline_model,
)
from skulk.worker.tests.unittests.test_mlx.conftest import MockLayer

PipelineResult = tuple[int, bool, mx.array | str]


class _FakeCacheEntry:
    def __init__(self) -> None:
        self.keys = mx.array([1.0])


def test_arrays_cache_make_mask_accepts_attention_kwargs() -> None:
    cache = ArraysCache(1)
    make_mask = cast(Callable[..., object], cache.make_mask)

    assert make_mask(1, return_array=False, window_size=None) is None


def run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Queue[PipelineResult],
) -> None:
    """Run the real two-rank wrapper and report its actual result or failure."""
    import os

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    class MockLayerInner(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom_attr = "test_value"

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            return x * 2

    class MockModel(mlx_nn.Module):
        def __init__(self, layers: list[mlx_nn.Module]) -> None:
            super().__init__()
            self.layers = layers

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
            return x

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        mock = MockLayerInner()
        first = PipelineFirstLayer(mock, r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)

        # Wrap in a mock model, then wrap in PipelineParallelModel for all_gather
        inner_model = MockModel([composed])
        model = patch_pipeline_model(inner_model, group)

        x = mx.ones((1, 4))
        result = model(x)
        mx.eval(result)
        success = result.shape == x.shape
        result_queue.put((rank, success, result))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def test_single_wrapper_delegates_attributes() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    assert wrapped.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert wrapped.use_sliding is True  # type: ignore[attr-defined]


def test_composed_wrappers_delegate_attributes() -> None:
    mock = MockLayer()
    group = mx.distributed.init()

    first = PipelineFirstLayer(mock, r=0, group=group)
    composed = PipelineLastLayer(first, r=0, s=1, group=group)

    assert composed.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert composed.use_sliding is True  # type: ignore[attr-defined]


def test_missing_attribute_raises() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    with pytest.raises(AttributeError):
        _ = wrapped.nonexistent_attr  # type: ignore[attr-defined]


def test_composed_call_works() -> None:
    """Wait for actual child results through cold starts and always reap workers."""
    ctx = mp.get_context("spawn")

    world_size = 2
    base_port = 29500

    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    result_queue: Queue[PipelineResult] = ctx.Queue()
    processes: list[BaseProcess] = []
    deadline = time.monotonic() + 90
    try:
        for rank in range(world_size):
            p = ctx.Process(
                target=run_pipeline_device,
                args=(rank, world_size, hostfile_path, result_queue),
            )
            p.start()
            processes.append(p)

        results: dict[int, mx.array] = {}
        errors: dict[int, str] = {}
        # Queue.empty() is not a completion barrier. Collect while the children
        # run: cold imports may exceed ten seconds, and joining writers before
        # draining their queue can prevent their feeder threads from exiting.
        while len(results) + len(errors) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rank, success, value = result_queue.get(timeout=min(0.5, remaining))
            except Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue
            if success:
                assert isinstance(value, mx.array)
                results[rank] = value
            else:
                assert isinstance(value, str)
                errors[rank] = value

        assert len(results) == world_size, (
            f"Expected {world_size} results, got {len(results)}. Errors: {errors}. "
            f"Exit codes: {[process.exitcode for process in processes]}"
        )

        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            assert not process.is_alive(), (
                "Pipeline worker did not drain after its result"
            )

        for rank in range(world_size):
            assert rank in results, (
                f"Device {rank} failed: {errors.get(rank, 'unknown')}"
            )
            result_array = results[rank]
            # Both devices see the final result (4.0) after all_gather
            assert (result_array == 4.0).all(), (
                f"Device {rank}: expected 4.0, got {result_array}"
            )
    finally:
        try:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                process.join(timeout=5)
        finally:
            result_queue.close()
            result_queue.join_thread()
            os.unlink(hostfile_path)


def test_patch_pipeline_model_patches_nested_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLanguageModel(mlx_nn.Module):
        def __call__(self, x: mx.array, cache: list[object] | None = None) -> mx.array:
            return x * 2

    class FakeOuterModel(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = FakeLanguageModel()

        def __call__(self, x: mx.array, cache: list[object] | None = None) -> mx.array:
            return self.language_model(x, cache=cache)

    depends_calls: list[tuple[mx.array, mx.array]] = []

    def _fake_depends(lhs: mx.array, rhs: mx.array) -> mx.array:
        depends_calls.append((lhs, rhs))
        return lhs

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.auto_parallel.mx.depends", _fake_depends
    )

    model = FakeOuterModel()
    patch_pipeline_model(model, mx.distributed.init())

    cache_entry = _FakeCacheEntry()
    _ = model.language_model(mx.ones((1, 1)), cache=[cache_entry])

    assert depends_calls, "Expected nested language_model call to be pipeline-patched"
