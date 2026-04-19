"""Tests for pipeline-aware prefill path selection."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.mlx import KVCacheType, Model
from exo.worker.engines.mlx.generator import generate as generate_module


class _FakeCache:
    def __init__(self) -> None:
        self.trim_calls: list[int] = []

    def trim(self, count: int) -> None:
        self.trim_calls.append(count)


class _FakeGroup:
    def __init__(self, rank: int = 0, size: int = 3) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _FakeModel:
    def __init__(self) -> None:
        self.layers: list[object] = []


def _identity_sampler(logits: mx.array) -> mx.array:
    return logits


def _fake_model() -> Model:
    return cast(Model, cast(object, _FakeModel()))


def _fake_tokenizer() -> TokenizerWrapper:
    return cast(TokenizerWrapper, object())


def _fake_group() -> mx.distributed.Group:
    return cast(mx.distributed.Group, cast(object, _FakeGroup()))


def _fake_cache_list(cache: _FakeCache) -> KVCacheType:
    return cast(KVCacheType, [cache])


def _noop_barrier(_group: object) -> None:
    return None


def _record_prefill_mode(target: list[bool]):
    def _set_prefill(_model: object, *, is_prefill: bool) -> None:
        target.append(is_prefill)

    return _set_prefill


def _record_queue_sends(target: list[bool]):
    def _set_queue_sends(_model: object, *, queue_sends: bool) -> None:
        target.append(queue_sends)

    return _set_queue_sends


def _pipeline_enabled(_model: object) -> bool:
    return True


def _pipeline_disabled(_model: object) -> bool:
    return False


def _empty_stream(*args: object, **kwargs: object) -> Iterator[object]:
    if False:
        yield args, kwargs


def _record_pipeline_prefill(calls: list[str]) -> Callable[..., None]:
    def _pipeline_prefill(**_kwargs: object) -> None:
        calls.append("pipeline_parallel_prefill")

    return _pipeline_prefill


def test_prefill_uses_pipeline_parallel_path_for_long_pipeline_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long prompts on pipeline-parallel models should use pipeline_parallel_prefill.

    The threshold is two effective per-rank chunks. With group_size=3 and
    prefill_step_size=4096, the effective per-rank chunk size is 1365, so a
    5000-token prompt produces 4 chunks and qualifies.
    """
    calls: list[str] = []
    fake_cache = _FakeCache()
    prefill_mode_calls: list[bool] = []

    monkeypatch.setattr(generate_module, "mx_barrier", _noop_barrier)
    monkeypatch.setattr(
        generate_module,
        "set_pipeline_prefill",
        _record_prefill_mode(prefill_mode_calls),
    )
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        _pipeline_enabled,
    )
    monkeypatch.setattr(generate_module, "stream_generate", _empty_stream)

    def _pipeline_parallel_prefill(**kwargs: object) -> None:
        calls.append("pipeline_parallel_prefill")
        prompt_progress_callback = cast(
            Callable[[int, int], None], kwargs["prompt_progress_callback"]
        )
        prompt = cast(mx.array, kwargs["prompt"])
        prompt_progress_callback(len(prompt), len(prompt))

    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        _pipeline_parallel_prefill,
    )

    prefill_tps, prefill_tokens, snapshots = generate_module.prefill(
        model=_fake_model(),
        tokenizer=_fake_tokenizer(),
        sampler=_identity_sampler,
        prompt_tokens=mx.array(list(range(5000))),
        cache=_fake_cache_list(fake_cache),
        group=_fake_group(),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert calls == ["pipeline_parallel_prefill"]
    assert prefill_tokens == 5000
    assert snapshots == []
    assert prefill_tps >= 0.0
    assert fake_cache.trim_calls == [2]
    assert prefill_mode_calls == [True, False]


def test_prefill_uses_pipeline_parallel_path_for_short_pipeline_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short pipeline prompts must avoid upstream stream_generate prefill.

    ``mlx_lm.generate_step`` prefetches a decode step before yielding even with
    ``max_tokens=1``. In pipeline mode that hidden prefetch can leave sends/recvs
    in flight, so short prompts use Skulk's explicit pipeline prefill path while
    suppressing distributed progress callbacks.
    """
    calls: list[str] = []
    fake_cache = _FakeCache()
    prefill_mode_calls: list[bool] = []

    monkeypatch.setattr(generate_module, "mx_barrier", _noop_barrier)
    monkeypatch.setattr(
        generate_module,
        "set_pipeline_prefill",
        _record_prefill_mode(prefill_mode_calls),
    )
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        _pipeline_enabled,
    )

    def _pipeline_parallel_prefill(**kwargs: object) -> None:
        calls.append("pipeline_parallel_prefill")
        assert kwargs["distributed_prompt_progress_callback"] is None
        prompt_progress_callback = cast(
            Callable[[int, int], None], kwargs["prompt_progress_callback"]
        )
        prompt = cast(mx.array, kwargs["prompt"])
        prompt_progress_callback(len(prompt), len(prompt))

    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        _pipeline_parallel_prefill,
    )

    def _stream_generate(*args: object, **kwargs: object) -> Iterator[object]:
        calls.append("stream_generate")
        yield object()

    monkeypatch.setattr(generate_module, "stream_generate", _stream_generate)

    def _fail_distributed_callback() -> None:
        raise AssertionError("single-chunk pipeline prefill should not poll tasks")

    prefill_tps, prefill_tokens, snapshots = generate_module.prefill(
        model=_fake_model(),
        tokenizer=_fake_tokenizer(),
        sampler=_identity_sampler,
        prompt_tokens=mx.array(list(range(23))),  # exactly the gemma 4 warmup prompt size
        cache=_fake_cache_list(fake_cache),
        group=_fake_group(),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=_fail_distributed_callback,
    )

    assert calls == ["pipeline_parallel_prefill"]
    assert "stream_generate" not in calls
    assert prefill_tokens == 23
    assert snapshots == []
    assert prefill_tps >= 0.0
    assert fake_cache.trim_calls == [2]
    assert prefill_mode_calls == [True, False]


def test_prefill_uses_pipeline_parallel_path_at_single_chunk_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompts with one real pipeline chunk still use explicit pipeline prefill."""
    calls: list[str] = []
    fake_cache = _FakeCache()
    prefill_mode_calls: list[bool] = []

    monkeypatch.setattr(generate_module, "mx_barrier", _noop_barrier)
    monkeypatch.setattr(
        generate_module,
        "set_pipeline_prefill",
        _record_prefill_mode(prefill_mode_calls),
    )
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        _pipeline_enabled,
    )
    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        _record_pipeline_prefill(calls),
    )

    def _stream_generate(*args: object, **kwargs: object) -> Iterator[object]:
        calls.append("stream_generate")
        yield object()

    monkeypatch.setattr(generate_module, "stream_generate", _stream_generate)

    prefill_tps, prefill_tokens, snapshots = generate_module.prefill(
        model=_fake_model(),
        tokenizer=_fake_tokenizer(),
        sampler=_identity_sampler,
        prompt_tokens=mx.array(list(range(1366))),
        cache=_fake_cache_list(fake_cache),
        group=_fake_group(),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert calls == ["pipeline_parallel_prefill"]
    assert prefill_tokens == 1366
    assert snapshots == []
    assert prefill_tps >= 0.0
    assert fake_cache.trim_calls == [2]
    assert prefill_mode_calls == [True, False]


def test_prefill_resets_pipeline_flags_after_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected prefill errors must restore pipeline-mode flags before bubbling."""
    fake_cache = _FakeCache()
    prefill_mode_calls: list[bool] = []
    queue_send_calls: list[bool] = []

    monkeypatch.setattr(generate_module, "mx_barrier", _noop_barrier)
    monkeypatch.setattr(
        generate_module,
        "set_pipeline_prefill",
        _record_prefill_mode(prefill_mode_calls),
    )
    monkeypatch.setattr(
        generate_module,
        "set_pipeline_queue_sends",
        _record_queue_sends(queue_send_calls),
    )
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        _pipeline_enabled,
    )
    monkeypatch.setattr(generate_module, "stream_generate", _empty_stream)

    def _pipeline_parallel_prefill(**kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        _pipeline_parallel_prefill,
    )

    with pytest.raises(RuntimeError, match="boom"):
        generate_module.prefill(
            model=_fake_model(),
            tokenizer=_fake_tokenizer(),
            sampler=_identity_sampler,
            prompt_tokens=mx.array(list(range(5000))),
            cache=_fake_cache_list(fake_cache),
            group=_fake_group(),
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

    assert queue_send_calls == [True, False]
    assert prefill_mode_calls == [True, False]


def test_prefill_uses_stream_generate_when_model_is_not_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_cache = _FakeCache()

    monkeypatch.setattr(generate_module, "mx_barrier", _noop_barrier)
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        _pipeline_disabled,
    )
    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        _record_pipeline_prefill(calls),
    )

    def _stream_generate(*args: object, **kwargs: object) -> Iterator[object]:
        calls.append("stream_generate")
        yield object()

    monkeypatch.setattr(generate_module, "stream_generate", _stream_generate)

    prefill_tps, prefill_tokens, snapshots = generate_module.prefill(
        model=_fake_model(),
        tokenizer=_fake_tokenizer(),
        sampler=_identity_sampler,
        prompt_tokens=mx.array(list(range(128))),
        cache=_fake_cache_list(fake_cache),
        group=_fake_group(),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert calls == ["stream_generate"]
    assert prefill_tokens == 128
    assert snapshots == []
    assert prefill_tps >= 0.0
    assert fake_cache.trim_calls == [2]
