"""Tests for pipeline-aware prefill path selection."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

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

    monkeypatch.setattr(generate_module, "mx_barrier", lambda _group: None)
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        lambda _model: True,
    )
    monkeypatch.setattr(
        generate_module,
        "stream_generate",
        lambda *args, **kwargs: (_ for _ in () if False),
    )

    def _pipeline_parallel_prefill(**kwargs: object) -> None:
        calls.append("pipeline_parallel_prefill")
        prompt_progress_callback = kwargs["prompt_progress_callback"]
        assert callable(prompt_progress_callback)
        prompt = kwargs["prompt"]
        assert isinstance(prompt, list)
        prompt_progress_callback(len(prompt), len(prompt))

    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        _pipeline_parallel_prefill,
    )

    prefill_tps, prefill_tokens, snapshots = generate_module.prefill(
        model=_FakeModel(),
        tokenizer=object(),
        sampler=object(),
        prompt_tokens=list(range(5000)),
        cache=[fake_cache],
        group=_FakeGroup(),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert calls == ["pipeline_parallel_prefill"]
    assert prefill_tokens == 5000
    assert snapshots == []
    assert prefill_tps >= 0.0
    assert fake_cache.trim_calls == [2]


def test_prefill_uses_stream_generate_for_short_pipeline_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short prompts that fit in a single per-rank chunk must avoid pipeline prefill.

    Regression for the Gemma 4 warmup wedge: pipeline_parallel_prefill hangs
    when ``n_real == 1`` (the prompt fits inside a single per-rank chunk),
    so any such prompt — including the 23-token warmup prompt — must route
    through stream_generate even on pipeline-parallel models. PR #101's
    pipeline-prefill widening removed this guard and reintroduced the hang
    on Gemma 4; this test locks the guard back in.
    """
    calls: list[str] = []
    fake_cache = _FakeCache()

    monkeypatch.setattr(generate_module, "mx_barrier", lambda _group: None)
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        lambda _model: True,
    )

    def _pipeline_parallel_prefill(**kwargs: object) -> None:
        calls.append("pipeline_parallel_prefill")

    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        _pipeline_parallel_prefill,
    )

    def _stream_generate(*args: object, **kwargs: object) -> Iterator[object]:
        calls.append("stream_generate")
        yield object()

    monkeypatch.setattr(generate_module, "stream_generate", _stream_generate)

    prefill_tps, prefill_tokens, snapshots = generate_module.prefill(
        model=_FakeModel(),
        tokenizer=object(),
        sampler=object(),
        prompt_tokens=list(range(23)),  # exactly the gemma 4 warmup prompt size
        cache=[fake_cache],
        group=_FakeGroup(),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert calls == ["stream_generate"]
    assert "pipeline_parallel_prefill" not in calls
    assert prefill_tokens == 23
    assert snapshots == []
    assert prefill_tps >= 0.0
    assert fake_cache.trim_calls == [2]


def test_prefill_uses_stream_generate_when_model_is_not_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_cache = _FakeCache()

    monkeypatch.setattr(generate_module, "mx_barrier", lambda _group: None)
    monkeypatch.setattr(
        generate_module,
        "_has_pipeline_communication_layer",
        lambda _model: False,
    )
    monkeypatch.setattr(
        generate_module,
        "pipeline_parallel_prefill",
        lambda **kwargs: calls.append("pipeline_parallel_prefill"),
    )

    def _stream_generate(*args: object, **kwargs: object) -> Iterator[object]:
        calls.append("stream_generate")
        yield object()

    monkeypatch.setattr(generate_module, "stream_generate", _stream_generate)

    prefill_tps, prefill_tokens, snapshots = generate_module.prefill(
        model=_FakeModel(),
        tokenizer=object(),
        sampler=object(),
        prompt_tokens=list(range(128)),
        cache=[fake_cache],
        group=_FakeGroup(),
        on_prefill_progress=None,
        distributed_prompt_progress_callback=None,
    )

    assert calls == ["stream_generate"]
    assert prefill_tokens == 128
    assert snapshots == []
    assert prefill_tps >= 0.0
    assert fake_cache.trim_calls == [2]
