"""Sliding-window caches during the speculative loop: buffered while it runs, plain after.

A plain ``RotatingKVCache`` appends each multi-position verify write by
concatenating, so every round allocated a buffer one round larger than the
last and MLX kept the freed ones (about 9.5 GB over a 400-token Gemma 4
response). The loop now runs on mlx-vlm's ``BufferedRotatingKVCache`` and puts
the plain caches back when it ends.
"""

from collections.abc import Callable, Generator
from typing import Protocol, cast
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_vlm.models.cache import BufferedRotatingKVCache

from skulk.worker.engines.mlx import cache as cache_module
from skulk.worker.engines.mlx.cache import (
    KVPrefixCache,
    buffer_rotating_caches_for_speculation,
    restore_plain_rotating_caches,
)
from skulk.worker.engines.mlx.generator import generate as generate_module

WINDOW = 8


class _WindowCache(Protocol):
    """The sliding-window cache surface these tests drive (both classes have it)."""

    offset: int

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]: ...

    def trim(self, n: int) -> int: ...

    @property
    def state(self) -> tuple[mx.array, mx.array]: ...


def positions(start: int, count: int) -> mx.array:
    """Keys whose value is their own position, so a window is easy to read."""
    return mx.arange(start, start + count, dtype=mx.float32).reshape(1, 1, count, 1)


def write(entry: object, start: int, count: int) -> None:
    span = positions(start, count)
    _ = cast(_WindowCache, entry).update_and_fetch(span, span)


def window(entry: object) -> list[int]:
    """The positions a cache holds, oldest first, limited to the attention window."""
    keys, _ = cast(_WindowCache, entry).state
    if type(entry) is RotatingKVCache:
        # A plain cache past its window holds a ring; put it in temporal order.
        temporal = cast(
            Callable[[mx.array], mx.array],
            getattr(entry, "_temporal_order"),  # noqa: B009 - private upstream method
        )
        keys = temporal(keys)
    values = cast(list[float], keys.reshape(-1).tolist())
    return [int(value) for value in values][-WINDOW:]


def test_a_converted_entry_writes_in_place_and_restores_the_same_window() -> None:
    plain = RotatingKVCache(max_size=WINDOW)
    write(plain, 0, 3)
    caches: list[object] = [plain, KVCache()]

    assert buffer_rotating_caches_for_speculation(caches, tokens_per_round=3)
    assert isinstance(caches[0], BufferedRotatingKVCache)
    assert isinstance(caches[1], KVCache), "full-attention caches are left alone"

    buffered = cast(_WindowCache, cast(object, caches[0]))
    committed = 3
    # Verify rounds of three positions with a varying rejected tail, and a
    # plain single-position write now and then, far past the window and the
    # 32-position slack so the buffered cache compacts more than once.
    for round_index in range(40):
        count = 1 if round_index % 7 == 6 else 3
        rejected = (0, 1, 2)[round_index % 3] if count == 3 else 0
        write(buffered, committed, count)
        if rejected:
            _ = buffered.trim(rejected)
        committed += count - rejected
        assert window(buffered) == list(range(max(0, committed - WINDOW), committed))
    assert committed > WINDOW + 32
    expected = list(range(committed - WINDOW, committed))

    restore_plain_rotating_caches(caches)

    restored = caches[0]
    assert type(restored) is RotatingKVCache
    assert restored.offset == committed
    assert window(restored) == expected
    # The restored cache keeps working as a plain one.
    write(restored, committed, 1)
    assert window(restored) == expected[1:] + [committed]


def test_only_plain_unkept_rotating_entries_are_converted() -> None:
    kept = RotatingKVCache(max_size=WINDOW, keep=2)
    caches: list[object] = [kept, KVCache()]

    assert not buffer_rotating_caches_for_speculation(caches, tokens_per_round=2)
    assert caches[0] is kept


def test_a_tuple_cache_is_left_as_it_is() -> None:
    caches = (RotatingKVCache(max_size=WINDOW),)

    assert not buffer_rotating_caches_for_speculation(caches, tokens_per_round=2)
    assert type(caches[0]) is RotatingKVCache


def test_the_prefix_cache_stores_a_converted_entry_plain() -> None:
    """The consumer stores the cache while the loop is still suspended."""
    plain = RotatingKVCache(max_size=WINDOW)
    write(plain, 0, 5)
    caches: list[object] = [plain]
    _ = buffer_rotating_caches_for_speculation(caches, tokens_per_round=2)
    write(caches[0], 5, 2)

    prefix = KVPrefixCache(None)
    prefix.add_kv_cache(mx.arange(7), caches)

    stored = prefix.caches[0][0]
    assert type(stored) is RotatingKVCache
    assert window(stored) == list(range(7))
    # The live cache is untouched by the copy.
    assert isinstance(caches[0], BufferedRotatingKVCache)


def _fake_rounds(seen: list[type]) -> Callable[..., Generator[int, None, None]]:
    def rounds(**kwargs: object) -> Generator[int, None, None]:
        seen.extend(type(entry) for entry in cast(list[object], kwargs["prompt_cache"]))
        yield 1
        yield 2

    return rounds


def _run_wrapper(caches: list[object], *, close_early: bool) -> None:
    generator = generate_module._stream_generate_with_mtp(  # pyright: ignore[reportPrivateUsage]
        model=MagicMock(),
        tokenizer=MagicMock(),
        drafter=MagicMock(),
        trunk_fn=MagicMock(),
        head_fn=MagicMock(),
        prompt=mx.array([1]),
        max_tokens=2,
        sampler=MagicMock(),
        logits_processors=[],
        prompt_cache=caches,
        kv_group_size=None,
        kv_bits=None,
        depth=1,
    )
    _ = next(generator)
    if close_early:
        generator.close()
    else:
        for _ in generator:
            pass


@pytest.mark.parametrize("close_early", [False, True])
def test_the_loop_runs_buffered_and_hands_back_plain_caches(
    monkeypatch: pytest.MonkeyPatch, close_early: bool
) -> None:
    seen: list[type] = []
    monkeypatch.setattr(generate_module, "_stream_generate_with_mtp_rounds", _fake_rounds(seen))
    caches: list[object] = [RotatingKVCache(max_size=WINDOW), KVCache()]

    _run_wrapper(caches, close_early=close_early)

    assert seen == [BufferedRotatingKVCache, KVCache]
    assert type(caches[0]) is RotatingKVCache
    assert isinstance(caches[1], KVCache)


def test_without_mlx_vlm_the_loop_keeps_plain_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[type] = []
    monkeypatch.setattr(generate_module, "_stream_generate_with_mtp_rounds", _fake_rounds(seen))
    monkeypatch.setattr(cache_module, "_buffered_rotating_cache_class", lambda: None)
    caches: list[object] = [RotatingKVCache(max_size=WINDOW)]

    _run_wrapper(caches, close_early=False)

    assert seen == [RotatingKVCache]
