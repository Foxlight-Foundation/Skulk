# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownLambdaType=false, reportArgumentType=false
# pyright: reportMissingTypeStubs=false
"""Positional speculative rollback for attention-only caches.

The speculative loop drops rejected verify positions from KV and rotating
sliding-window caches by trimming instead of snapshotting (Gemma 4 targets
load through mlx-lm, which ships no ``rollback_speculative_cache``). These
tests pin three contracts:

- which cache lists qualify (recurrent and unknown entries never do);
- a rotating cache rolled back after multi-token writes keeps exactly the
  committed positions, including after its window wraps and when a later
  single-token write (plain-decode fallback) follows a rollback;
- end to end, the loop's emitted tokens equal the target's greedy sequence
  and the target cache ends holding only committed tokens.
"""

from __future__ import annotations

import importlib.util
import random
from typing import cast
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from skulk.worker.engines.mlx.cache import (
    rollback_speculative_positions,
    supports_positional_rollback,
)

_REJECTED = -1.0


def _positions(values: list[float]) -> tuple[mx.array, mx.array]:
    """Keys/values whose single channel carries each entry's position."""
    array = mx.array(values, dtype=mx.float32).reshape(1, 1, len(values), 1)
    return array, array


def _flat(array: mx.array) -> list[float]:
    return [float(value) for value in cast("list[float]", array.reshape(-1).tolist())]


class TestSupportsPositionalRollback:
    def test_kv_and_rotating_caches_qualify(self) -> None:
        assert supports_positional_rollback([KVCache(), RotatingKVCache(max_size=8)])

    def test_recurrent_state_disqualifies(self) -> None:
        assert not supports_positional_rollback([KVCache(), ArraysCache(size=1)])

    def test_unknown_entry_disqualifies(self) -> None:
        class _Opaque:
            pass

        assert not supports_positional_rollback([KVCache(), _Opaque()])

    def test_entry_reporting_untrimmable_disqualifies(self) -> None:
        class _Untrimmable:
            def is_trimmable(self) -> bool:
                return False

            def trim(self, n: int) -> int:
                return n

        assert not supports_positional_rollback([_Untrimmable()])

    def test_mlx_vlm_caches_qualify(self) -> None:
        if importlib.util.find_spec("mlx_vlm") is None:
            pytest.skip("mlx-vlm is not installed on this platform")
        from mlx_vlm.models.cache import KVCache as VlmKVCache
        from mlx_vlm.models.cache import RotatingKVCache as VlmRotatingKVCache

        assert supports_positional_rollback(
            [VlmKVCache(), VlmRotatingKVCache(max_size=8)]
        )


class TestRotatingRollback:
    """Randomized rounds against a tiny window, checking what attention sees."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_rotating_window_never_exposes_rejected_positions(self, seed: int) -> None:
        rng = random.Random(seed)
        window = 6
        cache = RotatingKVCache(max_size=window)
        committed = 5
        _ = cache.update_and_fetch(*_positions([float(p) for p in range(committed)]))
        for _round in range(60):
            if rng.random() < 0.15:
                # Plain-decode fallback: one committed token, in-place write.
                keys, _ = cache.update_and_fetch(*_positions([float(committed)]))
                committed += 1
                visible = _flat(keys)
                assert _REJECTED not in visible
                expected = set(range(max(0, committed - window), committed))
                assert {int(v) for v in visible} == expected
                continue
            width = rng.randint(2, 4)
            accepted = rng.randint(0, width - 1)  # bonus + accepted drafts kept
            written = [float(committed + i) for i in range(accepted + 1)]
            written += [_REJECTED] * (width - accepted - 1)
            keys, _ = cache.update_and_fetch(*_positions(written))
            visible = _flat(keys)
            # The verify sees an ordered prefix of committed positions, at
            # least a full window of them once enough have been committed.
            prefix = visible[: len(visible) - width]
            assert _REJECTED not in prefix
            assert prefix == [float(p) for p in range(committed - len(prefix), committed)]
            assert len(prefix) >= min(committed, window - 1)
            rollback_speculative_positions([cache], width - accepted - 1)
            committed += accepted + 1
            state_keys = cast(mx.array, cache.state[0])
            assert _flat(state_keys) == [
                float(p) for p in range(committed - state_keys.shape[2], committed)
            ]
            assert cache.offset == committed

    def test_kv_cache_rollback_trims_offset(self) -> None:
        cache = KVCache()
        _ = cache.update_and_fetch(*_positions([0.0, 1.0, 2.0, 3.0]))
        rollback_speculative_positions([cache], 2)
        assert cache.offset == 2
        keys, _ = cache.state
        assert _flat(keys) == [0.0, 1.0]

    def test_zero_rejections_is_a_no_op(self) -> None:
        cache = RotatingKVCache(max_size=4)
        _ = cache.update_and_fetch(*_positions([0.0, 1.0, 2.0]))
        rollback_speculative_positions([cache], 0)
        assert cache.offset == 3


# ---------------------------------------------------------------------------
# End to end through the speculative loop with real caches
# ---------------------------------------------------------------------------

_VOCAB = 32
_HIDDEN = 4
_EOS = 0


def _successor(token: int) -> int:
    """The fake target's greedy next token (never EOS)."""
    return (token * 5 + 3) % (_VOCAB - 1) + 1


class _CacheWritingTarget:
    """Trunk/head pair whose cache writes and hiddens encode token ids."""

    def __init__(self, successor: dict[int, int] | None = None) -> None:
        self._successor = successor
        self.trunk_calls = 0

    def next_token(self, token: int) -> int:
        if self._successor is not None:
            return self._successor.get(token, 1)
        return _successor(token)

    def trunk(self, tokens: mx.array, cache: list[object] | None = None) -> mx.array:
        self.trunk_calls += 1
        ids = tokens.reshape(-1).astype(mx.float32)
        for entry in cache or []:
            kv = ids.reshape(1, 1, -1, 1)
            _ = cast(KVCache, entry).update_and_fetch(kv, kv)
        hidden = mx.zeros((1, ids.shape[0], _HIDDEN))
        return hidden + ids.reshape(1, -1, 1)

    def head(self, hidden: mx.array) -> mx.array:
        ids = [int(v) for v in cast("list[float]", hidden[0, :, 0].tolist())]
        rows = [
            mx.where(mx.arange(_VOCAB) == self.next_token(i), 100.0, 0.0) for i in ids
        ]
        return mx.stack(rows)[None]


class _PatternDrafter:
    """Drafts the true chain, then corrupts one position by a fixed pattern.

    Round ``r`` corrupts draft ``r % 3`` (2 = no corruption at depth 2), so
    rounds cycle through first-draft rejects, second-draft rejects, and full
    accepts. An EOS successor is followed by a junk draft that must never
    be accepted.
    """

    def __init__(self, target: _CacheWritingTarget) -> None:
        self._target = target
        self._round = 0

    def begin_request(self, prompt_cache: object) -> None:
        del prompt_cache

    def observe(self, hiddens: mx.array, next_tokens: mx.array) -> None:
        del hiddens, next_tokens

    def draft(self, hidden: mx.array, next_token: int, depth: int = 1) -> mx.array:
        del hidden
        corrupt_at = self._round % 3
        self._round += 1
        rows: list[mx.array] = []
        token = next_token
        for index in range(depth):
            token = self._target.next_token(token) if token != _EOS else 6
            proposal = token if index != corrupt_at else (token + 7) % (_VOCAB - 1) + 1
            rows.append(mx.where(mx.arange(_VOCAB) == proposal, 100.0, 0.0))
        return mx.stack(rows).astype(mx.float32)


def _tokenizer(eos_ids: list[int]) -> MagicMock:
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    tokenizer = MagicMock(spec=TokenizerWrapper)
    detokenizer = MagicMock()
    detokenizer.last_segment = ""
    tokenizer.detokenizer = detokenizer
    tokenizer.eos_token_ids = eos_ids
    return tokenizer


def _run_loop(
    target: _CacheWritingTarget,
    caches: list[object],
    *,
    prompt: list[int],
    max_tokens: int,
    eos_ids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], MagicMock]:
    from skulk.worker.engines.mlx.generator import generate

    def _no_snapshot(cache: object) -> object:
        raise AssertionError("attention-only caches must not be snapshotted")

    monkeypatch.setattr(generate, "snapshot_ssm_states", _no_snapshot)
    native_rollback = MagicMock()
    language_model = MagicMock()
    language_model.rollback_speculative_cache = native_rollback
    model = MagicMock()
    model.language_model = language_model
    outputs = list(
        generate._stream_generate_with_mtp(
            model=model,
            tokenizer=_tokenizer(eos_ids),
            drafter=_PatternDrafter(target),
            trunk_fn=target.trunk,
            head_fn=target.head,
            prompt=mx.array(prompt),
            max_tokens=max_tokens,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            logits_processors=[],
            prompt_cache=caches,
            kv_group_size=None,
            kv_bits=None,
            depth=2,
        )
    )
    return [int(output.token) for output in outputs], native_rollback


class TestLoopPositionalRollback:
    def test_greedy_output_and_cache_match_the_target_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = _CacheWritingTarget()
        full_cache = KVCache()
        window_cache = RotatingKVCache(max_size=4)
        prompt = [3, 9, 4]
        emitted, native_rollback = _run_loop(
            target,
            [full_cache, window_cache],
            prompt=prompt,
            max_tokens=24,
            eos_ids=[],
            monkeypatch=monkeypatch,
        )

        # The target's own greedy sequence, a few tokens past max_tokens:
        # accepted drafts beyond the last emitted token stay committed.
        truth: list[int] = []
        token = prompt[-1]
        for _ in range(24 + 4):
            token = target.next_token(token)
            truth.append(token)
        assert emitted == truth[:24]
        # The loop owns the rollback; the model hook stays unused.
        native_rollback.assert_not_called()
        # One trunk call per round plus prefill and first bonus: no replay.
        assert target.trunk_calls <= 2 + len(emitted)

        sequence = [float(t) for t in prompt + truth]
        full_keys, _ = full_cache.state
        cached = _flat(full_keys)
        # Every cached position is a committed token of the true sequence.
        assert cached == sequence[: len(cached)]
        assert len(cached) >= len(prompt) + len(emitted) - 1
        # The loop's last cache write was a verify or a rollback, both of
        # which leave the rotating buffer in temporal order.
        window = _flat(cast(mx.array, window_cache.state[0]))[-window_cache.max_size :]
        assert window == cached[-len(window) :]

    def test_accepted_eos_ends_the_stream_and_drops_later_drafts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 4 -> 7 -> 12 -> 5 -> 9 -> EOS; the drafter follows EOS with junk.
        successor = {4: 7, 7: 12, 12: 5, 5: 9, 9: _EOS}
        target = _CacheWritingTarget(successor)
        full_cache = KVCache()
        emitted, _ = _run_loop(
            target,
            [full_cache, RotatingKVCache(max_size=4)],
            prompt=[2, 4],
            max_tokens=32,
            eos_ids=[_EOS],
            monkeypatch=monkeypatch,
        )
        assert emitted == [7, 12, 5, 9, _EOS]
        keys, _ = full_cache.state
        assert 6.0 not in _flat(keys)

