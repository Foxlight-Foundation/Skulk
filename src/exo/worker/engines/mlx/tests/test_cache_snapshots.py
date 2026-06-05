# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportIndexIssue=false, reportArgumentType=false
"""Unit tests for the reference-snapshot semantics of SSM cache rollback.

``snapshot_ssm_states`` reference-clones ArraysCache entries instead of
deep-copying them (the MTP verify loop snapshots every round, and deepcopy
forces a recurrent-state buffer copy + GPU sync per round). That is only
correct because SSM layers *replace* cache slots wholesale and never mutate
the stored arrays in place. These tests pin that contract.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.engines.mlx.cache import (
    snapshot_ssm_states,
    trim_cache,
)


def _arrays_cache(values: list[float]) -> ArraysCache:
    cache = ArraysCache(len(values))
    for slot, value in enumerate(values):
        cache[slot] = mx.full((1, 4), value)
    return cache


def _assert_slots_equal(entry: object, values: list[float]) -> None:
    assert isinstance(entry, ArraysCache)
    for slot, value in enumerate(values):
        assert mx.allclose(entry[slot], mx.full((1, 4), value)), (
            f"slot {slot} expected {value}"
        )


class TestArraysCacheSnapshot:
    def test_restore_undoes_slot_replacement(self) -> None:
        ssm = _arrays_cache([1.0, 2.0])
        live: list[object] = [ssm]
        snapshot = snapshot_ssm_states(live)

        # SSM forward passes replace slots wholesale — simulate one.
        ssm[0] = mx.full((1, 4), 9.0)
        ssm[1] = mx.full((1, 4), 8.0)

        trim_cache(live, 1, snapshot)
        _assert_slots_equal(live[0], [1.0, 2.0])

    def test_snapshot_survives_multiple_restores(self) -> None:
        # KVPrefixCache restores the same snapshot across requests; the live
        # cache must never BE the snapshot's stored object.
        live: list[object] = [_arrays_cache([1.0, 2.0])]
        snapshot = snapshot_ssm_states(live)

        for round_value in (5.0, 6.0):
            restored = live[0]
            assert isinstance(restored, ArraysCache)
            restored[0] = mx.full((1, 4), round_value)
            trim_cache(live, 1, snapshot)
            _assert_slots_equal(live[0], [1.0, 2.0])
            assert live[0] is not snapshot.states[0]

    def test_snapshot_shares_array_references(self) -> None:
        # The point of the change: no buffer copies at snapshot time.
        ssm = _arrays_cache([1.0])
        snapshot = snapshot_ssm_states([ssm])
        snapshot_entry = snapshot.states[0]
        assert isinstance(snapshot_entry, ArraysCache)
        assert snapshot_entry[0] is ssm[0]

    def test_plain_kv_entries_are_trimmed_not_snapshotted(self) -> None:
        kv = KVCache()
        keys = mx.zeros((1, 2, 6, 4))
        values = mx.zeros((1, 2, 6, 4))
        kv.update_and_fetch(keys, values)
        live: list[object] = [kv]

        snapshot = snapshot_ssm_states(live)
        assert snapshot.states[0] is None

        trim_cache(live, 2, snapshot)
        assert kv.offset == 4


class TestRotatingKVCacheSnapshot:
    def test_snapshot_is_a_real_copy(self) -> None:
        # RotatingKVCache writes into preallocated buffers in place, so it
        # must NOT share references with the snapshot.
        rotating = RotatingKVCache(max_size=8)
        keys = mx.ones((1, 2, 4, 4))
        values = mx.ones((1, 2, 4, 4))
        rotating.update_and_fetch(keys, values)
        live: list[object] = [rotating]

        snapshot = snapshot_ssm_states(live)

        # In-place mutation of the live buffers must not leak into the
        # snapshot.
        rotating.update_and_fetch(
            mx.full((1, 2, 1, 4), 7.0), mx.full((1, 2, 1, 4), 7.0)
        )

        trim_cache(live, 1, snapshot)
        restored = live[0]
        assert isinstance(restored, RotatingKVCache)
        assert restored.offset == 4
        restored_keys = restored.state[0]
        assert restored_keys is not None
        assert mx.allclose(restored_keys[..., :4, :], mx.ones((1, 2, 4, 4)))
