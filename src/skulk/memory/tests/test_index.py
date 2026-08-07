# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Behavioral suite for MemoryIndex: both implementations pass the same gates.

These reproduce the Phase 0 experiment ceilings from deterministic seeded
fixtures (no external corpus) and assert the invariants the whole architecture
rests on: reliable recall below capacity, zero cross-episode false-confidence at
the gate, order-independent convergence, and exact forgetting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from skulk.memory.hrr import DTYPE, Vector, random_vectors
from skulk.memory.index import ExactIndex, HolographicField, MemoryIndex

DIM = 4096  # D/20 capacity ~200 traces; tests run comfortably below the ceiling


def make_exact() -> MemoryIndex:
    return ExactIndex()


def make_field() -> MemoryIndex:
    return HolographicField(dim=DIM)


INDEX_FACTORIES: list[Callable[[], MemoryIndex]] = [make_exact, make_field]


def _noisy(vec: Vector, level: float, seed: int) -> Vector:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(vec.shape).astype(DTYPE) * (level / np.sqrt(vec.shape[-1]))
    return (vec + noise).astype(DTYPE)


@pytest.mark.parametrize("factory", INDEX_FACTORIES)
def test_single_write_recalls_exactly(factory: Callable[[], MemoryIndex]) -> None:
    index = factory()
    keys = random_vectors(1, DIM, seed=10)
    values = random_vectors(1, DIM, seed=11)
    index.write("t0", keys[0], values[0])
    result = index.probe(keys[0])
    assert result.hit
    assert result.trace_id == "t0"


@pytest.mark.parametrize("factory", INDEX_FACTORIES)
def test_capacity_below_ceiling(factory: Callable[[], MemoryIndex]) -> None:
    # 80 traces at D=4096 is well below the ~D/20 ceiling; noisy cues (the
    # imperfect cache->cue projector) must still recall the right trace >= 90%.
    n = 80
    keys = random_vectors(n, DIM, seed=20)
    values = random_vectors(n, DIM, seed=21)
    index = factory()
    for i in range(n):
        index.write(f"t{i}", keys[i], values[i])

    correct = 0
    for i in range(n):
        cue = _noisy(keys[i], level=0.25, seed=1000 + i)
        result = index.probe(cue)
        if result.trace_id == f"t{i}":
            correct += 1
    assert correct / n >= 0.90


@pytest.mark.parametrize("factory", INDEX_FACTORIES)
def test_never_confabulates_across_episodes(factory: Callable[[], MemoryIndex]) -> None:
    # The load-bearing invariant: probing with cues that were NEVER written must
    # not produce a confident recall. Phase 0 measured this at exactly 0.000.
    n = 80
    keys = random_vectors(n, DIM, seed=30)
    values = random_vectors(n, DIM, seed=31)
    index = factory()
    for i in range(n):
        index.write(f"t{i}", keys[i], values[i])

    fresh = random_vectors(300, DIM, seed=999)  # unrelated cues
    false_confident = sum(1 for cue in fresh if index.probe(cue).hit)
    assert false_confident == 0


@pytest.mark.parametrize("factory", INDEX_FACTORIES)
def test_convergence_is_order_independent(factory: Callable[[], MemoryIndex]) -> None:
    # Writing the same traces in any order yields the same recall (superposition
    # is commutative), so replicas that apply deltas in different orders agree.
    n = 40
    keys = random_vectors(n, DIM, seed=40)
    values = random_vectors(n, DIM, seed=41)

    forward = factory()
    for i in range(n):
        forward.write(f"t{i}", keys[i], values[i])
    reverse = factory()
    for i in reversed(range(n)):
        reverse.write(f"t{i}", keys[i], values[i])

    for i in range(n):
        cue = _noisy(keys[i], level=0.2, seed=2000 + i)
        a = forward.probe(cue)
        b = reverse.probe(cue)
        assert a.trace_id == b.trace_id
        assert a.confidence == pytest.approx(b.confidence, abs=1e-4)


@pytest.mark.parametrize("factory", INDEX_FACTORIES)
def test_subtract_forgets_exactly(factory: Callable[[], MemoryIndex]) -> None:
    keys = random_vectors(3, DIM, seed=50)
    values = random_vectors(3, DIM, seed=51)
    index = factory()
    for i in range(3):
        index.write(f"t{i}", keys[i], values[i])
    assert index.probe(keys[1]).trace_id == "t1"

    index.subtract("t1", keys[1], values[1])
    assert len(index) == 2
    result = index.probe(keys[1])
    # The forgotten trace can never be returned again.
    assert result.trace_id != "t1"


@pytest.mark.parametrize("factory", INDEX_FACTORIES)
def test_decay_to_zero_clears_recall(factory: Callable[[], MemoryIndex]) -> None:
    keys = random_vectors(5, DIM, seed=60)
    values = random_vectors(5, DIM, seed=61)
    index = factory()
    for i in range(5):
        index.write(f"t{i}", keys[i], values[i])
    index.decay(0.0)
    for i in range(5):
        assert not index.probe(keys[i]).hit


@pytest.mark.parametrize("factory", INDEX_FACTORIES)
def test_rewrite_is_idempotent(factory: Callable[[], MemoryIndex]) -> None:
    keys = random_vectors(2, DIM, seed=70)
    values = random_vectors(2, DIM, seed=71)
    index = factory()
    index.write("t0", keys[0], values[0])
    index.write("t1", keys[1], values[1])
    # Rewriting t0 must not double-count it in the field.
    index.write("t0", keys[0], values[0])
    assert len(index) == 2
    assert index.probe(keys[0]).trace_id == "t0"

def test_holographic_rewrite_with_new_key_leaves_no_residue() -> None:
    """Rewriting a trace under a different key must subtract the stored binding.

    Regression test for the review finding on #782: the rewrite path
    previously subtracted with the caller's NEW key against the old value,
    leaving the original binding smeared in the superposition forever.
    """
    from skulk.memory.hrr import random_vectors
    from skulk.memory.index import HolographicField

    dim = 2048
    field = HolographicField(dim=dim)
    keys = random_vectors(2, dim, seed=7)
    values = random_vectors(2, dim, seed=8)
    field.write("t1", keys[0], values[0])
    field.write("t1", keys[1], values[1])
    field.subtract("t1", keys[1], values[1])
    assert len(field) == 0
    # An exact lifecycle drains the field to near-zero energy; the old bug
    # left the original binding smeared in the superposition here.
    assert field.energy() < 1e-3


def test_holographic_subtract_ignores_caller_vectors() -> None:
    """Subtract removes the stored binding even when the caller passes junk."""
    from skulk.memory.hrr import random_vectors
    from skulk.memory.index import HolographicField

    dim = 2048
    field = HolographicField(dim=dim)
    keys = random_vectors(2, dim, seed=9)
    values = random_vectors(2, dim, seed=10)
    field.write("t1", keys[0], values[0])
    field.subtract("t1", keys[1], values[1])
    assert len(field) == 0
    assert field.energy() < 1e-3


def test_whitener_fit_rejects_degenerate_input() -> None:
    import pytest

    from skulk.memory.separation import Whitener

    single = np.ones((1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        Whitener.fit(single)
    corpus = np.random.default_rng(0).normal(size=(16, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        Whitener.fit(corpus, alpha=1.5)
    with pytest.raises(ValueError):
        Whitener.fit(corpus, shrinkage=-0.1)


def test_holographic_decay_gates_out_like_exact_index() -> None:
    """Amplitude folds into field confidence, matching ExactIndex semantics."""
    from skulk.memory.hrr import random_vectors
    from skulk.memory.index import ExactIndex, HolographicField

    dim = 2048
    keys = random_vectors(1, dim, seed=21)
    values = random_vectors(1, dim, seed=22)
    field = HolographicField(dim=dim)
    exact = ExactIndex()
    field.write("t1", keys[0], values[0])
    exact.write("t1", keys[0], values[0])
    assert field.probe(keys[0]).hit
    assert exact.probe(keys[0]).hit
    field.decay(0.05)
    exact.decay(0.05)
    assert not field.probe(keys[0]).hit
    assert not exact.probe(keys[0]).hit


def test_holographic_probe_disambiguates_shared_value_codes() -> None:
    """Two traces with one value code resolve to the trace whose key matches."""
    from skulk.memory.hrr import random_vectors
    from skulk.memory.index import HolographicField

    dim = 2048
    keys = random_vectors(2, dim, seed=31)
    values = random_vectors(1, dim, seed=32)
    field = HolographicField(dim=dim)
    field.write("t1", keys[0], values[0])
    field.write("t2", keys[1], values[0])
    assert field.probe(keys[0]).trace_id == "t1"
    assert field.probe(keys[1]).trace_id == "t2"


def test_holographic_mixed_age_decay_counts_once() -> None:
    """A decayed trace beside a fresh one keeps its true (single) attenuation.

    Regression test for the review finding on #782: scoring a NORMALIZED
    projection and multiplying tracked amplitude double-counted decay once
    the field held mixed ages, so a 0.3-amplitude trace scored ~0.09 and
    missed the gate it should clear.
    """
    from skulk.memory.hrr import random_vectors
    from skulk.memory.index import HolographicField

    dim = 8192
    keys = random_vectors(2, dim, seed=41)
    values = random_vectors(2, dim, seed=42)
    field = HolographicField(dim=dim)
    field.write("old", keys[0], values[0])
    field.decay(0.3)
    field.write("fresh", keys[1], values[1])
    old_probe = field.probe(keys[0])
    fresh_probe = field.probe(keys[1])
    assert fresh_probe.trace_id == "fresh" and fresh_probe.hit
    assert old_probe.trace_id == "old"
    # Single-counted decay: confidence near 0.3, comfortably above the gate;
    # the old double-count landed near 0.09 and missed.
    assert old_probe.hit
    assert 0.2 < old_probe.confidence < 0.45
