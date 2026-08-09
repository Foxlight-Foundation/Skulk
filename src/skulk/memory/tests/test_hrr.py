# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Unit tests for the HRR primitives."""

from __future__ import annotations

import numpy as np

from skulk.memory.hrr import (
    bind,
    cosine,
    make_unitary,
    normalize,
    random_vectors,
    superpose,
    unbind,
)


def test_unbind_inverts_bind() -> None:
    v = random_vectors(2, 1024, seed=1)
    a, b = v[0], v[1]
    recovered = unbind(bind(a, b), a)
    # With a plain Gaussian key, unbinding is only approximate: recall lands much
    # closer to b than to anything unrelated, which cleanup then snaps exactly.
    other = random_vectors(1, 1024, seed=2)[0]
    assert cosine(recovered, b) > 0.5
    assert cosine(recovered, b) > cosine(recovered, other) + 0.3


def test_unitary_key_gives_exact_recovery() -> None:
    a = make_unitary(random_vectors(1, 1024, seed=1)[0])
    b = random_vectors(1, 1024, seed=2)[0]
    # A unitary key makes correlation the exact inverse of convolution.
    assert cosine(unbind(bind(a, b), a), b) > 0.99


def test_bind_is_dissimilar_to_operands() -> None:
    v = random_vectors(2, 1024, seed=3)
    bound = bind(v[0], v[1])
    assert abs(cosine(bound, v[0])) < 0.2
    assert abs(cosine(bound, v[1])) < 0.2


def test_bind_commutes() -> None:
    v = random_vectors(2, 512, seed=4)
    assert cosine(bind(v[0], v[1]), bind(v[1], v[0])) > 0.999


def test_superpose_sums() -> None:
    v = random_vectors(5, 256, seed=5)
    assert np.allclose(superpose(v), v.sum(axis=0), atol=1e-5)


def test_normalize_unit_length() -> None:
    v = random_vectors(3, 128, seed=6)
    n = normalize(v)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-5)


def test_random_vectors_are_deterministic_under_seed() -> None:
    assert np.array_equal(random_vectors(4, 64, seed=7), random_vectors(4, 64, seed=7))