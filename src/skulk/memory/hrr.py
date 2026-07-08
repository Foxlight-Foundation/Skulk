# pyright: reportAny=false
"""Holographic Reduced Representation primitives (Plate, 1995).

The core mechanic behind the holographic memory field: bind a key to a value by
circular convolution, superpose many bindings into one fixed-size vector by
summation, and recall a value by circular correlation (the approximate inverse
of binding). These are pure numpy functions with no state, no I/O, and no Skulk
imports; the memory index (:mod:`skulk.memory.index`) composes them.

The viability of this representation was measured in the fabric-memory Phase 0
experiments: reliable recall at roughly ``N = D/20`` superposed traces with
graceful degradation, and cross-episode false-confident recall of zero at the
confidence gate. See ``initiatives/fabric-memory`` in the foxlight-docs repo.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# All vectors and fields are stored as float32: embeddings arrive as float32 and
# a D=32768 field is then 128 KB in RAM / 64 KB fp16 on the wire. The FFT upcasts
# internally; results are cast back so callers only ever see float32.
Vector = NDArray[np.float32]

DTYPE = np.float32


def random_vectors(count: int, dim: int, *, seed: int | None = None) -> Vector:
    """Return ``count`` i.i.d. Gaussian HRR vectors of dimension ``dim``.

    Scaled by ``1/sqrt(dim)`` so a binding of two such vectors has unit-ish
    scale, matching the standard HRR construction. A ``seed`` makes the draw
    deterministic (used by tests and by role/time vectors that must be stable).
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((count, dim)) / np.sqrt(dim)
    return v.astype(DTYPE)


def make_unitary(v: Vector) -> Vector:
    """Project ``v`` onto the nearest unitary vector (unit magnitude per bin).

    A unitary vector has magnitude 1 in every frequency bin, which makes
    circular correlation the *exact* inverse of circular convolution, so
    ``unbind(bind(unitary_key, value), unitary_key) == value``. Plate (1995)
    uses unitary vectors for the key/role so recall fidelity does not erode with
    field density; binding keys this way is what lets a D=32768 field hold
    400-800 traces at >90% accuracy, matching the Phase 0 capacity law.
    """
    spectrum = np.fft.rfft(v)
    magnitude = np.abs(spectrum)
    spectrum = spectrum / np.where(magnitude == 0, 1, magnitude)
    return np.fft.irfft(spectrum, n=v.shape[-1]).astype(DTYPE)


def bind(a: Vector, b: Vector) -> Vector:
    """Circular convolution of ``a`` and ``b`` via FFT.

    ``bind(a, b)`` is dissimilar to both operands and associates a key with a
    value. Binding is commutative and associative.
    """
    n = a.shape[-1]
    out = np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=n)
    return out.astype(DTYPE)


def unbind(c: Vector, a: Vector) -> Vector:
    """Circular correlation: the approximate inverse of :func:`bind`.

    ``unbind(bind(a, b), a) ~ b``. Probing a superposed field with a key cue
    recovers a noisy estimate of the bound value, cleaned up against the value
    codebook by the caller.
    """
    n = c.shape[-1]
    out = np.fft.irfft(np.fft.rfft(c) * np.conj(np.fft.rfft(a)), n=n)
    return out.astype(DTYPE)


def superpose(vectors: Vector) -> Vector:
    """Sum a stack of vectors ``(count, dim)`` into one field vector ``(dim,)``.

    Superposition is how many key/value bindings share one fixed-size field.
    """
    return vectors.sum(axis=0).astype(DTYPE)


def normalize(v: Vector, *, axis: int = -1) -> Vector:
    """L2-normalize along ``axis``; zero vectors are left unchanged."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return (v / np.where(norm == 0, 1, norm)).astype(DTYPE)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)