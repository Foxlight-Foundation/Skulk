# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Tests for the fractional-ZCA pattern-separation stage."""

from __future__ import annotations

import numpy as np

from skulk.memory.hrr import DTYPE, Vector, normalize
from skulk.memory.separation import Whitener


def _anisotropic_episodes(seed: int) -> tuple[Vector, np.ndarray]:
    """Synthesize embeddings with the anisotropy real sentence embeddings show.

    Six episodes of six spans each, every vector loaded heavily on one shared
    dominant axis so that unrelated episodes sit at high cosine (the failure
    mode Phase 0 measured). Returns the vectors and their episode ids.
    """
    # Many more spans than dimensions so the sample covariance is full rank and
    # its condition number is a stable measure of anisotropy.
    rng = np.random.default_rng(seed)
    dim = 32
    # The anisotropy axis is a property of the embedding model, so it is shared
    # across every corpus (fixed seed); only the episodes and spans vary. This is
    # what lets a whitener fit on one corpus generalize to unseen conversations.
    aniso_axis = normalize(np.random.default_rng(0).standard_normal(dim).astype(DTYPE))
    vectors: list[Vector] = []
    episode_ids: list[int] = []
    for episode in range(20):
        center = rng.standard_normal(dim).astype(DTYPE) * 0.5
        for _ in range(15):
            span = center + rng.standard_normal(dim).astype(DTYPE) * 0.3
            span = span + aniso_axis * float(rng.standard_normal()) * 3.0
            vectors.append(span.astype(DTYPE))
            episode_ids.append(episode)
    return np.asarray(vectors, dtype=DTYPE), np.asarray(episode_ids)


def _anisotropy(vectors: Vector) -> float:
    """Condition number of the covariance: how anisotropic the cloud is.

    A perfectly isotropic (white) cloud has ratio 1; a strong shared axis blows
    the top eigenvalue up and the ratio with it. Reducing this ratio is exactly
    what the separation stage does.
    """
    x = np.asarray(vectors, dtype=np.float64)
    cov = np.cov(x, rowvar=False)
    eigval = np.linalg.eigvalsh(cov)
    return float(eigval[-1] / max(eigval[0], 1e-9))


def test_whitener_reduces_anisotropy() -> None:
    vectors, _ = _anisotropic_episodes(seed=100)
    raw = _anisotropy(vectors)
    separated = _anisotropy(Whitener.fit(vectors, alpha=0.5).transform(vectors))
    # The synthetic cloud is strongly anisotropic; alpha=0.5 flattens it a lot.
    assert raw > 10.0
    assert separated < raw / 2.0


def test_alpha_is_the_separation_dial() -> None:
    # Anisotropy must fall monotonically as alpha goes centering -> half -> full,
    # and full ZCA must land essentially isotropic. This is the measured dial.
    vectors, _ = _anisotropic_episodes(seed=101)
    centering = _anisotropy(Whitener.fit(vectors, alpha=0.0).transform(vectors))
    half = _anisotropy(Whitener.fit(vectors, alpha=0.5).transform(vectors))
    full = _anisotropy(Whitener.fit(vectors, alpha=1.0).transform(vectors))
    assert full < half < centering
    assert full < 2.0  # full ZCA whitens to near-isotropy


def test_whitener_generalizes_to_held_out_vectors() -> None:
    # Fit on one corpus, apply to unseen vectors: the transform must still flatten
    # their covariance (fit-once / freeze / reuse is the design).
    fit_vectors, _ = _anisotropic_episodes(seed=102)
    holdout, _ = _anisotropic_episodes(seed=103)
    whitener = Whitener.fit(fit_vectors, alpha=0.5)
    raw = _anisotropy(holdout)
    separated = _anisotropy(whitener.transform(holdout))
    assert separated < raw


def test_whitener_shape_and_identity() -> None:
    vectors, _ = _anisotropic_episodes(seed=104)
    whitener = Whitener.fit(
        vectors, embedding_model_id="bge-small-en-v1.5", version="test-v1"
    )
    assert whitener.input_dim == 32
    assert whitener.transform(vectors).shape == vectors.shape
    assert whitener.transform(vectors[0]).shape == (32,)
    # Identity travels with the whitener so a mismatched cue space is detectable.
    assert whitener.embedding_model_id == "bge-small-en-v1.5"
    assert whitener.version == "test-v1"