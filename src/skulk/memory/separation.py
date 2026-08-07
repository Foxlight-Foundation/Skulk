# pyright: reportAny=false
"""Pattern separation: the fractional ZCA whitener.

Raw sentence embeddings are too anisotropic to use as HRR cues: unrelated
conversations sit at ~0.46 cosine, which swamps the recall signal (Phase 0
experiment 3 measured a ceiling of zero). Fitting a fractional ZCA whitener and
mapping every key and cue through it is the mandatory separation stage that
makes real cues usable.

Fractional whitening always subtracts the mean, then scales each eigendirection
of the covariance by ``lambda ** (-alpha/2)``:

    alpha = 0   ->  centering only (no whitening)
    alpha = 1   ->  full ZCA (over-corrects; amplifies cue perturbation)

The measured operating point is ``alpha = 0.5`` with shrinkage ``0.05``. The
whitener fits once on a generic conversation corpus, freezes, and is versioned;
Phase 0 holdout confirmed it generalizes to unseen conversations, so the same
fit serves new traffic. The whitener version is part of memory identity: a field
built under one whitener cannot be probed with cues from another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skulk.memory.hrr import DTYPE, Vector

DEFAULT_ALPHA = 0.5
DEFAULT_SHRINKAGE = 0.05


@dataclass(frozen=True)
class Whitener:
    """A frozen, versioned fractional-ZCA separation map.

    ``mu`` is the corpus mean, ``matrix`` the symmetric whitening operator. Apply
    with :meth:`transform`. Construct with :meth:`fit`; the ``version`` string and
    ``embedding_model_id`` travel with any field built through this whitener so a
    mismatched cue space is detectable rather than silently wrong.
    """

    mu: NDArray[np.float32]
    matrix: NDArray[np.float32]
    alpha: float = DEFAULT_ALPHA
    shrinkage: float = DEFAULT_SHRINKAGE
    embedding_model_id: str = "unknown"
    version: str = "v1"
    _extra: dict[str, str] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        """Embedding dimension this whitener maps from/to."""
        return int(self.mu.shape[0])

    @classmethod
    def fit(
        cls,
        embeddings: Vector,
        *,
        alpha: float = DEFAULT_ALPHA,
        shrinkage: float = DEFAULT_SHRINKAGE,
        embedding_model_id: str = "unknown",
        version: str = "v1",
    ) -> "Whitener":
        """Fit a fractional whitener on a corpus of embeddings ``(n, d)``.

        Shrinkage blends the empirical covariance toward a scaled identity for
        numerical stability on limited corpora. The result is frozen; refitting
        (e.g. by a future consolidation daemon) mints a new version.
        """
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n, d); got shape {x.shape}")
        if len(x) < 2:
            raise ValueError(
                f"whitener needs at least 2 embeddings to fit a covariance; got {len(x)}"
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1]; got {shrinkage}")
        mu = x.mean(axis=0)
        xc = x - mu
        cov = (xc.T @ xc) / len(xc)
        trace_scale = np.trace(cov) / cov.shape[0]
        cov = (1.0 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * trace_scale
        eigval, eigvec = np.linalg.eigh(cov)
        scale = np.power(np.maximum(eigval, 1e-10), -alpha / 2.0)
        matrix = eigvec @ np.diag(scale) @ eigvec.T
        return cls(
            mu=mu.astype(DTYPE),
            matrix=matrix.astype(DTYPE),
            alpha=alpha,
            shrinkage=shrinkage,
            embedding_model_id=embedding_model_id,
            version=version,
        )

    def transform(self, embeddings: Vector) -> Vector:
        """Map raw embeddings into the separated cue space.

        Accepts a single vector ``(d,)`` or a batch ``(n, d)`` and preserves the
        input's rank. Centering is applied before the whitening map.
        """
        x = np.asarray(embeddings, dtype=np.float64)
        out = (x - self.mu) @ self.matrix
        return out.astype(DTYPE)