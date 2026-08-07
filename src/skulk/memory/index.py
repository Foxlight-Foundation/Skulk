"""The MemoryIndex interface and its two implementations.

An index maps a *key* cue to the *trace* whose key best matches it, over a
population of superposed writes, and never confidently returns a trace that was
not written. Two implementations satisfy the same contract:

- :class:`ExactIndex` -- the flat baseline. Stores each key/value and scores a
  probe by cosine to every key. This is the honest bar the holographic field
  must beat; at short-term-memory scale a flat index often wins on points, which
  is exactly why the bench-off (fabric-memory Phase 5) picks the default from
  measurement rather than assumption.
- :class:`HolographicField` -- the HRR field. Binds each key to its value and
  superposes all bindings into one fixed-size vector; a probe unbinds the cue
  and cleans up against the value codebook. Field size is O(D) regardless of how
  many traces it holds.

Both are decay-aware (a per-trace amplitude fades with age) and support exact
subtraction (deliberate forgetting). Each probe returns the index's best guess
plus a confidence and a gate flag; the caller surfaces a recall only when the
gate is cleared, which is where the "never confabulate" guarantee lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from skulk.memory.hrr import DTYPE, Vector, bind, cosine, normalize, unbind

# Confidence threshold measured safe in Phase 0: cross-episode false-confident
# recall was 0.000 at every field density at and above this cosine. The
# production value is re-derived from live statistics in Phase 5.
DEFAULT_CONFIDENCE_GATE = 0.15


_SCORE_TIE_TOLERANCE = 1e-6
"""Projection scores within this tolerance count as tied (shared value codes)."""


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of probing an index with a cue.

    ``trace_id`` is the index's best guess (the cleanup argmax), or ``None`` only
    when the index is empty. ``confidence`` is that guess's score. ``hit`` is
    ``confidence >= gate``: it is the surfacing decision. A caller must gate on
    ``hit`` and never surface a miss, because below the gate the best guess is
    just the nearest neighbour of noise. Recall accuracy is measured on
    ``trace_id``; surfacing precision (never confabulate) is measured on ``hit``.
    """

    trace_id: str | None
    confidence: float
    hit: bool


@runtime_checkable
class MemoryIndex(Protocol):
    """Write/probe/decay/subtract contract shared by every representation."""

    def write(self, trace_id: str, key: Vector, value: Vector) -> None:
        """Associate ``key`` with ``value`` under ``trace_id`` at full strength."""
        ...

    def probe(self, cue: Vector) -> ProbeResult:
        """Return the best-matching trace for ``cue``, gated by confidence."""
        ...

    def decay(self, retention: float) -> None:
        """Scale every trace's strength by ``retention`` (0 < retention <= 1)."""
        ...

    def subtract(self, trace_id: str, key: Vector, value: Vector) -> None:
        """Exactly remove a trace's current contribution (deliberate forgetting)."""
        ...

    def __len__(self) -> int:
        """Number of live traces."""
        ...


@dataclass
class ExactIndex:
    """Flat baseline index: stores unit keys and scores probes by cosine.

    Confidence folds the direction match with the trace's current amplitude, so
    decayed traces score lower and eventually fall below the gate.
    """

    gate: float = DEFAULT_CONFIDENCE_GATE

    def __post_init__(self) -> None:
        self._keys: dict[str, Vector] = {}
        self._values: dict[str, Vector] = {}
        self._amplitude: dict[str, float] = {}

    def write(self, trace_id: str, key: Vector, value: Vector) -> None:
        """Store (or overwrite) a trace at full amplitude."""
        self._keys[trace_id] = normalize(np.asarray(key, dtype=DTYPE))
        self._values[trace_id] = normalize(np.asarray(value, dtype=DTYPE))
        self._amplitude[trace_id] = 1.0

    def probe(self, cue: Vector) -> ProbeResult:
        """Score the cue against every key; hit iff best score clears the gate."""
        if not self._keys:
            return ProbeResult(trace_id=None, confidence=0.0, hit=False)
        cue_n = normalize(np.asarray(cue, dtype=DTYPE))
        best_id: str | None = None
        best_score = -np.inf
        for trace_id, key in self._keys.items():
            score = cosine(cue_n, key) * self._amplitude[trace_id]
            if score > best_score:
                best_score = score
                best_id = trace_id
        confidence = float(max(best_score, 0.0))
        hit = confidence >= self.gate
        return ProbeResult(trace_id=best_id, confidence=confidence, hit=hit)

    def decay(self, retention: float) -> None:
        """Multiply every trace's amplitude by ``retention`` (0..1)."""
        for trace_id in list(self._amplitude):
            self._amplitude[trace_id] *= retention

    def subtract(self, trace_id: str, key: Vector, value: Vector) -> None:
        """Forget a trace. Exact for a flat store: the row is simply dropped."""
        del key, value  # exact for a flat store: just drop the row
        self._keys.pop(trace_id, None)
        self._values.pop(trace_id, None)
        self._amplitude.pop(trace_id, None)

    def __len__(self) -> int:
        """Number of live traces."""
        return len(self._keys)


@dataclass
class HolographicField:
    """HRR holographic field: one fixed-size vector holding all bindings.

    The field is the superposition of ``amplitude * bind(key, value)`` over every
    live trace. A probe unbinds the cue and scores each codebook value with
    ``min(direction cosine, raw projection)``: the cosine bounds cross-episode
    noise independent of density (the Phase 0 zero-false-confabulation
    invariant) while the raw projection is an unbiased estimate of current
    amplitude, so decay gates traces out whether they sit alone in the field
    or beside fresh writes.
    """

    dim: int
    gate: float = DEFAULT_CONFIDENCE_GATE

    def __post_init__(self) -> None:
        self._field: Vector = np.zeros(self.dim, dtype=DTYPE)
        self._codebook: dict[str, Vector] = {}  # trace_id -> unit value
        self._keys: dict[str, Vector] = {}  # trace_id -> unit key (authoritative)
        self._amplitude: dict[str, float] = {}

    def _key(self, key: Vector) -> Vector:
        """Unit-normalize a key to field dimension.

        A plain L2-normalized key is what the Phase 0 capacity experiment used
        and it reaches the ``N ~ D/20`` ceiling with cleanup; unitary keys are a
        stronger option (see :func:`skulk.memory.hrr.make_unitary`) but they are
        sensitive to noise entering *before* the projection, which is exactly how
        a real cue arrives, so the normalized key is the robust default here.
        """
        arr = np.asarray(key, dtype=DTYPE)
        if arr.shape[-1] != self.dim:
            raise ValueError(f"key dim {arr.shape[-1]} != field dim {self.dim}")
        return normalize(arr)

    def write(self, trace_id: str, key: Vector, value: Vector) -> None:
        """Superpose a trace into the field (rewrites subtract the old binding)."""
        key_u = self._key(key)
        value_n = normalize(np.asarray(value, dtype=DTYPE))
        if value_n.shape[-1] != self.dim:
            raise ValueError(f"value dim {value_n.shape[-1]} != field dim {self.dim}")
        # If the trace already exists, remove its old contribution first so a
        # rewrite is idempotent rather than doubly-superposed. The STORED key
        # and value are authoritative here: subtracting with the caller's new
        # key would leave the old binding smeared in the field forever.
        if trace_id in self._amplitude:
            self._field = (
                self._field
                - self._amplitude[trace_id]
                * bind(self._keys[trace_id], self._codebook[trace_id])
            ).astype(DTYPE)
        self._field = self._field + bind(key_u, value_n)
        self._codebook[trace_id] = value_n
        self._keys[trace_id] = key_u
        self._amplitude[trace_id] = 1.0

    def probe(self, cue: Vector) -> ProbeResult:
        """Unbind the cue and project onto the value codebook; gate the score."""
        if not self._codebook:
            return ProbeResult(trace_id=None, confidence=0.0, hit=False)
        cue_u = self._key(cue)
        # Normalized cosine of the recovered direction to each unit value: this
        # is exactly the score the Phase 0 experiments gated at 0.15 and measured
        # zero cross-episode false-confidence for, so the invariant transfers.
        # Confidence is min(direction cosine, raw strength), because each
        # term is the tight bound in a different regime and both must clear
        # the gate:
        #  - The NORMALIZED cosine bounds cross-episode noise independent of
        #    field density (the Phase 0 zero-false-confabulation invariant;
        #    raw projection noise grows with sqrt(N/D) and would confabulate
        #    near capacity).
        #  - The RAW projection is an unbiased estimate of the trace's
        #    current amplitude, so decay gates an ISOLATED trace out (where
        #    normalization cancels it) and mixed-age fields count decay
        #    exactly once (where cosine-times-amplitude double-counted it).
        recalled_raw = unbind(self._field, cue_u)
        recalled_direction = normalize(recalled_raw)
        best_id: str | None = None
        best_score = -np.inf
        best_key_match = -np.inf
        for trace_id, value_n in self._codebook.items():
            direction = float(np.dot(recalled_direction, value_n))  # pyright: ignore[reportAny] - np.dot stub gap
            strength = float(np.dot(recalled_raw, value_n))  # pyright: ignore[reportAny] - np.dot stub gap
            score = min(direction, strength)
            # Traces sharing one value code score identically on projection;
            # the cue's match against each trace's own stored key breaks the
            # tie so a confident hit names the right trace.
            key_match = float(np.dot(cue_u, self._keys[trace_id]))  # pyright: ignore[reportAny] - np.dot stub gap
            if score > best_score + _SCORE_TIE_TOLERANCE or (
                abs(score - best_score) <= _SCORE_TIE_TOLERANCE
                and key_match > best_key_match
            ):
                best_score = score
                best_key_match = key_match
                best_id = trace_id
        confidence = float(max(best_score, 0.0))
        hit = confidence >= self.gate
        return ProbeResult(trace_id=best_id, confidence=confidence, hit=hit)

    def decay(self, retention: float) -> None:
        """Multiply every trace's amplitude by ``retention`` (0..1)."""
        # Scaling the field scales every binding's contribution to the raw
        # projection the probe scores, so decayed traces gate out in both
        # index implementations; interference from newer full-amplitude
        # writes remains the second forgetting force. The per-trace amplitude
        # dict mirrors the field's bookkeeping for exact rewrite/subtract.
        # Absolute idle-trace expiry and rehearsal refresh are later operator
        # controls.
        self._field = (self._field * retention).astype(DTYPE)
        for trace_id in list(self._amplitude):
            self._amplitude[trace_id] *= retention

    def subtract(self, trace_id: str, key: Vector, value: Vector) -> None:
        """Remove a trace's exact contribution from the field.

        The binding stored at write time is authoritative: caller-supplied
        vectors are accepted for :class:`MemoryIndex` protocol compatibility
        but deliberately ignored, since subtracting anything other than the
        stored pair would leave a residue in the superposition.
        """
        del key, value
        if trace_id not in self._amplitude:
            return
        self._field = (
            self._field
            - self._amplitude[trace_id]
            * bind(self._keys[trace_id], self._codebook[trace_id])
        ).astype(DTYPE)
        self._codebook.pop(trace_id, None)
        self._keys.pop(trace_id, None)
        self._amplitude.pop(trace_id, None)

    def energy(self) -> float:
        """L2 norm of the superposition.

        A diagnostic for tests and future maintenance cadences: an exact
        write/subtract lifecycle drains the field back to (near) zero energy,
        while inexact bookkeeping leaves residue here even when the trace
        bookkeeping looks empty.
        """
        return float(np.linalg.norm(self._field))

    def __len__(self) -> int:
        """Number of live traces in the superposition."""
        return len(self._codebook)