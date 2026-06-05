"""The Drafter protocol — the seam between the generation loop and any
speculative-decoding draft mechanism.

Design rules (read before adding methods):

- The loop OWNS verification, accept/reject, target-cache trimming, and SSM
  snapshot/restore. Those are mechanism-independent and must never leak into
  drafter implementations.
- The drafter OWNS whatever private state its mechanism needs (a KV cache for
  block-based heads, nothing for projection-only heads, references to the
  target's caches for assistant-model drafters).
- Family-specific facts (norm conventions, concat orders) are resolved at
  construction time by the builder; a constructed drafter is self-contained.

The pair-stream contract
------------------------

Hidden-conditioned drafters consume an ordered stream of *(hidden, next
token)* pairs: the trunk's pre-final-norm hidden state at position ``t``
paired with the committed token at ``t + 1``. The loop guarantees every
committed position's pair is fed **exactly once, in order**, through either
:meth:`Drafter.observe` (bulk or single skipped positions) or
:meth:`Drafter.draft` (which consumes its input pair before predicting).
Stateful drafters key RoPE/attention positions off this stream, so a missed
or duplicated pair silently corrupts drafting — the loop's observe calls on
the accept and reject paths exist precisely to keep the stream gapless.

Pairs only ever reference committed tokens, so drafter state never needs
rollback on reject — by construction, not by luck.

Future extensions (do not pre-build, but do not preclude):

- Block drafting (D > 1): a ``draft_block`` returning several candidates.
- Mid-trunk feature taps (EAGLE-style): extra fields on the observe payload.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import mlx.core as mx


@runtime_checkable
class Drafter(Protocol):
    """Protocol for speculative-decoding draft mechanisms.

    Implementations must be cheap to call per decode step and must never
    touch the target model's caches except through handles received in
    :meth:`begin_request`.
    """

    def begin_request(self, prompt_cache: Sequence[object]) -> None:
        """Reset drafter state for a new request and bind target caches.

        Args:
            prompt_cache: The per-request target-model cache list. Drafters
                that attend over the target's KV (e.g. Gemma 4 assistants)
                keep references; sidecar-head drafters ignore it.

        Side effects: discards all per-request drafter state (private KV
        caches, position counters).
        """
        ...

    def observe(self, hiddens: mx.array, next_tokens: mx.array) -> None:
        """Feed committed (hidden, next-token) pairs into drafter state.

        Used by the loop to (a) bulk-ingest the prompt's pairs after the
        first trunk forward and (b) feed the single skipped pair after each
        verify resolution (see the pair-stream contract in the module
        docstring). Stateless drafters implement this as a no-op.

        Args:
            hiddens: ``(T, hidden_size)`` pre-final-norm trunk hidden states
                for positions ``p .. p+T-1``.
            next_tokens: ``(T,)`` int32 committed token ids for positions
                ``p+1 .. p+T``.

        Side effects: advances the drafter's position/state by ``T``.
        """
        ...

    def draft(self, hidden: mx.array, next_token: int) -> mx.array:
        """Consume one pair and return draft logits for the following position.

        Equivalent to ``observe(hidden[None], [next_token])`` followed by
        predicting the token *after* ``next_token`` — implementations advance
        their state by exactly one position.

        Args:
            hidden: ``(hidden_size,)`` pre-final-norm trunk hidden state at
                the current position.
            next_token: Token id of the next committed token.

        Returns:
            ``(vocab_size,)`` float32 logits for the position after
            ``next_token``.
        """
        ...
