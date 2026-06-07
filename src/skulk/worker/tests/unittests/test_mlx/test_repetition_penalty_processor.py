"""Regression: bare repetition_penalty must not crash the penalty processor.

Clients routinely send ``repetition_penalty`` without
``repetition_context_size``. Our call sites passed the request's None
straight into ``make_logits_processors``, overriding mlx-lm's default of
20 — and the processor's ``tokens[-context_size:]`` slice then raised
``bad operand type for unary -`` and killed the runner (found by the
2026-06-06 before/after benchmark matrix on gemma E2B; applies to every
model).
"""

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors


def _coerced_context_size(request_value: int | None) -> int:
    """Mirror of the call-site coercion in generate.py / batch_generate.py."""
    return request_value if request_value is not None else 20


def test_bare_repetition_penalty_processes_without_crashing() -> None:
    processors = make_logits_processors(
        repetition_penalty=1.05,
        repetition_context_size=_coerced_context_size(None),
    )
    assert processors, "penalty must produce a processor"

    tokens = mx.array([1, 2, 3, 2, 1])
    logits = mx.zeros((1, 16))
    for processor in processors:
        logits = processor(tokens, logits)
    mx.eval(logits)
    assert logits.shape == (1, 16)


def test_none_context_size_reproduces_the_crash() -> None:
    """Documents WHY the coercion exists: passing None through crashes."""
    processors = make_logits_processors(
        repetition_penalty=1.05,
        repetition_context_size=None,
    )
    tokens = mx.array([1, 2, 3])
    logits = mx.zeros((1, 16))
    crashed = False
    try:
        for processor in processors:
            logits = processor(tokens, logits)
        mx.eval(logits)
    except TypeError:
        crashed = True
    assert crashed, (
        "mlx-lm accepted a None context size — the call-site coercion may "
        "no longer be necessary; revisit before removing it"
    )


def test_explicit_context_size_is_respected() -> None:
    processors = make_logits_processors(
        repetition_penalty=1.05,
        repetition_context_size=_coerced_context_size(64),
    )
    tokens = mx.array([1, 2, 3, 2, 1])
    logits = mx.zeros((1, 16))
    for processor in processors:
        logits = processor(tokens, logits)
    mx.eval(logits)
    assert logits.shape == (1, 16)
