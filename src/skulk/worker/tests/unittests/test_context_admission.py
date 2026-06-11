"""Unit coverage for request-time context-length admission (#145)."""

import pytest

from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX
from skulk.worker.engines.mlx.generator.context_admission import (
    ContextLengthExceededError,
    admit_max_tokens,
)


def test_no_limit_preserves_legacy_behavior():
    assert (
        admit_max_tokens(
            prompt_tokens=1_000_000,
            requested_max_output_tokens=None,
            default_max_output_tokens=4096,
            context_token_limit=None,
        )
        == 4096
    )
    assert (
        admit_max_tokens(
            prompt_tokens=1_000_000,
            requested_max_output_tokens=7,
            default_max_output_tokens=4096,
            context_token_limit=None,
        )
        == 7
    )


def test_omitted_max_tokens_is_clamped_to_remaining_context():
    assert (
        admit_max_tokens(
            prompt_tokens=80,
            requested_max_output_tokens=None,
            default_max_output_tokens=4096,
            context_token_limit=100,
        )
        == 20
    )


def test_omitted_max_tokens_keeps_default_when_it_fits():
    assert (
        admit_max_tokens(
            prompt_tokens=80,
            requested_max_output_tokens=None,
            default_max_output_tokens=10,
            context_token_limit=100,
        )
        == 10
    )


def test_explicit_max_tokens_that_fits_is_honored_exactly():
    assert (
        admit_max_tokens(
            prompt_tokens=80,
            requested_max_output_tokens=20,
            default_max_output_tokens=4096,
            context_token_limit=100,
        )
        == 20
    )


def test_explicit_max_tokens_overflow_is_rejected_not_clamped():
    with pytest.raises(ContextLengthExceededError):
        admit_max_tokens(
            prompt_tokens=80,
            requested_max_output_tokens=21,
            default_max_output_tokens=4096,
            context_token_limit=100,
        )


def test_prompt_filling_the_window_is_rejected_even_without_max_tokens():
    with pytest.raises(ContextLengthExceededError):
        admit_max_tokens(
            prompt_tokens=100,
            requested_max_output_tokens=None,
            default_max_output_tokens=4096,
            context_token_limit=100,
        )


def test_zero_limit_rejects_everything():
    with pytest.raises(ContextLengthExceededError):
        admit_max_tokens(
            prompt_tokens=1,
            requested_max_output_tokens=None,
            default_max_output_tokens=4096,
            context_token_limit=0,
        )


def test_rejection_message_carries_the_wire_sentinel():
    with pytest.raises(ContextLengthExceededError) as excinfo:
        admit_max_tokens(
            prompt_tokens=200,
            requested_max_output_tokens=50,
            default_max_output_tokens=4096,
            context_token_limit=100,
        )
    assert str(excinfo.value).startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX)
    assert "200" in str(excinfo.value)
    assert "50" in str(excinfo.value)
    assert "100" in str(excinfo.value)
