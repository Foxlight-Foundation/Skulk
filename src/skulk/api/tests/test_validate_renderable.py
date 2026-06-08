"""Tests for the text-generation renderability guard (#233).

An empty ``messages`` array was accepted with 200 and then crashed the
runner inside ``apply_chat_template([])`` with an IndexError. The guard at
the single dispatch chokepoint rejects un-renderable requests with 400 so
malformed input never reaches a runner.
"""

import pytest
from fastapi import HTTPException

from skulk.api.main import validate_renderable_text_generation
from skulk.shared.types.common import ModelId
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def _params(**overrides: object) -> TextGenerationTaskParams:
    base: dict[str, object] = {
        "model": ModelId("test-model"),
        "input": [InputMessage(role="user", content="hi")],
    }
    base.update(overrides)
    return TextGenerationTaskParams(**base)  # pyright: ignore[reportArgumentType]


def test_normal_request_passes() -> None:
    validate_renderable_text_generation(_params())


def test_chat_template_only_request_passes() -> None:
    # Some adapters carry chat_template_messages instead of input; a request
    # with either is renderable.
    validate_renderable_text_generation(
        _params(input=[], chat_template_messages=[{"role": "user", "content": "hi"}])
    )


def test_empty_messages_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_renderable_text_generation(_params(input=[]))
    assert exc.value.status_code == 400
    assert "content" in str(exc.value.detail).lower()


def test_empty_input_and_empty_template_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_renderable_text_generation(
            _params(input=[], chat_template_messages=[])
        )
    assert exc.value.status_code == 400


def test_blank_content_message_rejected() -> None:
    # The chat adapter substitutes [InputMessage(role="user", content="")]
    # for an empty `messages` array; the runner filters blank content back
    # to an empty list and crashes. The guard must catch this masked case,
    # not just a literally-empty input list (#233 — the bug the first fix
    # missed because `input` looked non-empty).
    with pytest.raises(HTTPException) as exc:
        validate_renderable_text_generation(
            _params(input=[InputMessage(role="user", content="")])
        )
    assert exc.value.status_code == 400


def test_whitespace_only_content_passes() -> None:
    # The guard mirrors the runner's filter exactly (`if not msg.content`):
    # whitespace-only content is truthy, so the runner renders it and the
    # guard must NOT reject it — being stricter than the runner would 400
    # inputs the runner can handle (review catch on PR #235).
    validate_renderable_text_generation(
        _params(input=[InputMessage(role="user", content="   \n\t ")])
    )


def test_instructions_only_request_passes() -> None:
    # A system-only request renders fine (instructions become a system turn).
    validate_renderable_text_generation(
        _params(input=[InputMessage(role="user", content="")], instructions="be brief")
    )


def test_zero_max_tokens_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_renderable_text_generation(_params(max_output_tokens=0))
    assert exc.value.status_code == 400
    assert "max_tokens" in str(exc.value.detail).lower()


def test_negative_max_tokens_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_renderable_text_generation(_params(max_output_tokens=-5))
    assert exc.value.status_code == 400


def test_none_max_tokens_allowed() -> None:
    # Unset max_tokens is valid (runner applies its own default).
    validate_renderable_text_generation(_params(max_output_tokens=None))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
