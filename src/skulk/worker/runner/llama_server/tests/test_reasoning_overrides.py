"""Tests for thinking-control forwarding to llama-server (#428/#420)."""

from __future__ import annotations

from skulk.shared.types.common import ModelId
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
)
from skulk.worker.runner.llama_server.runner import reasoning_request_overrides


def _params(**kwargs: object) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("m"),
        input=[InputMessage(role="user", content="hi")],
        **kwargs,  # type: ignore[arg-type]
    )


def test_enable_thinking_false_forwards_chat_template_kwargs() -> None:
    # The throughput-cell case: enable_thinking=False must reach llama-server, or
    # the model reasons through the whole budget and returns empty content (#428).
    overrides = reasoning_request_overrides(_params(enable_thinking=False))
    assert overrides["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in overrides


def test_enable_thinking_true_forwards_toggle() -> None:
    overrides = reasoning_request_overrides(_params(enable_thinking=True))
    assert overrides["chat_template_kwargs"] == {"enable_thinking": True}


def test_reasoning_effort_forwarded_but_none_dropped() -> None:
    assert reasoning_request_overrides(_params(reasoning_effort="high")) == {
        "reasoning_effort": "high"
    }
    # "none" is not a valid server effort; disabling is expressed via
    # enable_thinking, so it must not be forwarded as reasoning_effort.
    assert reasoning_request_overrides(_params(reasoning_effort="none")) == {}


def test_no_controls_yields_no_overrides() -> None:
    # Neither set -> let the model's own default behavior stand.
    assert reasoning_request_overrides(_params()) == {}


def test_both_controls_combine() -> None:
    overrides = reasoning_request_overrides(
        _params(enable_thinking=True, reasoning_effort="medium")
    )
    assert overrides == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "medium",
    }
