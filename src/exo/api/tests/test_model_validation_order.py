"""Tests that request validation happens before capability-driven card loading."""

from typing import Any

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types import ChatCompletionMessage, ChatCompletionRequest
from exo.api.types.openai_responses import ResponsesRequest
from exo.shared.types.common import ModelId


@pytest.mark.anyio
async def test_chat_completions_validates_model_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid chat models should fail locally before the adapter tries to build task params."""

    async def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("chat_request_to_text_generation should not run before validation")

    async def _raise_not_found(self: API, model_id: ModelId) -> ModelId:
        raise HTTPException(status_code=404, detail=f"No instance found for model {model_id}")

    monkeypatch.setattr("exo.api.main.chat_request_to_text_generation", _fail_if_called)
    monkeypatch.setattr(API, "_resolve_and_validate_text_model", _raise_not_found)

    api = object.__new__(API)
    payload = ChatCompletionRequest(
        model=ModelId("missing/model"),
        messages=[ChatCompletionMessage(role="user", content="hello")],
    )

    with pytest.raises(HTTPException) as exc_info:
        await api.chat_completions(payload)

    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_openai_responses_validates_model_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid responses models should fail locally before the adapter tries to build task params."""

    async def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "responses_request_to_text_generation should not run before validation"
        )

    async def _raise_not_found(self: API, model_id: ModelId) -> ModelId:
        raise HTTPException(status_code=404, detail=f"No instance found for model {model_id}")

    monkeypatch.setattr("exo.api.main.responses_request_to_text_generation", _fail_if_called)
    monkeypatch.setattr(API, "_resolve_and_validate_text_model", _raise_not_found)

    api = object.__new__(API)
    payload = ResponsesRequest(
        model=ModelId("missing/model"),
        input="hello",
    )

    with pytest.raises(HTTPException) as exc_info:
        await api.openai_responses(payload)

    assert exc_info.value.status_code == 404
