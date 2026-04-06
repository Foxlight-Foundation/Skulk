"""Tests that request validation happens before capability-driven card loading."""

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types import (
    BenchChatCompletionRequest,
    ChatCompletionMessage,
    ChatCompletionRequest,
)
from exo.api.types.openai_responses import ResponsesRequest
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


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


@pytest.mark.anyio
async def test_running_text_requests_use_in_memory_model_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running text requests should not depend on ModelCard.load cache/fetch behavior."""

    running_card = ModelCard(
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "vision", "thinking"],
        family="gemma",
    )

    async def _fail_if_called(model_id: ModelId) -> ModelCard:
        raise AssertionError(f"ModelCard.load should not be called for running model {model_id}")

    monkeypatch.setattr("exo.api.main.ModelCard.load", _fail_if_called)

    api = object.__new__(API)
    api.state = SimpleNamespace(
        instances={
            "running": SimpleNamespace(
                shard_assignments=SimpleNamespace(
                    model_id=running_card.model_id,
                    model_card=running_card,
                )
            )
        }
    )

    resolved = await api._get_running_model_card(running_card.model_id)

    assert resolved == running_card


@pytest.mark.anyio
async def test_bench_chat_completions_uses_running_model_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """Benchmark chat should use the same model-aware defaults as normal chat."""

    running_card = ModelCard(
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text", "vision", "thinking"],
        family="gemma",
    )

    api = object.__new__(API)
    api.state = SimpleNamespace(
        instances={
            "running": SimpleNamespace(
                shard_assignments=SimpleNamespace(
                    model_id=running_card.model_id,
                    model_card=running_card,
                )
            )
        }
    )

    captured: dict[str, object] = {}

    async def _capture_adapter(
        request: BenchChatCompletionRequest,
        *,
        model_card: ModelCard | None = None,
    ):
        captured["request_model"] = request.model
        captured["model_card"] = model_card
        return SimpleNamespace(
            model=request.model,
            stream=False,
            bench=False,
            model_copy=lambda update: SimpleNamespace(
                model=update.get("model", request.model),
                stream=update.get("stream", False),
                bench=update.get("bench", False),
                model_copy=lambda nested_update: SimpleNamespace(
                    model=nested_update.get("model", update.get("model", request.model)),
                    stream=nested_update.get("stream", update.get("stream", False)),
                    bench=nested_update.get("bench", update.get("bench", False)),
                ),
            ),
        )

    async def _send_task(task_params):
        captured["task_params"] = task_params
        return SimpleNamespace(command_id="cmd-1")

    async def _collect_stats(command_id: str):
        captured["command_id"] = command_id
        return SimpleNamespace(model=str(running_card.model_id))

    monkeypatch.setattr("exo.api.main.chat_request_to_text_generation", _capture_adapter)
    monkeypatch.setattr(api, "_send_text_generation_with_images", _send_task)
    monkeypatch.setattr(api, "_collect_text_generation_with_stats", _collect_stats)

    payload = BenchChatCompletionRequest(
        model=running_card.model_id,
        messages=[ChatCompletionMessage(role="user", content="hello")],
    )

    response = await api.bench_chat_completions(payload)

    assert captured["request_model"] == running_card.model_id
    assert captured["model_card"] == running_card
    assert captured["command_id"] == "cmd-1"
    assert response.model == str(running_card.model_id)
