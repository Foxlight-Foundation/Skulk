"""Tests that request validation happens before capability-driven card loading."""

from collections.abc import Awaitable, Callable
from typing import Never, cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from exo.api.main import API
from exo.api.types import (
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
)
from exo.api.types.claude_api import ClaudeMessage, ClaudeMessagesRequest
from exo.api.types.ollama_api import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaMessage,
)
from exo.api.types.openai_responses import ResponsesRequest
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import TextGeneration
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _get_running_model_card_fn(api: API) -> Callable[[ModelId], Awaitable[ModelCard]]:
    """Return the protected running-card helper with an explicit callable type."""
    return cast(
        Callable[[ModelId], Awaitable[ModelCard]],
        object.__getattribute__(api, "_get_running_model_card"),
    )


def _build_running_state(running_card: ModelCard) -> State:
    """Build a minimal typed State containing one running pipeline instance."""
    runner_id = RunnerId("runner-1")
    node_id = NodeId("node-1")
    return State(
        instances={
            InstanceId("instance-1"): MlxRingInstance(
                instance_id=InstanceId("instance-1"),
                shard_assignments=ShardAssignments(
                    model_id=running_card.model_id,
                    runner_to_shard={
                        runner_id: PipelineShardMetadata(
                            model_card=running_card,
                            device_rank=0,
                            world_size=1,
                            start_layer=0,
                            end_layer=1,
                            n_layers=1,
                        )
                    },
                    node_to_runner={node_id: runner_id},
                ),
                hosts_by_node={node_id: []},
                ephemeral_port=52415,
            )
        }
    )


@pytest.mark.anyio
async def test_chat_completions_validates_model_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid chat models should fail locally before the adapter tries to build task params."""

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
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

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
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
async def test_claude_messages_validates_model_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid Claude models should fail before adapter conversion runs."""

    async def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("claude_request_to_text_generation should not run before validation")

    async def _raise_not_found(self: API, model_id: ModelId) -> ModelId:
        raise HTTPException(status_code=404, detail=f"No instance found for model {model_id}")

    monkeypatch.setattr("exo.api.main.claude_request_to_text_generation", _fail_if_called)
    monkeypatch.setattr(API, "_resolve_and_validate_text_model", _raise_not_found)

    api = object.__new__(API)
    payload = ClaudeMessagesRequest(
        model=ModelId("missing/model"),
        max_tokens=100,
        messages=[ClaudeMessage(role="user", content="hello")],
    )

    with pytest.raises(HTTPException) as exc_info:
        await api.claude_messages(payload)

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
    api.state = _build_running_state(running_card)

    resolved = await _get_running_model_card_fn(api)(running_card.model_id)

    assert resolved == running_card


@pytest.mark.anyio
async def test_ollama_chat_validates_model_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid Ollama chat models should fail locally before adapter conversion."""

    def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("ollama_request_to_text_generation should not run before validation")

    async def _raise_not_found(self: API, model_id: ModelId) -> ModelId:
        raise HTTPException(status_code=404, detail=f"No instance found for model {model_id}")

    monkeypatch.setattr("exo.api.main.ollama_request_to_text_generation", _fail_if_called)
    monkeypatch.setattr(API, "_resolve_and_validate_text_model", _raise_not_found)

    api = object.__new__(API)
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
    }
    body = OllamaChatRequest(
        model=ModelId("missing/model"),
        messages=[OllamaMessage(role="user", content="hello")],
    ).model_dump_json().encode()

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive=receive)

    with pytest.raises(HTTPException) as exc_info:
        await api.ollama_chat(request)

    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_ollama_generate_validates_model_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid Ollama generate models should fail locally before adapter conversion."""

    def _fail_if_called(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("ollama_generate_request_to_text_generation should not run before validation")

    async def _raise_not_found(self: API, model_id: ModelId) -> ModelId:
        raise HTTPException(status_code=404, detail=f"No instance found for model {model_id}")

    monkeypatch.setattr("exo.api.main.ollama_generate_request_to_text_generation", _fail_if_called)
    monkeypatch.setattr(API, "_resolve_and_validate_text_model", _raise_not_found)

    api = object.__new__(API)
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
    }
    body = OllamaGenerateRequest(
        model=ModelId("missing/model"),
        prompt="hello",
    ).model_dump_json().encode()

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive=receive)

    with pytest.raises(HTTPException) as exc_info:
        await api.ollama_generate(request)

    assert exc_info.value.status_code == 404


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
    api.state = _build_running_state(running_card)

    captured: dict[str, object] = {}

    async def _capture_adapter(
        request: BenchChatCompletionRequest,
        *,
        model_card: ModelCard | None = None,
    ) -> TextGenerationTaskParams:
        captured["request_model"] = request.model
        captured["model_card"] = model_card
        return TextGenerationTaskParams(
            model=request.model,
            input=[InputMessage(role="user", content="hello")],
            stream=False,
            bench=False,
        )

    async def _send_task(task_params: TextGenerationTaskParams) -> TextGeneration:
        captured["task_params"] = task_params
        return TextGeneration(
            command_id=CommandId("cmd-1"),
            task_params=task_params,
        )

    async def _collect_stats(command_id: CommandId) -> BenchChatCompletionResponse:
        captured["command_id"] = command_id
        return BenchChatCompletionResponse(
            id=str(command_id),
            created=0,
            model=str(running_card.model_id),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="done"),
                    finish_reason="stop",
                )
            ],
        )

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
    assert captured["command_id"] == CommandId("cmd-1")
    assert response.model == str(running_card.model_id)
