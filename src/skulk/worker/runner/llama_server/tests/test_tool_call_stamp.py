# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false, reportUnknownLambdaType=false
"""Tool-call generations stamp runner attribution for the envelope tap (#596).

The streaming path stamps ``serving_node``/``serving_backend``/
``in_flight_at_admission`` onto its terminal stats; the non-streamed tool-call
path (``_generate_with_tools``) must do the same, or served tool workloads (the
agentic audience) reach the API un-attributed and fall back to the ambiguous
API-side offered-count. This drives that path with a canned server response.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import httpx

from skulk.shared.types.chunks import TokenChunk, ToolCallChunk
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.events import ChunkGenerated
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import TextGeneration
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.runner.llama_server.runner import Runner


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def post(self, _url: str, json: dict[str, Any]) -> _FakeResponse:  # noqa: A002
        return _FakeResponse(self._payload)


def _tool_task() -> TextGeneration:
    return TextGeneration(
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("m"),
            input=[],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        ),
        instance_id=InstanceId("i"),
    )


def test_tool_call_generation_stamps_runner_attribution(monkeypatch: Any) -> None:
    task = _tool_task()
    events: list[object] = []

    runner: Any = object.__new__(Runner)
    runner.base_url = "http://127.0.0.1:0"
    runner._max_concurrency = 4  # a batching config
    runner.bound_instance = SimpleNamespace(bound_node_id=NodeId("node-9"))
    runner.shard_metadata = SimpleNamespace(resolved_backend="llama_server-vulkan")
    runner._admission_inflight = {task.task_id: 3}
    runner._status_lock = threading.Lock()
    runner._inflight = 0
    runner._uses_channel_parser = False
    runner.event_sender = SimpleNamespace(send=events.append)
    # This request is not cancelled; the model returns a real tool call.
    monkeypatch.setattr(runner, "_is_cancelled", lambda _tid: False)
    monkeypatch.setattr(runner, "_server_peak_memory", lambda: Memory(in_bytes=0))

    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    monkeypatch.setattr(httpx, "Client", lambda **_kw: _FakeClient(payload))

    runner._generate_with_tools(task, {}, ModelId("m"), task.command_id)

    tool_chunks = [
        e.chunk
        for e in events
        if isinstance(e, ChunkGenerated) and isinstance(e.chunk, ToolCallChunk)
    ]
    assert len(tool_chunks) == 1
    stats = tool_chunks[0].stats
    assert stats is not None
    # The runner attribution is stamped, exactly like the streaming path.
    assert stats.serving_node == "node-9"
    assert stats.serving_backend == "llama_server-vulkan"
    assert stats.in_flight_at_admission == 3  # from the admission capture
    assert stats.serving_batches is True  # max_concurrency 4 > 1


def test_prose_answer_via_tool_path_stamps_terminal_stats(monkeypatch: Any) -> None:
    # A tools request the model answers in prose closes with a terminal token
    # that must carry the same stamped stats.
    task = _tool_task()
    events: list[object] = []

    runner: Any = object.__new__(Runner)
    runner.base_url = "http://127.0.0.1:0"
    runner._max_concurrency = 1  # a serial config
    runner.bound_instance = SimpleNamespace(bound_node_id=NodeId("node-2"))
    runner.shard_metadata = SimpleNamespace(resolved_backend="llama_server-vulkan")
    runner._admission_inflight = {task.task_id: 1}
    runner._status_lock = threading.Lock()
    runner._inflight = 0
    runner._uses_channel_parser = False
    runner.event_sender = SimpleNamespace(send=events.append)
    monkeypatch.setattr(runner, "_is_cancelled", lambda _tid: False)
    monkeypatch.setattr(runner, "_server_peak_memory", lambda: Memory(in_bytes=0))

    payload = {
        "choices": [
            {"message": {"content": "sunny"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    monkeypatch.setattr(httpx, "Client", lambda **_kw: _FakeClient(payload))

    runner._generate_with_tools(task, {}, ModelId("m"), task.command_id)

    terminal = [
        e.chunk
        for e in events
        if isinstance(e, ChunkGenerated)
        and isinstance(e.chunk, TokenChunk)
        and e.chunk.stats is not None
    ]
    assert terminal, "the prose tool path must emit a terminal chunk carrying stats"
    stats = terminal[-1].stats
    assert stats is not None
    assert stats.serving_node == "node-2"
    assert stats.serving_batches is False  # serial config
    assert stats.in_flight_at_admission == 1
