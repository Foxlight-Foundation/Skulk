# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false, reportUnknownLambdaType=false
"""vLLM tool calling: launch flags, parser resolution, and the round trip.

Mirrors the llama_server tool-call test harness: the runner is driven with a
canned server response, so these tests cover the proxy conversion without a
GPU or a live server.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    RuntimeCapabilityCardConfig,
    ToolingCardConfig,
)
from skulk.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.events import ChunkGenerated
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import TextGeneration
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.runner.vllm.runner import (
    Runner,
    build_vllm_serve_args,
    resolve_vllm_tool_call_parser,
)


def _serve_args(**overrides: object) -> list[str]:
    kwargs: dict[str, object] = dict(
        binary="/opt/vllm/bin/vllm",
        model_dir=Path("/models/org--repo"),
        served_model_name="org/repo",
        host="127.0.0.1",
        port=51234,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
    )
    kwargs.update(overrides)
    return build_vllm_serve_args(**kwargs)  # type: ignore[arg-type]


def _card(
    family: str = "qwen",
    tooling: ToolingCardConfig | None = None,
    runtime: RuntimeCapabilityCardConfig | None = None,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("org/repo"),
        storage_size=Memory.from_mb(100),
        n_layers=2,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family=family,
        capabilities=["text"],
        tooling=tooling,
        runtime=runtime,
    )


def test_serve_args_pair_tool_parser_flags() -> None:
    args = _serve_args(tool_call_parser="hermes")
    # vLLM refuses --tool-call-parser without --enable-auto-tool-choice.
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "hermes"


def test_serve_args_omit_tool_flags_without_parser() -> None:
    args = _serve_args()
    assert "--enable-auto-tool-choice" not in args
    assert "--tool-call-parser" not in args


def test_parser_resolution_explicit_runtime_wins() -> None:
    card = _card(
        family="somefamily",
        runtime=RuntimeCapabilityCardConfig(vllm_tool_call_parser="mistral"),
    )
    assert resolve_vllm_tool_call_parser(card) == "mistral"


def test_parser_resolution_family_default_needs_tooling() -> None:
    declared = _card(tooling=ToolingCardConfig(supports_tool_calling=True))
    assert resolve_vllm_tool_call_parser(declared) == "hermes"
    undeclared = _card(tooling=ToolingCardConfig(supports_tool_calling=False))
    assert resolve_vllm_tool_call_parser(undeclared) is None
    unknown_family = _card(
        family="mysteryfamily",
        tooling=ToolingCardConfig(supports_tool_calling=True),
    )
    assert resolve_vllm_tool_call_parser(unknown_family) is None


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict[str, Any], seen: dict[str, Any]) -> None:
        self._payload = payload
        self._seen = seen

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def post(self, _url: str, **kwargs: Any) -> _FakeResponse:
        self._seen.update(kwargs.get("json") or {})
        return _FakeResponse(self._payload)


def _tool_task() -> TextGeneration:
    return TextGeneration(
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("m"),
            input=[],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
        ),
        instance_id=InstanceId("i"),
    )


def _runner(events: list[object], monkeypatch: Any, task: TextGeneration) -> Any:
    runner: Any = object.__new__(Runner)
    runner.base_url = "http://127.0.0.1:0"
    runner.bound_instance = SimpleNamespace(bound_node_id=NodeId("node-7"))
    runner.shard_metadata = SimpleNamespace(resolved_backend="vllm-cuda")
    runner._admission_inflight = {task.task_id: 2}
    runner._admission_lock = threading.Lock()
    runner._status_lock = threading.Lock()
    runner._inflight = 0
    runner._max_concurrency = 4
    runner._tool_call_parser = "hermes"
    runner.event_sender = SimpleNamespace(send=events.append)
    monkeypatch.setattr(runner, "_is_cancelled", lambda _tid: False)
    monkeypatch.setattr(runner, "_server_peak_memory", lambda: Memory(in_bytes=0))
    return runner


def test_tool_round_trip_emits_stamped_tool_call_chunk(monkeypatch: Any) -> None:
    task = _tool_task()
    events: list[object] = []
    runner = _runner(events, monkeypatch, task)
    seen: dict[str, Any] = {}
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
    monkeypatch.setattr(httpx, "Client", lambda **_kw: _FakeClient(payload, seen))

    runner._generate_with_tools(task, {}, ModelId("m"), task.command_id)

    # The request carried tools, tool_choice, and forced non-streaming.
    assert seen["tools"] == task.task_params.tools
    assert seen["tool_choice"] == "auto"
    assert seen["stream"] is False

    tool_chunks = [
        e.chunk
        for e in events
        if isinstance(e, ChunkGenerated) and isinstance(e.chunk, ToolCallChunk)
    ]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_calls[0].name == "get_weather"
    stats = tool_chunks[0].stats
    assert stats is not None
    # Runner attribution (#596), exactly like the streaming path.
    assert stats.serving_node == "node-7"
    assert stats.serving_backend == "vllm-cuda"
    assert stats.in_flight_at_admission == 2


def test_prose_answer_falls_back_to_tokens(monkeypatch: Any) -> None:
    task = _tool_task()
    events: list[object] = []
    runner = _runner(events, monkeypatch, task)
    payload = {
        "choices": [
            {
                "message": {"content": "no tool needed", "reasoning_content": "hmm"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    monkeypatch.setattr(httpx, "Client", lambda **_kw: _FakeClient(payload, {}))

    runner._generate_with_tools(task, {}, ModelId("m"), task.command_id)

    tokens = [
        e.chunk
        for e in events
        if isinstance(e, ChunkGenerated) and isinstance(e.chunk, TokenChunk)
    ]
    assert [t.text for t in tokens if t.is_thinking] == ["hmm"]
    assert [t.text for t in tokens if not t.is_thinking and t.text] == [
        "no tool needed"
    ]
    # Truncation is preserved, not collapsed to "stop".
    assert tokens[-1].finish_reason == "length"


def test_mid_flight_cancel_emits_nothing(monkeypatch: Any) -> None:
    task = _tool_task()
    events: list[object] = []
    runner = _runner(events, monkeypatch, task)
    payload: dict[str, Any] = {"choices": [{"message": {"content": "x"}}], "usage": {}}
    monkeypatch.setattr(httpx, "Client", lambda **_kw: _FakeClient(payload, {}))
    # Not cancelled at entry, cancelled after the POST returns.
    calls = iter([False, True])
    monkeypatch.setattr(runner, "_is_cancelled", lambda _tid: next(calls))

    runner._generate_with_tools(task, {}, ModelId("m"), task.command_id)

    assert events == []


def test_tools_without_parser_surface_error_chunk(monkeypatch: Any) -> None:
    """A tool request against a server launched without the parser pair must
    fail loudly (#385 no-silent-empty), not silently drop the tools."""
    task = _tool_task()
    events: list[object] = []
    runner = _runner(events, monkeypatch, task)
    runner._tool_call_parser = None
    runner.shard_metadata = SimpleNamespace(
        resolved_backend="vllm-cuda",
        model_card=SimpleNamespace(model_id=ModelId("m"), runtime=None),
    )
    monkeypatch.setattr(runner, "_was_cancelled", lambda _tid: False)

    runner._generate(task)

    errors = [
        e.chunk
        for e in events
        if isinstance(e, ChunkGenerated) and isinstance(e.chunk, ErrorChunk)
    ]
    assert len(errors) == 1
    assert "tool" in errors[0].error_message.lower()
