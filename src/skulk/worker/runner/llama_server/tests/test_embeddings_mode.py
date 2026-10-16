# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
"""llama-server embeddings mode: a dedicated server, truncation, ordered vectors.

Invariants:

1. The embedding server launches with ``--embeddings``, one slot and batch
   sizes covering the window, and none of the chat lane's template,
   reasoning, speculative, or projector flags.
2. Inputs longer than the window are truncated through the server's own
   tokenizer (llama.cpp would refuse them); shorter ones pass untouched, and
   order is preserved.
3. Vectors come back in input order whatever order the server lists them in.
4. A server failure is an ErrorChunk and a failed task, never partial vectors.
5. An embedding card without a declared ``context_length`` is refused at load.
"""

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from skulk.shared.types.chunks import EmbeddingChunk, ErrorChunk
from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.embedding import TextEmbeddingTaskParams
from skulk.shared.types.events import ChunkGenerated, Event
from skulk.shared.types.tasks import TextEmbedding
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.runner.llama_server import runner as runner_module
from skulk.worker.runner.llama_server.embeddings import (
    embedding_server_args,
    parse_embeddings_response,
    text_token_budget,
    truncate_inputs,
)
from skulk.worker.runner.llama_server.runner import Runner

# --- pure helpers -------------------------------------------------------------


def test_server_args_cover_the_window_with_one_slot() -> None:
    assert embedding_server_args(512) == [
        "--embeddings", "--parallel", "1", "-b", "512", "-ub", "512",
    ]


def test_text_budget_reserves_special_tokens_and_a_margin() -> None:
    assert text_token_budget(512, 2) == 508
    assert text_token_budget(4, 8) == 1


def test_truncation_only_tokenizes_texts_that_could_overflow() -> None:
    tokenized: list[str] = []

    def tokenize(text: str) -> list[int]:
        tokenized.append(text)
        return list(range(len(text.split())))

    def detokenize(tokens: list[int]) -> str:
        return " ".join(f"w{token}" for token in tokens)

    long_text = " ".join(f"w{index}" for index in range(10))
    one_long_word = "x" * 40
    result = truncate_inputs(
        ["hi", long_text, one_long_word],
        max_text_tokens=4,
        tokenize=tokenize,
        detokenize=detokenize,
    )
    # "hi" is shorter than the budget in bytes, so it never reaches the tokenizer.
    assert tokenized == [long_text, one_long_word]
    assert result == ["hi", "w0 w1 w2 w3", one_long_word]


def test_reply_is_reordered_by_index_and_counts_prompt_tokens() -> None:
    payload = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0], "object": "embedding"},
            {"index": 0, "embedding": [1.0, 0.0], "object": "embedding"},
        ],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }
    assert parse_embeddings_response(payload, expected=2) == ([[1.0, 0.0], [0.0, 1.0]], 7)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"data": "nope"},
        {"data": [{"index": 0, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]},
        {"data": [{"embedding": [1.0]}, {"index": 1, "embedding": [1.0]}]},
    ],
)
def test_reply_that_does_not_cover_every_input_is_refused(payload: object) -> None:
    with pytest.raises(ValueError):
        parse_embeddings_response(payload, expected=2)


# --- runner -------------------------------------------------------------------


class _Capture:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def _runner(budget: int = 4) -> tuple[Runner, _Capture]:
    runner = Runner.__new__(Runner)
    runner._init_concurrent_dispatch(1, "test")
    capture = _Capture()
    runner.event_sender = cast(Any, capture)
    runner.shard_metadata = cast(
        Any,
        SimpleNamespace(
            model_card=SimpleNamespace(model_id=ModelId("org/embed-GGUF"), context_length=512),
            resolved_backend="llama_server-vulkan",
        ),
    )
    runner.base_url = "http://embed.test"
    runner._embedding_text_budget = budget
    return runner, capture


def _task(texts: list[str]) -> TextEmbedding:
    return TextEmbedding(
        instance_id=InstanceId("i1"),
        command_id=CommandId("c1"),
        task_params=TextEmbeddingTaskParams(model=ModelId("org/embed-GGUF"), input_texts=texts),
    )


def _serve(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[dict[str, Any]]:
    """Route the runner's httpx clients to ``handler``; return the requests seen."""
    seen: list[dict[str, Any]] = []
    original = httpx.Client

    def record(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content or b"{}")})
        return cast(httpx.Response, handler(request))

    transport = httpx.MockTransport(record)

    def client_factory(**kwargs: Any) -> httpx.Client:
        return original(transport=transport, **kwargs)

    monkeypatch.setattr(runner_module.httpx, "Client", client_factory)
    return seen


def test_embed_truncates_then_returns_vectors_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(len(body["content"].split())))})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": " ".join(f"w{t}" for t in body["tokens"])})
        inputs = body["input"]
        data = [
            {"index": index, "embedding": [float(index)], "object": "embedding"}
            for index in reversed(range(len(inputs)))
        ]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 9}})

    seen = _serve(monkeypatch, handler)
    runner, capture = _runner(budget=4)
    long_text = " ".join(f"w{index}" for index in range(10))

    runner._embed(_task(["short", long_text]))

    embed_call = [call for call in seen if call["path"] == "/v1/embeddings"]
    assert embed_call[0]["body"]["input"] == ["short", "w0 w1 w2 w3"]
    chunk = cast(ChunkGenerated, capture.events[-1]).chunk
    assert isinstance(chunk, EmbeddingChunk)
    assert chunk.embeddings == [[0.0], [1.0]]
    assert chunk.token_count == 9
    assert not runner._failed_generation_ids


def test_server_failure_is_an_error_and_a_failed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _serve(monkeypatch, fail)
    runner, capture = _runner()
    task = _task(["hi"])

    runner._embed(task)

    chunk = cast(ChunkGenerated, capture.events[-1]).chunk
    assert isinstance(chunk, ErrorChunk)
    assert task.task_id in runner._failed_generation_ids


def test_embedding_launch_carries_no_chat_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(runner_module.LLAMA_SERVER_BIN_ENV, str(binary))
    runner, _ = _runner()
    launched: list[list[str]] = []
    monkeypatch.setattr(runner, "_pick_port", lambda: 4321)
    def capture_launch(_binary: str, cmd: list[str], _port: int) -> None:
        launched.append(cmd)

    monkeypatch.setattr(runner, "_launch_server", capture_launch)

    runner._spawn_server(Path("/models/embed.gguf"), 512, None, True, embeddings=True)

    cmd = launched[0]
    assert cmd[0] == str(binary)
    for flag, value in (("-m", "/models/embed.gguf"), ("-c", "512"), ("-b", "512"), ("-ub", "512"), ("--parallel", "1"), ("--port", "4321")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--embeddings" in cmd
    for chat_flag in ("--jinja", "--reasoning-format", "--spec-type", "--mmproj", "--kv-unified"):
        assert chat_flag not in cmd


def test_embedding_card_without_context_length_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _ = _runner()
    runner.shard_metadata = cast(
        Any,
        SimpleNamespace(
            model_card=SimpleNamespace(model_id=ModelId("org/embed-GGUF"), context_length=0),
            resolved_backend="llama_server-vulkan",
        ),
    )
    runner._serving_context_tokens = 32768
    torn_down: list[bool] = []
    monkeypatch.setattr(runner, "_teardown_server", lambda: torn_down.append(True))
    task = SimpleNamespace(task_id="t1")

    with pytest.raises(RuntimeError, match="declares no context_length"):
        runner._load_embedding_server(cast(Any, task), Path("/models/embed.gguf"))


def test_macos_launch_runs_under_the_parent_watchdog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """macOS has no PDEATHSIG, so the server runs under the shared watchdog."""
    runner, _ = _runner()
    runner.runner_id = cast(Any, "r1")
    launched: list[list[str]] = []

    class _Process:
        def __init__(self, cmd: list[str], **_: object) -> None:
            launched.append(cmd)

    monkeypatch.setattr(
        runner_module.os, "uname", lambda: cast(Any, SimpleNamespace(sysname="Darwin"))
    )
    monkeypatch.setattr(runner_module.subprocess, "Popen", _Process)
    monkeypatch.setattr(runner_module.tempfile, "gettempdir", lambda: str(tmp_path))

    runner._launch_server("/engines/llama-server", ["/engines/llama-server", "--embeddings"], 4321)

    cmd = launched[0]
    assert cmd[1].endswith("parent_watchdog.py")
    assert cmd[2].isdigit()
    assert cmd[3:] == ["/engines/llama-server", "--embeddings"]
    assert runner.base_url == "http://127.0.0.1:4321"


def test_embedding_dispatch_is_serial_like_its_one_slot_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wider pool would acknowledge requests that then queue inside the server."""

    def sixteen_slots(_card: object) -> int:
        return 16

    monkeypatch.setattr(runner_module, "_effective_server_parallel", sixteen_slots)
    embedding = cast(Any, SimpleNamespace(tasks=["TextEmbedding"]))
    chat = cast(Any, SimpleNamespace(tasks=["TextGeneration"]))
    assert runner_module._dispatch_width(embedding) == 1
    assert runner_module._dispatch_width(chat) == 16
