# pyright: reportPrivateUsage=false
"""The generic engine-package runner against a really installed package.

The package's adapter below is a standard-library program that also plays the
engine: it serves the pinned chat dialect from its own loopback port. The
runner is driven through its real load, generation, health and teardown code
over real pipes, processes and HTTP.
"""

import json
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import httpx
import psutil
import pytest

from skulk.engines.tests.package_fixtures import install_test_package
from skulk.provisioning.engine_install import InstalledEngine
from skulk.shared.backends import (
    ENGINE_LOAD_REFUSED_FAILURE_MARKER,
    ENGINE_UNAVAILABLE_FAILURE_MARKER,
    engine_load_refusal_code,
)
from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.chunks import ErrorChunk, TokenChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import (
    LoadModel,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
)
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerReady
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import MpSender, mp_channel
from skulk.worker.plan import _retried_on_failure
from skulk.worker.runner import bootstrap
from skulk.worker.runner.engine_package import request as request_module
from skulk.worker.runner.engine_package import runner as runner_module
from skulk.worker.runner.engine_package.runner import (
    Runner,
    chat_request_body,
    raise_for_engine_status,
    raise_for_stream_error,
    response_id,
    unsupported_request_reason,
)
from skulk.worker.runner.llama_cpp.runner import serving_n_ctx
from skulk.worker.tests.unittests.conftest import get_mlx_ring_instance

NODE = NodeId("node-a")
RUNNER = RunnerId("runner-a")
MODEL = ModelId("org/model")
INSTANCE = InstanceId("instance-a")

ENGINE_ADAPTER = rb"""
import json, os, struct, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

digest = os.environ["SKULK_ENGINE_BUILD_DIGEST"]
config = {"mode": "normal", "record": None}
cancelled = threading.Event()
record_lock = threading.Lock()


def record(entry):
    if config["record"] is None:
        return
    with record_lock, open(config["record"], "a") as handle:
        handle.write(json.dumps(entry) + "\n")


def event(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode()


class Engine(BaseHTTPRequestHandler):
    def log_message(self, *_arguments):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        record({"request": body})
        if config["mode"] == "overflow":
            payload = json.dumps({"error": {"message": "the prompt is too long",
                                            "type": "context_length_exceeded"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def piece(delta, finish=None):
            return event({"id": "chat-1", "choices": [{"delta": delta, "finish_reason": finish}]})

        try:
            if config["mode"] == "stream-overflow":
                # Some engines open the stream and then refuse inside it.
                self.wfile.write(event({"id": "chat-1", "error": {
                    "message": "the prompt is too long", "type": "invalid_request_error",
                    "code": "context_length_exceeded", "param": "messages"}}))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if config["mode"] == "stream-failure":
                self.wfile.write(piece({"content": "Hel"}))
                self.wfile.write(event({"error": {"message": "the decoder failed",
                                                  "type": "server_error"}}))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if config["mode"] == "slow":
                deadline = time.monotonic() + 20
                while not cancelled.is_set() and time.monotonic() < deadline:
                    self.wfile.write(piece({"content": "tick "}))
                    self.wfile.flush()
                    time.sleep(0.02)
                return
            self.wfile.write(piece({"reasoning_content": "thinking"}))
            self.wfile.write(piece({"content": "Hello"}))
            self.wfile.write(piece({"content": " world"}))
            self.wfile.write(piece({}, "stop"))
            self.wfile.write(event({"id": "chat-1", "choices": [], "usage": {
                "prompt_tokens": 11, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 4}}}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def read_frame():
    head = sys.stdin.buffer.read(4)
    if not head:
        return None
    (length,) = struct.unpack(">I", head)
    return json.loads(sys.stdin.buffer.read(length))


def send(message):
    body = json.dumps(message, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
    sys.stdout.buffer.flush()


server = None
while True:
    request = read_frame()
    if request is None:
        break
    kind = request["type"]
    base = {"type": kind, "sequence": request["sequence"], "build_digest": digest}
    if kind == "hello":
        send({**base, "version": max(request["versions"])})
    elif kind == "load":
        with open(os.path.join(request["model"]["directory"], "mode.json")) as handle:
            config.update(json.load(handle))
        record({"load": request})
        if config["mode"] == "refuse":
            send({**base, "type": "error", "code": "load_refused",
                  "message": "cannot map /private/weights/layer.bin"})
            continue
        applied = dict(request["settings"])
        if config["mode"] == "shrink":
            applied["context_tokens"] //= 2
        server = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        send({**base, "endpoint_port": server.server_address[1], "applied": applied})
    elif kind == "health":
        send({**base, "ready": config["mode"] != "unhealthy"})
    elif kind == "cancel":
        record({"cancel": request["request_id"]})
        cancelled.set()
        send({**base, "request_id": request["request_id"], "acknowledged": True})
    elif kind == "unload":
        record({"unload": request["instance_id"]})
        send({**base, "released": True})
    elif kind == "stop":
        record({"stop": True})
        send(base)
        break
"""


def _bound(task: ModelTask = ModelTask.TextGeneration) -> BoundInstance:
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=MODEL,
            storage_size=Memory.from_mb(1000),
            n_layers=4,
            hidden_size=64,
            supports_tensor=False,
            tasks=[task],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
        resolved_backend="tensorfold-rocm",
    )
    instance = get_mlx_ring_instance(INSTANCE, MODEL, {NODE: RUNNER}, {RUNNER: shard})
    return BoundInstance(instance=instance, bound_runner_id=RUNNER, bound_node_id=NODE)


class _Harness:
    """A real runner over a real installed package, with its events captured."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
        *,
        task: ModelTask = ModelTask.TextGeneration,
        installing: bool = False,
        drafting: bool = False,
    ) -> None:
        self.model_directory = tmp_path / "model"
        self.model_directory.mkdir()
        self.record_path = tmp_path / "record.jsonl"
        (self.model_directory / "mode.json").write_text(
            json.dumps({"mode": mode, "record": str(self.record_path)})
        )
        self.installed: InstalledEngine = install_test_package(
            tmp_path, tmp_path / "packages", adapter=ENGINE_ADAPTER, drafting=drafting
        )
        self.state_root = tmp_path / "state"
        self.events: list[Event] = []
        events_lock = threading.Lock()

        def capture(event: Event) -> None:
            with events_lock:
                self.events.append(event)

        event_sender, self._event_receiver = mp_channel[Event]()
        self._task_sender, task_receiver = mp_channel[Task]()
        self.cancel_sender, cancel_receiver = mp_channel[TaskId]()
        monkeypatch.setattr(event_sender, "send", capture)

        def model_path(*_arguments: object, **_keywords: object) -> Path:
            return self.model_directory

        monkeypatch.setattr(request_module, "build_model_path", model_path)
        self._senders: tuple[MpSender[Event], MpSender[Task], MpSender[TaskId]] = (
            event_sender,
            self._task_sender,
            self.cancel_sender,
        )
        # ``installing`` is a runner the worker started before the package was
        # on disk: it is handed no package and finds it when asked to load.
        self.on_disk: InstalledEngine | None = None if installing else self.installed
        self.runner = Runner(
            _bound(task),
            event_sender,
            task_receiver,
            cancel_receiver,
            None if installing else self.installed,
            context_token_limit=8192,
            find_installed=lambda: self.on_disk,
            state_root=self.state_root,
        )

    def load(self) -> None:
        self.runner.handle_task(LoadModel(instance_id=INSTANCE))

    def record(self) -> list[dict[str, object]]:
        if not self.record_path.exists():
            return []
        return [
            cast("dict[str, object]", json.loads(line))
            for line in self.record_path.read_text().splitlines()
        ]

    def chunks(self) -> list[TokenChunk | ErrorChunk]:
        return [
            event.chunk
            for event in list(self.events)
            if isinstance(event, ChunkGenerated)
            and isinstance(event.chunk, (TokenChunk, ErrorChunk))
        ]

    def adapter_pid(self) -> int:
        session = self.runner._session
        assert session is not None and session.process_id is not None
        return session.process_id

    def close(self) -> None:
        self.runner._teardown_server()
        for sender in self._senders:
            sender.close()


def _generation(**overrides: object) -> TextGeneration:
    fields: dict[str, object] = {
        "model": MODEL,
        "input": [InputMessage(role="user", content="Say hello")],
    }
    fields.update(overrides)
    return TextGeneration(
        command_id=CommandId(),
        task_params=TextGenerationTaskParams.model_validate(fields),
        instance_id=INSTANCE,
    )


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def test_a_build_that_says_it_drafts_is_loaded_with_drafting_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the installed build's signed manifest turns drafting on."""
    harness = _Harness(tmp_path, monkeypatch, "normal", drafting=True)
    harness.load()
    try:
        assert isinstance(harness.runner.current_status, RunnerReady)
        load = cast("dict[str, dict[str, object]]", harness.record()[0]["load"])
        assert load["settings"] == {
            "context_tokens": serving_n_ctx(8192, logits_all=False),
            "parallel_streams": 1,
            "cache_precision": "default",
            "drafting": True,
        }
    finally:
        harness.close()


def test_a_package_loads_serves_text_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load states the admitted settings, text streams back, teardown leaves nothing."""
    harness = _Harness(tmp_path, monkeypatch, "normal")
    harness.load()
    pid = harness.adapter_pid()
    try:
        assert isinstance(harness.runner.current_status, RunnerReady)
        load = cast("dict[str, dict[str, object]]", harness.record()[0]["load"])
        assert load["model"] == {
            "model_id": str(MODEL),
            "source_revision": None,
            "directory": str(harness.model_directory.resolve()),
        }
        assert load["settings"] == {
            "context_tokens": serving_n_ctx(8192, logits_all=False),
            "parallel_streams": 1,
            "cache_precision": "default",
            "drafting": False,
        }
        assert load["instance_id"] == str(INSTANCE)

        harness.runner._generate(
            _generation(max_output_tokens=32, enable_thinking=False)
        )

        chunks = [chunk for chunk in harness.chunks() if isinstance(chunk, TokenChunk)]
        assert [chunk.text for chunk in chunks if chunk.is_thinking] == ["thinking"]
        assert "".join(c.text for c in chunks if not c.is_thinking) == "Hello world"
        terminal = chunks[-1]
        assert terminal.finish_reason == "stop" and terminal.stats is not None
        assert terminal.stats.prompt_tokens == 11
        assert terminal.stats.generation_tokens == 3
        request = cast("dict[str, object]", harness.record()[1]["request"])
        assert request["model"] == str(MODEL)
        assert request["stream"] is True
        assert request["stream_options"] == {"include_usage": True}
        assert request["max_tokens"] == 32
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        assert "temperature" not in request
    finally:
        harness.close()

    kinds = [next(iter(entry)) for entry in harness.record()]
    assert kinds[-2:] == ["unload", "stop"]
    assert _gone(pid)
    assert not (
        harness.state_root / harness.installed.digest / "instances"
    ).exists() or not any(
        (harness.state_root / harness.installed.digest / "instances").iterdir()
    )


def test_a_runner_started_before_its_package_was_installed_loads_once_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker starts the runner, installs the package, then asks for the load."""
    harness = _Harness(tmp_path, monkeypatch, "normal", installing=True)
    # Nothing is read from disk, and no directory is made, before the load.
    assert not harness.state_root.exists()
    harness.on_disk = harness.installed
    harness.load()
    pid = harness.adapter_pid()
    instances = harness.state_root / harness.installed.digest / "instances"
    try:
        assert isinstance(harness.runner.current_status, RunnerReady)
        assert [entry.name for entry in instances.iterdir()] == [str(INSTANCE)]
        harness.runner._generate(_generation(max_output_tokens=32))
        chunks = [chunk for chunk in harness.chunks() if isinstance(chunk, TokenChunk)]
        assert "".join(c.text for c in chunks if not c.is_thinking) == "Hello world"
    finally:
        harness.close()

    assert _gone(pid)
    assert not any(instances.iterdir())


def test_a_load_whose_package_never_arrived_is_a_missing_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker gives such an instance up once; it does not relaunch the runner."""
    harness = _Harness(tmp_path, monkeypatch, "normal", installing=True)
    try:
        with pytest.raises(RuntimeError) as failed:
            harness.load()
    finally:
        harness.close()

    assert str(failed.value).startswith(f"{ENGINE_UNAVAILABLE_FAILURE_MARKER}: ")
    assert "tensorfold-rocm" in str(failed.value)
    assert not _retried_on_failure(RunnerFailed(error_message=str(failed.value)))
    assert harness.record() == []
    assert not harness.state_root.exists()


def test_a_runner_needs_a_package_or_a_way_to_find_one() -> None:
    event_sender, _ = mp_channel[Event]()
    _, task_receiver = mp_channel[Task]()
    _, cancel_receiver = mp_channel[TaskId]()

    with pytest.raises(ValueError, match="a package or a lookup"):
        _ = Runner(_bound(), event_sender, task_receiver, cancel_receiver, None)


def test_a_refused_load_is_terminal_and_carries_only_the_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's own reason stays in the failure text; the worker reads the code."""
    harness = _Harness(tmp_path, monkeypatch, "refuse")
    try:
        with pytest.raises(RuntimeError) as failed:
            harness.load()
    finally:
        harness.close()

    message = str(failed.value)
    assert message.startswith(f"{ENGINE_LOAD_REFUSED_FAILURE_MARKER}[load_refused]: ")
    assert "/private/weights/layer.bin" in message
    assert engine_load_refusal_code(message) == "load_refused"
    assert not _retried_on_failure(RunnerFailed(error_message=message))
    assert harness.runner._session is None and harness.runner.base_url is None


def test_settings_the_engine_changed_fail_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine that sized itself differently from admission is not served."""
    harness = _Harness(tmp_path, monkeypatch, "shrink")
    try:
        with pytest.raises(RuntimeError) as failed:
            harness.load()
    finally:
        harness.close()

    assert engine_load_refusal_code(str(failed.value)) == "settings_changed"
    assert [next(iter(entry)) for entry in harness.record()][-2:] == ["unload", "stop"]


def test_cancel_reaches_the_engine_through_the_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled stream is cancelled at the engine by its own request id."""
    harness = _Harness(tmp_path, monkeypatch, "slow")
    harness.load()
    task = _generation()
    try:
        worker = threading.Thread(target=harness.runner._generate, args=(task,))
        worker.start()
        deadline = time.monotonic() + 10
        while not harness.chunks() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert harness.chunks(), "the engine never streamed"
        harness.cancel_sender.send(task.task_id)
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        harness.close()

    assert {"cancel": "chat-1"} in harness.record()
    assert all(
        isinstance(chunk, TokenChunk) and chunk.finish_reason is None
        for chunk in harness.chunks()
    )


def _named(name: str) -> TextGeneration:
    """A request whose prompt is ``name``, so the engine's record says which ran."""
    return _generation(input=[InputMessage(role="user", content=name)])


def _until(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def _prompts_at_the_engine(harness: _Harness) -> list[object]:
    prompts: list[object] = []
    for entry in harness.record():
        request = entry.get("request")
        if isinstance(request, dict):
            messages = cast("list[dict[str, object]]", request["messages"])
            prompts.append(messages[-1]["content"])
    return prompts


def _acknowledged(harness: _Harness, task: TextGeneration) -> bool:
    return any(
        isinstance(event, TaskAcknowledged) and event.task_id == task.task_id
        for event in list(harness.events)
    )


def _has_status(harness: _Harness, task: TextGeneration, status: TaskStatus) -> bool:
    return any(
        isinstance(event, TaskStatusUpdated)
        and event.task_id == task.task_id
        and event.task_status is status
        for event in list(harness.events)
    )


def test_a_request_cancelled_while_it_waits_never_reaches_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two requests wait behind a running one; the cancelled one is not generated.

    The worker waits for each request's acknowledgement before it plans again.
    A loop that stops reading while its one slot is busy never acknowledges
    the second waiting request, so the worker cannot deliver a cancel for the
    first, and that request is answered for a client that has gone.
    """
    harness = _Harness(tmp_path, monkeypatch, "slow")
    harness.load()
    running, abandoned, patient = _named("running"), _named("abandoned"), _named("patient")
    loop = threading.Thread(target=harness.runner.run_dispatch_loop)
    loop.start()
    try:
        harness._task_sender.send(running)
        assert _until(lambda: bool(harness.chunks())), "the engine never streamed"
        harness._task_sender.send(abandoned)
        harness._task_sender.send(patient)
        assert _until(
            lambda: _acknowledged(harness, abandoned) and _acknowledged(harness, patient)
        ), "a waiting request was not acknowledged while another was being answered"

        harness.cancel_sender.send(abandoned.task_id)
        assert _until(
            lambda: _has_status(harness, abandoned, TaskStatus.Cancelled)
        ), "the cancel did not reach the waiting request"
        assert _prompts_at_the_engine(harness) == ["running"]

        harness.cancel_sender.send(running.task_id)
        assert _until(lambda: "patient" in _prompts_at_the_engine(harness))
    finally:
        harness._task_sender.close()
        loop.join(timeout=30)
        harness.cancel_sender.close()

    assert not loop.is_alive()
    assert _prompts_at_the_engine(harness) == ["running", "patient"]
    assert not _has_status(harness, abandoned, TaskStatus.Running)


def test_a_request_past_the_waiting_limit_fails_at_once_with_its_own_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue is bounded, and the refusal names the engine package."""
    monkeypatch.setattr(runner_module, "MAX_WAITING_REQUESTS", 1)
    harness = _Harness(tmp_path, monkeypatch, "slow")
    harness.load()
    running, waiting, excess = _named("running"), _named("waiting"), _named("excess")
    loop = threading.Thread(target=harness.runner.run_dispatch_loop)
    loop.start()
    try:
        harness._task_sender.send(running)
        assert _until(lambda: bool(harness.chunks())), "the engine never streamed"
        harness._task_sender.send(waiting)
        harness._task_sender.send(excess)
        assert _until(lambda: _has_status(harness, excess, TaskStatus.Failed))
        refusals = [chunk for chunk in harness.chunks() if isinstance(chunk, ErrorChunk)]
        assert [chunk.error_message for chunk in refusals] == [
            request_module.WAITING_LIMIT_MESSAGE
        ]
        assert not _has_status(harness, waiting, TaskStatus.Failed)
        harness.cancel_sender.send(waiting.task_id)
        harness.cancel_sender.send(running.task_id)
        assert _until(lambda: _has_status(harness, waiting, TaskStatus.Cancelled))
    finally:
        harness._task_sender.close()
        loop.join(timeout=30)
        harness.cancel_sender.close()

    assert not loop.is_alive()
    assert _prompts_at_the_engine(harness) == ["running"]


@pytest.mark.parametrize(
    ("error", "overflow"),
    [
        ({"message": "too long", "type": "context_length_exceeded"}, True),
        (
            {
                "message": "too long",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "messages",
            },
            True,
        ),
        ({"message": "bad field", "type": "invalid_request_error", "code": None}, False),
    ],
)
def test_a_context_overflow_is_read_from_the_code_or_the_type(
    error: dict[str, object], overflow: bool
) -> None:
    """OpenAI names the overflow in ``code``; the first dialect draft used ``type``."""
    response = httpx.Response(400, json={"error": error})

    with pytest.raises(RuntimeError) as raised:
        raise_for_engine_status(response)

    assert str(raised.value).startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX) is overflow
    assert str(error["message"]) in str(raised.value)


def test_a_context_overflow_keeps_the_api_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's refusal reaches the caller as a context error, not a crash."""
    harness = _Harness(tmp_path, monkeypatch, "overflow")
    harness.load()
    try:
        harness.runner._generate(_generation())
    finally:
        harness.close()

    errors = [chunk for chunk in harness.chunks() if isinstance(chunk, ErrorChunk)]
    assert len(errors) == 1
    assert errors[0].error_message.startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX)
    assert "the prompt is too long" in errors[0].error_message


def test_an_overflow_refused_inside_the_stream_is_not_an_empty_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine that answers 200 and then refuses must not read as a finished reply."""
    harness = _Harness(tmp_path, monkeypatch, "stream-overflow")
    harness.load()
    try:
        harness.runner._generate(_generation())
    finally:
        harness.close()

    chunks = harness.chunks()
    errors = [chunk for chunk in chunks if isinstance(chunk, ErrorChunk)]
    assert len(errors) == 1
    assert errors[0].error_message.startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX)
    assert "the prompt is too long" in errors[0].error_message
    assert not [chunk for chunk in chunks if isinstance(chunk, TokenChunk)]


def test_a_failure_inside_the_stream_ends_the_reply_with_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text already sent stays sent; the reply ends as an error, not as a stop."""
    harness = _Harness(tmp_path, monkeypatch, "stream-failure")
    harness.load()
    try:
        harness.runner._generate(_generation())
    finally:
        harness.close()

    chunks = harness.chunks()
    errors = [chunk for chunk in chunks if isinstance(chunk, ErrorChunk)]
    tokens = [chunk for chunk in chunks if isinstance(chunk, TokenChunk)]
    assert len(errors) == 1
    assert "the decoder failed" in errors[0].error_message
    assert not errors[0].error_message.startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX)
    assert [chunk.text for chunk in tokens] == ["Hel"]
    assert all(chunk.finish_reason is None for chunk in tokens)


@pytest.mark.parametrize(
    "line",
    [
        'data: {"id": "c", "choices": [{"delta": {"content": "an \\"error\\" here"}}]}',
        'data: {"id": "c", "choices": [], "usage": {"prompt_tokens": 1}}',
        'data: {"id": "c", "error": null, "choices": [{"delta": {"content": "x"}}]}',
        "data: [DONE]",
        ": keep-alive",
        'data: {"error"',
    ],
)
def test_ordinary_stream_lines_are_not_read_as_errors(line: str) -> None:
    raise_for_stream_error(line)


def test_a_request_outside_the_dialect_never_reaches_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tools are refused with a reason before any request is sent."""
    harness = _Harness(tmp_path, monkeypatch, "normal")
    harness.load()
    try:
        harness.runner._generate(
            _generation(tools=[{"type": "function", "function": {"name": "f"}}])
        )
    finally:
        harness.close()

    errors = [chunk for chunk in harness.chunks() if isinstance(chunk, ErrorChunk)]
    assert len(errors) == 1 and "tool" in errors[0].error_message
    assert not any("request" in entry for entry in harness.record())


def test_an_engine_that_stops_serving_ends_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A not-ready health answer is a crash for the supervisor, not a ready runner."""
    harness = _Harness(tmp_path, monkeypatch, "unhealthy")
    harness.load()
    try:
        with pytest.raises(RuntimeError, match="no longer serving"):
            harness.runner._ensure_server_alive()
    finally:
        harness.close()


def test_a_dead_adapter_ends_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter that died between requests is found by the idle health check."""
    harness = _Harness(tmp_path, monkeypatch, "normal")
    harness.load()
    try:
        harness.runner._ensure_server_alive()
        os.kill(harness.adapter_pid(), signal.SIGKILL)
        with pytest.raises(RuntimeError, match="health check"):
            harness.runner._ensure_server_alive()
    finally:
        harness.close()


def test_a_placement_that_is_not_text_is_refused_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedding placement stamped onto a package starts no adapter."""
    with pytest.raises(RuntimeError) as failed:
        _Harness(tmp_path, monkeypatch, "normal", task=ModelTask.TextEmbedding)

    assert engine_load_refusal_code(str(failed.value)) == "unsupported_task"
    assert "embedding" in str(failed.value)
    assert not (tmp_path / "state").exists()


def test_the_chat_body_sends_only_what_the_caller_set() -> None:
    """Unset sampling fields are left to the engine's own defaults."""
    params = _generation(temperature=0.2, stop=["END"], seed=7).task_params

    body = chat_request_body(params, MODEL)

    assert body["temperature"] == 0.2 and body["stop"] == ["END"] and body["seed"] == 7
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert not {"max_tokens", "top_p", "top_k", "chat_template_kwargs"} & set(body)


def test_images_and_logprobs_are_named_as_unsupported() -> None:
    """Each request kind outside the dialect gets its own reason."""
    assert unsupported_request_reason(_generation().task_params) is None
    images = unsupported_request_reason(_generation(images=["abc"]).task_params)
    logprobs = unsupported_request_reason(_generation(logprobs=True).task_params)
    assert images is not None and "images" in images
    assert logprobs is not None and "logprobs" in logprobs


@pytest.mark.parametrize(
    ("line", "found"),
    [
        ('data: {"id": "chat-9", "choices": []}', "chat-9"),
        ("data: [DONE]", None),
        ("event: ping", None),
        ("data: not json", None),
        ('data: {"id": 7}', None),
        ('data: {"id": "' + "x" * 200 + '"}', None),
    ],
)
def test_the_response_id_is_read_only_from_a_well_formed_event(
    line: str, found: str | None
) -> None:
    assert response_id(line) == found


def test_bootstrap_finds_only_packages_this_node_holds(tmp_path: Path) -> None:
    """Built-in and unheld backends are left to the existing checks."""
    packages = tmp_path / "packages"
    installed = install_test_package(tmp_path, packages)

    def find(backend: str | None, build: str | None) -> InstalledEngine | None:
        return bootstrap.installed_engine_package(
            backend, build, packages_root=packages
        )

    assert find(None, None) is None
    assert find("mlx-metal", None) is None
    assert find("tensorfold-cuda", None) is None
    found = find("tensorfold-rocm", installed.digest)
    assert found is not None and found.digest == installed.digest


def test_a_placement_stamped_with_a_removed_generation_is_not_moved(
    tmp_path: Path,
) -> None:
    """A stamped build that is gone fails clearly; the active build is not substituted."""
    packages = tmp_path / "packages"
    install_test_package(tmp_path, packages)

    with pytest.raises(RuntimeError) as failed:
        bootstrap.installed_engine_package(
            "tensorfold-rocm", "f" * 64, packages_root=packages
        )

    assert str(failed.value).startswith(f"{ENGINE_UNAVAILABLE_FAILURE_MARKER}: ")
    assert not _retried_on_failure(RunnerFailed(error_message=str(failed.value)))
