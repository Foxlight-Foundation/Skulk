# pyright: reportPrivateUsage=false, reportMissingParameterType=false
"""llama.cpp runner cancellation safety (#123).

The tool-calling path runs one blocking, uninterruptible
``create_chat_completion``, so cancellation is honored at the boundaries around
it: an already-cancelled task skips the call, and a cancel that lands while the
call runs suppresses the result. In both cases no ``ChunkGenerated`` is emitted
and the task ends ``Cancelled`` rather than ``Complete``. A request that is not
cancelled still emits exactly one terminal chunk and completes.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from anyio import WouldBlock

from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.events import ChunkGenerated, Event, TaskStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import LoadModel, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerId, RunnerReady
from skulk.worker.runner.llama_cpp.runner import Runner


class _CaptureSender:
    """Stand-in for the runner's MpSender that records every emitted event."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, item: Event) -> None:
        self.events.append(item)


class _OneShotReceiver:
    """Stand-in MpReceiver that yields a fixed task list once, then stops."""

    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __enter__(self):
        return iter(self._items)

    def __exit__(self, *_: object) -> bool:
        return False


class _CancelReceiver:
    """MpReceiver stand-in: receive_nowait pops queued ids, else raises WouldBlock."""

    def __init__(self) -> None:
        self._queue: list[object] = []

    def push(self, task_id: object) -> None:
        self._queue.append(task_id)

    def receive_nowait(self) -> object:
        if self._queue:
            return self._queue.pop(0)
        raise WouldBlock


def _make_runner(cancel: _CancelReceiver) -> tuple[Runner, _CaptureSender]:
    sender = _CaptureSender()
    card = ModelCard(
        model_id=ModelId("some/gguf-model"),
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model.gguf",
    )
    bound = SimpleNamespace(
        instance=SimpleNamespace(),
        bound_runner_id=RunnerId("r1"),
        bound_shard=SimpleNamespace(
            world_size=1,
            model_card=card,
            device_rank=0,
        ),
        bound_node_id=NodeId("n1"),
    )
    runner = Runner(
        bound_instance=cast("object", bound),  # pyright: ignore[reportArgumentType]
        event_sender=cast("object", sender),  # pyright: ignore[reportArgumentType]
        task_receiver=cast("object", _OneShotReceiver([])),  # pyright: ignore[reportArgumentType]
        cancel_receiver=cast("object", cancel),  # pyright: ignore[reportArgumentType]
    )
    return runner, sender


def test_load_model_explicitly_disables_full_swa_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dependency defaults must not expand SWA beyond admission's estimate."""
    from skulk.download import download_utils
    from skulk.shared.models.memory_estimate import LLAMA_CPP_FULL_SWA_CACHE

    captured: dict[str, object] = {}

    class _FakeLlama:
        def __init__(self, *, swa_full: bool, **kwargs: object) -> None:
            captured["swa_full"] = swa_full
            captured.update(kwargs)

    fake_module = ModuleType("llama_cpp")
    monkeypatch.setattr(fake_module, "Llama", _FakeLlama, raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    def _model_path(*_args: object) -> Path:
        return tmp_path

    monkeypatch.setattr(download_utils, "build_model_path", _model_path)
    (tmp_path / "model.gguf").touch()

    runner, _sender = _make_runner(_CancelReceiver())
    runner._load_model(LoadModel(instance_id=InstanceId("i1")))

    assert LLAMA_CPP_FULL_SWA_CACHE is False
    assert captured["swa_full"] is LLAMA_CPP_FULL_SWA_CACHE


def test_load_model_rejects_binding_that_silently_ignores_swa_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older ``**kwargs``-only binding must fail before allocating memory."""

    class _LegacyLlama:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("legacy binding must not be constructed")

    fake_module = ModuleType("llama_cpp")
    monkeypatch.setattr(fake_module, "Llama", _LegacyLlama, raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    runner, _sender = _make_runner(_CancelReceiver())
    with pytest.raises(RuntimeError, match="does not explicitly support swa_full"):
        runner._load_model(LoadModel(instance_id=InstanceId("i1")))


def _tool_task() -> TextGeneration:
    return TextGeneration(
        instance_id=InstanceId("i1"),
        command_id=CommandId("c1"),
        task_params=TextGenerationTaskParams.model_validate(
            {
                "model": ModelId("some/gguf-model"),
                "input": [InputMessage(role="user", content="weather in SF?")],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            }
        ),
    )


def _tool_result() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _run_one(runner: Runner, sender: _CaptureSender, task: TextGeneration) -> TaskStatus:
    runner.current_status = RunnerReady()
    runner.task_receiver = cast("object", _OneShotReceiver([task]))  # pyright: ignore[reportAttributeAccessIssue]
    runner.main()
    return next(
        e.task_status
        for e in sender.events
        if isinstance(e, TaskStatusUpdated)
        and e.task_status in (TaskStatus.Complete, TaskStatus.Cancelled)
    )


def test_tool_request_cancelled_during_call_emits_nothing() -> None:
    cancel = _CancelReceiver()
    runner, sender = _make_runner(cancel)
    task = _tool_task()

    # The cancel arrives *while* the blocking call runs (modeled as a side effect
    # of create_chat_completion). The assembled result must be suppressed.
    def fake_ccc(**_kw: object) -> dict[str, Any]:
        cancel.push(task.task_id)
        return _tool_result()

    runner.model = SimpleNamespace(create_chat_completion=fake_ccc)
    terminal = _run_one(runner, sender, task)

    assert not any(isinstance(e, ChunkGenerated) for e in sender.events)
    assert terminal == TaskStatus.Cancelled


def test_tool_request_already_cancelled_skips_call() -> None:
    cancel = _CancelReceiver()
    runner, sender = _make_runner(cancel)
    task = _tool_task()
    cancel.push(task.task_id)  # cancelled before generation even starts

    calls = {"n": 0}

    def fake_ccc(**_kw: object) -> dict[str, Any]:
        calls["n"] += 1
        return _tool_result()

    runner.model = SimpleNamespace(create_chat_completion=fake_ccc)
    terminal = _run_one(runner, sender, task)

    assert calls["n"] == 0  # the uninterruptible call was skipped entirely
    assert not any(isinstance(e, ChunkGenerated) for e in sender.events)
    assert terminal == TaskStatus.Cancelled


def test_tool_request_not_cancelled_emits_one_tool_chunk() -> None:
    cancel = _CancelReceiver()
    runner, sender = _make_runner(cancel)
    task = _tool_task()

    def fake_ccc(**_kw: object) -> dict[str, Any]:
        return _tool_result()

    runner.model = SimpleNamespace(create_chat_completion=fake_ccc)
    terminal = _run_one(runner, sender, task)

    chunks = [e for e in sender.events if isinstance(e, ChunkGenerated)]
    assert len(chunks) == 1
    assert terminal == TaskStatus.Complete
