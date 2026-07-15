# pyright: reportPrivateUsage=false, reportAny=false
"""Unit tests for the pure helpers of the vLLM served-backend runner.

The live subprocess + streaming path is validated on GPU hardware; these cover
the pure, engine-specific logic: the ``vllm serve`` argument builder, the OpenAI
SSE parser, and the GPU-memory-utilization knob.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skulk.shared.types.common import CommandId
from skulk.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TaskStatus
from skulk.shared.types.worker.runners import RunnerReady, RunnerRunning
from skulk.worker.runner.vllm.runner import (
    _DEFAULT_GPU_MEMORY_UTILIZATION,
    _DEFAULT_MAX_CONCURRENT_REQUESTS,
    _GPU_MEMORY_UTILIZATION_ENV,
    _MAX_CONCURRENT_REQUESTS_ENV,
    _gpu_memory_utilization,
    _max_concurrent_requests,
    build_vllm_serve_args,
    parse_openai_sse_line,
    vllm_generation_kwargs,
    vllm_reasoning_overrides,
)
from skulk.worker.runner.vllm.runner import (
    Runner as VllmRunner,
)


def _params(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = dict(
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        stop=None,
        seed=None,
        enable_thinking=None,
        reasoning_effort=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_vllm_generation_kwargs_uses_vllm_parameter_names() -> None:
    kwargs = vllm_generation_kwargs(
        _params(
            max_output_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.1,
            stop=["</s>"],
            seed=7,
        )
    )
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 40
    assert kwargs["min_p"] == 0.05
    # vLLM's name, not llama.cpp's repeat_penalty (which vLLM would ignore).
    assert kwargs["repetition_penalty"] == 1.1
    assert "repeat_penalty" not in kwargs
    assert kwargs["stop"] == ["</s>"]
    assert kwargs["seed"] == 7


def test_vllm_generation_kwargs_omits_unset() -> None:
    assert vllm_generation_kwargs(_params()) == {}


def test_vllm_reasoning_overrides_maps_thinking_controls() -> None:
    assert vllm_reasoning_overrides(_params(enable_thinking=False)) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert vllm_reasoning_overrides(_params(reasoning_effort="high")) == {
        "reasoning_effort": "high"
    }
    # "none" effort is not a valid server value; disabling goes via enable_thinking.
    assert vllm_reasoning_overrides(_params(reasoning_effort="none")) == {}
    assert vllm_reasoning_overrides(_params()) == {}


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


def test_build_vllm_serve_args_shape() -> None:
    args = _serve_args()
    assert args[0] == "/opt/vllm/bin/vllm"
    assert args[1] == "serve"
    assert args[2] == "/models/org--repo"
    # served-model-name decouples the addressed id from the on-disk path.
    assert args[args.index("--served-model-name") + 1] == "org/repo"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "51234"
    assert args[args.index("--max-model-len") + 1] == "8192"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.90"
    # single-node in this slice.
    assert args[args.index("--tensor-parallel-size") + 1] == "1"


def test_build_vllm_serve_args_trust_remote_code() -> None:
    assert "--trust-remote-code" not in _serve_args(trust_remote_code=False)
    assert "--trust-remote-code" in _serve_args(trust_remote_code=True)


def test_parse_sse_content_delta() -> None:
    line = 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.content == "hello"
    assert delta.reasoning == ""
    assert delta.finish is None
    assert delta.done is False


def test_parse_sse_reasoning_delta() -> None:
    line = 'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.reasoning == "think"
    assert delta.content == ""


def test_parse_sse_finish_reason_mapped() -> None:
    line = 'data: {"choices":[{"delta":{"content":""},"finish_reason":"length"}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.finish == "length"


def test_parse_sse_preserves_content_filter() -> None:
    # vLLM can emit content_filter; it must not be collapsed to a normal stop.
    line = 'data: {"choices":[{"delta":{"content":""},"finish_reason":"content_filter"}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.finish == "content_filter"


def test_parse_sse_done_sentinel() -> None:
    delta = parse_openai_sse_line("data: [DONE]")
    assert delta is not None
    assert delta.done is True


@pytest.mark.parametrize(
    "line",
    [
        "event: ping",  # non-data line
        "data: {not json}",  # malformed json
        'data: {"choices":[]}',  # choice-less payload
        "",  # blank
    ],
)
def test_parse_sse_skips_non_deltas(line: str) -> None:
    assert parse_openai_sse_line(line) is None


def test_gpu_memory_utilization_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_GPU_MEMORY_UTILIZATION_ENV, raising=False)
    assert _gpu_memory_utilization() == _DEFAULT_GPU_MEMORY_UTILIZATION


def test_gpu_memory_utilization_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_GPU_MEMORY_UTILIZATION_ENV, "0.75")
    assert _gpu_memory_utilization() == 0.75


@pytest.mark.parametrize("bad", ["nonsense", "0", "1.5", "-0.2"])
def test_gpu_memory_utilization_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # Unparseable or out-of-(0,1] values fall back to the default rather than
    # passing vLLM a fraction that would fail the server at spawn.
    monkeypatch.setenv(_GPU_MEMORY_UTILIZATION_ENV, bad)
    assert _gpu_memory_utilization() == _DEFAULT_GPU_MEMORY_UTILIZATION


def test_max_concurrent_requests_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_MAX_CONCURRENT_REQUESTS_ENV, raising=False)
    assert _max_concurrent_requests() == _DEFAULT_MAX_CONCURRENT_REQUESTS


def test_max_concurrent_requests_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_MAX_CONCURRENT_REQUESTS_ENV, "8")
    assert _max_concurrent_requests() == 8


@pytest.mark.parametrize("bad", ["nonsense", "0", "-3", "1.5"])
def test_max_concurrent_requests_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # Unparseable or below-1 values fall back to the default rather than
    # disabling concurrency or crashing the pool at construction.
    monkeypatch.setenv(_MAX_CONCURRENT_REQUESTS_ENV, bad)
    assert _max_concurrent_requests() == _DEFAULT_MAX_CONCURRENT_REQUESTS


# --- concurrent dispatch --------------------------------------------------
#
# These exercise the runner's dispatch orchestration (in-flight counting, status
# transitions, terminal-status classification, stale-cancel recovery) with a fake
# ``_generate`` -- no server, no GPU. The live streaming path is validated on
# hardware. The runner is built with ``__new__`` so only the dispatch state is
# set up; ``update_status`` / ``send_task_status`` are stubbed to record calls
# (and keep ``current_status`` in sync) rather than construct wire events.


def _fake_task() -> Any:
    """A duck-typed stand-in for a TextGeneration (only ids are read here)."""
    return SimpleNamespace(task_id=TaskId(), command_id=CommandId())


def _bare_runner(max_concurrency: int = 4) -> Any:
    runner: Any = VllmRunner.__new__(VllmRunner)
    runner._status_lock = threading.Lock()
    runner._cancel_lock = threading.Lock()
    runner._inflight = 0
    runner._max_concurrency = max_concurrency
    runner.cancelled_tasks = set()
    runner.seen = set()
    runner.current_status = RunnerReady()
    runner.status_updates = []
    runner.task_statuses = []

    def _record_status(status: Any) -> None:
        runner.status_updates.append(status)
        runner.current_status = status

    def _record_task_status(task: Any, status: Any) -> None:
        runner.task_statuses.append((task.task_id, status))

    runner.update_status = _record_status
    runner.send_task_status = _record_task_status
    return runner


def test_dispatch_runs_generations_concurrently() -> None:
    runner = _bare_runner(max_concurrency=4)
    started = threading.Semaphore(0)
    release = threading.Event()

    def fake_generate(_task: Any) -> None:
        started.release()  # signal this generation has started
        release.wait(5)  # hold it open until the test releases all at once

    runner._generate = fake_generate
    tasks = [_fake_task() for _ in range(3)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for task in tasks:
            # Must return immediately: dispatch is non-blocking so the loop can
            # keep receiving while prior generations stream.
            runner._dispatch_generation(task, pool)
        # All three run at once (proving they are not serialized).
        for _ in range(3):
            assert started.acquire(timeout=5)
        assert runner._inflight_count() == 3
        assert isinstance(runner.current_status, RunnerRunning)
        release.set()
        pool.shutdown(wait=True)

    # Done-callbacks (terminal status + inflight decrement) run on the worker
    # threads as their futures resolve; all have run by the time shutdown returns.
    assert runner._inflight_count() == 0
    assert isinstance(runner.current_status, RunnerReady)
    running = [s for _, s in runner.task_statuses if s is TaskStatus.Running]
    complete = [s for _, s in runner.task_statuses if s is TaskStatus.Complete]
    assert len(running) == 3
    assert len(complete) == 3


def test_finish_generation_marks_cancelled_and_returns_ready() -> None:
    runner = _bare_runner()
    runner._inflight = 1
    runner.current_status = RunnerRunning()
    task = _fake_task()
    runner.cancelled_tasks.add(task.task_id)
    future: Future[None] = Future()
    future.set_result(None)

    runner._finish_generation(task, future)

    assert (task.task_id, TaskStatus.Cancelled) in runner.task_statuses
    assert runner._inflight == 0
    assert isinstance(runner.current_status, RunnerReady)


def test_finish_generation_cancel_all_marks_cancelled() -> None:
    runner = _bare_runner()
    runner._inflight = 1
    runner.current_status = RunnerRunning()
    runner.cancelled_tasks.add(CANCEL_ALL_TASKS)
    task = _fake_task()
    future: Future[None] = Future()
    future.set_result(None)

    runner._finish_generation(task, future)

    assert (task.task_id, TaskStatus.Cancelled) in runner.task_statuses


def test_dispatch_clears_stale_cancel_all_when_idle() -> None:
    # A lingering cluster-wide cancel must not kill a fresh request admitted when
    # nothing else is in flight.
    runner = _bare_runner()
    runner.cancelled_tasks.add(CANCEL_ALL_TASKS)
    release = threading.Event()

    def fake_generate(_task: Any) -> None:
        release.wait(5)

    runner._generate = fake_generate
    task = _fake_task()

    with ThreadPoolExecutor(max_workers=1) as pool:
        runner._dispatch_generation(task, pool)
        assert CANCEL_ALL_TASKS not in runner.cancelled_tasks
        release.set()
        pool.shutdown(wait=True)

    assert (task.task_id, TaskStatus.Complete) in runner.task_statuses


def test_note_generation_status_transitions() -> None:
    runner = _bare_runner()
    assert isinstance(runner.current_status, RunnerReady)

    runner._note_generation_started()
    assert runner._inflight == 1
    assert isinstance(runner.current_status, RunnerRunning)

    # A second concurrent generation does not re-flip status.
    runner._note_generation_started()
    assert runner._inflight == 2
    assert isinstance(runner.current_status, RunnerRunning)

    # Only the LAST one to drain returns the runner to Ready.
    runner._note_generation_finished()
    assert isinstance(runner.current_status, RunnerRunning)
    runner._note_generation_finished()
    assert runner._inflight == 0
    assert isinstance(runner.current_status, RunnerReady)


def test_inflight_never_goes_negative() -> None:
    # Defensive: an extra finish (double done-callback) must not drive the count
    # below zero or spuriously toggle status.
    runner = _bare_runner()
    runner._note_generation_finished()
    assert runner._inflight == 0
    assert isinstance(runner.current_status, RunnerReady)


def test_main_broadcasts_ready_after_load_model() -> None:
    # Regression: the concurrent main() must re-broadcast the runner status after
    # a lifecycle task, because _load_model sets current_status = RunnerReady() by
    # DIRECT ASSIGNMENT (no event). Without the broadcast the runner loads but
    # never announces Ready, so the worker never dispatches a generation to it.
    # (Caught live: a probe saw statuses stop at [Idle, Loading] though the server
    # loaded fine.)
    #
    # ORDER is also load-bearing and asserted here: RunnerSupervisor._forward_events
    # asserts the runner is in an active state (Loading/Running/...) when a terminal
    # task status arrives, so the LoadModel Complete must precede the RunnerReady
    # broadcast (Loading -> Complete -> Ready), not follow it. This drives the real
    # main() loop over genuine mp channels.
    from skulk.shared.types.events import (
        Event,
        RunnerStatusUpdated,
        TaskStatusUpdated,
    )
    from skulk.shared.types.tasks import LoadModel, Shutdown, Task, TaskStatus
    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId, RunnerIdle
    from skulk.utils.channels import mp_channel

    runner: Any = VllmRunner.__new__(VllmRunner)
    runner._status_lock = threading.Lock()
    runner._cancel_lock = threading.Lock()
    runner._inflight = 0
    runner._max_concurrency = 2
    runner.cancelled_tasks = set()
    runner.seen = set()
    runner.runner_id = RunnerId("vllm-test")
    runner.current_status = RunnerIdle()

    evt_s, evt_r = mp_channel[Event]()
    task_s, task_r = mp_channel[Task]()
    _cancel_s, cancel_r = mp_channel[TaskId]()
    runner.event_sender = evt_s
    runner.task_receiver = task_r
    runner.cancel_receiver = cancel_r

    # Fakes for the lifecycle path: no real vllm serve. _load_model mimics the
    # real one's direct assignment (the exact behavior that made the bug latent).
    def fake_load(_task: Any) -> None:
        runner.current_status = RunnerReady()

    runner._load_model = fake_load
    runner._teardown_server = lambda: None
    runner._ensure_server_alive = lambda: None

    thread = threading.Thread(target=runner.main, daemon=True)
    thread.start()
    try:
        iid = InstanceId("probe-inst")
        task_s.send(LoadModel(instance_id=iid))
        load_complete_at: int | None = None
        ready_at: int | None = None
        seq = 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and ready_at is None:
            try:
                event = evt_r.receive_timeout(0.5)
            except Exception:
                continue
            seq += 1
            if (
                isinstance(event, TaskStatusUpdated)
                and event.task_status is TaskStatus.Complete
                and load_complete_at is None
            ):
                load_complete_at = seq
            if isinstance(event, RunnerStatusUpdated) and isinstance(
                event.runner_status, RunnerReady
            ):
                ready_at = seq
        assert ready_at is not None, "runner never broadcast RunnerReady after LoadModel"
        assert load_complete_at is not None, "LoadModel never reported Complete"
        # Ready must come AFTER the terminal Complete (the supervisor assertion).
        assert load_complete_at < ready_at, (
            "RunnerReady was broadcast before the LoadModel Complete; the supervisor "
            "asserts an active runner state on terminal task status"
        )
    finally:
        task_s.send(
            Shutdown(instance_id=InstanceId("probe-inst"), runner_id=runner.runner_id)
        )
        thread.join(timeout=5)
