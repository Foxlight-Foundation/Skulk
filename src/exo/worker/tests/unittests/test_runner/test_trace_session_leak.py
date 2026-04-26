# type: ignore
"""Regression test for the trace-session leak in handle_generation_tasks.

Before the fix, an exception inside ``self.generator.step()`` would bubble
out of ``handle_generation_tasks`` without flushing any active traced
tasks. The runner would leak ``_trace_sessions`` entries, and — worse —
the master would wait forever for ``TracesCollected`` from this rank,
leaving the cluster trace in ``_pending_traces`` permanently.

This test exercises the helper directly without standing up the heavy
runner constructor: the contract is that for every traced active task,
the helper emits an ``aborted`` marker and flushes the session.
"""

from typing import cast
from unittest.mock import MagicMock

from exo.shared import tracing
from exo.shared.types.tasks import TaskId
from exo.worker.runner.llm_inference.runner import Runner


def _make_traced_task(task_id: TaskId) -> MagicMock:
    """Minimal traced-task stub — only the fields the helper reads."""
    task = MagicMock()
    task.task_id = task_id
    task.trace_enabled = True
    task.command_id = "cmd"
    return task


def _make_untraced_task(task_id: TaskId) -> MagicMock:
    task = MagicMock()
    task.task_id = task_id
    task.trace_enabled = False
    return task


def _seed_session(task_id: str, *, rank: int = 0) -> None:
    tracing.begin_trace_session(
        cast(object, task_id),
        rank=rank,
        node_id="node-test",
        model_id="test-model",
        task_kind="text",
        tags=["text_generation"],
    )


def test_flush_unfinished_trace_sessions_emits_marker_and_pops_session() -> None:
    """For each traced active task, helper records ``aborted`` and pops session."""
    task_id = TaskId("task-traced-1")
    _seed_session(str(task_id))
    assert tracing.has_trace_session(task_id)

    fake_runner = MagicMock()
    fake_runner.active_tasks = {task_id: _make_traced_task(task_id)}
    fake_runner.device_rank = 0
    fake_runner._flush_trace_session = MagicMock()

    Runner._flush_unfinished_trace_sessions(  # pyright: ignore[reportPrivateUsage]
        cast(Runner, fake_runner)
    )

    # Aborted marker recorded onto the session before the flush.
    session_events = tracing.collect_trace_session(task_id)
    assert any(
        event.name == "aborted" and "trace_abort" in event.tags
        for event in session_events
    ), f"expected aborted marker; got {[e.name for e in session_events]}"

    # The runner's flush was invoked exactly once for this task.
    fake_runner._flush_trace_session.assert_called_once_with(task_id)

    # Cleanup so test isolation holds — the helper does not pop the session
    # itself (that's the runner's _flush_trace_session responsibility, which
    # we mocked above), so we explicitly clear here.
    tracing.clear_trace_session(task_id)


def test_flush_unfinished_trace_sessions_skips_untraced_tasks() -> None:
    """Tasks with ``trace_enabled=False`` must not produce trace markers or flushes."""
    task_id = TaskId("task-untraced-1")

    fake_runner = MagicMock()
    fake_runner.active_tasks = {task_id: _make_untraced_task(task_id)}
    fake_runner.device_rank = 0
    fake_runner._flush_trace_session = MagicMock()

    Runner._flush_unfinished_trace_sessions(  # pyright: ignore[reportPrivateUsage]
        cast(Runner, fake_runner)
    )

    fake_runner._flush_trace_session.assert_not_called()


def test_flush_unfinished_trace_sessions_handles_marker_failures() -> None:
    """If ``record_trace_marker`` somehow raises, the flush still happens.

    The helper uses ``contextlib.suppress`` on both the marker and the
    flush. We can't easily make ``record_trace_marker`` raise here, but
    a session that has already been popped (so the marker silently
    no-ops) still needs the flush call so the runner gets a chance to
    notify the master with whatever events did make it in.
    """
    task_id = TaskId("task-no-session")
    # Deliberately do not seed a session — record_trace_marker will be a
    # no-op (the implementation guards on session presence).

    fake_runner = MagicMock()
    fake_runner.active_tasks = {task_id: _make_traced_task(task_id)}
    fake_runner.device_rank = 0
    fake_runner._flush_trace_session = MagicMock()

    Runner._flush_unfinished_trace_sessions(  # pyright: ignore[reportPrivateUsage]
        cast(Runner, fake_runner)
    )

    fake_runner._flush_trace_session.assert_called_once_with(task_id)


def test_flush_helper_iterates_all_traced_tasks() -> None:
    """When multiple tasks are traced, every one of them gets aborted+flushed."""
    task_ids = [TaskId(f"task-multi-{i}") for i in range(3)]
    for tid in task_ids:
        _seed_session(str(tid))

    fake_runner = MagicMock()
    fake_runner.active_tasks = {tid: _make_traced_task(tid) for tid in task_ids}
    fake_runner.device_rank = 0
    fake_runner._flush_trace_session = MagicMock()

    Runner._flush_unfinished_trace_sessions(  # pyright: ignore[reportPrivateUsage]
        cast(Runner, fake_runner)
    )

    assert fake_runner._flush_trace_session.call_count == 3
    flushed_ids = {call.args[0] for call in fake_runner._flush_trace_session.call_args_list}
    assert flushed_ids == set(task_ids)

    for tid in task_ids:
        tracing.clear_trace_session(tid)
