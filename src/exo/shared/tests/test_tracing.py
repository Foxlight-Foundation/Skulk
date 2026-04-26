"""Tests for task-scoped tracing helpers."""

from pathlib import Path

from exo.shared.tracing import (
    begin_trace_session,
    bind_trace_session,
    clear_trace_session,
    export_trace,
    load_trace_file,
    pop_trace_session,
    record_shared_span,
    record_trace_marker,
    trace,
)


def test_task_scoped_sessions_do_not_mix_events() -> None:
    """Concurrent trace sessions should keep their buffered spans isolated."""

    task_a = "task-a"
    task_b = "task-b"
    begin_trace_session(
        task_a,
        rank=0,
        node_id="kite-dev",
        model_id="mlx-community/model-a",
        task_kind="text",
        tags=["chat"],
    )
    begin_trace_session(
        task_b,
        rank=1,
        node_id="kite-2",
        model_id="mlx-community/model-b",
        task_kind="embedding",
    )

    try:
        with bind_trace_session(task_a):
            record_trace_marker("queued", rank=0)
            with trace("prefill", rank=0, category="compute"):
                pass

        with bind_trace_session(task_b):
            record_trace_marker("tokenize", rank=1)

        task_a_events = pop_trace_session(task_a)
        task_b_events = pop_trace_session(task_b)
    finally:
        clear_trace_session(task_a)
        clear_trace_session(task_b)

    assert [event.name for event in task_a_events] == ["queued", "prefill"]
    assert [event.name for event in task_b_events] == ["tokenize"]
    assert all(event.node_id == "kite-dev" for event in task_a_events)
    assert all(event.model_id == "mlx-community/model-a" for event in task_a_events)
    assert all(event.task_kind == "text" for event in task_a_events)
    assert all("chat" in event.tags for event in task_a_events)
    assert task_b_events[0].node_id == "kite-2"
    assert task_b_events[0].task_kind == "embedding"


def test_shared_spans_are_copied_into_each_participant_trace() -> None:
    """Shared batch work should appear in each participating task trace."""

    task_ids = ["task-one", "task-two"]
    for task_id in task_ids:
        begin_trace_session(
            task_id,
            rank=0,
            node_id="kite-dev",
            model_id="mlx-community/gemma",
            task_kind="text",
        )

    try:
        record_shared_span(
            task_ids,
            name="decode_step",
            start_us=123,
            duration_us=456,
            rank=2,
            category="decode",
            tags=["tool_call"],
            attrs={
                "shared_span": True,
                "batch_size": 2,
                "participant_task_ids": task_ids,
            },
        )
        task_one_events = pop_trace_session(task_ids[0])
        task_two_events = pop_trace_session(task_ids[1])
    finally:
        for task_id in task_ids:
            clear_trace_session(task_id)

    for event in [*task_one_events, *task_two_events]:
        assert event.name == "decode_step"
        assert event.duration_us == 456
        assert event.rank == 2
        assert event.category == "decode"
        assert "tool_call" in event.tags
        assert event.attrs["shared_span"] is True
        assert event.attrs["batch_size"] == 2
        assert event.attrs["participant_task_ids"] == task_ids


def test_export_trace_round_trips_metadata(tmp_path: Path) -> None:
    """Chrome trace export should preserve task metadata in the args payload."""

    task_id = "task-roundtrip"
    begin_trace_session(
        task_id,
        rank=4,
        node_id="kite3.local",
        model_id="mlx-community/gemma-4-26b-a4b-it-4bit",
        task_kind="image",
        tags=["tool_call"],
        attrs={"batch_size": 1},
    )

    try:
        with bind_trace_session(task_id):
            record_trace_marker("finish", rank=4, attrs={"success": True})
        output_path = tmp_path / "trace_roundtrip.json"
        export_trace(pop_trace_session(task_id), output_path)
    finally:
        clear_trace_session(task_id)

    loaded = load_trace_file(output_path)

    assert len(loaded) == 1
    event = loaded[0]
    assert event.name == "finish"
    assert event.node_id == "kite3.local"
    assert event.model_id == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert event.task_kind == "image"
    assert "tool_call" in event.tags
    assert event.attrs["batch_size"] == 1
    assert event.attrs["success"] is True
