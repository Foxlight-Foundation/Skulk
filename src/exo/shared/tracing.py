from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, final

from exo.worker.runner.bootstrap import logger

# Context variable to track the current trace category for hierarchical nesting
_current_category: ContextVar[str | None] = ContextVar("current_category", default=None)
_current_trace_task_id: ContextVar[str | None] = ContextVar(
    "current_trace_task_id", default=None
)

TraceTaskKind = Literal["image", "text", "embedding"]
TraceAttrValue = str | int | float | bool | list[str]


@final
@dataclass(frozen=True)
class TraceEvent:
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str
    node_id: str | None = None
    model_id: str | None = None
    task_kind: TraceTaskKind | None = None
    tags: tuple[str, ...] = ()
    attrs: dict[str, TraceAttrValue] = field(default_factory=dict)


@final
@dataclass
class CategoryStats:
    total_us: int = 0
    count: int = 0
    min_us: int = 0
    max_us: int = 0

    def add(self, duration_us: int) -> None:
        if self.count == 0:
            self.min_us = duration_us
            self.max_us = duration_us
        else:
            self.min_us = min(self.min_us, duration_us)
            self.max_us = max(self.max_us, duration_us)
        self.total_us += duration_us
        self.count += 1

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count > 0 else 0.0


@final
@dataclass
class TraceStats:
    total_wall_time_us: int = 0
    by_category: dict[str, CategoryStats] = field(default_factory=dict)
    by_rank: dict[int, dict[str, CategoryStats]] = field(default_factory=dict)


@final
@dataclass
class TraceSession:
    task_id: str
    rank: int
    node_id: str
    model_id: str
    task_kind: TraceTaskKind
    tags: tuple[str, ...] = ()
    attrs: dict[str, TraceAttrValue] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)


_trace_sessions: dict[str, TraceSession] = {}


def now_us() -> int:
    """Return the current wall-clock time in microseconds."""

    return int(time.time() * 1_000_000)


def _merge_tags(
    session_tags: Sequence[str], event_tags: Sequence[str] | None
) -> tuple[str, ...]:
    merged: list[str] = []
    for tag in (*session_tags, *(event_tags or ())):
        if tag not in merged:
            merged.append(tag)
    return tuple(merged)


def _merge_attrs(
    session_attrs: Mapping[str, TraceAttrValue],
    event_attrs: Mapping[str, TraceAttrValue] | None,
) -> dict[str, TraceAttrValue]:
    merged = dict(session_attrs)
    if event_attrs is not None:
        merged.update(event_attrs)
    return merged


def begin_trace_session(
    task_id: object,
    *,
    rank: int,
    node_id: str,
    model_id: str,
    task_kind: TraceTaskKind,
    tags: Sequence[str] | None = None,
    attrs: Mapping[str, TraceAttrValue] | None = None,
) -> None:
    """Create or replace a task-scoped trace session.

    Runtime tracing is task-scoped so concurrent requests can emit spans without
    contaminating one another, even when they share a batch engine.
    """

    normalized_task_id = str(task_id)
    _trace_sessions[normalized_task_id] = TraceSession(
        task_id=normalized_task_id,
        rank=rank,
        node_id=node_id,
        model_id=model_id,
        task_kind=task_kind,
        tags=tuple(tags or ()),
        attrs=dict(attrs or {}),
    )


def has_trace_session(task_id: object) -> bool:
    """Return whether *task_id* currently has an active trace session."""

    return str(task_id) in _trace_sessions


def tracing_active(task_id: object | None = None) -> bool:
    """Return whether tracing is currently enabled for a task context."""

    if task_id is not None:
        return has_trace_session(task_id)
    current_task_id = _current_trace_task_id.get()
    return current_task_id is not None and current_task_id in _trace_sessions


@contextmanager
def bind_trace_session(task_id: object) -> Generator[None, None, None]:
    """Bind a trace session to the current execution context."""

    normalized_task_id = str(task_id)
    if normalized_task_id not in _trace_sessions:
        yield
        return

    task_token = _current_trace_task_id.set(normalized_task_id)
    category_token = _current_category.set(None)
    try:
        yield
    finally:
        _current_category.reset(category_token)
        _current_trace_task_id.reset(task_token)


def _record_span(
    task_id: str,
    name: str,
    start_us: int,
    duration_us: int,
    rank: int,
    category: str,
    tags: Sequence[str] | None = None,
    attrs: Mapping[str, TraceAttrValue] | None = None,
) -> None:
    session = _trace_sessions.get(task_id)
    if session is None:
        return

    session.events.append(
        TraceEvent(
            name=name,
            start_us=start_us,
            duration_us=duration_us,
            rank=rank,
            category=category,
            node_id=session.node_id,
            model_id=session.model_id,
            task_kind=session.task_kind,
            tags=_merge_tags(session.tags, tags),
            attrs=_merge_attrs(session.attrs, attrs),
        )
    )


@contextmanager
def trace(
    name: str,
    rank: int,
    category: str = "compute",
    *,
    task_id: object | None = None,
    tags: Sequence[str] | None = None,
    attrs: Mapping[str, TraceAttrValue] | None = None,
) -> Generator[None, None, None]:
    """Context manager to trace any operation.

    Nested traces automatically inherit the parent category, creating hierarchical
    categories like "sync/compute" or "async/comms".

    Args:
        name: Name of the operation (e.g., "recv 0", "send 1", "joint_blocks")
        rank: This rank's ID
        category: Category for grouping in trace viewer ("comm", "compute", "step")

    Example:
        with trace(f"sync {t}", rank, "sync"):
            with trace("joint_blocks", rank, "compute"):
                # Recorded with category "sync/compute"
                hidden_states = some_computation(...)
    """
    normalized_task_id = (
        str(task_id) if task_id is not None else _current_trace_task_id.get()
    )
    if normalized_task_id is None or normalized_task_id not in _trace_sessions:
        yield
        return

    # Combine with parent category if nested
    parent = _current_category.get()
    full_category = f"{parent}/{category}" if parent else category

    # Set as current for nested traces
    token = _current_category.set(full_category)

    try:
        start_us = now_us()
        start_perf = time.perf_counter()
        yield
        duration_us = int((time.perf_counter() - start_perf) * 1_000_000)
        _record_span(
            normalized_task_id,
            name,
            start_us,
            duration_us,
            rank,
            full_category,
            tags=tags,
            attrs=attrs,
        )
    finally:
        _current_category.reset(token)


def record_trace_marker(
    name: str,
    rank: int,
    category: str = "lifecycle",
    *,
    task_id: object | None = None,
    tags: Sequence[str] | None = None,
    attrs: Mapping[str, TraceAttrValue] | None = None,
) -> None:
    """Record a zero-duration marker in the current or explicit trace session."""

    normalized_task_id = (
        str(task_id) if task_id is not None else _current_trace_task_id.get()
    )
    if normalized_task_id is None or normalized_task_id not in _trace_sessions:
        return

    parent = _current_category.get()
    full_category = f"{parent}/{category}" if parent else category
    _record_span(
        normalized_task_id,
        name,
        now_us(),
        0,
        rank,
        full_category,
        tags=tags,
        attrs=attrs,
    )


def record_shared_span(
    task_ids: Sequence[object],
    *,
    name: str,
    start_us: int,
    duration_us: int,
    rank: int,
    category: str,
    tags: Sequence[str] | None = None,
    attrs: Mapping[str, TraceAttrValue] | None = None,
) -> None:
    """Duplicate a shared span into multiple task traces.

    Batch text generation can advance several requests in one decode step. We
    duplicate that shared work into each participating task trace and mark it
    explicitly so the dashboard can filter or explain it later.
    """

    for task_id in task_ids:
        _record_span(
            str(task_id),
            name,
            start_us,
            duration_us,
            rank,
            category,
            tags=tags,
            attrs=attrs,
        )


def collect_trace_session(task_id: object) -> list[TraceEvent]:
    """Return the current buffered events for *task_id* without clearing them."""

    session = _trace_sessions.get(str(task_id))
    return list(session.events) if session is not None else []


def pop_trace_session(task_id: object) -> list[TraceEvent]:
    """Return and clear the buffered events for *task_id*."""

    session = _trace_sessions.pop(str(task_id), None)
    return list(session.events) if session is not None else []


def clear_trace_session(task_id: object) -> None:
    """Discard any buffered trace state for *task_id*."""

    _trace_sessions.pop(str(task_id), None)


def export_trace(traces: list[TraceEvent], output_path: Path) -> None:
    trace_events: list[dict[str, object]] = []

    for event in traces:
        # Chrome trace format uses "X" for complete events (with duration)
        chrome_event: dict[str, object] = {
            "name": event.name,
            "cat": event.category,
            "ph": "X",
            "ts": event.start_us,
            "dur": event.duration_us,
            "pid": 0,
            "tid": event.rank,
            "args": {
                "rank": event.rank,
                "node_id": event.node_id,
                "model_id": event.model_id,
                "task_kind": event.task_kind,
                "tags": list(event.tags),
                "attrs": event.attrs,
            },
        }
        trace_events.append(chrome_event)

    ranks_seen = set(t.rank for t in traces)
    for rank in ranks_seen:
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",  # Metadata event
                "pid": 0,
                "tid": rank,
                "args": {"name": f"Rank {rank}"},
            }
        )

    chrome_trace = {"traceEvents": trace_events}

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(chrome_trace, f, indent=2)
    except OSError as e:
        logger.warning("Failed to export trace to %s: %s", output_path, e)


def load_trace_file(path: Path) -> list[TraceEvent]:
    with open(path) as f:
        data = cast(dict[str, list[dict[str, object]]], json.load(f))

    events = data.get("traceEvents", [])
    traces: list[TraceEvent] = []

    for event in events:
        # Skip metadata events
        if event.get("ph") == "M":
            continue

        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        ts_value = event.get("ts", 0)
        dur_value = event.get("dur", 0)
        tid_value = event.get("tid", 0)
        start_us = int(ts_value) if isinstance(ts_value, (int, float, str)) else 0
        duration_us = int(dur_value) if isinstance(dur_value, (int, float, str)) else 0

        # Get rank from tid or args
        rank = int(tid_value) if isinstance(tid_value, (int, float, str)) else 0
        args = event.get("args")
        node_id: str | None = None
        model_id: str | None = None
        task_kind: TraceTaskKind | None = None
        tags: tuple[str, ...] = ()
        attrs: dict[str, TraceAttrValue] = {}
        if isinstance(args, dict):
            args_dict = cast(dict[str, object], args)
            rank_from_args = args_dict.get("rank")
            if isinstance(rank_from_args, (int, float, str)):
                rank = int(rank_from_args)
            node_id_value = args_dict.get("node_id")
            if isinstance(node_id_value, str) and node_id_value:
                node_id = node_id_value
            model_id_value = args_dict.get("model_id")
            if isinstance(model_id_value, str) and model_id_value:
                model_id = model_id_value
            task_kind_value = args_dict.get("task_kind")
            if task_kind_value in ("image", "text", "embedding"):
                task_kind = task_kind_value
            tags_value = args_dict.get("tags")
            if isinstance(tags_value, list):
                tag_items = cast(list[object], tags_value)
                tags = tuple(
                    tag if isinstance(tag, str) else str(tag) for tag in tag_items
                )
            attrs_value = args_dict.get("attrs")
            if isinstance(attrs_value, dict):
                attrs = cast(dict[str, TraceAttrValue], attrs_value)

        traces.append(
            TraceEvent(
                name=name,
                start_us=start_us,
                duration_us=duration_us,
                rank=rank,
                category=category,
                node_id=node_id,
                model_id=model_id,
                task_kind=task_kind,
                tags=tags,
                attrs=attrs,
            )
        )

    return traces


def compute_stats(traces: list[TraceEvent]) -> TraceStats:
    stats = TraceStats()

    if not traces:
        return stats

    # Calculate wall time from earliest start to latest end
    min_start = min(t.start_us for t in traces)
    max_end = max(t.start_us + t.duration_us for t in traces)
    stats.total_wall_time_us = max_end - min_start

    # Initialize nested dicts
    by_category: dict[str, CategoryStats] = defaultdict(CategoryStats)
    by_rank: dict[int, dict[str, CategoryStats]] = defaultdict(
        lambda: defaultdict(CategoryStats)
    )

    for event in traces:
        # By category
        by_category[event.category].add(event.duration_us)

        # By rank and category
        by_rank[event.rank][event.category].add(event.duration_us)

    stats.by_category = dict(by_category)
    stats.by_rank = {k: dict(v) for k, v in by_rank.items()}

    return stats
