"""Local-only runner diagnostic flight-recorder helpers.

This module intentionally bypasses the event-sourced cluster state. It is an
always-on, bounded, best-effort signal path from runner subprocesses back to
their supervisors so wedged native/MLX phases remain visible even when traces
are disabled or cannot flush.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import cast

from anyio import ClosedResourceError, WouldBlock

from exo.shared.types.diagnostics import (
    MlxMemorySnapshot,
    RunnerDiagnosticContext,
    RunnerDiagnosticUpdate,
    RunnerDiagnosticValue,
    RunnerPhaseName,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId
from exo.utils.channels import MpSender

_diagnostic_sender: MpSender[RunnerDiagnosticUpdate] | None = None
_diagnostic_context: RunnerDiagnosticContext | None = None
_known_wired_limit_bytes: int | None = None


def _now_utc_iso() -> str:
    """Return the current UTC timestamp for diagnostic events."""

    return datetime.now(tz=timezone.utc).isoformat()


def configure_runner_diagnostics(
    sender: MpSender[RunnerDiagnosticUpdate] | None,
    context: RunnerDiagnosticContext,
) -> None:
    """Configure process-local runner diagnostic emission.

    Args:
        sender: Multiprocessing sender owned by the runner subprocess.
        context: Stable identity fields to attach to every diagnostic event.
    """

    global _diagnostic_context, _diagnostic_sender
    _diagnostic_sender = sender
    _diagnostic_context = context


def remember_wired_limit_bytes(value: int | None) -> None:
    """Remember the last configured MLX wired limit when the caller knows it."""

    global _known_wired_limit_bytes
    _known_wired_limit_bytes = value


def _memory_from_callable(module: object, name: str) -> Memory | None:
    getter = getattr(module, name, None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except Exception:
        return None
    if isinstance(value, (int, float)):
        return Memory.from_bytes(int(value))
    return None


def capture_mlx_memory_snapshot() -> MlxMemorySnapshot | None:
    """Return a best-effort MLX/Metal memory snapshot for the current process."""

    for module_name in ("mlx.core", "mlx.core.metal"):
        try:
            module = import_module(module_name)
        except Exception:
            continue

        active = _memory_from_callable(module, "get_active_memory")
        cache = _memory_from_callable(module, "get_cache_memory")
        peak = _memory_from_callable(module, "get_peak_memory")
        if active is None and cache is None and peak is None:
            continue
        return MlxMemorySnapshot(
            generated_at=_now_utc_iso(),
            active=active,
            cache=cache,
            peak=peak,
            wired_limit=Memory.from_bytes(_known_wired_limit_bytes)
            if _known_wired_limit_bytes is not None
            else None,
            source=module_name,
        )
    return None


def _normalize_attr_value(value: object) -> RunnerDiagnosticValue | None:
    """Coerce diagnostic attrs into the bounded JSON-safe schema."""

    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, (list, tuple)):
        normalized = [str(item) for item in cast(Sequence[object], value)]
        return normalized
    if value is None:
        return None
    return str(value)


def normalize_attrs(attrs: dict[str, object] | None) -> dict[str, RunnerDiagnosticValue]:
    """Return attrs filtered to values supported by diagnostic models."""

    if attrs is None:
        return {}
    normalized: dict[str, RunnerDiagnosticValue] = {}
    for key, value in attrs.items():
        normalized_value = _normalize_attr_value(value)
        if normalized_value is not None:
            normalized[str(key)] = normalized_value
    return normalized


def _task_id_string(task_id: TaskId | str | None) -> str | None:
    return None if task_id is None else str(task_id)


def record_runner_phase(
    phase: RunnerPhaseName,
    *,
    event: str = "phase",
    detail: str | None = None,
    attrs: dict[str, object] | None = None,
    task_id: TaskId | str | None = None,
    command_id: str | None = None,
    include_memory: bool = False,
) -> None:
    """Emit one non-blocking runner diagnostic update if configured."""

    if _diagnostic_sender is None or _diagnostic_context is None:
        return

    update = RunnerDiagnosticUpdate(
        at=_now_utc_iso(),
        phase=phase,
        event=event,
        detail=detail,
        attrs=normalize_attrs(attrs),
        context=_diagnostic_context.model_copy(update={"pid": os.getpid()}),
        task_id=_task_id_string(task_id),
        command_id=command_id,
        mlx_memory=capture_mlx_memory_snapshot() if include_memory else None,
    )
    with contextlib.suppress(WouldBlock, ClosedResourceError, OSError, ValueError):
        _diagnostic_sender.send_nowait(update)


@contextlib.contextmanager
def runner_phase(
    phase: RunnerPhaseName,
    *,
    detail: str | None = None,
    attrs: dict[str, object] | None = None,
    task_id: TaskId | str | None = None,
    command_id: str | None = None,
    include_memory: bool = False,
) -> Iterator[None]:
    """Record phase entry and exit around a runner operation."""

    record_runner_phase(
        phase,
        event="enter",
        detail=detail,
        attrs=attrs,
        task_id=task_id,
        command_id=command_id,
        include_memory=include_memory,
    )
    try:
        yield
    except BaseException as exc:
        if type(exc).__name__ == "PrefillCancelled":
            record_runner_phase(
                "cancel_observed",
                event="prefill_cancelled",
                detail=detail,
                attrs=attrs,
                task_id=task_id,
                command_id=command_id,
                include_memory=True,
            )
        else:
            record_runner_phase(
                "error",
                event="exception",
                detail=f"{type(exc).__name__}: {exc}",
                attrs=attrs,
                task_id=task_id,
                command_id=command_id,
                include_memory=True,
            )
        raise
    else:
        record_runner_phase(
            phase,
            event="exit",
            detail=detail,
            attrs=attrs,
            task_id=task_id,
            command_id=command_id,
            include_memory=include_memory,
        )
