from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Literal, cast

import anyio
from anyio import BrokenResourceError, ClosedResourceError, to_thread
from loguru import logger

from exo.download.download_utils import (
    build_vindex_path,
    create_http_session,
    resolve_vindex_in_path,
)
from exo.shared.types.diagnostics import (
    RunnerFlightRecorderEntry,
    RunnerLifecycleMilestone,
    RunnerPhaseName,
    RunnerSupervisorDiagnostics,
    RunnerTaskDiagnostics,
)
from exo.shared.types.events import (
    Event,
    LarqlRunnerReadinessUpdated,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import (
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.larql import LarqlRunnerReadiness
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from exo.shared.types.worker.shards import LarqlShardMetadata
from exo.utils.channels import Sender
from exo.utils.task_group import TaskGroup

ProcessFactory = Callable[[Sequence[str]], subprocess.Popen[str]]


def _now_utc_iso() -> str:
    """Return the current UTC time for lifecycle diagnostics."""

    return datetime.now(tz=timezone.utc).isoformat()


def allocate_larql_port(host: str = "127.0.0.1") -> int:
    """Reserve and release a free local TCP port for a LARQL server process."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = cast(tuple[str, int], sock.getsockname())
        return port


def build_larql_serve_command(
    shard: LarqlShardMetadata,
    *,
    vindex_path: Path,
    port: int,
) -> tuple[str, ...]:
    """Build the deterministic LARQL serve command for one cold-tier shard."""

    command = [
        "larql",
        "serve",
        str(vindex_path),
        "--host",
        shard.server_host,
        "--port",
        str(port),
        "--ffn-only",
        "--layers",
        f"{shard.start_layer}-{shard.end_layer}",
        "--preset",
        shard.preset,
    ]
    if shard.expert_range is not None:
        command.extend(
            [
                "--experts",
                f"{shard.expert_range.start_expert}-{shard.expert_range.end_expert}",
            ]
        )
    if shard.units_manifest_path is not None:
        command.extend(["--units", shard.units_manifest_path])
    return tuple(command)


def _default_process_factory(command: Sequence[str]) -> subprocess.Popen[str]:
    """Start a LARQL child process with text stdout/stderr pipes."""

    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _resolve_larql_vindex_path(shard: LarqlShardMetadata) -> Path:
    """Resolve the staged local vindex path for a LARQL shard."""

    if shard.local_vindex_path is not None:
        return Path(shard.local_vindex_path).expanduser()
    found = resolve_vindex_in_path(shard.model_card.model_id)
    if found is not None:
        return found
    return build_vindex_path(shard.model_card.model_id)


@dataclass(eq=False)
class LarqlRunnerSupervisor:
    """Worker-managed supervisor for one upstream `larql serve` child process."""

    bound_instance: BoundInstance
    shard_metadata: LarqlShardMetadata
    _event_sender: Sender[Event]
    _process_factory: ProcessFactory = _default_process_factory
    _readiness_poll_interval: float = 0.25
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)
    in_progress: dict[TaskId, Task] = field(default_factory=dict, init=False)
    completed: set[TaskId] = field(default_factory=set, init=False)
    cancelled: set[TaskId] = field(default_factory=set, init=False)
    _process: subprocess.Popen[str] | None = field(default=None, init=False)
    _port: int | None = field(default=None, init=False)
    _shutdown_requested: bool = field(default=False, init=False)
    _status_since: str = field(default_factory=_now_utc_iso, init=False)
    _status_since_monotonic: float = field(default_factory=time.monotonic, init=False)
    _phase: RunnerPhaseName = field(default="created", init=False)
    _phase_started_at: str = field(default_factory=_now_utc_iso, init=False)
    _phase_started_monotonic: float = field(default_factory=time.monotonic, init=False)
    _last_progress_at: str | None = field(default=None, init=False)
    _last_task_sent_at: str | None = field(default=None, init=False)
    _last_event_received_at: str | None = field(default=None, init=False)
    _last_event_type: str | None = field(default=None, init=False)
    _phase_detail: str | None = field(default=None, init=False)
    _milestones: deque[RunnerLifecycleMilestone] = field(
        default_factory=lambda: deque(maxlen=32), init=False
    )
    _flight_recorder: deque[RunnerFlightRecorderEntry] = field(
        default_factory=lambda: deque(maxlen=128), init=False
    )

    @classmethod
    def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        process_factory: ProcessFactory = _default_process_factory,
    ) -> "LarqlRunnerSupervisor":
        """Construct a LARQL supervisor for an assigned LARQL shard."""

        shard = bound_instance.bound_shard
        if not isinstance(shard, LarqlShardMetadata):
            raise TypeError("LarqlRunnerSupervisor requires LarqlShardMetadata")
        return cls(
            bound_instance=bound_instance,
            shard_metadata=shard,
            _event_sender=event_sender,
            _process_factory=process_factory,
        )

    async def run(self) -> None:
        """Keep the supervisor task group alive until the worker shuts it down."""

        self._record_milestone("supervisor_created", self.status.__class__.__name__)
        try:
            async with self._tg:
                await anyio.sleep_forever()
        finally:
            await self._terminate_child()
            with contextlib.suppress(ClosedResourceError):
                self._event_sender.close()

    def shutdown(self) -> None:
        """Request local LARQL child shutdown and cancel supervisor tasks."""

        self._shutdown_requested = True
        self._record_milestone("shutdown_requested")
        self._tg.cancel_tasks()

    async def start_task(self, task: Task) -> None:
        """Execute lifecycle tasks understood by the LARQL supervisor."""

        self._last_task_sent_at = _now_utc_iso()
        self.pending[task.task_id] = anyio.Event()
        self.in_progress[task.task_id] = task
        await self._send_event(TaskAcknowledged(task_id=task.task_id))
        self.pending.pop(task.task_id, None)
        try:
            if isinstance(task, LoadModel):
                await self._start_until_ready()
                self.completed.add(task.task_id)
                await self._send_event(
                    TaskStatusUpdated(
                        task_id=task.task_id,
                        task_status=TaskStatus.Complete,
                    )
                )
            elif isinstance(task, StartWarmup):
                self.completed.add(task.task_id)
                await self._send_event(
                    TaskStatusUpdated(
                        task_id=task.task_id,
                        task_status=TaskStatus.Complete,
                    )
                )
            elif isinstance(task, Shutdown):
                self._shutdown_requested = True
                await self._terminate_child()
                self.completed.add(task.task_id)
                await self._send_event(
                    TaskStatusUpdated(
                        task_id=task.task_id,
                        task_status=TaskStatus.Complete,
                    )
                )
                await self._set_status(RunnerShutdown())
            else:
                raise RuntimeError(
                    f"LarqlRunner does not execute {task.__class__.__name__} tasks"
                )
        except Exception as exc:
            await self._mark_failed(str(exc))
            await self._send_event(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Failed)
            )
        finally:
            self.in_progress.pop(task.task_id, None)

    async def cancel_task(self, task_id: TaskId) -> None:
        """Mark a task cancellation request for diagnostics."""

        self.cancelled.add(task_id)

    async def _start_until_ready(self) -> None:
        vindex_path = _resolve_larql_vindex_path(self.shard_metadata)
        port = self.shard_metadata.server_port or allocate_larql_port(
            self.shard_metadata.server_host
        )
        self._port = port
        command = build_larql_serve_command(
            self.shard_metadata,
            vindex_path=vindex_path,
            port=port,
        )
        for attempt in range(self.shard_metadata.max_crash_restarts + 1):
            await self._set_status(RunnerLoading())
            self._record_milestone("larql_start_requested", " ".join(command))
            self._process = self._process_factory(command)
            self._tg.start_soon(self._forward_stream, self._process.stdout, "stdout")
            self._tg.start_soon(self._forward_stream, self._process.stderr, "stderr")
            try:
                await self._wait_until_ready()
                await self._set_status(RunnerReady())
                await self._send_readiness("ready")
                assert self._process is not None
                self._tg.start_soon(self._watch_process_exit, self._process)
                return
            except Exception as exc:
                await self._terminate_child()
                if attempt >= self.shard_metadata.max_crash_restarts:
                    raise RuntimeError(
                        "LARQL child failed readiness after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
                self._record_milestone(
                    "larql_restart",
                    f"attempt={attempt + 1}: {type(exc).__name__}",
                )

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.shard_metadata.readiness_timeout_seconds
        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                raise RuntimeError("LARQL process was not started")
            if process.poll() is not None:
                raise RuntimeError(f"LARQL process exited with {process.returncode}")
            if await self._health_check():
                return
            await anyio.sleep(self._readiness_poll_interval)
        raise TimeoutError("Timed out waiting for LARQL readiness")

    async def _watch_process_exit(self, process: subprocess.Popen[str]) -> None:
        """Report a ready LARQL child that exits outside intentional shutdown."""

        return_code = await to_thread.run_sync(process.wait, abandon_on_cancel=True)
        if self._shutdown_requested or self._process is not process:
            return
        message = f"LARQL child exited unexpectedly with exit code {return_code}"
        self._record_milestone("larql_exited", message)
        await self._mark_failed(message)

    async def _health_check(self) -> bool:
        port = self._port
        if port is None:
            return False
        url = f"http://{self.shard_metadata.server_host}:{port}/v1/health"
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.get(url) as response,
            ):
                return response.status == 200
        except Exception:
            return False

    async def _send_readiness(
        self,
        status: Literal["ready", "not_ready", "failed"],
        error_message: str | None = None,
    ) -> None:
        port = self._port or 0
        readiness = LarqlRunnerReadiness(
            runner_id=self.bound_instance.bound_runner_id,
            vindex_uri=self.shard_metadata.vindex_uri,
            preset=self.shard_metadata.preset,
            start_layer=self.shard_metadata.start_layer,
            end_layer=self.shard_metadata.end_layer,
            expert_range=self.shard_metadata.expert_range,
            units_manifest_path=self.shard_metadata.units_manifest_path,
            host=self.shard_metadata.server_host,
            port=max(port, 1),
            status=status,
            ram_footprint=await self._process_memory(),
            error_message=error_message,
        )
        await self._send_event(LarqlRunnerReadinessUpdated(readiness=readiness))

    async def _process_memory(self) -> Memory | None:
        process = self._process
        if process is None:
            return None
        try:
            import psutil

            info = await to_thread.run_sync(
                lambda: psutil.Process(process.pid).memory_info()
            )
            return Memory.from_bytes(int(info.rss))
        except Exception:
            return None

    async def _forward_stream(
        self,
        stream: IO[str] | None,
        stream_name: Literal["stdout", "stderr"],
    ) -> None:
        if stream is None:
            return
        while True:
            line = await to_thread.run_sync(stream.readline, abandon_on_cancel=True)
            if not line:
                return
            logger.info(f"larql-server[{stream_name}]: {line.rstrip()}")

    async def _terminate_child(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        await self._set_status(RunnerShuttingDown())
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            await to_thread.run_sync(
                lambda: process.wait(timeout=5),
                abandon_on_cancel=True,
            )
        if process.poll() is None:
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                await to_thread.run_sync(
                    lambda: process.wait(timeout=5),
                    abandon_on_cancel=True,
                )

    async def _mark_failed(self, message: str) -> None:
        await self._set_status(RunnerFailed(error_message=message))
        await self._send_readiness("failed", message)

    async def _set_status(self, status: RunnerStatus) -> None:
        self.status = status
        self._status_since = _now_utc_iso()
        self._status_since_monotonic = time.monotonic()
        self._record_milestone("status_changed", status.__class__.__name__)
        await self._send_event(
            RunnerStatusUpdated(
                runner_id=self.bound_instance.bound_runner_id,
                runner_status=status,
            )
        )

    async def _send_event(self, event: Event) -> None:
        self._last_event_received_at = _now_utc_iso()
        self._last_event_type = event.__class__.__name__
        try:
            await self._event_sender.send(event)
        except (ClosedResourceError, BrokenResourceError):
            logger.warning("LarqlRunner event sender closed")

    def _record_milestone(self, name: str, detail: str | None = None) -> None:
        self._milestones.append(
            RunnerLifecycleMilestone(at=_now_utc_iso(), name=name, detail=detail)
        )
        self._last_progress_at = _now_utc_iso()

    def _task_diagnostics(self, task: Task) -> RunnerTaskDiagnostics:
        return RunnerTaskDiagnostics(
            task_id=str(task.task_id),
            task_kind=task.__class__.__name__,
            task_status=str(task.task_status.value),
            instance_id=str(task.instance_id),
            command_id=None,
            runner_id=str(self.bound_instance.bound_runner_id),
            model_id=str(self.shard_metadata.model_card.model_id),
        )

    def diagnostics(self) -> RunnerSupervisorDiagnostics:
        """Return live read-only diagnostics for this LARQL supervisor."""

        process = self._process
        return RunnerSupervisorDiagnostics(
            runner_id=str(self.bound_instance.bound_runner_id),
            instance_id=str(self.bound_instance.instance.instance_id),
            node_id=str(self.bound_instance.bound_node_id),
            model_id=str(self.shard_metadata.model_card.model_id),
            device_rank=self.shard_metadata.device_rank,
            world_size=self.shard_metadata.world_size,
            start_layer=self.shard_metadata.start_layer,
            end_layer=self.shard_metadata.end_layer,
            n_layers=self.shard_metadata.n_layers,
            pid=process.pid if process is not None else None,
            process_alive=process is not None and process.poll() is None,
            exit_code=process.returncode if process is not None else None,
            status_kind=self.status.__class__.__name__,
            status_since=self._status_since,
            seconds_in_status=time.monotonic() - self._status_since_monotonic,
            phase=self._phase,
            phase_started_at=self._phase_started_at,
            seconds_in_phase=time.monotonic() - self._phase_started_monotonic,
            last_progress_at=self._last_progress_at,
            active_task_id=None,
            active_command_id=None,
            phase_detail=self._phase_detail,
            last_mlx_memory=None,
            flight_recorder=list(self._flight_recorder),
            pending_task_ids=[str(task_id) for task_id in self.pending],
            in_progress_tasks=[
                self._task_diagnostics(task) for task in self.in_progress.values()
            ],
            completed_task_count=len(self.completed),
            cancelled_task_ids=[str(task_id) for task_id in self.cancelled],
            last_task_sent_at=self._last_task_sent_at,
            last_event_received_at=self._last_event_received_at,
            last_event_type=self._last_event_type,
            milestones=list(self._milestones),
        )
