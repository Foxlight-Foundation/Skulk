# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Served-backend text-generation runner: launches and proxies ``vllm serve``.

The second *served* engine (after ``llama_server``), reusing that engine's
generic shape -- a managed inference-server subprocess plus an OpenAI HTTP proxy
-- with the vLLM CLI instead of ``llama-server``. vLLM is the GPU-serving fast
path: its continuous batching and paged attention hold latency flat and grow
aggregate throughput under concurrent load, where the single-stream engines
(``llama_cpp`` / ``llama_server``) collapse. It coexists with those engines and
is selected per model by the card's ``compatible_backends`` on a node that set
``SKULK_VLLM_BIN``.

Single-node only in this first slice (no ring / warmup / RPC), mirroring the
in-process runners. Linux-oriented: the subprocess is reaped on parent death via
``PR_SET_PDEATHSIG`` so a runner crash never orphans a ``vllm serve`` process
holding GPU memory. Per-request cancellation aborts the proxied HTTP connection
(stopping server-side generation); ``SIGTERM`` is for whole-server teardown.

Scope of this slice: streamed chat completions only. Tool calling and per-token
logprobs are rejected loudly rather than silently mismeasured (the OpenAI SSE
proxy does not surface logprobs, and tool-call round-tripping is a follow-up).

Reasoning is best-effort in this slice. Thinking control (``enable_thinking`` /
``reasoning_effort``) is forwarded so the model thinks, and both ``reasoning_content``
and ``reasoning`` SSE deltas are parsed into ``is_thinking`` chunks. But vLLM only
SPLITS reasoning from content when the server is launched with a family-specific
``--reasoning-parser`` (e.g. ``qwen3`` / ``deepseek_r1`` / ``openai_gptoss``); this
slice does not yet map the card to that flag, so on a reasoning model the thinking
text arrives inline in ``content`` (raw markers) rather than as a separated
reasoning stream. Threading the card's reasoning family into ``--reasoning-parser``
is a follow-up (alongside tool calling and logprobs).

Concurrent dispatch: unlike the in-process runners (which serialize one task at a
time), ``main()`` keeps up to N ``TextGeneration`` requests in flight at once,
each streaming its own HTTP request to the one shared ``vllm serve`` on its own
thread. That is what actually lets vLLM's continuous batching + paged attention
activate: the server sees concurrent in-flight requests and batches their decode
steps, holding latency flat and growing aggregate throughput under load (the
whole reason this engine exists). N is bounded by a thread pool sized from
``SKULK_VLLM_MAX_CONCURRENT_REQUESTS`` (vLLM itself caps at ``--max-num-seqs``, so
this is a client-side admission bound, not the batch width). Runner status is
``RunnerRunning`` while any generation is in flight and ``RunnerReady`` when the
last one drains; ``MpSender`` event sends and the diagnostic emitter are already
thread-safe, and each ``DataChunk`` carries ``command_id`` + ``sequence`` so the
API demultiplexes interleaved token streams. The lifecycle tasks (``LoadModel``,
``Shutdown``) run inline on the dispatch thread; shutdown cancels every in-flight
generation and drains the pool before tearing down the server.
"""

import contextlib
import ctypes
import json
import os
import random
import signal
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import httpx
from anyio import ClosedResourceError, EndOfStream, WouldBlock

from skulk.api.types import GenerationStats
from skulk.download.download_utils import build_model_path
from skulk.shared.backends import VLLM_BIN_ENV
from skulk.shared.types.chunks import ErrorChunk, TokenChunk
from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    LoadModel,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.diagnostics import record_runner_phase, runner_phase
from skulk.worker.runner.generation_stats import (
    StreamStatsClock,
    subprocess_peak_memory,
)
from skulk.worker.runner.llama_cpp.runner import (
    map_finish_reason,
    messages_for_llama,
    serving_n_ctx,
    wants_logprobs,
)

# vLLM startup can be slow: weight load + torch.compile + CUDA-graph capture on a
# large model runs to a couple of minutes, so allow a generous health deadline.
_HEALTH_DEADLINE_S: Final = 600.0
# Between tasks the runner wakes at this cadence to verify the server subprocess
# is still alive (a dead server between requests must crash the runner, not wedge
# it Ready), mirroring the llama_server runner.
_LIVENESS_POLL_S: Final = 2.0

# Fraction of GPU VRAM vLLM may use for weights + KV cache. Operator-tunable via
# env; vLLM's own default is 0.90. Placement admits against the same usable-VRAM
# figure, so this stays a node-local serving knob for now (a card-level override
# is a follow-up when vLLM-aware admission lands).
_GPU_MEMORY_UTILIZATION_ENV: Final = "SKULK_VLLM_GPU_MEMORY_UTILIZATION"
_DEFAULT_GPU_MEMORY_UTILIZATION: Final = 0.90

# Upper bound on concurrent in-flight generations the runner streams to the one
# ``vllm serve`` at once. This is a client-side admission bound (the thread-pool
# width), NOT vLLM's batch width -- the server batches up to its own
# ``--max-num-seqs`` (default 256). Kept below that so queued requests wait in the
# runner's bounded pool rather than piling unbounded threads against the server.
_MAX_CONCURRENT_REQUESTS_ENV: Final = "SKULK_VLLM_MAX_CONCURRENT_REQUESTS"
_DEFAULT_MAX_CONCURRENT_REQUESTS: Final = 32


def _max_concurrent_requests() -> int:
    """The concurrent in-flight generation cap, from env or the default.

    An unparseable or below-1 value falls back to the default rather than
    disabling concurrency (0) or crashing the pool at construction.
    """
    raw = os.environ.get(_MAX_CONCURRENT_REQUESTS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_CONCURRENT_REQUESTS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            f"{_MAX_CONCURRENT_REQUESTS_ENV}={raw!r} is not an integer; "
            f"using {_DEFAULT_MAX_CONCURRENT_REQUESTS}"
        )
        return _DEFAULT_MAX_CONCURRENT_REQUESTS
    if value < 1:
        logger.warning(
            f"{_MAX_CONCURRENT_REQUESTS_ENV}={value} is below 1; "
            f"using {_DEFAULT_MAX_CONCURRENT_REQUESTS}"
        )
        return _DEFAULT_MAX_CONCURRENT_REQUESTS
    return value


class _StreamDelta(NamedTuple):
    """One parsed SSE delta from the proxied ``/v1/chat/completions`` stream."""

    reasoning: str
    content: str
    finish: Literal["stop", "length", "content_filter"] | None
    done: bool  # the terminal ``data: [DONE]`` sentinel


def _gpu_memory_utilization() -> float:
    """The ``--gpu-memory-utilization`` fraction, from env or the 0.90 default.

    An unparseable or out-of-range (0, 1] value falls back to the default rather
    than passing vLLM a nonsense fraction that would fail the server at spawn.
    """
    raw = os.environ.get(_GPU_MEMORY_UTILIZATION_ENV, "").strip()
    if not raw:
        return _DEFAULT_GPU_MEMORY_UTILIZATION
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            f"{_GPU_MEMORY_UTILIZATION_ENV}={raw!r} is not a number; "
            f"using {_DEFAULT_GPU_MEMORY_UTILIZATION}"
        )
        return _DEFAULT_GPU_MEMORY_UTILIZATION
    if not 0.0 < value <= 1.0:
        logger.warning(
            f"{_GPU_MEMORY_UTILIZATION_ENV}={value} is outside (0, 1]; "
            f"using {_DEFAULT_GPU_MEMORY_UTILIZATION}"
        )
        return _DEFAULT_GPU_MEMORY_UTILIZATION
    return value


def build_vllm_serve_args(
    binary: str,
    model_dir: Path,
    served_model_name: str,
    host: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
) -> list[str]:
    """Build the ``vllm serve`` command line. Pure, so it is unit-testable.

    vLLM auto-detects the platform (CUDA vs ROCm) from its own install, so unlike
    llama-server there is no per-compute-backend flag to set -- the node advertised
    ``vllm-cuda`` / ``vllm-rocm`` only because the matching vLLM build is present.
    ``--served-model-name`` pins the model id callers address (the Skulk model id),
    decoupled from the on-disk directory path. ``--trust-remote-code`` is added when
    the card permits it (the ModelCard default; required by custom-code HF repos)
    since vLLM's flag defaults off and those models would otherwise fail at startup.
    """
    args = [
        binary,
        "serve",
        str(model_dir),
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.2f}",
        "--tensor-parallel-size",
        "1",
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    return args


def vllm_generation_kwargs(task_params: Any) -> dict[str, Any]:
    """Translate Skulk sampling params into vLLM ``/v1/chat/completions`` fields.

    Distinct from the llama.cpp mapper (``generation_kwargs``): vLLM's OpenAI server
    uses OpenAI/HF parameter names, so the repetition control is ``repetition_penalty``
    (llama.cpp's ``repeat_penalty`` would be silently ignored by vLLM). ``top_k`` /
    ``min_p`` are vLLM sampling extensions passed through by name. Pure, so the
    mapping is unit-testable. Thinking control is layered separately by
    :func:`vllm_reasoning_overrides`; logprobs are rejected before this is called.
    """
    kwargs: dict[str, Any] = {}
    if task_params.max_output_tokens is not None:
        kwargs["max_tokens"] = task_params.max_output_tokens
    if task_params.temperature is not None:
        kwargs["temperature"] = task_params.temperature
    if task_params.top_p is not None:
        kwargs["top_p"] = task_params.top_p
    if task_params.top_k is not None:
        kwargs["top_k"] = task_params.top_k
    if task_params.min_p is not None:
        kwargs["min_p"] = task_params.min_p
    if task_params.repetition_penalty is not None:
        kwargs["repetition_penalty"] = task_params.repetition_penalty
    if task_params.stop is not None:
        kwargs["stop"] = task_params.stop
    if task_params.seed is not None:
        kwargs["seed"] = task_params.seed
    return kwargs


def vllm_reasoning_overrides(task_params: Any) -> dict[str, Any]:
    """Map Skulk's thinking controls onto vLLM request fields.

    vLLM's OpenAI server exposes the same two levers as llama-server:
    ``chat_template_kwargs`` (the model's jinja template reads ``enable_thinking``,
    the Qwen3 / Gemma toggle; a template that ignores it is harmless) and
    ``reasoning_effort`` (OpenAI-style effort for gpt-oss). Without this the sampling
    body carries no thinking control, so ``enable_thinking=False`` would be silently
    ignored and a reasoning model would think on every request. ``"none"`` effort is
    not a valid server value (disabling goes through ``enable_thinking=False``), so
    it is dropped.
    """
    overrides: dict[str, Any] = {}
    enable_thinking = getattr(task_params, "enable_thinking", None)
    if enable_thinking is not None:
        overrides["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    effort = getattr(task_params, "reasoning_effort", None)
    if effort is not None and effort != "none":
        overrides["reasoning_effort"] = effort
    return overrides


def parse_openai_sse_line(line: str) -> _StreamDelta | None:
    """Parse one OpenAI SSE line into a ``_StreamDelta``, or ``None`` to skip it.

    Handles the standard streaming shape vLLM emits: ``data: {json}`` lines whose
    first choice carries a ``delta`` (``content`` and/or ``reasoning_content`` when
    a reasoning parser is configured) plus an optional ``finish_reason``, and the
    terminal ``data: [DONE]``. Returns ``None`` for non-``data:`` lines; ``[DONE]``
    is reported via ``done=True``; malformed JSON or a choice-less payload is
    skipped (``None``) so a stray line never breaks the stream. Pure (no I/O).
    """
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return _StreamDelta("", "", None, done=True)
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(chunk, dict):
        return None
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    raw_delta = choice.get("delta")
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    # Preserve OpenAI's `content_filter` finish reason, which vLLM can emit but the
    # shared llama.cpp `map_finish_reason` collapses to `stop` (llama.cpp never
    # emits it); otherwise a filtered response is misreported as a normal stop.
    raw_finish = choice.get("finish_reason")
    finish = (
        "content_filter"
        if raw_finish == "content_filter"
        else map_finish_reason(raw_finish)
    )
    return _StreamDelta(
        # vLLM has streamed reasoning under `reasoning_content` (DeepSeek-style)
        # and, in newer versions, `reasoning`; accept both so a reasoning model's
        # thinking stream is not silently dropped.
        reasoning=delta.get("reasoning_content") or delta.get("reasoning") or "",
        content=delta.get("content") or "",
        finish=finish,
        done=False,
    )


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL this child when its parent (the runner) dies.

    Runs in the forked child before ``exec`` (``preexec_fn``). Linux-only,
    best-effort: a runner-process crash must never leave an orphaned ``vllm serve``
    holding GPU memory. Any failure is swallowed (the explicit teardown path still
    applies on graceful shutdown).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0)
    except Exception:  # noqa: BLE001 - best-effort; non-Linux or no libc
        pass


class Runner:
    """Single-node served-backend runner that proxies an external ``vllm serve``.

    Lifecycle mirrors the ``llama_server`` runner: it skips the ring
    (``ConnectToGroup`` / ``StartWarmup``), spawns the server on ``LoadModel``, and
    serves ``TextGeneration`` by streaming the server's SSE output back as
    ``ChunkGenerated`` events.
    """

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
        context_token_limit: int | None = None,
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance
        self.context_token_limit = context_token_limit
        self.instance, self.runner_id, self.shard_metadata = (
            bound_instance.instance,
            bound_instance.bound_runner_id,
            bound_instance.bound_shard,
        )
        # vLLM is single-node in this slice: vLLM's own tensor/pipeline parallelism
        # is a later track, so any multi-node placement reaching here is a bug.
        if self.shard_metadata.world_size != 1:
            raise RuntimeError(
                "vllm runner requires single-node placement, got "
                f"world_size={self.shard_metadata.world_size}"
            )
        self.setup_start_time = time.time()
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        # Concurrent-dispatch state. Worker threads mutate the cancel set and the
        # in-flight counter, so both are lock-guarded; never hold both locks at
        # once (no nested acquisition) to keep the ordering trivially deadlock-free.
        self._max_concurrency: int = _max_concurrent_requests()
        self._status_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        # Backpressure: caps SUBMITTED generations to _max_concurrency. A bare
        # ThreadPoolExecutor bounds active threads but has an UNBOUNDED submit
        # queue, so without this the runner would accumulate an unbounded backlog
        # of queued TextGeneration jobs under sustained load. The dispatch loop
        # acquires a permit before submitting and blocks (applying backpressure to
        # the task receiver) when all are held; a finished generation releases one.
        self._dispatch_permits = threading.Semaphore(self._max_concurrency)
        self._inflight: int = 0
        self.server_proc: subprocess.Popen[bytes] | None = None
        self.server_log: Any = None
        self.server_log_path: Path | None = None
        self.base_url: str | None = None
        self.current_status: RunnerStatus = RunnerIdle()
        logger.info("vllm runner created")
        self.update_status(RunnerIdle())

    # --- runner-contract plumbing (mirrors the llama_server runner) ------------

    def update_status(self, status: RunnerStatus) -> None:
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def acknowledge_task(self, task: Task) -> None:
        record_runner_phase(
            "task_submission",
            event="task_acknowledged",
            detail=task.__class__.__name__,
            task_id=task.task_id,
        )
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def _drain_cancellations(self) -> None:
        # Concurrent generation threads all poll cancellation, so serialize the
        # single-consumer cancel pipe and the shared set behind the cancel lock.
        with self._cancel_lock:
            while True:
                try:
                    cancelled = self.cancel_receiver.receive_nowait()
                except WouldBlock:
                    break
                self.cancelled_tasks.add(cancelled)

    def _is_cancelled(self, task_id: TaskId) -> bool:
        self._drain_cancellations()
        return self._was_cancelled(task_id)

    def _was_cancelled(self, task_id: TaskId) -> bool:
        """Whether ``task_id`` is cancelled, WITHOUT draining the pipe.

        Used to classify a finished generation's terminal status: draining here
        could pull in a cancellation for a *different, still-running* task and,
        combined with a stale ``CANCEL_ALL``, misreport this one.
        """
        with self._cancel_lock:
            return (
                task_id in self.cancelled_tasks
                or CANCEL_ALL_TASKS in self.cancelled_tasks
            )

    def main(self) -> None:
        # One thread per in-flight generation. The pool caps ACTIVE threads; the
        # _dispatch_permits semaphore (acquired below before each submit) caps
        # SUBMITTED jobs to the same bound, so the pool's otherwise-unbounded submit
        # queue never accumulates a backlog -- excess load backpressures the task
        # receiver instead.
        pool = ThreadPoolExecutor(
            max_workers=self._max_concurrency, thread_name_prefix="vllm-gen"
        )
        try:
            with self.task_receiver as tasks:
                while True:
                    try:
                        task = tasks.receive_timeout(_LIVENESS_POLL_S)
                    except WouldBlock:
                        # No task within the poll window: verify the vllm serve
                        # subprocess is still alive. Without this a server that
                        # dies BETWEEN requests leaves the runner gossiping Ready
                        # forever while every future request fails.
                        self._ensure_server_alive()
                        continue
                    except (EndOfStream, ClosedResourceError):
                        break
                    if task.task_id in self.seen:
                        logger.warning("repeat task - potential error")
                        continue
                    self.seen.add(task.task_id)
                    match task:
                        case TextGeneration() if isinstance(
                            self.current_status, (RunnerReady, RunnerRunning)
                        ):
                            # Acknowledge acceptance NOW, before any backpressure
                            # block: the supervisor's start_task waits on the ack
                            # before the worker can plan again, so deferring it until
                            # a pool slot frees would stall dispatch of cancellations
                            # / shutdown / other work behind the first over-capacity
                            # request. Ack means "accepted", not "running".
                            self.acknowledge_task(task)
                            # Backpressure: block until a dispatch slot frees so the
                            # runner never accumulates an unbounded backlog in the
                            # pool's submit queue (max_concurrency caps ACTIVE
                            # threads, not the queue). Blocking here stops pulling
                            # from the receiver, backpressuring the master. Wake
                            # periodically while saturated so a dead server is still
                            # caught by the liveness check.
                            while not self._dispatch_permits.acquire(
                                timeout=_LIVENESS_POLL_S
                            ):
                                self._ensure_server_alive()
                            # Dispatch to the pool and loop back to receive the next
                            # task, so N generations run concurrently and vLLM
                            # batches them.
                            self._dispatch_generation(task, pool)
                        case Shutdown():
                            self._handle_shutdown(task, pool)
                            break
                        case _:
                            # Lifecycle (LoadModel) or an out-of-state task: run it
                            # inline on this thread. LoadModel only occurs once,
                            # before any generation, so serial handling is correct.
                            self.send_task_status(task, TaskStatus.Running)
                            self.handle_task(task)
                            # _load_model sets current_status = RunnerReady() by
                            # direct assignment (no broadcast); broadcast it here so
                            # the worker learns the runner is ready to serve -- but
                            # ONLY AFTER the terminal Complete. Order is load-bearing:
                            # RunnerSupervisor._forward_events asserts the runner is
                            # in an active state (Loading/Running/...) when a terminal
                            # task status arrives, so Ready must not precede Complete
                            # (else Loading -> Ready -> Complete trips that assertion
                            # and aborts the forwarder). This matches the serial loop,
                            # which sent Complete then update_status(current_status).
                            self.send_task_status(task, TaskStatus.Complete)
                            self.update_status(self.current_status)
        finally:
            # Drain in-flight generations, then stop the server. Shutdown already
            # cancels them (so the pool drains promptly); this also covers the
            # EndOfStream / crash exits. PR_SET_PDEATHSIG is the SIGKILL backstop.
            pool.shutdown(wait=True)
            self._teardown_server()

    # --- concurrent dispatch --------------------------------------------------

    def _dispatch_generation(self, task: TextGeneration, pool: ThreadPoolExecutor) -> None:
        """Admit a generation and run it on the pool without blocking the loop."""
        # Recover from a stale cluster-wide cancel: with nothing in flight, a
        # lingering CANCEL_ALL must not kill this fresh request. (While requests
        # are in flight it is left set so those observe it.)
        if self._inflight_count() == 0:
            with self._cancel_lock:
                self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self.send_task_status(task, TaskStatus.Running)
        self._note_generation_started()
        try:
            future = pool.submit(self._run_one_generation, task)
        except RuntimeError as exc:
            # The pool rejected the job (already shut down / broken). The
            # done-callback that normally releases the permit and decrements the
            # in-flight count will never run, so undo them here and surface a
            # terminal error rather than leaking a slot and wedging RunnerRunning.
            logger.opt(exception=exc).error("vllm dispatch submit failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.shard_metadata.model_card.model_id,
                        error_message=f"runner could not dispatch generation: {exc}",
                    ),
                )
            )
            self.send_task_status(task, TaskStatus.Failed)
            self._note_generation_finished()
            self._dispatch_permits.release()
            return
        future.add_done_callback(lambda f: self._finish_generation(task, f))

    def _run_one_generation(self, task: TextGeneration) -> None:
        """Pool-worker body: stream one generation. Runs on a worker thread.

        ``_generate`` catches its own errors and surfaces them as an ErrorChunk;
        this outer guard only covers an unexpected escape so a crashed worker
        never swallows the stream silently (its terminal status is still emitted
        by the done-callback).
        """
        try:
            self._generate(task)
        except Exception as exc:  # noqa: BLE001 - defensive; keep the pool alive
            logger.opt(exception=exc).error("vllm generation worker crashed")
            with contextlib.suppress(Exception):
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=ErrorChunk(
                            model=self.shard_metadata.model_card.model_id,
                            error_message=str(exc),
                        ),
                    )
                )

    def _finish_generation(self, task: TextGeneration, future: "Future[None]") -> None:
        """Done-callback: emit the terminal task status and drop the in-flight count.

        Runs on the worker thread once the generation future completes. The
        terminal status classification does NOT drain the cancel pipe (see
        ``_was_cancelled``); ``_note_generation_finished`` flips the runner back to
        Ready when this was the last in-flight generation.
        """
        try:
            was_cancelled = self._was_cancelled(task.task_id)
            self.send_task_status(
                task,
                TaskStatus.Cancelled if was_cancelled else TaskStatus.Complete,
            )
        finally:
            drained_to_idle = self._note_generation_finished()
            # Clear a stale cluster-wide cancel the moment the last in-flight
            # generation drains, not only on the next dispatch: a CANCEL_ALL that
            # arrived while requests were in flight would otherwise linger until a
            # new task happens to arrive (and the dispatch-time clear runs), and a
            # request that arrives before that could be spuriously cancelled. Not
            # under _status_lock (lock ordering); skipped during shutdown, which
            # sets CANCEL_ALL deliberately to break the draining streams.
            if drained_to_idle:
                with self._cancel_lock:
                    if not isinstance(
                        self.current_status, (RunnerShuttingDown, RunnerShutdown)
                    ):
                        self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
            # Release the backpressure slot this generation held (acquired in the
            # dispatch loop), letting a queued task be pulled and submitted.
            self._dispatch_permits.release()

    def _note_generation_started(self) -> None:
        with self._status_lock:
            self._inflight += 1
            # First in-flight generation flips Ready -> Running.
            if self._inflight == 1 and isinstance(self.current_status, RunnerReady):
                self.update_status(RunnerRunning())

    def _note_generation_finished(self) -> bool:
        """Drop the in-flight count; return True if this drained to idle (0)."""
        with self._status_lock:
            self._inflight = max(0, self._inflight - 1)
            # Last in-flight generation flips Running -> Ready. A shutdown in
            # progress (ShuttingDown/Shutdown) is left alone.
            if self._inflight == 0 and isinstance(self.current_status, RunnerRunning):
                self.update_status(RunnerReady())
            return self._inflight == 0

    def _inflight_count(self) -> int:
        with self._status_lock:
            return self._inflight

    def _handle_shutdown(self, task: Task, pool: ThreadPoolExecutor) -> None:
        """Cancel in-flight generations, drain the pool, then tear down the server."""
        logger.info("vllm runner shutting down")
        # Emit Running before the terminal Complete: the worker's task-lifecycle
        # contract expects Running -> Complete for every task, shutdown included
        # (the serial loop sent it around handle_task; the concurrent loop must
        # keep that ordering for shutdown too).
        self.send_task_status(task, TaskStatus.Running)
        record_runner_phase(
            "shutdown_cleanup",
            event="runner_shutdown_requested",
            task_id=task.task_id,
        )
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        # Break every in-flight stream: their loops poll _is_cancelled.
        with self._cancel_lock:
            self.cancelled_tasks.add(CANCEL_ALL_TASKS)
        pool.shutdown(wait=True)
        self._teardown_server()
        record_runner_phase(
            "shutdown_cleanup",
            event="server_teardown_complete",
            task_id=task.task_id,
        )
        self.current_status = RunnerShutdown()
        self.send_task_status(task, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def _ensure_server_alive(self) -> None:
        """Raise if the spawned ``vllm serve`` exited behind our back.

        Raising kills the runner process; the supervisor observes the crash and
        the peer-failure cascade tears the instance down instead of leaving a
        wedged Ready runner.
        """
        proc = self.server_proc
        if proc is None or isinstance(
            self.current_status, (RunnerShuttingDown, RunnerShutdown)
        ):
            return
        if proc.poll() is not None:
            record_runner_phase(
                "error",
                event="server_exited",
                detail=f"vllm serve exited unexpectedly (code {proc.returncode})",
            )
            raise RuntimeError(
                f"vllm serve exited unexpectedly (code {proc.returncode}); "
                f"log tail:\n{self._server_log_tail()}"
            )

    def handle_task(self, task: Task) -> None:
        # TextGeneration and Shutdown are handled directly by the concurrent
        # dispatch loop in main(); this serves the inline lifecycle path (LoadModel).
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case _:
                raise RuntimeError(
                    f"vllm runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )

    # --- model load: spawn + health-check the server --------------------------

    def _load_model(self, task: Task) -> None:
        self.update_status(RunnerLoading())
        self.acknowledge_task(task)

        card = self.shard_metadata.model_card
        model_id = card.model_id
        model_dir = build_model_path(ModelId(model_id))
        n_ctx = serving_n_ctx(self.context_token_limit, logits_all=False)
        try:
            with runner_phase(
                "load_model",
                detail="spawn_vllm_serve",
                task_id=task.task_id,
                attrs={"model_dir": model_dir.name, "n_ctx": n_ctx},
            ):
                self._spawn_server(model_dir, str(model_id), n_ctx)
                self._await_health()
        except Exception:
            self._teardown_server()
            raise
        self.current_status = RunnerReady()
        record_runner_phase("idle", event="runner_ready", task_id=task.task_id)
        logger.info(
            f"vllm runner ready in {time.time() - self.setup_start_time:.1f}s "
            f"(url={self.base_url})"
        )

    def _spawn_server(
        self, model_dir: Path, served_model_name: str, n_ctx: int
    ) -> None:
        binary = os.environ.get(VLLM_BIN_ENV, "").strip()
        if not binary or not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            # Validate up front (like the llama_server runner) so a misconfigured or
            # vanished-since-probe binary surfaces as a clear runner error rather
            # than a bare FileNotFoundError/PermissionError out of subprocess.Popen.
            raise RuntimeError(
                f"{VLLM_BIN_ENV}={binary!r} is not an existing executable; the vllm "
                "runner cannot spawn a server. This node should not have been a "
                "placement candidate for a vLLM card."
            )
        host = "127.0.0.1"
        port = self._pick_port()
        self.base_url = f"http://{host}:{port}"
        args = build_vllm_serve_args(
            binary,
            model_dir,
            served_model_name,
            host,
            port,
            n_ctx,
            _gpu_memory_utilization(),
            self.shard_metadata.model_card.trust_remote_code,
        )
        # Deterministic log path keyed by runner_id (matching llama_server), so
        # postmortem debugging is easy and restarts truncate rather than pile up
        # random temp files.
        self.server_log_path = (
            Path(tempfile.gettempdir()) / f"skulk-vllm-serve-{self.runner_id}.log"
        )
        self.server_log = open(self.server_log_path, "wb")  # noqa: SIM115
        logger.info(f"spawning vllm serve: {' '.join(args)} (log={self.server_log_path})")
        self.server_proc = subprocess.Popen(
            args,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            preexec_fn=_set_pdeathsig if os.name == "posix" else None,
        )

    def _pick_port(self) -> int:
        """Pick a free ephemeral port for the server, avoiding the API port."""
        for _ in range(30):
            port = random.randint(49153, 65535)
            if port == 52415:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise RuntimeError("could not find a free port for vllm serve")

    def _await_health(self) -> None:
        assert self.server_proc is not None and self.base_url is not None
        deadline = time.time() + _HEALTH_DEADLINE_S
        with httpx.Client(timeout=5.0) as client:
            while time.time() < deadline:
                if self.server_proc.poll() is not None:
                    raise RuntimeError(
                        "vllm serve exited during startup (code "
                        f"{self.server_proc.returncode}); log tail:\n"
                        f"{self._server_log_tail()}"
                    )
                try:
                    # vLLM's /health returns 200 with an empty body once the engine
                    # is up (no JSON status field, unlike llama-server).
                    if client.get(f"{self.base_url}/health").status_code == 200:
                        return
                except Exception:  # noqa: BLE001 - not up yet; keep polling
                    pass
                time.sleep(2)
        raise RuntimeError(
            f"vllm serve did not become healthy within {_HEALTH_DEADLINE_S:.0f}s; "
            f"log tail:\n{self._server_log_tail()}"
        )

    def _server_log_tail(self, lines: int = 30) -> str:
        if self.server_log_path is None or not self.server_log_path.exists():
            return "(no log)"
        try:
            text = self.server_log_path.read_text(errors="replace")
        except OSError:
            return "(log unreadable)"
        return "\n".join(text.splitlines()[-lines:])

    def _teardown_server(self) -> None:
        proc = self.server_proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            self.server_proc = None
        if self.server_log is not None:
            with contextlib.suppress(Exception):
                self.server_log.close()
            self.server_log = None

    # --- generation: proxy the server's OpenAI streaming API ------------------

    def _generate(self, task: Task) -> None:
        # Runs on a pool worker thread. Runner status (Running/Ready) is owned by
        # the dispatch loop's in-flight counter, not this per-request path, so a
        # finishing generation never flips the runner to Ready while others run.
        # The task was already acknowledged at acceptance in the dispatch loop
        # (before backpressure), so it is not re-acknowledged here.
        assert isinstance(task, TextGeneration)
        assert self.base_url is not None

        model_id = self.shard_metadata.model_card.model_id
        command_id = task.command_id
        body: dict[str, Any] = vllm_generation_kwargs(task.task_params)
        # vLLM's OpenAI server requires `model` in the request body (unlike
        # llama-server, which serves one model and ignores it); it must match the
        # server's --served-model-name, which the runner sets to the Skulk model id.
        body["model"] = str(model_id)
        body["messages"] = messages_for_llama(task.task_params)
        # Forward thinking control (enable_thinking / reasoning_effort) to vLLM;
        # without it a reasoning model thinks on every request regardless of the
        # request's toggle.
        body.update(vllm_reasoning_overrides(task.task_params))

        record_runner_phase(
            "task_submission",
            event="submit_text_generation",
            task_id=task.task_id,
            command_id=str(command_id),
            attrs={"tools": bool(task.task_params.tools)},
        )
        try:
            # Tool calling and per-token logprobs are out of scope for this slice:
            # the tool-call round trip and logprob surfacing over the SSE proxy are
            # follow-ups. Fail loud rather than silently drop them (matching the
            # llama_server runner's #385 no-silent-empty contract).
            if task.task_params.tools:
                raise RuntimeError(
                    "Tool calling is not yet supported on the vllm engine; retry "
                    "without tools or use a llama_cpp/llama_server card."
                )
            if wants_logprobs(
                task.task_params.logprobs, task.task_params.top_logprobs
            ):
                body.pop("logprobs", None)
                body.pop("top_logprobs", None)
                raise RuntimeError(
                    "Per-token logprobs are not supported on the vllm engine: the "
                    "OpenAI SSE proxy does not surface them. Retry without "
                    "logprobs/top_logprobs."
                )
            record_runner_phase(
                "decode_stream",
                event="request_started",
                task_id=task.task_id,
                command_id=str(command_id),
            )
            self._generate_streaming(task, body, model_id, command_id)
        except Exception as exc:  # noqa: BLE001 - surface as an ErrorChunk
            record_runner_phase(
                "error",
                event="generation_failed",
                detail=f"{type(exc).__name__}: {exc}",
                task_id=task.task_id,
                command_id=str(command_id),
            )
            logger.opt(exception=exc).warning("vllm generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(model=model_id, error_message=str(exc)),
                )
            )
        else:
            # Read the shared cancel set through the lock-guarded helper: generations
            # run concurrently, so an unlocked membership read here races the pool
            # workers mutating cancelled_tasks.
            was_cancelled = self._was_cancelled(task.task_id)
            record_runner_phase(
                "cancel_observed" if was_cancelled else "completion",
                event="generation_finished",
                task_id=task.task_id,
                command_id=str(command_id),
            )
        # Status is NOT flipped here: the dispatch loop returns the runner to
        # Ready only when the LAST in-flight generation drains (see
        # _note_generation_finished), so a peer generation still streaming keeps
        # the runner Running.

    def _generate_streaming(
        self,
        task: TextGeneration,
        body: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        body["stream"] = True
        assert self.base_url is not None
        clock = StreamStatsClock()

        def final_stats() -> GenerationStats:
            # Peak memory always comes from the server child (weights + KV live
            # there), never this proxy. Prompt count is unknowable from the SSE
            # stream, reported as 0 (a zero reads as "unmeasured").
            return clock.stats(
                prompt_tokens=0, generation_tokens=clock.pieces
            ).model_copy(update={"peak_memory_usage": self._server_peak_memory()})

        emitted_finish = False
        # No read timeout: generation can pause between tokens on a busy GPU. The
        # connection is closed (aborting server generation) when we break out.
        timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=None)
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=body
            ) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if self._is_cancelled(task.task_id):
                    logger.info(f"vllm generation cancelled: {task.task_id}")
                    break
                delta = parse_openai_sse_line(line)
                if delta is None:
                    continue
                if delta.done:
                    break
                if delta.reasoning or delta.content:
                    # One SSE delta per generated token piece.
                    clock.mark_piece()
                if delta.reasoning:
                    self._send_token(
                        command_id, model_id, delta.reasoning, is_thinking=True
                    )
                if delta.content:
                    self._send_token(command_id, model_id, delta.content)
                if delta.finish is not None:
                    self._send_token(
                        command_id,
                        model_id,
                        "",
                        finish_reason=delta.finish,
                        stats=final_stats(),
                    )
                    emitted_finish = True
        # Guarantee a terminal chunk so the consumer's stream closes even if the
        # server ended without an explicit finish_reason.
        if not emitted_finish and not self._is_cancelled(task.task_id):
            self._send_token(
                command_id, model_id, "", finish_reason="stop", stats=final_stats()
            )

    def _server_peak_memory(self) -> Memory:
        """Peak RSS of the ``vllm serve`` child, or zero when unmeasurable.

        The weights and KV cache live in the external server process, so the
        proxy's own RSS would misattribute memory in telemetry. Zero means
        "unmeasured" (non-Linux, or the child already exited).
        """
        proc = self.server_proc
        if proc is None:
            return Memory.from_bytes(0)
        return subprocess_peak_memory(proc.pid) or Memory.from_bytes(0)

    def _send_token(
        self,
        command_id: CommandId,
        model_id: ModelId,
        text: str,
        *,
        is_thinking: bool = False,
        finish_reason: Any = None,
        stats: GenerationStats | None = None,
    ) -> None:
        """Emit one TokenChunk; skip empty non-terminal chunks."""
        if not text and finish_reason is None:
            return
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=TokenChunk(
                    model=model_id,
                    text=text,
                    token_id=-1,  # the OpenAI proxy stream does not expose ids
                    usage=None,
                    finish_reason=finish_reason,
                    is_thinking=is_thinking,
                    stats=stats,
                ),
            )
        )
