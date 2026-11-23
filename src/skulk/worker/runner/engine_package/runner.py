# pyright: reportAny=false
"""Generic runner for an installed engine package.

One runner serves any packaged engine. It holds no engine name: it starts the
package's adapter, asks it to load the placement's model with the admitted
settings stated outright, and then proxies text generation straight to the
loopback endpoint the load returned. Lifecycle (load, health, cancel, unload,
stop) travels the adapter's control channel; tokens never do.

The data endpoint speaks one pinned chat dialect, version 1:

* ``POST /v1/chat/completions`` with ``stream`` true and
  ``stream_options.include_usage`` true.
* Request fields: ``model``, ``messages``, ``max_tokens``, ``temperature``,
  ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``, ``stop``, ``seed``
  and, when the caller set it, ``chat_template_kwargs.enable_thinking``.
* Response: OpenAI server-sent events. Each event carries the response ``id``;
  a delta carries ``content`` and may carry ``reasoning_content``; the final
  event before ``[DONE]`` carries ``usage`` with an empty ``choices`` list.
* A refusal is an HTTP error whose JSON body is ``{"error": {"message",
  "type", "code"}}``. A 400 whose ``code`` or ``type`` is
  ``context_length_exceeded`` is a context overflow. OpenAI puts that name in
  ``code`` beside the type ``invalid_request_error``; both places are read.

The dialect carries no tools, images or per-token log probabilities, so a
request asking for one is answered with a clear error instead of a degraded
reply. An engine that speaks differently is translated by its own adapter.

A load the engine refuses, or one that took settings other than the admitted
ones, fails with ``ENGINE_LOAD_REFUSED_FAILURE_MARKER`` so the worker gives the
instance up without relaunching: loading again on this node and build repeats
the answer.
"""

import contextlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx

from skulk.api.types import GenerationStats
from skulk.engines.adapter import (
    AdapterFailedError,
    AdapterRefusedError,
    AdapterSession,
)
from skulk.engines.launch import (
    ENGINE_STATE_DIR,
    adapter_session,
    engine_state_directory,
    plan_adapter_launch,
)
from skulk.provisioning.engine_install import InstalledEngine
from skulk.shared.backends import engine_load_refusal
from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX
from skulk.shared.models.model_cards import card_serves_speech
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
    LoadModel,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from skulk.shared.types.worker.shards import RpcDonorShardMetadata
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.diagnostics import record_runner_phase, runner_phase
from skulk.worker.runner.engine_package.request import (
    PARALLEL_STREAMS,
    package_load_request,
)
from skulk.worker.runner.generation_stats import StreamStatsClock
from skulk.worker.runner.llama_cpp.runner import (
    messages_for_llama,
    wants_logprobs,
)
from skulk.worker.runner.llm_inference.scaffolding_scrub import (
    StreamingScaffoldingScrub,
)
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch
from skulk.worker.runner.vllm.runner import parse_openai_sse_line

_HEALTH_DEADLINE_SECONDS: Final = 10.0
_CANCEL_DEADLINE_SECONDS: Final = 10.0
_UNLOAD_DEADLINE_SECONDS: Final = 20.0
_MAX_RESPONSE_ID_CHARS: Final = 128


def chat_request_body(
    params: TextGenerationTaskParams, model_id: ModelId
) -> dict[str, Any]:
    """Build a dialect version 1 chat request from Skulk's task parameters.

    Only fields the caller set are sent, so an engine's own defaults apply to
    the rest. Pure, so the mapping is testable without an engine.
    """
    body: dict[str, Any] = {
        "model": str(model_id),
        "messages": messages_for_llama(params),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    optional: dict[str, Any] = {
        "max_tokens": params.max_output_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "stop": params.stop,
        "seed": params.seed,
    }
    body.update({name: value for name, value in optional.items() if value is not None})
    if params.enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": params.enable_thinking}
    return body


def unsupported_request_reason(params: TextGenerationTaskParams) -> str | None:
    """Say why a request is outside the pinned dialect, or return ``None``."""
    if params.tools:
        return (
            "This model is served by an engine package, whose chat dialect "
            "carries no tool calls. Retry without tools."
        )
    if params.images or params.image_count:
        return (
            "This model is served by an engine package, whose chat dialect "
            "carries no images. Retry with text only."
        )
    if wants_logprobs(params.logprobs, params.top_logprobs):
        return (
            "Per-token logprobs are not supported by an engine package: its "
            "chat dialect does not carry them. Retry without logprobs."
        )
    return None


def response_id(line: str) -> str | None:
    """Read the response ``id`` from one server-sent event line, if it has one."""
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    found = cast("dict[str, object]", event).get("id")
    if isinstance(found, str) and 0 < len(found) <= _MAX_RESPONSE_ID_CHARS:
        return found
    return None


def raise_for_engine_status(response: httpx.Response) -> None:
    """Raise for an HTTP refusal, keeping the engine's own bounded reason.

    Reads the error body before the streamed connection closes. A context
    overflow is prefixed with the API's sentinel so the caller receives a 400.
    """
    if response.is_success:
        return
    response.read()
    try:
        body = cast(object, response.json())
    except ValueError:
        body = None
    detail = (
        cast("dict[str, object]", body).get("error") if isinstance(body, dict) else None
    )
    kind = ""
    if isinstance(detail, dict):
        fields = cast("dict[str, object]", detail)
        message = str(fields.get("message") or "")
        kind = (
            "context_length_exceeded"
            if "context_length_exceeded" in (fields.get("code"), fields.get("type"))
            else ""
        )
    else:
        message = detail if isinstance(detail, str) else response.text
    message = " ".join(message.split())[:300]
    explanation = f"the engine answered {response.status_code}: " + (
        message or response.reason_phrase
    )
    if response.status_code == 400 and kind == "context_length_exceeded":
        explanation = f"{CONTEXT_LENGTH_EXCEEDED_PREFIX} {explanation}"
    raise RuntimeError(explanation)


def _count(usage: dict[str, Any] | None, key: str) -> int | None:
    """A non-negative integer token count from a usage object, or ``None``."""
    value = None if usage is None else usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def unsupported_placement_reason(bound_instance: BoundInstance) -> str | None:
    """Say why a placement is outside what a packaged engine serves, if it is.

    Packaged engines serve single-node text generation. Any other kind of
    placement stamped onto one is refused before an adapter is started.
    """
    shard = bound_instance.bound_shard
    if shard.world_size != 1 or isinstance(shard, RpcDonorShardMetadata):
        return "an engine package serves a model on one node"
    kinds = (
        ("source separation", bound_instance.is_separation_model),
        ("music", bound_instance.is_music_model),
        ("video", bound_instance.is_video_model),
        ("image", bound_instance.is_image_model),
        ("embedding", bound_instance.is_embedding_model),
        ("speech", card_serves_speech(shard.model_card)),
    )
    for name, matches in kinds:
        if matches:
            return f"an engine package serves text generation, not {name}"
    return None


class Runner(ServedConcurrentDispatch):
    """Serve one text placement on an installed engine package.

    Shares the served engines' dispatch loop: generation is admitted against
    ``PARALLEL_STREAMS``, lifecycle tasks run on the dispatch thread, and an
    idle poll asks the adapter for the instance's health so a dead engine ends
    the runner instead of leaving it advertised as ready.
    """

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
        installed: InstalledEngine,
        context_token_limit: int | None = None,
        *,
        state_root: Path = ENGINE_STATE_DIR,
    ) -> None:
        """Prepare a runner for one placement. Nothing is started until ``LoadModel``.

        Raises:
            RuntimeError: Marked as a refused load when the placement is of a
                kind a packaged engine does not serve.
        """
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
        unsupported = unsupported_placement_reason(bound_instance)
        if unsupported is not None:
            raise RuntimeError(engine_load_refusal("unsupported_task", unsupported))
        self._installed = installed
        self._instance_key = str(self.instance.instance_id)
        # One directory per instance, so two instances of one build never
        # share a home, a temporary directory or a log.
        self._state_directory = (
            engine_state_directory(installed.digest, root=state_root)
            / "instances"
            / self._instance_key
        )
        self.setup_start_time = time.time()
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self._init_concurrent_dispatch(PARALLEL_STREAMS, "engine-package-gen")
        self._stack = contextlib.ExitStack()
        self._session: AdapterSession | None = None
        # The control channel carries one exchange at a time. Health comes
        # from the dispatch thread and cancel from generation threads.
        self._control_lock = threading.Lock()
        self.base_url: str | None = None
        self.current_status: RunnerStatus = RunnerIdle()
        logger.info(
            f"engine package runner created for {installed.claims.engine_id} "
            f"({installed.claims.lane}, build {installed.digest[:12]})"
        )
        self.update_status(RunnerIdle())

    # --- runner-contract plumbing ---------------------------------------------

    def update_status(self, status: RunnerStatus) -> None:
        """Record and broadcast the runner's status."""
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        """Broadcast one task's status."""
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def acknowledge_task(self, task: Task) -> None:
        """Tell the worker this runner has taken a task."""
        record_runner_phase(
            "task_submission",
            event="task_acknowledged",
            detail=task.__class__.__name__,
            task_id=task.task_id,
        )
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self) -> None:
        """Run the shared dispatch loop until shutdown."""
        self.run_dispatch_loop()

    def handle_task(self, task: Task) -> None:
        """Serve the inline lifecycle path; generation and shutdown are the loop's."""
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case _:
                raise RuntimeError(
                    "engine package runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )

    # --- load, health, teardown -----------------------------------------------

    def _load_model(self, task: Task) -> None:
        self.update_status(RunnerLoading())
        self.acknowledge_task(task)
        model, settings = package_load_request(
            self.shard_metadata, self.context_token_limit
        )
        try:
            with runner_phase(
                "load_model",
                detail="engine_package_load",
                task_id=task.task_id,
                attrs={
                    "engine": self._installed.claims.engine_id,
                    "n_ctx": settings.context_tokens,
                },
            ):
                launch = plan_adapter_launch(
                    self._installed, state_directory=self._state_directory
                )
                session = self._stack.enter_context(adapter_session(launch))
                self._session = session
                try:
                    loaded = session.load(
                        instance_id=self._instance_key, model=model, settings=settings
                    )
                except AdapterRefusedError as refused:
                    raise RuntimeError(
                        engine_load_refusal(refused.code, refused.reason)
                    ) from None
                if loaded.applied != settings:
                    # The engine sized itself differently from what admission
                    # counted. Serving on would overbook or under-serve.
                    raise RuntimeError(
                        engine_load_refusal(
                            "settings_changed",
                            f"the engine took {loaded.applied.model_dump()} where "
                            f"{settings.model_dump()} was admitted",
                        )
                    )
        except Exception:
            self._teardown_server()
            raise
        self.base_url = f"http://127.0.0.1:{loaded.endpoint_port}"
        self.current_status = RunnerReady()
        record_runner_phase("idle", event="runner_ready", task_id=task.task_id)
        logger.info(
            "engine package runner ready in "
            f"{time.time() - self.setup_start_time:.1f}s (url={self.base_url})"
        )

    def _ensure_server_alive(self) -> None:
        """Raise when the adapter or its engine is no longer serving.

        Raising ends the runner process; the supervisor reports the crash
        instead of leaving a ready runner in front of a dead engine.
        """
        session = self._session
        if session is None or isinstance(
            self.current_status, (RunnerShuttingDown, RunnerShutdown)
        ):
            return
        try:
            with self._control_lock:
                health = session.health(
                    instance_id=self._instance_key,
                    deadline_seconds=_HEALTH_DEADLINE_SECONDS,
                )
        except (AdapterFailedError, AdapterRefusedError) as error:
            record_runner_phase("error", event="server_exited", detail=str(error))
            raise RuntimeError(
                f"engine package adapter failed its health check: {error}"
            ) from None
        if not health.ready:
            record_runner_phase("error", event="server_exited", detail="not ready")
            raise RuntimeError("engine package reports its model is no longer serving")

    def _teardown_server(self) -> None:
        """Unload, stop the adapter's whole process group, and clear instance state."""
        session, self._session = self._session, None
        self.base_url = None
        if session is not None and session.alive:
            with (
                contextlib.suppress(AdapterFailedError, AdapterRefusedError),
                self._control_lock,
            ):
                session.unload(
                    instance_id=self._instance_key,
                    deadline_seconds=_UNLOAD_DEADLINE_SECONDS,
                )
        stack, self._stack = self._stack, contextlib.ExitStack()
        try:
            stack.close()
        except Exception as error:  # noqa: BLE001 - reported; teardown continues
            # An unconfirmed process-group drain keeps its leader unreaped and
            # is the one teardown failure an operator must see.
            logger.opt(exception=error).error(
                "engine package adapter did not stop cleanly"
            )
            return
        # A stopped instance keeps nothing; a failed stop keeps its log.
        shutil.rmtree(self._state_directory, ignore_errors=True)

    # --- generation -----------------------------------------------------------

    def _generate(self, task: Task) -> None:
        # Runs on a pool worker thread; runner status is the dispatch loop's.
        assert isinstance(task, TextGeneration)
        model_id = self.shard_metadata.model_card.model_id
        command_id = task.command_id
        record_runner_phase(
            "task_submission",
            event="submit_text_generation",
            task_id=task.task_id,
            command_id=str(command_id),
        )
        try:
            unsupported = unsupported_request_reason(task.task_params)
            if unsupported is not None:
                raise RuntimeError(unsupported)
            record_runner_phase(
                "decode_stream",
                event="request_started",
                task_id=task.task_id,
                command_id=str(command_id),
            )
            self._stream(task, chat_request_body(task.task_params, model_id))
        except Exception as error:  # noqa: BLE001 - surfaced as an ErrorChunk
            record_runner_phase(
                "error",
                event="generation_failed",
                detail=f"{type(error).__name__}: {error}",
                task_id=task.task_id,
                command_id=str(command_id),
            )
            logger.opt(exception=error).warning("engine package generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(model=model_id, error_message=str(error)),
                )
            )
        else:
            record_runner_phase(
                "cancel_observed"
                if self._was_cancelled(task.task_id)
                else "completion",
                event="generation_finished",
                task_id=task.task_id,
                command_id=str(command_id),
            )

    def _stream(self, task: TextGeneration, body: dict[str, Any]) -> None:
        base_url = self.base_url
        if base_url is None:
            raise RuntimeError("the engine package is not loaded")
        model_id = self.shard_metadata.model_card.model_id
        command_id = task.command_id
        clock = StreamStatsClock()
        admission_in_flight = self._admission_concurrency(task.task_id)
        last_usage: dict[str, Any] | None = None
        finish_reason: Literal["stop", "length", "content_filter"] | None = None
        engine_request: str | None = None
        # The dialect has no tools, so a model writing a call anyway would
        # leak its markers as content.
        scrub = StreamingScaffoldingScrub()
        # No read timeout: generation can pause between tokens under load.
        timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=None)
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream(
                "POST", f"{base_url}/v1/chat/completions", json=body
            ) as response,
        ):
            raise_for_engine_status(response)
            for line in response.iter_lines():
                if engine_request is None:
                    engine_request = response_id(line)
                if self._is_cancelled(task.task_id):
                    logger.info(f"engine package generation cancelled: {task.task_id}")
                    # Cancel is a control message: the engine stops the
                    # request and releases its slot before the stream closes.
                    self._cancel_at_engine(engine_request)
                    break
                delta = parse_openai_sse_line(line)
                if delta is None:
                    continue
                if delta.done:
                    break
                if delta.usage is not None:
                    last_usage = delta.usage
                if delta.reasoning or delta.content:
                    clock.mark_piece()
                if delta.reasoning:
                    self._send_token(
                        command_id, model_id, delta.reasoning, is_thinking=True
                    )
                if delta.content:
                    emitted = scrub.feed(delta.content)
                    if emitted:
                        self._send_token(command_id, model_id, emitted)
                if delta.finish is not None:
                    # Usage arrives after the finish event; the terminal chunk
                    # waits for it so its stats are complete.
                    finish_reason = delta.finish
        if self._is_cancelled(task.task_id):
            return
        tail = scrub.flush()
        if tail:
            self._send_token(command_id, model_id, tail)
        self._send_token(
            command_id,
            model_id,
            "",
            finish_reason=finish_reason or "stop",
            stats=self._final_stats(clock, last_usage, admission_in_flight),
        )

    def _final_stats(
        self,
        clock: StreamStatsClock,
        usage: dict[str, Any] | None,
        admission_in_flight: int,
    ) -> GenerationStats:
        """Build a generation's stats from the engine's usage and the stream clock.

        The prompt rate covers only newly processed tokens, so a cached prefix
        cannot inflate it, while the reported prompt count stays the request's
        true size. Peak memory is reported as zero, meaning unmeasured: the
        weights live in the engine process, which the adapter owns.
        """
        prompt_total = _count(usage, "prompt_tokens") or 0
        completion = _count(usage, "completion_tokens")
        processed = prompt_total
        details = None if usage is None else usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = _count(cast("dict[str, Any]", details), "cached_tokens")
            if cached is not None and 0 < cached <= prompt_total:
                processed = prompt_total - cached
        base = clock.stats(
            prompt_tokens=processed,
            generation_tokens=completion if completion is not None else clock.pieces,
        ).model_copy(
            update={
                "prompt_tokens": prompt_total,
                "peak_memory_usage": Memory.from_bytes(0),
            }
        )
        return self.stamp_runner_stats(base, admission_in_flight)

    def _cancel_at_engine(self, engine_request: str | None) -> None:
        """Ask the adapter to cancel one request at the engine, if it has started."""
        session = self._session
        if engine_request is None or session is None:
            return
        # A refusal means the engine no longer knows the request; a failed
        # channel is found by the next health check.
        with (
            contextlib.suppress(AdapterFailedError, AdapterRefusedError),
            self._control_lock,
        ):
            session.cancel(
                instance_id=self._instance_key,
                request_id=engine_request,
                deadline_seconds=_CANCEL_DEADLINE_SECONDS,
            )

    def _send_token(
        self,
        command_id: CommandId,
        model_id: ModelId,
        text: str,
        *,
        is_thinking: bool = False,
        finish_reason: Literal["stop", "length", "content_filter"] | None = None,
        stats: GenerationStats | None = None,
    ) -> None:
        """Emit one token chunk; an empty chunk is sent only as the terminal one."""
        if not text and finish_reason is None:
            return
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=TokenChunk(
                    model=model_id,
                    text=text,
                    token_id=-1,  # the proxied stream does not expose token ids
                    usage=None,
                    finish_reason=finish_reason,
                    is_thinking=is_thinking,
                    stats=stats,
                ),
            )
        )
