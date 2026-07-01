# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Served-backend text-generation runner: launches and proxies ``llama-server``.

Unlike the in-process ``llama_cpp`` runner (which loads the GGUF via
``llama-cpp-python`` and calls ``create_chat_completion``), this runner launches
an external ``llama-server`` subprocess pointed at the staged GGUF and proxies its
OpenAI-compatible HTTP API. That is the only way to reach llama.cpp's native
multi-token-prediction speculative decoding (``--spec-type draft-mtp``): the MTP
orchestration lives in the server application (``tools/server``), not in the
``libllama`` C API or the Python binding, so it cannot be driven in-process.

This is the first *served* engine. Its shape (managed inference server + OpenAI
proxy) is deliberately generic so vLLM and other OpenAI-compatible servers can
become additional served backends without new runner architecture.

Single-node only (no ring / ConnectToGroup / warmup), mirroring the in-process
llama.cpp and embeddings runners. Linux-oriented: the subprocess is reaped on
parent death via ``PR_SET_PDEATHSIG`` so a runner crash never orphans a server
holding GPU memory. Per-request cancellation aborts the proxied HTTP connection
(which stops server-side generation); ``SIGTERM`` is for instance teardown of the
whole server, not a single request. The server emits structured
``reasoning_content`` and ``tool_calls`` itself, so the in-process text parsers
(harmony / think / tool) are not used here.
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
import time
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import httpx
from anyio import WouldBlock

from skulk.shared.backends import LLAMA_SERVER_BIN_ENV
from skulk.shared.models.model_cards import OutputParserType
from skulk.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
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
from skulk.worker.runner.llama_cpp.runner import (
    generation_kwargs,
    map_finish_reason,
    messages_for_llama,
    select_gguf_file,
    serving_n_ctx,
    tool_calls_from_message,
    wants_logprobs,
)
from skulk.worker.runner.llama_server.channel_text_parser import (
    GemmaChannelTextParser,
)

# Card ``served_spec_type`` value -> the ``llama-server --spec-type`` token.
# ``draft_mtp`` uses the model's own built-in MTP heads (no draft model needed).
_SPEC_TYPE_FLAG: Final[dict[str, str]] = {
    "draft_mtp": "draft-mtp",
    "draft_eagle3": "draft-eagle3",
    "draft_simple": "draft-simple",
    "ngram": "ngram-cache",
}

# Served spec modes that REQUIRE a separate `--model-draft` GGUF. ``draft_mtp`` is
# optional (Qwen/DeepSeek/GLM bake the heads into the base GGUF; Gemma 4 instead
# supplies its assistant as a draft), and ``ngram`` needs no model at all.
_DRAFT_MODEL_REQUIRED: Final[frozenset[str]] = frozenset(
    {"draft_simple", "draft_eagle3"}
)


def _force_no_spec() -> bool:
    """True if this node is configured to serve without speculative decoding.

    ``SKULK_LLAMA_SERVER_FORCE_NO_SPEC`` makes the served runner ignore a card's
    ``served_spec_type`` and launch ``llama-server`` without any ``--spec-type``
    flags, so the same GGUF serves as a plain-decode baseline. Intended for an
    MTP on-vs-off throughput comparison and for debugging a misbehaving spec
    pairing; unset in normal operation.
    """
    return os.environ.get("SKULK_LLAMA_SERVER_FORCE_NO_SPEC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _draft_model_args(runtime: Any, spec_type: str) -> list[str]:
    """Resolve the ``--model-draft`` args for a served spec mode.

    When the card declares a draft GGUF (``served_spec_draft_repo`` +
    ``served_spec_draft_file``), resolve its on-disk path and return
    ``["--model-draft", path]`` (Gemma 4 draft_mtp, draft_simple, draft_eagle3).
    Modes in ``_DRAFT_MODEL_REQUIRED`` raise loudly when no draft is configured;
    ``draft_mtp`` without a draft is fine (built-in heads), and ``ngram`` needs
    none. Returns ``[]`` when no draft applies. Pure except for the on-disk path
    resolution, so the validation branches are unit-testable.
    """
    draft_repo = getattr(runtime, "served_spec_draft_repo", None) if runtime else None
    draft_file = getattr(runtime, "served_spec_draft_file", None) if runtime else None
    if draft_repo:
        if not draft_file:
            raise RuntimeError(
                "served_spec_draft_repo is set but served_spec_draft_file is "
                "missing; both are required to pass --model-draft"
            )
        from skulk.download.download_utils import build_model_path

        draft_dir = build_model_path(ModelId(draft_repo))
        draft_path = (draft_dir / draft_file).resolve()
        if not draft_path.is_file() or not draft_path.is_relative_to(
            draft_dir.resolve()
        ):
            raise RuntimeError(
                f"served draft GGUF {draft_file!r} not found under {draft_dir}"
            )
        return ["--model-draft", str(draft_path)]
    if spec_type in _DRAFT_MODEL_REQUIRED:
        raise RuntimeError(
            f"served_spec_type={spec_type!r} requires a draft model; set "
            "served_spec_draft_repo + served_spec_draft_file on the card"
        )
    return []


def _model_declares_reasoning(card: Any) -> bool:
    """Whether the card advertises a reasoning/thinking capability.

    Drives ``--reasoning-format``: a reasoning model keeps llama-server's default
    (``auto``) so thoughts land in ``message.reasoning_content`` (which the runner
    flags as ``is_thinking``); a non-reasoning model is served with
    ``--reasoning-format none`` so all output stays in ``message.content``.
    Without that, llama-server's ``auto`` can extract a plain model's prose into
    ``reasoning_content`` (observed with Gemma 4 served via ``--jinja``), leaving
    ``message.content`` empty for the client. Detection mirrors the capability
    spine: an explicit ``reasoning`` card section or a ``thinking`` capability.
    """
    if getattr(card, "reasoning", None) is not None:
        return True
    return "thinking" in (getattr(card, "capabilities", None) or [])


def reasoning_request_overrides(task_params: Any) -> dict[str, Any]:
    """Map Skulk's thinking controls onto llama-server request fields.

    ``generation_kwargs`` carries sampling params but NOT thinking control, so
    without this the served runner never tells llama-server to suppress reasoning.
    A reasoning model then thinks on every request regardless of
    ``enable_thinking=False``, and on a bounded ``max_tokens`` it can spend the
    whole budget thinking and return EMPTY content (#428/#420).

    llama-server exposes two levers, forwarded here:

    - ``chat_template_kwargs`` -> the model's jinja chat template. ``enable_thinking``
      is the canonical Qwen3 / Gemma toggle; a template that doesn't understand it
      simply ignores it, so forwarding is safe across families.
    - ``reasoning_effort`` -> OpenAI-style effort for harmony models (gpt-oss).
      ``"none"`` is not a valid server value; disabling is expressed via
      ``enable_thinking=False`` instead, so it is dropped here.
    """
    overrides: dict[str, Any] = {}
    enable_thinking = getattr(task_params, "enable_thinking", None)
    if enable_thinking is not None:
        overrides["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    effort = getattr(task_params, "reasoning_effort", None)
    if effort is not None and effort != "none":
        overrides["reasoning_effort"] = effort
    return overrides


# How long to wait for the server to finish loading the model and report healthy.
# A large GGUF on a GPU node can take a while to map + warm up.
_HEALTH_DEADLINE_S: Final = 600.0


class _StreamDelta(NamedTuple):
    """One parsed SSE delta from the proxied ``/v1/chat/completions`` stream."""

    reasoning: str
    content: str
    finish: Literal["stop", "length", "content_filter"] | None
    done: bool  # the terminal ``data: [DONE]`` sentinel


def _gpu_layers_for_backend(resolved_backend: str | None) -> str:
    """The ``-ngl`` (n-gpu-layers) value to pass llama-server for a backend tag.

    Mirrors the master's VRAM admission exactly (placement_utils
    ``_has_gpu_offload_backend``): offload every layer (``"99"``) only for a
    recognized GPU compute tag (``llama_server-<gpu>``). A ``-cpu`` tag OR a bare
    ``llama_server`` tag was admitted against system RAM, not VRAM, so use
    ``"0"`` rather than grabbing a GPU that was not budgeted (or may not exist). A
    missing resolution (a manual / fallback launch off the placement path)
    defaults to full GPU offload, the common served case.
    """
    if resolved_backend is None:
        return "99"
    if resolved_backend.startswith("llama_server-") and not resolved_backend.endswith(
        "-cpu"
    ):
        return "99"
    return "0"


def _parse_sse_line(line: str) -> _StreamDelta | None:
    """Parse one SSE line into a ``_StreamDelta``, or ``None`` to skip it.

    Handles the OpenAI streaming shape llama-server emits: ``data: {json}`` lines
    whose first choice carries a ``delta`` (``content`` and/or ``reasoning_content``)
    and an optional ``finish_reason``, plus the terminal ``data: [DONE]``. Returns
    ``None`` for non-``data:`` lines, ``[DONE]`` is reported via ``done=True``, and
    malformed JSON or a choice-less payload is skipped (``None``) so a stray line
    never breaks the stream. Pure (no I/O) so the parse is unit-testable.
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
    choices = chunk.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    delta = choice.get("delta") or {}
    return _StreamDelta(
        reasoning=delta.get("reasoning_content") or "",
        content=delta.get("content") or "",
        finish=map_finish_reason(choice.get("finish_reason")),
        done=False,
    )


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL this child when its parent (the runner) dies.

    Runs in the forked child before ``exec`` (``preexec_fn``). Linux-only; a
    best-effort guard so a runner-process crash never leaves an orphaned
    ``llama-server`` holding GPU memory. Any failure is swallowed (the explicit
    teardown path still applies on graceful shutdown).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0)
    except Exception:  # noqa: BLE001 - best-effort; non-Linux or no libc
        pass


class Runner:
    """Single-node served-backend runner that proxies an external ``llama-server``.

    Lifecycle mirrors the in-process llama.cpp runner: it skips the ring
    (``ConnectToGroup`` / ``StartWarmup``), spawns the server on ``LoadModel``,
    and serves ``TextGeneration`` by streaming the server's SSE output back as
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
        if self.shard_metadata.world_size != 1:
            raise RuntimeError(
                "llama-server runner requires single-node placement, got "
                f"world_size={self.shard_metadata.world_size}"
            )
        self.setup_start_time = time.time()
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self.server_proc: subprocess.Popen[bytes] | None = None
        self.server_log: Any = None
        self.server_log_path: Path | None = None
        self.base_url: str | None = None
        # Set at load: the card declares a channel output parser (Gemma 4), so the
        # served runner reparses ``<|channel>`` markers out of the content stream
        # itself (llama-server can't), splitting reasoning from the answer.
        self._uses_channel_parser: bool = False
        self.current_status: RunnerStatus = RunnerIdle()
        logger.info("llama-server runner created")
        self.update_status(RunnerIdle())

    # --- runner-contract plumbing (mirrors the llama.cpp runner) ---------------

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
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def _drain_cancellations(self) -> None:
        while True:
            try:
                cancelled = self.cancel_receiver.receive_nowait()
            except WouldBlock:
                break
            self.cancelled_tasks.add(cancelled)

    def _is_cancelled(self, task_id: TaskId) -> bool:
        self._drain_cancellations()
        return (
            task_id in self.cancelled_tasks or CANCEL_ALL_TASKS in self.cancelled_tasks
        )

    def main(self) -> None:
        try:
            with self.task_receiver as tasks:
                for task in tasks:
                    if task.task_id in self.seen:
                        logger.warning("repeat task - potential error")
                    self.seen.add(task.task_id)
                    self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
                    self.send_task_status(task, TaskStatus.Running)
                    self.handle_task(task)
                    was_cancelled = (
                        task.task_id in self.cancelled_tasks
                        or CANCEL_ALL_TASKS in self.cancelled_tasks
                    )
                    self.send_task_status(
                        task,
                        TaskStatus.Cancelled if was_cancelled else TaskStatus.Complete,
                    )
                    self.update_status(self.current_status)
                    if isinstance(self.current_status, RunnerShutdown):
                        break
        finally:
            # Never leave the server subprocess running past the runner loop, even
            # on an unexpected exit (PR_SET_PDEATHSIG is the SIGKILL backstop).
            self._teardown_server()

    def handle_task(self, task: Task) -> None:
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                self._generate(task)
            case Shutdown():
                logger.info("llama-server runner shutting down")
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self._teardown_server()
                self.current_status = RunnerShutdown()
            case _:
                raise RuntimeError(
                    f"llama-server runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )

    # --- model load: spawn + health-check the server --------------------------

    def _load_model(self, task: Task) -> None:
        self.update_status(RunnerLoading())
        self.acknowledge_task(task)

        from skulk.download.download_utils import build_model_path

        card = self.shard_metadata.model_card
        model_id = card.model_id
        model_dir = build_model_path(ModelId(model_id))
        # Load the file the card pinned (the selected quant); fall back to scanning
        # so download / sizing / loading stay in agreement. Reject an absolute or
        # ``..`` path that escapes the model dir.
        pinned = card.gguf_file
        gguf_path: Path | None = None
        if pinned:
            candidate = (model_dir / pinned).resolve()
            if candidate.is_file() and candidate.is_relative_to(model_dir.resolve()):
                gguf_path = candidate
            else:
                logger.warning(
                    f"card gguf_file {pinned!r} is missing or outside the model "
                    f"dir; scanning {model_dir} instead"
                )
        if gguf_path is None:
            gguf_path = select_gguf_file(model_dir)

        # When the card declares a channel output parser we strip reasoning
        # markers ourselves, so llama-server must hand back raw text
        # (--reasoning-format none) regardless of whether the model "reasons":
        # its own reasoning parsers don't understand Gemma 4's <|channel> tokens.
        self._uses_channel_parser = (
            card.runtime is not None
            and card.runtime.output_parser == OutputParserType.Gemma4
        )
        reasoning_format_none = self._uses_channel_parser or not (
            _model_declares_reasoning(card)
        )
        n_ctx = serving_n_ctx(self.context_token_limit, logits_all=False)
        try:
            self._spawn_server(gguf_path, n_ctx, card.runtime, reasoning_format_none)
            self._await_health()
        except Exception:
            self._teardown_server()
            raise
        self.current_status = RunnerReady()
        logger.info(
            f"llama-server runner ready in {time.time() - self.setup_start_time:.1f}s "
            f"(url={self.base_url})"
        )

    def _spawn_server(
        self,
        gguf_path: Path,
        n_ctx: int,
        runtime: Any,
        reasoning_format_none: bool,
    ) -> None:
        binary = os.environ.get(LLAMA_SERVER_BIN_ENV, "").strip()
        if not binary:
            raise RuntimeError(
                f"{LLAMA_SERVER_BIN_ENV} is not set; cannot launch llama-server"
            )
        # Validate the binary up front (same check the probe uses) so a
        # misconfigured path fails with a clear, actionable error rather than a
        # bare FileNotFoundError/PermissionError from Popen later.
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(
                f"{LLAMA_SERVER_BIN_ENV}={binary!r} is not an executable file; "
                "point it at a llama-server binary built >= b9196 (for draft-mtp)"
            )
        port = self._pick_port()
        n_gpu_layers = _gpu_layers_for_backend(self.shard_metadata.resolved_backend)
        cmd: list[str] = [
            binary,
            "-m",
            str(gguf_path),
            "-ngl",
            n_gpu_layers,
            "-c",
            str(n_ctx),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            # --jinja enables the GGUF's chat template path, which is what makes
            # tool calling and reasoning-content extraction work server-side.
            "--jinja",
        ]
        # --reasoning-format none hands back raw text in message.content. We use
        # it for (a) plain non-reasoning models (otherwise llama-server's default
        # `auto` extracts their prose into reasoning_content, leaving content
        # empty) and (b) models we parse ourselves (Gemma 4's <|channel> markers,
        # which llama-server's parsers mishandle). A reasoning model llama-server
        # *can* parse keeps the default so its thoughts land in reasoning_content.
        if reasoning_format_none:
            cmd += ["--reasoning-format", "none"]
        spec_type = getattr(runtime, "served_spec_type", None) if runtime else None
        # Operator/benchmark override: force plain decode for a served model whose
        # card asks for speculation. Serving the SAME GGUF with the spec flags
        # omitted is the apples-to-apples "MTP off" baseline for an on-vs-off
        # throughput comparison (identical weights, speculation disabled), and a
        # debug lever when a spec pairing misbehaves. Node-level, read at launch.
        if spec_type and spec_type != "none" and _force_no_spec():
            logger.info(
                f"SKULK_LLAMA_SERVER_FORCE_NO_SPEC set; serving {spec_type!r} model "
                "with speculative decoding disabled (plain decode)"
            )
            spec_type = None
        if spec_type and spec_type != "none":
            flag = _SPEC_TYPE_FLAG.get(spec_type)
            if flag is None:
                logger.warning(
                    f"unknown served_spec_type {spec_type!r}; serving without "
                    "speculative decoding"
                )
            else:
                cmd += ["--spec-type", flag]
                n_max = getattr(runtime, "served_spec_n_max", None)
                if n_max is not None:
                    cmd += ["--spec-draft-n-max", str(n_max)]
                cmd += _draft_model_args(runtime, spec_type)

        self.server_log_path = (
            Path(tempfile.gettempdir()) / f"skulk-llama-server-{self.runner_id}.log"
        )
        self.server_log = open(self.server_log_path, "wb")  # noqa: SIM115
        # Modern llama.cpp links libllama.so / libggml*.so from the binary's own
        # directory (rpath $ORIGIN). Add that dir to LD_LIBRARY_PATH too so the
        # shared libs resolve regardless of the runner's working directory.
        env = os.environ.copy()
        bin_dir = str(Path(binary).resolve().parent)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir
        logger.info("launching llama-server: " + " ".join(cmd))
        self.server_proc = subprocess.Popen(  # noqa: S603 - args are built here, not user input
            cmd,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=_set_pdeathsig,  # noqa: PLW1509 - Linux reap-on-parent-death
        )
        self.base_url = f"http://127.0.0.1:{port}"

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
        raise RuntimeError("could not find a free port for llama-server")

    def _await_health(self) -> None:
        assert self.server_proc is not None and self.base_url is not None
        deadline = time.time() + _HEALTH_DEADLINE_S
        with httpx.Client(timeout=5.0) as client:
            while time.time() < deadline:
                if self.server_proc.poll() is not None:
                    raise RuntimeError(
                        "llama-server exited during startup (code "
                        f"{self.server_proc.returncode}); log tail:\n"
                        f"{self._server_log_tail()}"
                    )
                try:
                    resp = client.get(f"{self.base_url}/health")
                    if (
                        resp.status_code == 200
                        and (resp.json() or {}).get("status") == "ok"
                    ):
                        return
                except Exception:  # noqa: BLE001 - not up yet; keep polling
                    pass
                time.sleep(2)
        raise RuntimeError(
            f"llama-server did not become healthy within {_HEALTH_DEADLINE_S:.0f}s; "
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
        assert isinstance(task, TextGeneration)
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)
        assert self.base_url is not None

        model_id = self.shard_metadata.model_card.model_id
        command_id = task.command_id
        body: dict[str, Any] = generation_kwargs(task.task_params)
        body["messages"] = messages_for_llama(task.task_params)
        # Forward thinking-control (enable_thinking / reasoning_effort) to
        # llama-server. Without this a reasoning model thinks on every request and
        # can return empty content under a bounded budget (#428/#420).
        body.update(reasoning_request_overrides(task.task_params))

        try:
            # Per-token logprobs are not wired over the SSE proxy yet. Fail loud
            # rather than return a successful response with logprobs silently
            # missing, matching the in-process runner's #385 no-silent-empty
            # contract (the raise surfaces as an ErrorChunk below).
            if wants_logprobs(task.task_params.logprobs, task.task_params.top_logprobs):
                body.pop("logprobs", None)
                body.pop("top_logprobs", None)
                raise RuntimeError(
                    "Per-token logprobs are not supported on the served "
                    "(llama_server) engine: the OpenAI SSE proxy does not surface "
                    "them. Retry without logprobs/top_logprobs."
                )
            if task.task_params.tools:
                self._generate_with_tools(task, body, model_id, command_id)
                self.current_status = RunnerReady()
                return
            self._generate_streaming(task, body, model_id, command_id)
        except Exception as exc:  # noqa: BLE001 - surface as an ErrorChunk
            logger.opt(exception=exc).warning("llama-server generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(model=model_id, error_message=str(exc)),
                )
            )
        self.current_status = RunnerReady()

    def _generate_streaming(
        self,
        task: TextGeneration,
        body: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        body["stream"] = True
        assert self.base_url is not None
        emitted_finish = False
        # Gemma 4 emits its reasoning as literal <|channel> markers in content;
        # reparse them here (llama-server can't) into reasoning/content chunks.
        parser = GemmaChannelTextParser() if self._uses_channel_parser else None
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
                    logger.info(f"llama-server generation cancelled: {task.task_id}")
                    break
                delta = _parse_sse_line(line)
                if delta is None:
                    continue
                if delta.done:
                    break
                if delta.reasoning:
                    self._send_token(
                        command_id, model_id, delta.reasoning, is_thinking=True
                    )
                if delta.content:
                    if parser is not None:
                        for text, is_thinking in parser.feed(delta.content):
                            self._send_token(
                                command_id, model_id, text, is_thinking=is_thinking
                            )
                    else:
                        self._send_token(command_id, model_id, delta.content)
                if delta.finish is not None:
                    if parser is not None:
                        for text, is_thinking in parser.flush():
                            self._send_token(
                                command_id, model_id, text, is_thinking=is_thinking
                            )
                    self._send_token(
                        command_id, model_id, "", finish_reason=delta.finish
                    )
                    emitted_finish = True
        # Guarantee a terminal chunk so the consumer's stream closes even if the
        # server ended without an explicit finish_reason; drain any held tail.
        if not emitted_finish and not self._is_cancelled(task.task_id):
            if parser is not None:
                for text, is_thinking in parser.flush():
                    self._send_token(
                        command_id, model_id, text, is_thinking=is_thinking
                    )
            self._send_token(command_id, model_id, "", finish_reason="stop")

    def _generate_with_tools(
        self,
        task: TextGeneration,
        body: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        # Tool calls are requested non-streamed (the caller wants the assembled
        # call): llama-server parses the model's native tool-call format and
        # returns structured ``tool_calls`` via --jinja, so no text parsing here.
        body["stream"] = False
        body["tools"] = task.task_params.tools
        tool_choice = getattr(task.task_params, "tool_choice", None)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        assert self.base_url is not None
        if self._is_cancelled(task.task_id):
            return
        timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=None)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{self.base_url}/v1/chat/completions", json=body)
            resp.raise_for_status()
            result = resp.json()
        # A cancel that arrived while the (non-streamed) request was in flight:
        # drain it (the streaming path checks every chunk; this blocking path has
        # no mid-flight checkpoint) and skip emission so main() marks the task
        # Cancelled, not Complete, and no tool call is surfaced for it.
        if self._is_cancelled(task.task_id):
            logger.info(f"llama-server tool generation cancelled: {task.task_id}")
            return
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = tool_calls_from_message(message)
        if tool_calls:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ToolCallChunk(
                        model=model_id, tool_calls=tool_calls, usage=None
                    ),
                )
            )
            return
        # The model answered in prose: emit its reasoning + content, then close.
        # Preserve the server's finish_reason (e.g. "length" when the answer hit
        # max_tokens) rather than hard-coding "stop", so a truncated prose answer
        # still signals truncation to the client.
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        if reasoning:
            self._send_token(command_id, model_id, reasoning, is_thinking=True)
        if content:
            if self._uses_channel_parser:
                # Reparse Gemma 4's <|channel> markers out of the prose answer.
                parser = GemmaChannelTextParser()
                for text, is_thinking in parser.feed(content) + parser.flush():
                    self._send_token(
                        command_id, model_id, text, is_thinking=is_thinking
                    )
            else:
                self._send_token(command_id, model_id, content)
        finish = map_finish_reason(choice.get("finish_reason")) or "stop"
        self._send_token(command_id, model_id, "", finish_reason=finish)

    def _send_token(
        self,
        command_id: CommandId,
        model_id: ModelId,
        text: str,
        *,
        is_thinking: bool = False,
        finish_reason: Any = None,
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
                ),
            )
        )
