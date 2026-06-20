# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false
"""In-process text-generation runner backed by llama.cpp (``llama-cpp-python``).

This is the first non-MLX inference engine: it serves GGUF models on GPU nodes
that ``mlx-lm`` cannot target (e.g. an AMD Strix Halo box via the Vulkan or ROCm
llama.cpp backend). It is selected by the model card's backend tags resolving to
the ``llama_cpp`` engine on this node (``bootstrap._resolve_text_engine``).

llama.cpp is autoregressive token-by-token, so it maps directly onto Skulk's
existing per-token data plane: the runner emits one ``TokenChunk`` per decoded
token via ``ChunkGenerated`` (exactly like the MLX runner), so no new chunk
contract is needed. It is **single-node only** (no ring / ``ConnectToGroup`` /
warmup), mirroring the embeddings runner's group-less lifecycle.

``llama_cpp`` is imported lazily inside ``LoadModel`` so this module imports
cleanly on nodes (e.g. Macs) where the binding is not installed.
"""

import os
import time
from pathlib import Path
from typing import Any, Literal

from anyio import WouldBlock

from skulk.api.types import ToolCallItem, TopLogprobItem
from skulk.shared.models.memory_estimate import KV_CONTEXT_BUDGET_TOKENS
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
from skulk.shared.types.text_generation import TextGenerationTaskParams
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


def select_gguf_file(model_dir: Path) -> Path:
    """Pick the GGUF weights file to load from a staged model directory.

    Fallback for when the card does not pin a file (``gguf_file``): scans
    recursively, skips ``mmproj*`` projectors, and ranks by the same
    quant preference the card uses (``gguf_quant_rank``: a real quant over BF16),
    then basename, so this fallback agrees with the card's sizing
    (``gguf_shard_group_size``) and selection (``select_preferred_gguf``) and
    never silently loads the full-precision BF16. Raises ``FileNotFoundError``
    when the directory has no usable GGUF.
    """
    from skulk.shared.models.model_cards import gguf_quant_rank

    candidates = sorted(
        (
            path
            for path in model_dir.glob("**/*.gguf")
            if "mmproj" not in path.name.lower()
        ),
        key=lambda path: (gguf_quant_rank(path.name), path.name),
    )
    if not candidates:
        raise FileNotFoundError(f"no .gguf weights file found in {model_dir}")
    return candidates[0]


def messages_for_llama(task_params: TextGenerationTaskParams) -> list[dict[str, Any]]:
    """Build the chat-completion messages llama.cpp's chat template expects.

    Prefers ``chat_template_messages`` (already-rendered OpenAI-style dicts that
    the API populates on the chat path). Falls back to reconstructing a minimal
    message list from ``instructions`` + ``input`` when that is absent.
    """
    if task_params.chat_template_messages:
        return task_params.chat_template_messages
    messages: list[dict[str, Any]] = []
    if task_params.instructions:
        messages.append({"role": "system", "content": task_params.instructions})
    for message in task_params.input:
        role = getattr(message, "role", "user")
        content = getattr(message, "content", "")
        messages.append(
            {
                "role": role,
                "content": content if isinstance(content, str) else str(content),
            }
        )
    return messages


def _generation_kwargs(task_params: TextGenerationTaskParams) -> dict[str, Any]:
    """Translate Skulk task params into llama.cpp ``create_chat_completion`` kwargs."""
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
        kwargs["repeat_penalty"] = task_params.repetition_penalty
    if task_params.stop is not None:
        kwargs["stop"] = task_params.stop
    if task_params.seed is not None:
        kwargs["seed"] = task_params.seed
    if task_params.logprobs:
        kwargs["logprobs"] = True
        if task_params.top_logprobs is not None:
            kwargs["top_logprobs"] = task_params.top_logprobs
    return kwargs


def _tool_calls_from_message(message: dict[str, Any]) -> list[ToolCallItem]:
    """Extract Skulk ToolCallItems from a llama.cpp chat-completion message.

    llama.cpp returns OpenAI-shaped tool calls:
    ``{"tool_calls": [{"id", "type": "function", "function": {"name", "arguments"}}]}``
    where ``arguments`` is a JSON string. Returns [] when the message has none.
    """
    items: list[ToolCallItem] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        item_kwargs: dict[str, Any] = {
            "name": name,
            "arguments": function.get("arguments") or "",
        }
        if call.get("id"):
            item_kwargs["id"] = call["id"]
        items.append(ToolCallItem(**item_kwargs))
    return items


def _logprob_fields(
    choice: dict[str, Any],
) -> tuple[float | None, list[TopLogprobItem] | None]:
    """Pull (logprob, top_logprobs) for one streamed token from a chat chunk.

    llama.cpp mirrors the OpenAI chat-logprobs shape:
    ``choice["logprobs"]["content"] = [{"token", "logprob", "top_logprobs": [...]}]``.
    Best-effort: any shape mismatch yields ``(None, None)`` so a logprobs request
    never breaks generation. A streamed chunk carries one content entry.
    """
    try:
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content:
            return (None, None)
        entry = content[0]
        top = [
            TopLogprobItem(
                token=t["token"], logprob=t["logprob"], bytes=t.get("bytes")
            )
            for t in (entry.get("top_logprobs") or [])
        ] or None
        return (entry.get("logprob"), top)
    except (KeyError, TypeError, IndexError):
        return (None, None)


def _logits_all_enabled() -> bool:
    """Whether to load the model with ``logits_all=True`` (enables logprobs).

    llama-cpp-python gates all logprobs behind ``logits_all=True`` at model
    construction, and the runner is loaded once but serves logprobs per request,
    so the flag cannot be toggled per request.

    Defaults **off**. ``logits_all=True`` makes llama.cpp pre-allocate a logits
    buffer of ``n_ctx * vocab * 4`` bytes up front, which at a large context is
    enormous (e.g. 131072 * 152064 * 4 = 74 GiB for a Qwen2.5 vocab) and OOMs the
    node on load. So logprobs is opt-in via ``SKULK_LLAMA_CPP_LOGITS_ALL=1``, and
    when on the context is further capped (see ``_logits_all_n_ctx``) so the
    buffer stays bounded. With it off a logprobs request degrades to a clear
    error. Either way the serving context window is bounded by the instance's
    admission ceiling (see ``_serving_n_ctx``), never the model's full trained
    context.
    """
    return os.getenv("SKULK_LLAMA_CPP_LOGITS_ALL", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _logits_all_n_ctx() -> int:
    """Context cap (tokens) to use when ``logits_all`` is on, to bound the buffer.

    The ``logits_all`` logits buffer is ``n_ctx * vocab * 4`` bytes, so it must
    not ride a large context. This caps it to a modest window (default 8192,
    override with ``SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX``): at an ~150k vocab that is
    ~5 GiB, the price of opting into logprobs.
    """
    raw = os.getenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", "8192")
    try:
        value = int(raw)
    except ValueError:
        value = 8192
    return value if value > 0 else 8192


def _serving_n_ctx(context_token_limit: int | None, logits_all: bool) -> int:
    """Context window (tokens) to allocate for the llama.cpp KV cache on load.

    Never 0: ``n_ctx=0`` tells llama.cpp to size the KV cache for the model's
    FULL trained context (e.g. gemma-4's 128k), which far exceeds the per-instance
    budget placement reserved memory for and OOM-kills the whole node on load
    (observed loading gemma-4-31B on a Vulkan node: the process was oom-killed and
    the instance vanished). Instead allocate for the instance's static
    context-admission ceiling (#145) -- the value placement actually sized node
    memory against -- falling back to the placement KV budget when it is unset.
    When ``logits_all`` is on, also bound by the logits-buffer window since that
    buffer scales with ``n_ctx``.
    """
    ceiling = (
        context_token_limit
        if context_token_limit and context_token_limit > 0
        else KV_CONTEXT_BUDGET_TOKENS
    )
    if logits_all:
        return min(ceiling, _logits_all_n_ctx())
    return ceiling


def _map_finish_reason(
    reason: str | None,
) -> Literal["stop", "length", "content_filter"] | None:
    """Map a llama.cpp finish reason onto Skulk's TokenChunk finish reasons."""
    if reason is None:
        return None
    if reason == "length":
        return "length"
    # llama.cpp uses "stop" for both EOS and stop-string hits.
    return "stop"


class Runner:
    """Single-node llama.cpp text-generation runner.

    Lifecycle mirrors the embeddings runner: it skips ``ConnectToGroup`` and
    ``StartWarmup`` (no ring), loads on ``LoadModel``, and serves
    ``TextGeneration`` by streaming tokens.
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
        # Static per-instance context ceiling for request admission (#145),
        # computed by the worker from gossiped node memory before spawn. Bounds
        # the KV cache the runner allocates on load so it matches the memory
        # placement reserved (see _serving_n_ctx); never the model's full context.
        self.context_token_limit = context_token_limit
        self.instance, self.runner_id, self.shard_metadata = (
            bound_instance.instance,
            bound_instance.bound_runner_id,
            bound_instance.bound_shard,
        )
        if self.shard_metadata.world_size != 1:
            raise RuntimeError(
                "llama.cpp runner requires single-node placement, got "
                f"world_size={self.shard_metadata.world_size}"
            )
        self.setup_start_time = time.time()
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self.model: Any = None
        self.current_status: RunnerStatus = RunnerIdle()
        logger.info("llama.cpp runner created")
        self.update_status(RunnerIdle())

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
        """Move any pending cancellation task-ids into ``cancelled_tasks``."""
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
        with self.task_receiver as tasks:
            for task in tasks:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                self.seen.add(task.task_id)
                self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
                self.send_task_status(task, TaskStatus.Running)
                self.handle_task(task)
                # Use only cancellations OBSERVED during execution (the streaming
                # loop drains the cancel pipe via _is_cancelled as it runs). Do
                # NOT re-drain here: a cancel that loses the race with completion
                # must not retroactively flip an already-finished task (which has
                # streamed its tokens + finish chunk) to Cancelled.
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

    def handle_task(self, task: Task) -> None:
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                self._generate(task)
            case Shutdown():
                logger.info("llama.cpp runner shutting down")
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self.model = None
                self.current_status = RunnerShutdown()
            case _:
                raise RuntimeError(
                    f"llama.cpp runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )

    def _load_model(self, task: Task) -> None:
        self.update_status(RunnerLoading())
        self.acknowledge_task(task)

        from llama_cpp import Llama  # pyright: ignore[reportAttributeAccessIssue]

        from skulk.download.download_utils import build_model_path

        model_id = self.shard_metadata.model_card.model_id
        model_dir = build_model_path(ModelId(model_id))
        # Load the exact file the card pinned at creation (the selected quant);
        # fall back to scanning if it's absent (older card / manual staging), so
        # download, sizing, and loading stay in agreement.
        pinned = self.shard_metadata.model_card.gguf_file
        gguf_path: Path | None = None
        if pinned:
            candidate = (model_dir / pinned).resolve()
            # Reject a hand-edited card whose gguf_file is absolute or uses ".."
            # to escape the model directory; fall back to the in-dir scan.
            if candidate.is_file() and candidate.is_relative_to(model_dir.resolve()):
                gguf_path = candidate
            else:
                logger.warning(
                    f"card gguf_file {pinned!r} is missing or outside the model "
                    f"dir; scanning {model_dir} instead"
                )
        if gguf_path is None:
            gguf_path = select_gguf_file(model_dir)
        # n_gpu_layers=-1 offloads every layer to the GPU backend the binding was
        # built with (Vulkan/ROCm/CUDA). n_ctx is bounded by the instance's
        # admission ceiling (never 0/full-context, which OOM-kills the node on a
        # large-context model -- see _serving_n_ctx). logits_all (logprobs,
        # opt-in) further bounds it because it pre-allocates an n_ctx*vocab*4
        # logits buffer. See _logits_all_enabled / _logits_all_n_ctx.
        logits_all = _logits_all_enabled()
        n_ctx = _serving_n_ctx(self.context_token_limit, logits_all)
        logger.info(
            f"loading GGUF {gguf_path.name} for {model_id} "
            f"(n_ctx={n_ctx}, logits_all={logits_all})"
        )
        self.model = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            logits_all=logits_all,
            verbose=False,
        )
        self.current_status = RunnerReady()
        logger.info(
            f"llama.cpp runner ready in {time.time() - self.setup_start_time:.1f}s"
        )

    def _generate(self, task: Task) -> None:
        assert isinstance(task, TextGeneration)
        # Must be an ACTIVE status (not RunnerReady) for the whole task: the
        # supervisor asserts the runner is RunnerRunning/Loading/etc. when the
        # terminal TaskStatus arrives (runner_supervisor._forward_events). main()
        # sends Complete after this returns, so we stay RunnerRunning until then
        # and only flip current_status back to Ready (without an event) at the
        # end, so the Ready event is ordered after Complete.
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)
        assert self.model is not None

        model_id = self.shard_metadata.model_card.model_id
        command_id = task.command_id
        messages = messages_for_llama(task.task_params)
        kwargs = _generation_kwargs(task.task_params)

        try:
            # Tool calling can't be streamed token-by-token usefully (the caller
            # wants the assembled call), and accumulating OpenAI tool-call deltas
            # is fragile; run it non-streamed and emit one ToolCallChunk.
            if task.task_params.tools:
                self._generate_with_tools(task, messages, kwargs, model_id, command_id)
                self.current_status = RunnerReady()
                return

            want_logprobs = task.task_params.logprobs
            stream = self.model.create_chat_completion(
                messages=messages, stream=True, **kwargs
            )
            emitted_finish = False
            for chunk in stream:
                if self._is_cancelled(task.task_id):
                    logger.info(f"llama.cpp generation cancelled: {task.task_id}")
                    break
                choice = chunk["choices"][0]
                text = choice.get("delta", {}).get("content") or ""
                finish = _map_finish_reason(choice.get("finish_reason"))
                logprob, top_logprobs = (
                    _logprob_fields(choice) if want_logprobs else (None, None)
                )
                if not text and finish is None and logprob is None:
                    continue
                emitted_finish = emitted_finish or finish is not None
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=command_id,
                        chunk=TokenChunk(
                            model=model_id,
                            text=text,
                            token_id=-1,  # llama.cpp chat stream does not expose ids
                            usage=None,
                            finish_reason=finish,
                            logprob=logprob,
                            top_logprobs=top_logprobs,
                        ),
                    )
                )
            # Guarantee a terminal chunk on normal completion so the consumer's
            # stream closes even if llama.cpp ended without an explicit finish.
            if not emitted_finish and not self._is_cancelled(task.task_id):
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=command_id,
                        chunk=TokenChunk(
                            model=model_id,
                            text="",
                            token_id=-1,
                            usage=None,
                            finish_reason="stop",
                        ),
                    )
                )
        except Exception as exc:
            logger.opt(exception=exc).warning("llama.cpp generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(model=model_id, error_message=str(exc)),
                )
            )

        self.current_status = RunnerReady()

    def _generate_with_tools(
        self,
        task: TextGeneration,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        """Serve a tool-enabled request (non-streamed) and emit one terminal chunk.

        Passes the request's ``tools`` to llama.cpp. If the model returns tool
        calls, emits a ``ToolCallChunk``; otherwise it chose to answer in prose,
        so emits that content as a normal ``TokenChunk``. Either way a single
        terminal chunk closes the consumer's stream.

        Cancellation: unlike the streaming path (which checks per token), the
        tool call runs through one blocking ``create_chat_completion`` that
        cannot be interrupted mid-flight. So cancellation is honored at the two
        boundaries around it: skip the (possibly long) call entirely if the task
        is already cancelled, and suppress the result if a cancel landed while it
        ran. In both cases nothing is emitted and ``main`` reads the drained
        cancellation to mark the task ``Cancelled`` rather than ``Complete``.
        """
        if self._is_cancelled(task.task_id):
            logger.info(f"llama.cpp tool generation skipped (cancelled): {task.task_id}")
            return
        result = self.model.create_chat_completion(
            messages=messages,
            stream=False,
            tools=task.task_params.tools,
            **kwargs,
        )
        if self._is_cancelled(task.task_id):
            logger.info(f"llama.cpp tool generation cancelled: {task.task_id}")
            return
        choice = result["choices"][0]
        message = choice.get("message", {})
        tool_calls = _tool_calls_from_message(message)
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
        # No tool call: the model answered in prose. Emit it as a final token.
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=TokenChunk(
                    model=model_id,
                    text=message.get("content") or "",
                    token_id=-1,
                    usage=None,
                    finish_reason=_map_finish_reason(choice.get("finish_reason"))
                    or "stop",
                ),
            )
        )
