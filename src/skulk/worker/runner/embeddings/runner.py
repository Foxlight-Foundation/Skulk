# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Embedding model runner using HuggingFace transformers + torch.

BERT-based embedding models cannot be loaded by mlx_lm, so this runner
uses transformers.AutoModel directly. Single forward pass, no KV cache,
no generation loop, no streaming. Pooling, normalization and truncation come
from the artifact's own sentence-transformers metadata (see ``recipe``), so a
model's vectors match its reference implementation.
"""

import time
from pathlib import Path
from typing import Any, Final, cast

import torch
from transformers import AutoModel, AutoTokenizer

from skulk.shared.tracing import (
    begin_trace_session,
    bind_trace_session,
    pop_trace_session,
    record_trace_marker,
    trace,
)
from skulk.shared.types.chunks import EmbeddingChunk, ErrorChunk
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    LoadModel,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextEmbedding,
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
from skulk.worker.runner.embeddings.recipe import (
    EmbeddingRecipe,
    PoolingMode,
    resolve_embedding_recipe,
)


def _get_device() -> torch.device:
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_FORWARD_BATCH_TEXTS: Final = 32
"""Most texts per forward pass."""

_FORWARD_TOKEN_BUDGET: Final = 16384
"""Most padded tokens per forward pass (32 texts at the historical 512-token
cap). A request's texts are padded to their longest member, so the memory a pass
needs grows with texts x truncation length; a long-context recipe (8192 tokens)
gets two texts per pass instead of 32. Results are identical however the
request is split, since each text's vector depends only on its own tokens."""


def _texts_per_pass(max_tokens: int) -> int:
    """How many texts one forward pass may carry at this truncation length."""
    return max(1, min(_FORWARD_BATCH_TEXTS, _FORWARD_TOKEN_BUDGET // max_tokens))


def _pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: PoolingMode,
    *,
    left_padded: bool,
) -> torch.Tensor:
    """Collapse token states to one vector per text, ignoring padding.

    Args:
        hidden_states: ``[batch, tokens, dim]`` encoder output.
        attention_mask: ``[batch, tokens]`` with 1 for real tokens.
        pooling: The model's pooling mode.
        left_padded: Whether the tokenizer pads on the left, which moves the
            last real token to the final position.

    Returns:
        ``[batch, dim]`` pooled vectors.
    """
    if pooling == "cls":
        return hidden_states[:, 0]
    if pooling == "last":
        if left_padded:
            return hidden_states[:, -1]
        last_index = attention_mask.sum(dim=1).clamp(min=1) - 1
        return hidden_states[torch.arange(hidden_states.shape[0]), last_index]
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    if pooling == "max":
        masked = hidden_states.masked_fill(mask == 0, float("-inf"))
        return masked.max(dim=1).values
    summed = (hidden_states * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1)


def _forward(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    recipe: EmbeddingRecipe,
    *,
    rank: int,
) -> tuple[list[list[float]], int]:
    """Embed texts with the model's own recipe, in token-bounded chunks.

    Tokenize (truncating at the recipe's limit), encode, pool and optionally
    L2-normalize, as the model's sentence-transformers pipeline does.

    Returns:
        One vector per text, in input order, and the total input token count.
    """
    result: list[list[float]] = []
    token_count = 0
    left_padded = getattr(tokenizer, "padding_side", "right") == "left"
    step = _texts_per_pass(recipe.max_tokens)
    for start in range(0, len(texts), step):
        chunk = texts[start : start + step]
        with trace("tokenize", rank, "embedding"):
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=recipe.max_tokens,
            )
        token_count += int(inputs["attention_mask"].sum().item())
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with trace("forward", rank, "embedding"), torch.no_grad():
            outputs = model(**inputs)

        with trace("pooling", rank, "embedding"):
            embeddings = _pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
                recipe.pooling,
                left_padded=left_padded,
            )

        if recipe.normalize:
            with trace("normalization", rank, "embedding"):
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        result.extend(cast(list[list[float]], embeddings.tolist()))
    return result, token_count


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            bound_instance.instance,
            bound_instance.bound_runner_id,
            bound_instance.bound_shard,
        )

        logger.info("hello from the embedding runner")
        if self.shard_metadata.world_size != 1:
            raise RuntimeError(
                f"Embedding runner requires single-node placement, got world_size={self.shard_metadata.world_size}"
            )
        self.setup_start_time = time.time()
        self.cancelled_tasks = set[TaskId]()

        self.model: Any = None
        self.tokenizer: Any = None
        self.device: torch.device = torch.device("cpu")
        # The legacy recipe until LoadModel reads the artifact's own.
        self.recipe = EmbeddingRecipe(pooling="mean", normalize=True, max_tokens=512)

        self.current_status: RunnerStatus = RunnerIdle()
        logger.info("embedding runner created")
        self.update_status(RunnerIdle())
        self.seen = set[TaskId]()

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver as tasks:
            for task in tasks:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                self.seen.add(task.task_id)
                self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
                self.send_task_status(task, TaskStatus.Running)
                self.handle_task(task)
                was_cancelled = (task.task_id in self.cancelled_tasks) or (
                    CANCEL_ALL_TASKS in self.cancelled_tasks
                )
                if was_cancelled:
                    self.send_task_status(task, TaskStatus.Cancelled)
                else:
                    self.send_task_status(task, TaskStatus.Complete)
                self.update_status(self.current_status)

                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_task(self, task: Task):
        match task:
            # Embedding models are single-node, skip ConnectToGroup and StartWarmup
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                logger.info("embedding runner loading")
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)

                model_id = self.shard_metadata.model_card.model_id
                logger.info(f"loading embedding model: {model_id}")

                from skulk.download.download_utils import build_model_path
                from skulk.shared.types.common import ModelId

                local_path = str(
                    build_model_path(
                        ModelId(model_id),
                        self.shard_metadata.model_card.source_revision,
                        (
                            self.shard_metadata.model_card.artifact_bundle.root
                            if self.shard_metadata.model_card.artifact_bundle
                            is not None
                            else None
                        ),
                    )
                )
                logger.info(f"loading from local path: {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_path, local_files_only=True
                )
                self.device = _get_device()
                self.model = AutoModel.from_pretrained(
                    local_path, trust_remote_code=False, local_files_only=True
                )
                try:
                    self.model.to(self.device)
                except Exception as exc:
                    logger.opt(exception=exc).warning(
                        f"failed to move model to {self.device}, falling back to CPU"
                    )
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                self.model.eval()
                model_config = cast(object, self.model.config)
                max_positions = cast(
                    object, getattr(model_config, "max_position_embeddings", None)
                )
                self.recipe = resolve_embedding_recipe(
                    Path(local_path),
                    model_max_positions=(
                        max_positions if isinstance(max_positions, int) else None
                    ),
                )
                logger.info(
                    f"embedding model on device: {self.device}; pooling "
                    f"{self.recipe.pooling}, normalize {self.recipe.normalize}, "
                    f"max {self.recipe.max_tokens} tokens"
                )

                # Skip straight to Ready — no warmup needed for embedding models
                self.current_status = RunnerReady()
                logger.info(
                    f"embedding runner ready in {time.time() - self.setup_start_time:.1f}s"
                )

            case TextEmbedding(task_params=task_params, command_id=command_id) if (
                isinstance(self.current_status, RunnerReady)
            ):
                logger.info(f"embedding request: {len(task_params.input_texts)} texts")
                # Hold an active (RunnerRunning) status across the forward pass.
                # The supervisor asserts the runner is in an active state when a
                # task's terminal TaskStatus is forwarded; staying RunnerReady
                # here would trip that assert (#326). Mirrors the MLX and
                # llama.cpp text runners, which are RunnerRunning for the whole
                # task. RunnerReady is restored after the work below.
                self.update_status(RunnerRunning())
                self.acknowledge_task(task)

                assert self.model is not None and self.tokenizer is not None
                model_id = self.shard_metadata.model_card.model_id

                if task.trace_enabled:
                    begin_trace_session(
                        task.task_id,
                        rank=self.shard_metadata.device_rank,
                        node_id=str(self.bound_instance.bound_node_id),
                        model_id=str(model_id),
                        task_kind="embedding",
                        tags=["embedding"],
                    )
                    record_trace_marker(
                        "queued",
                        self.shard_metadata.device_rank,
                        task_id=task.task_id,
                    )

                try:
                    with bind_trace_session(task.task_id):
                        result_embeddings, token_count = _forward(
                            self.model,
                            self.tokenizer,
                            task_params.input_texts,
                            self.device,
                            self.recipe,
                            rank=self.shard_metadata.device_rank,
                        )
                    if task.trace_enabled:
                        record_trace_marker(
                            "finish",
                            self.shard_metadata.device_rank,
                            task_id=task.task_id,
                        )
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=EmbeddingChunk(
                                model=model_id,
                                embeddings=result_embeddings,
                                token_count=token_count,
                            ),
                        )
                    )
                except Exception as exc:
                    if task.trace_enabled:
                        record_trace_marker(
                            "error",
                            self.shard_metadata.device_rank,
                            task_id=task.task_id,
                            tags=["error"],
                            attrs={"message": str(exc)},
                        )
                    logger.opt(exception=exc).warning("embedding forward pass failed")
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(
                                model=model_id,
                                error_message=str(exc),
                            ),
                        )
                    )
                finally:
                    if task.trace_enabled:
                        traces = pop_trace_session(task.task_id)
                        self.event_sender.send(
                            TracesCollected(
                                task_id=task.task_id,
                                rank=self.shard_metadata.device_rank,
                                traces=[
                                    TraceEventData(
                                        name=trace_event.name,
                                        start_us=trace_event.start_us,
                                        duration_us=trace_event.duration_us,
                                        rank=trace_event.rank,
                                        category=trace_event.category,
                                        node_id=trace_event.node_id,
                                        model_id=trace_event.model_id,
                                        task_kind=trace_event.task_kind,
                                        tags=list(trace_event.tags),
                                        attrs=trace_event.attrs,
                                    )
                                    for trace_event in traces
                                ],
                            )
                        )

                self.current_status = RunnerReady()

            case Shutdown():
                logger.info("embedding runner shutting down")
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self.model = None
                self.tokenizer = None
                self.current_status = RunnerShutdown()

            case _:
                raise RuntimeError(
                    f"embedding runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )
