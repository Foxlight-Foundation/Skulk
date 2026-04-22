# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Embedding model runner using HuggingFace transformers + torch.

BERT-based embedding models cannot be loaded by mlx_lm, so this runner
uses transformers.AutoModel directly. Single forward pass, no KV cache,
no generation loop, no streaming.
"""

import time
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from exo.shared.tracing import (
    begin_trace_session,
    bind_trace_session,
    pop_trace_session,
    record_trace_marker,
    trace,
)
from exo.shared.types.chunks import EmbeddingChunk, ErrorChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    LoadModel,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextEmbedding,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner.bootstrap import logger


def _get_device() -> torch.device:
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _forward(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    *,
    rank: int,
) -> tuple[list[list[float]], int]:
    """Run embedding forward pass: tokenize -> model -> mean pool -> L2 normalize."""
    with trace("tokenize", rank, "embedding"):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
    token_count = int(inputs["attention_mask"].sum().item())
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with trace("forward", rank, "embedding"), torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling over non-padding tokens
    with trace("pooling", rank, "embedding"):
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden_states = outputs.last_hidden_state
        summed = (hidden_states * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts

    # L2 normalize
    with trace("normalization", rank, "embedding"):
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    result: list[list[float]] = embeddings.tolist()
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
                if not was_cancelled:
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

                from exo.download.download_utils import build_model_path
                from exo.shared.types.common import ModelId

                local_path = str(build_model_path(ModelId(model_id)))
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
                logger.info(f"embedding model on device: {self.device}")

                # Skip straight to Ready — no warmup needed for embedding models
                self.current_status = RunnerReady()
                logger.info(
                    f"embedding runner ready in {time.time() - self.setup_start_time:.1f}s"
                )

            case TextEmbedding(task_params=task_params, command_id=command_id) if (
                isinstance(self.current_status, RunnerReady)
            ):
                logger.info(f"embedding request: {len(task_params.input_texts)} texts")
                self.update_status(RunnerReady())
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
