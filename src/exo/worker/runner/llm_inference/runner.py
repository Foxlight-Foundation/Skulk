import os
import time
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
from anyio import WouldBlock
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    OutputParserType,
    PromptRendererType,
    ToolCallFormat,
)
from exo.shared.tracing import (
    begin_trace_session,
    pop_trace_session,
    record_trace_marker,
)
from exo.shared.types.chunks import (
    ErrorChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import ModelId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.cache import KVPrefixCache, get_kv_cache_backend
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    InferenceGenerator,
    SequentialGenerator,
)

from .batch_generator import Cancelled, Finished
from .tool_parsers import make_mlx_parser


def _should_skip_llm_warmup(
    group_size: int,
    model_id: ModelId,
    model_card: ModelCard | None,
) -> bool:
    """Return whether the synthetic warmup request should be bypassed.

    Gemma 4 distributed warmup has proven less reliable than the actual request
    path in MLX/VLM pipeline mode. Bypass only that known-problem family by
    default; the env escape hatch remains constrained to single-node runs so a
    partially configured cluster cannot strand peers in warmup collectives.
    """
    if group_size > 1 and _is_gemma4_model(model_id, model_card):
        logger.warning(
            "Skipping distributed Gemma 4 synthetic warmup; marking runner ready "
            f"(model_id={model_id}, group_size={group_size})"
        )
        return True

    # Temporary escape hatch for debug sessions where we want runners to reach
    # Ready without issuing the synthetic warmup request. Normal behavior keeps
    # warmup enabled unless SKULK_SKIP_LLM_WARMUP=1 is set explicitly.
    if os.environ.get("SKULK_SKIP_LLM_WARMUP") != "1":
        return False
    if group_size > 1:
        logger.warning(
            "Ignoring SKULK_SKIP_LLM_WARMUP=1 for distributed warmup "
            f"(group_size={group_size}); synthetic warmup must stay consistent "
            "across all ranks"
        )
        return False
    return True


def _is_gemma4_model(model_id: ModelId, model_card: ModelCard | None) -> bool:
    """Return whether a model should be treated as a Gemma 4 runtime."""
    normalized_model_id = str(model_id).lower().replace("_", "-")
    if "gemma-4" in normalized_model_id or "gemma4" in normalized_model_id:
        return True
    if model_card is None:
        return False

    if model_card.vision is not None and model_card.vision.model_type == "gemma4":
        return True

    runtime = model_card.runtime
    if runtime is not None and (
        runtime.prompt_renderer == PromptRendererType.Gemma4
        or runtime.output_parser == OutputParserType.Gemma4
    ):
        return True

    tooling = model_card.tooling
    return tooling is not None and tooling.tool_call_format == ToolCallFormat.Gemma4


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


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
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()

        self.generator: Builder | InferenceGenerator = Builder(
            self.model_id,
            self.event_sender,
            self.cancel_receiver,
            self.shard_metadata.model_card,
        )

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            TextGeneration,
        ] = {}

        logger.info("runner created")
        self.update_status(RunnerIdle())

    @staticmethod
    def _summarize_text_generation_task(task: TextGeneration) -> str:
        """Return a compact log summary for text generation tasks."""
        params = task.task_params
        return (
            "TextGeneration("
            f"task_id={task.task_id!r}, "
            f"command_id={task.command_id!r}, "
            f"model={params.model!r}, "
            f"input_messages={len(params.input)}, "
            f"chat_template_messages={len(params.chat_template_messages or [])}, "
            f"images={len(params.images)}, "
            f"cached_image_indices={sorted(params.image_hashes.keys())}, "
            f"total_input_chunks={params.total_input_chunks}, "
            f"image_count={params.image_count}, "
            f"stream={params.stream}, "
            f"reasoning_effort={params.reasoning_effort!r}, "
            f"enable_thinking={params.enable_thinking!r})"
        )

    def _lifecycle_context(self) -> str:
        """Return stable runner identity fields for lifecycle logs."""
        return (
            f"instance_id={self.instance.instance_id}, "
            f"runner_id={self.runner_id}, "
            f"node_id={self.bound_instance.bound_node_id}, "
            f"device_rank={self.shard_metadata.device_rank}, "
            f"world_size={self.shard_metadata.world_size}, "
            f"layers={self.shard_metadata.start_layer}:{self.shard_metadata.end_layer}"
        )

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_first_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                assert isinstance(self.generator, Builder)
                logger.info(f"runner connecting ({self._lifecycle_context()})")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.generator.group = initialize_mlx(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                (
                    isinstance(self.current_status, RunnerConnected)
                    and self.generator.group is not None
                )
                or (
                    isinstance(self.current_status, RunnerIdle)
                    and self.generator.group is None
                )
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                logger.info(f"runner loading ({self._lifecycle_context()})")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                def on_model_load_timeout() -> None:
                    self.update_status(
                        RunnerFailed(error_message="Model loading timed out")
                    )
                    time.sleep(0.5)

                def on_layer_loaded(layers_loaded: int, total: int) -> None:
                    self.update_status(
                        RunnerLoading(layers_loaded=layers_loaded, total_layers=total)
                    )

                assert (
                    ModelTask.TextGeneration in self.shard_metadata.model_card.tasks
                ), f"Incorrect model task(s): {self.shard_metadata.model_card.tasks}"
                (
                    self.generator.inference_model,
                    self.generator.tokenizer,
                    self.generator.vision_processor,
                ) = load_mlx_items(
                    self.bound_instance,
                    self.generator.group,
                    on_timeout=on_model_load_timeout,
                    on_layer_loaded=on_layer_loaded,
                )

                self.generator = self.generator.build()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info(f"runner loaded ({self._lifecycle_context()})")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, InferenceGenerator)
                logger.info(f"runner warming up ({self._lifecycle_context()})")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                warmup_generator = self.generator
                assert isinstance(warmup_generator, (SequentialGenerator, BatchGenerator))
                group_size = (
                    warmup_generator.group.size() if warmup_generator.group else 1
                )
                if _should_skip_llm_warmup(
                    group_size,
                    self.model_id,
                    self.shard_metadata.model_card,
                ):
                    logger.warning(
                        "Skipping LLM warmup and marking runner ready "
                        "(temporary debug bypass; unset SKULK_SKIP_LLM_WARMUP "
                        "or set it to 0 to restore synthetic warmup)"
                    )
                else:
                    warmup_generator.warmup()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())
                logger.info(f"runner ready ({self._lifecycle_context()})")

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        if isinstance(self.generator, InferenceGenerator):
            self.generator.close()
        mx.clear_cache()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_text_generation(self, task: TextGeneration):
        assert isinstance(self.generator, InferenceGenerator)
        if task.trace_enabled:
            begin_trace_session(
                task.task_id,
                rank=self.device_rank,
                node_id=str(self.bound_instance.bound_node_id),
                model_id=str(self.model_id),
                task_kind="text",
                tags=["text_generation"],
            )
            record_trace_marker("queued", self.device_rank, task_id=task.task_id)
        self.active_tasks[task.task_id] = task
        self.generator.submit(task)

    def _flush_trace_session(self, task_id: TaskId) -> None:
        traces = pop_trace_session(task_id)
        self.event_sender.send(
            TracesCollected(
                task_id=task_id,
                rank=self.device_rank,
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

    def handle_generation_tasks(self, starting_task: TextGeneration):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, InferenceGenerator)

        logger.info(
            "received chat request: "
            f"{self._summarize_text_generation_task(starting_task)}"
        )
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_text_generation(starting_task)

        while self.active_tasks:
            results = self.generator.step()

            finished: list[TaskId] = []
            for task_id, result in results:
                match result:
                    case Cancelled():
                        task = self.active_tasks.get(task_id)
                        if task is not None and task.trace_enabled:
                            record_trace_marker(
                                "cancel", self.device_rank, task_id=task_id
                            )
                        finished.append(task_id)
                    case Finished():
                        task = self.active_tasks.get(task_id)
                        if task is not None and task.trace_enabled:
                            record_trace_marker(
                                "finish", self.device_rank, task_id=task_id
                            )
                        self.send_task_status(task_id, TaskStatus.Complete)
                        finished.append(task_id)
                    case _:
                        self.send_response(
                            result, self.active_tasks[task_id]
                        )

            for task_id in finished:
                task = self.active_tasks.get(task_id)
                if task is not None and task.trace_enabled:
                    self._flush_trace_session(task_id)
                self.active_tasks.pop(task_id, None)

            try:
                task = self.task_receiver.receive_nowait()

                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)

                match task:
                    case TextGeneration():
                        self.acknowledge_task(task)
                        self.submit_text_generation(task)
                    case Shutdown():
                        self.shutdown(task)
                        return ExitCode.Shutdown
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                        )

            except WouldBlock:
                pass

        self.update_status(RunnerReady())
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def send_response(
        self,
        response: GenerationResponse | ToolCallResponse,
        task: TextGeneration,
    ):
        match response:
            case GenerationResponse():
                if response.finish_reason == "error" and task.trace_enabled:
                    record_trace_marker(
                        "error",
                        self.device_rank,
                        task_id=task.task_id,
                        tags=["error"],
                        attrs={"message": response.text},
                    )
                if self.device_rank == 0 and response.finish_reason == "error":
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ErrorChunk(
                                error_message=response.text,
                                model=self.model_id,
                            ),
                        )
                    )

                elif self.device_rank == 0:
                    assert response.finish_reason not in (
                        "error",
                        "tool_calls",
                        "function_call",
                    )
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=TokenChunk(
                                model=self.model_id,
                                text=response.text,
                                token_id=response.token,
                                usage=response.usage,
                                finish_reason=response.finish_reason,
                                stats=response.stats,
                                logprob=response.logprob,
                                top_logprobs=response.top_logprobs,
                                is_thinking=response.is_thinking,
                            ),
                        )
                    )
            case ToolCallResponse():
                if task.trace_enabled:
                    record_trace_marker(
                        "tool_call_emitted",
                        self.device_rank,
                        task_id=task.task_id,
                        category="tooling",
                        tags=["tool_call"],
                        attrs={"tool_call_count": len(response.tool_calls)},
                    )
                if self.device_rank == 0:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ToolCallChunk(
                                tool_calls=response.tool_calls,
                                model=self.model_id,
                                usage=response.usage,
                                stats=response.stats,
                            ),
                        )
                    )


@dataclass
class Builder:
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    model_card: ModelCard | None = None
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None

    def build(
        self,
    ) -> InferenceGenerator:
        assert self.model_id
        # Some valid MLX model wrappers can be falsey, so bootstrap should only
        # reject missing objects here rather than truthy-but-non-empty values.
        assert self.inference_model is not None
        assert self.tokenizer is not None

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)

        device_rank = 0 if self.group is None else self.group.rank()
        kv_backend = get_kv_cache_backend()
        # TODO: Remove this forced sequential fallback once quantized KV cache
        # backends support BatchGenerator history merge/extract semantics.
        force_sequential_for_kv_backend = kv_backend in (
            "mlx_quantized",
            "turboquant",
            "turboquant_adaptive",
            "optiq",
        )
        force_sequential_for_gemma4 = _is_gemma4_model(
            self.model_id, self.model_card
        )
        no_batch_requested = os.environ.get("SKULK_NO_BATCH") or os.environ.get(
            "EXO_NO_BATCH"
        )
        if (
            no_batch_requested
            or force_sequential_for_kv_backend
            or force_sequential_for_gemma4
        ):
            if force_sequential_for_kv_backend and not no_batch_requested:
                logger.warning(
                    "Quantized KV cache backend does not yet support "
                    "batch/history mode; forcing SequentialGenerator "
                    f"(kv_backend={kv_backend})"
                )
                logger.info(f"using SequentialGenerator (kv_backend={kv_backend})")
            elif force_sequential_for_gemma4 and not no_batch_requested:
                logger.warning(
                    "Gemma 4 is not compatible with distributed BatchGenerator "
                    "mode yet; forcing SequentialGenerator"
                )
                logger.info("using SequentialGenerator (model_family=gemma4)")
            else:
                logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_card=self.model_card,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
        logger.info("using BatchGenerator")
        return BatchGenerator(
            model=self.inference_model,
            tokenizer=self.tokenizer,
            group=self.group,
            tool_parser=tool_parser,
            kv_prefix_cache=kv_prefix_cache,
            model_card=self.model_card,
            model_id=self.model_id,
            device_rank=device_rank,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
            vision_processor=vision_processor,
        )
