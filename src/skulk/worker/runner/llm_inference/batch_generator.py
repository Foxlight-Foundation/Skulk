import itertools
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.constants import SKULK_MAX_CONCURRENT_REQUESTS
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.tracing import now_us, record_shared_span, record_trace_marker, trace
from skulk.shared.types.chunks import ErrorChunk, PrefillProgressChunk
from skulk.shared.types.common import ModelId
from skulk.shared.types.events import ChunkGenerated, Event
from skulk.shared.types.mlx import Model
from skulk.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TextGeneration
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.engines.mlx.cache import KVPrefixCache
from skulk.worker.engines.mlx.generator.batch_generate import SkulkBatchGenerator
from skulk.worker.engines.mlx.generator.context_admission import (
    ContextLengthExceededError,
)
from skulk.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    mlx_generate,
    warmup_inference,
)
from skulk.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    log_request_shape,
    mx_all_gather_tasks,
    mx_any,
)
from skulk.worker.engines.mlx.vision import (
    VisionPreprocessingError,
    VisionProcessor,
)
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.diagnostics import record_runner_phase, runner_phase

from .model_output_parsers import apply_all_parsers
from .tool_parsers import ToolParser


class Cancelled:
    pass


class Finished:
    pass


class GeneratorQueue[T]:
    def __init__(self):
        self._q = deque[T]()

    def push(self, t: T):
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


@dataclass
class _ActiveSequentialTask:
    task: TextGeneration
    mlx_generator: Generator[GenerationResponse]
    queue: GeneratorQueue[GenerationResponse]
    output_generator: Generator[GenerationResponse | ToolCallResponse | None]


@dataclass
class _ActiveBatchTask:
    task: TextGeneration
    queue: GeneratorQueue[GenerationResponse]
    output_generator: Generator[GenerationResponse | ToolCallResponse | None]


class InferenceGenerator(ABC):
    group: mx.distributed.Group | None
    _cancelled_tasks: set[TaskId]

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    @property
    def batches(self) -> bool:
        """Whether this generator decodes multiple requests together (#596).

        The performance-envelope tap keys concurrency behavior on this: a
        batching generator's aggregate throughput scales with concurrency, a
        serial one's does not. Serial by default; ``BatchGenerator`` overrides it,
        which is why MLX must report this rather than let the API guess (it cannot
        know from the outside which generator an in-process instance is running).
        """
        return False

    def admission_concurrency(self, task_id: TaskId) -> int:
        """In-flight requests this instance was serving when ``task_id`` was
        admitted (#596). 1 for a serial generator; ``BatchGenerator`` reports the
        batch position captured at admission so a burst spreads across buckets
        rather than collapsing onto the peak.
        """
        return 1

    def group_size(self) -> int:
        """Return the number of ranks participating in generation."""
        return self.group.size() if self.group is not None else 1

    def batches_for(self, task_id: TaskId) -> bool:
        """Return whether ``task_id`` used concurrent decode.

        Most generators have one serving mode for their lifetime, so the
        default delegates to :attr:`batches`. A modality-aware generator
        overrides this with task-local truth because text requests batch while
        image-bearing requests use the reference sequential path.
        """
        del task_id
        return self.batches

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: TextGeneration,
    ) -> None: ...

    @abstractmethod
    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, ToolCallResponse | GenerationResponse | Cancelled | Finished]
    ]: ...

    @abstractmethod
    def close(self) -> None: ...


SKULK_RUNNER_MUST_FAIL = "SKULK RUNNER MUST FAIL"
SKULK_RUNNER_MUST_OOM = "SKULK RUNNER MUST OOM"
SKULK_RUNNER_MUST_TIMEOUT = "SKULK RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input."""
    from skulk.worker.engines.mlx.utils_mlx import mlx_force_oom

    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if SKULK_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if SKULK_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if SKULK_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class SequentialGenerator(InferenceGenerator):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_card: ModelCard | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    mtp_weights: "dict[str, mx.array] | None" = None
    assistant_model: object | None = None
    check_for_cancel_every: int = 50
    # Static per-instance context ceiling for request admission (#145).
    context_token_limit: int | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: _ActiveSequentialTask | None = field(default=None, init=False)
    _logged_first_request_shape: bool = field(default=False, init=False)
    _decode_started_us: dict[TaskId, int] = field(default_factory=dict, init=False)
    _first_token_seen: set[TaskId] = field(default_factory=set, init=False)

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
            model_card=self.model_card,
        )

    def submit(
        self,
        task: TextGeneration,
    ) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        with runner_phase(
            "task_agreement",
            detail="mx_all_gather_tasks",
            attrs={"candidate_count": len(self._maybe_queue)},
        ):
            agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        self._queue.extend(task for task in self._maybe_queue if task in agreed)
        self._maybe_queue = [task for task in self._maybe_queue if task in different]
        record_runner_phase(
            "task_agreement",
            event="tasks_agreed",
            attrs={
                "agreed_count": len(agreed),
                "different_count": len(different),
                "queued_count": len(self._queue),
            },
        )

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])
                record_runner_phase(
                    "cancel_requested",
                    event="cancel_pipe_observed",
                    task_id=task_id,
                )

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)
        if agreed:
            record_runner_phase(
                "cancel_observed",
                event="cancellations_agreed",
                attrs={"task_ids": [str(task.task_id) for task in agreed]},
            )

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        if self._active is None:
            self.agree_on_tasks()

            if self._queue:
                self._start_next()
            else:
                # Report each cancellation EXACTLY ONCE, then forget it.
                # This idle path used to re-report every ever-cancelled id on
                # every step without clearing the set: an idle runner with N
                # cancelled tasks minted N fresh Cancelled results ~2x/s
                # forever, which the supervisor converted into an unbounded
                # TaskStatusUpdated(Cancelled)+TaskDeleted event storm that
                # drowned replicas and churned elections (#278). Ids in the
                # set here are neither queued nor active, so forgetting them
                # mirrors BatchGenerator._apply_cancellations; CANCEL_ALL is
                # forward-looking (cleared by the next submit) and survives.
                reported = [
                    (task_id, Cancelled())
                    for task_id in self._cancelled_tasks
                    if task_id != CANCEL_ALL_TASKS
                ]
                had_cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks
                self._cancelled_tasks.clear()
                if had_cancel_all:
                    self._cancelled_tasks.add(CANCEL_ALL_TASKS)
                return reported

        assert self._active is not None

        active = self._active
        task = active.task
        output: list[
            tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
        ] = []
        try:
            response = next(active.mlx_generator)
            active.queue.push(response)
            # drain potentially many responses every time
            while (parsed := next(active.output_generator, None)) is not None:
                output.append((task.task_id, parsed))
            if task.task_id not in self._first_token_seen:
                self._first_token_seen.add(task.task_id)
                self._decode_started_us[task.task_id] = now_us()
                record_runner_phase(
                    "decode_stream",
                    event="first_token",
                    task_id=task.task_id,
                    command_id=str(task.command_id),
                    include_memory=True,
                )
                record_trace_marker(
                    "first_token",
                    self.device_rank,
                    category="decode",
                    task_id=task.task_id,
                )

        except (StopIteration, PrefillCancelled):
            self._record_decode_span(task.task_id)
            record_runner_phase(
                "completion",
                event="decode_complete",
                task_id=task.task_id,
                command_id=str(task.command_id),
                include_memory=True,
            )
            output.append((task.task_id, Finished()))
            self._active = None
            if self._queue:
                self._start_next()

        except ContextLengthExceededError as e:
            # Deterministic pre-prefill rejection (#145): every rank raises
            # this in lockstep before any collective, so the task terminates
            # cleanly (error chunk + Finished) and the runner stays alive —
            # unlike the generic arm below, which escalates to a runner crash.
            self._record_decode_span(task.task_id)
            self._send_error(task, e)
            record_runner_phase(
                "completion",
                event="context_admission_rejected",
                task_id=task.task_id,
                command_id=str(task.command_id),
                include_memory=True,
            )
            output.append((task.task_id, Finished()))
            self._active = None
            if self._queue:
                self._start_next()

        except VisionPreprocessingError as e:
            # A request whose images could not be prepared fails as a request.
            # Every rank reaches the same verdict from the same images and the
            # same install, so terminating just this task keeps the ring in
            # lockstep, and the node goes on serving text and other vision
            # requests instead of crash-looping on one bad one.
            self._record_decode_span(task.task_id)
            self._send_error(task, e)
            record_runner_phase(
                "completion",
                event="vision_preprocessing_failed",
                task_id=task.task_id,
                command_id=str(task.command_id),
                include_memory=True,
            )
            output.append((task.task_id, Finished()))
            self._active = None
            if self._queue:
                self._start_next()

        except Exception as e:
            self._record_decode_span(task.task_id)
            self._send_error(task, e)
            self._active = None
            raise

        return itertools.chain(
            output,
            map(lambda task: (task, Cancelled()), self._cancelled_tasks),
        )

    def _start_next(self) -> None:
        task = self._queue.popleft()
        record_runner_phase(
            "task_submission",
            event="queue_admitted",
            task_id=task.task_id,
            command_id=str(task.command_id),
            attrs={"queue_depth": len(self._queue), "batch_size": 1},
        )
        record_trace_marker(
            "admitted_to_batch",
            self.device_rank,
            task_id=task.task_id,
            category="lifecycle",
            attrs={"batch_size": 1},
        )
        try:
            mlx_gen = self._build_generator(task)
        except Exception as e:
            self._send_error(task, e)
            raise
        queue = GeneratorQueue[GenerationResponse]()

        if task.task_params.bench:
            output_generator = queue.gen()
        else:
            with runner_phase(
                "parser",
                detail="apply_all_parsers",
                task_id=task.task_id,
                command_id=str(task.command_id),
            ):
                output_generator = apply_all_parsers(
                    queue.gen(),
                    apply_chat_template(
                        self.tokenizer, task.task_params, model_card=self.model_card
                    ),
                    self.tool_parser,
                    self.tokenizer,
                    type(self.model),
                    self.model_id,
                    task.task_params.tools,
                    self.model_card,
                    trace_task_id=task.task_id,
                    trace_rank=self.device_rank,
                )
        self._active = _ActiveSequentialTask(
            task=task,
            mlx_generator=mlx_gen,
            queue=queue,
            output_generator=output_generator,
        )

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        record_trace_marker(
            "error",
            self.device_rank,
            task_id=task.task_id,
            tags=["error"],
            attrs={"message": str(e)},
        )
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _build_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        _check_for_debug_prompts(task.task_params)
        with runner_phase(
            "prompt_build",
            detail="apply_chat_template",
            task_id=task.task_id,
            command_id=str(task.command_id),
            attrs={
                "image_count": task.task_params.image_count,
                "input_messages": len(task.task_params.input),
            },
        ), trace(
            "prompt_build",
            self.device_rank,
            "prompt",
            task_id=task.task_id,
        ):
            prompt = apply_chat_template(
                self.tokenizer, task.task_params, model_card=self.model_card
            )
        if not self._logged_first_request_shape:
            log_request_shape(
                "first-live-request",
                task.task_params,
                prompt,
                extra={
                    "command_id": str(task.command_id),
                    "device_rank": self.device_rank,
                    "generator": "sequential",
                    "task_id": str(task.task_id),
                },
            )
            self._logged_first_request_shape = True

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    raise PrefillCancelled()

                self.agree_on_tasks()

        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task.task_params,
            prompt=prompt,
            kv_prefix_cache=self.kv_prefix_cache,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            group=self.group,
            vision_processor=self.vision_processor,
            trace_task_id=task.task_id,
            trace_rank=self.device_rank,
            mtp_weights=self.mtp_weights,
            assistant_model=self.assistant_model,
            context_token_limit=self.context_token_limit,
            model_card=self.model_card,
        )

    def close(self) -> None:
        del self.model, self.tokenizer, self.group

    def _record_decode_span(self, task_id: TaskId) -> None:
        start_us = self._decode_started_us.pop(task_id, None)
        self._first_token_seen.discard(task_id)
        if start_us is None:
            return
        record_shared_span(
            [task_id],
            name="decode",
            start_us=start_us,
            duration_us=max(now_us() - start_us, 0),
            rank=self.device_rank,
            category="decode",
        )


@dataclass(eq=False)
class BatchGenerator(InferenceGenerator):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_card: ModelCard | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int = 50
    vision_processor: VisionProcessor | None = None
    mtp_weights: "dict[str, mx.array] | None" = None
    assistant_model: object | None = None
    # Static per-instance context ceiling for request admission (#145).
    context_token_limit: int | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _mlx_gen: SkulkBatchGenerator = field(init=False)
    _active_tasks: dict[int, _ActiveBatchTask] = field(default_factory=dict, init=False)
    _logged_first_request_shape: bool = field(default=False, init=False)
    _decode_started_us: dict[TaskId, int] = field(default_factory=dict, init=False)
    _first_token_seen: set[TaskId] = field(default_factory=set, init=False)
    # Batch position captured when each task was admitted, keyed by task (#596).
    # The runner reads this when it stamps the terminal stats; entries are pruned
    # at the top of the next step (see step()), by which point the runner has
    # already emitted the completed task's terminal chunk.
    _admission_concurrency: dict[TaskId, int] = field(
        default_factory=dict, init=False
    )

    @property
    def batches(self) -> bool:
        return True

    def admission_concurrency(self, task_id: TaskId) -> int:
        # Fall back to the live batch size if the admission capture is missing
        # (defensive; a dispatched task always has one), never below 1.
        return self._admission_concurrency.get(
            task_id, len(self._active_tasks) or 1
        )

    def __post_init__(self) -> None:
        self._mlx_gen = SkulkBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            vision_processor=self.vision_processor,
            context_token_limit=self.context_token_limit,
        )

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
            model_card=self.model_card,
        )

    def submit(
        self,
        task: TextGeneration,
    ) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        with runner_phase(
            "task_agreement",
            detail="batch_mx_all_gather_tasks",
            attrs={"candidate_count": len(self._maybe_queue)},
        ):
            agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        self._queue.extend(task for task in self._maybe_queue if task in agreed)
        self._maybe_queue = [task for task in self._maybe_queue if task in different]
        record_runner_phase(
            "task_agreement",
            event="batch_tasks_agreed",
            attrs={
                "agreed_count": len(agreed),
                "different_count": len(different),
                "queued_count": len(self._queue),
            },
        )

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])
                record_runner_phase(
                    "cancel_requested",
                    event="batch_cancel_pipe_observed",
                    task_id=task_id,
                )

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)
        if agreed:
            record_runner_phase(
                "cancel_observed",
                event="batch_cancellations_agreed",
                attrs={"task_ids": [str(task.task_id) for task in agreed]},
            )

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        # Drop admission records for tasks that completed or cancelled in the
        # previous step (#596): the runner has already read them when it emitted
        # those tasks' terminal chunks, so pruning here keeps the map bounded
        # without racing the read. Tasks admitted later in THIS step are recorded
        # after this prune, so they are unaffected.
        active_task_ids = {
            active.task.task_id for active in self._active_tasks.values()
        }
        self._admission_concurrency = {
            task_id: count
            for task_id, count in self._admission_concurrency.items()
            if task_id in active_task_ids
        }
        if not self._queue:
            self.agree_on_tasks()

        # Tasks rejected by context admission before reaching the engine
        # (#145): they terminate with an error chunk + Finished, and the
        # runner keeps serving.
        rejected: list[tuple[TaskId, Finished]] = []

        # Submit any queued tasks to the engine
        while self._queue and len(self._active_tasks) < SKULK_MAX_CONCURRENT_REQUESTS:
            task = self._queue.popleft()
            record_runner_phase(
                "task_submission",
                event="batch_queue_admitted",
                task_id=task.task_id,
                command_id=str(task.command_id),
                attrs={
                    "queue_depth": len(self._queue),
                    "batch_size": len(self._active_tasks) + 1,
                },
            )
            record_trace_marker(
                "admitted_to_batch",
                self.device_rank,
                task_id=task.task_id,
                category="lifecycle",
                attrs={"batch_size": len(self._active_tasks) + 1},
            )
            try:
                uid = self._start_task(task)
            except PrefillCancelled:
                continue
            except ContextLengthExceededError as e:
                # Deterministic pre-prefill rejection: all ranks agree (same
                # tokens, same static limit), so finishing the task here keeps
                # the batch engine in lockstep across the ring.
                self._send_error(task, e)
                record_runner_phase(
                    "completion",
                    event="context_admission_rejected",
                    task_id=task.task_id,
                    command_id=str(task.command_id),
                    include_memory=True,
                )
                rejected.append((task.task_id, Finished()))
                continue
            except VisionPreprocessingError as e:
                # Same lockstep argument as the admission arm above: the images
                # and the install are identical on every rank, so rejecting the
                # one request leaves the batch engine consistent.
                self._send_error(task, e)
                record_runner_phase(
                    "completion",
                    event="vision_preprocessing_failed",
                    task_id=task.task_id,
                    command_id=str(task.command_id),
                    include_memory=True,
                )
                rejected.append((task.task_id, Finished()))
                continue
            except Exception as e:
                self._send_error(task, e)
                raise

            queue = GeneratorQueue[GenerationResponse]()
            if task.task_params.bench:
                output_generator = queue.gen()
            else:
                output_generator = apply_all_parsers(
                    queue.gen(),
                    apply_chat_template(
                        self.tokenizer, task.task_params, model_card=self.model_card
                    ),
                    self.tool_parser,
                    self.tokenizer,
                    type(self.model),
                    self.model_id,
                    task.task_params.tools,
                    self.model_card,
                    trace_task_id=task.task_id,
                    trace_rank=self.device_rank,
                )
            # Capture this task's concurrency AT ADMISSION (#596): its position in
            # the batch is the current active count plus itself. Recording the
            # live batch size later (at completion) would collapse a burst onto the
            # peak; the admission value is the true throughput-vs-concurrency bucket.
            self._admission_concurrency[task.task_id] = len(self._active_tasks) + 1
            self._active_tasks[uid] = _ActiveBatchTask(
                task=task,
                queue=queue,
                output_generator=output_generator,
            )

        if not self._mlx_gen.has_work:
            return itertools.chain(rejected, self._apply_cancellations())

        shared_task_ids = [active.task.task_id for active in self._active_tasks.values()]
        decode_step_start_us = now_us()
        results = self._mlx_gen.step()
        decode_step_duration_us = max(now_us() - decode_step_start_us, 0)
        if shared_task_ids:
            record_shared_span(
                shared_task_ids,
                name="decode_step",
                start_us=decode_step_start_us,
                duration_us=decode_step_duration_us,
                rank=self.device_rank,
                category="decode",
                tags=["shared"],
                attrs={
                    "shared_span": True,
                    "batch_size": len(shared_task_ids),
                    "participant_task_ids": [str(task_id) for task_id in shared_task_ids],
                },
            )

        output: list[
            tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
        ] = []
        for uid, response in results:
            if uid not in self._active_tasks:
                # should we error here?
                logger.warning(f"{uid=} not found in active tasks")
                continue

            active = self._active_tasks[uid]
            task = active.task
            active.queue.push(response)
            # If a generator fails to parse for some reason and returns early, we should not crash
            while (parsed := next(active.output_generator, None)) is not None:
                output.append((task.task_id, parsed))
            if task.task_id not in self._first_token_seen:
                self._first_token_seen.add(task.task_id)
                self._decode_started_us[task.task_id] = now_us()
                record_runner_phase(
                    "decode_stream",
                    event="batch_first_token",
                    task_id=task.task_id,
                    command_id=str(task.command_id),
                    include_memory=True,
                )
                record_trace_marker(
                    "first_token",
                    self.device_rank,
                    category="decode",
                    task_id=task.task_id,
                )

            # check if original response was terminal and append a Finished()
            if response.finish_reason is not None:
                self._record_decode_span(task.task_id)
                record_runner_phase(
                    "completion",
                    event="batch_decode_complete",
                    task_id=task.task_id,
                    command_id=str(task.command_id),
                    include_memory=True,
                )
                output.append((task.task_id, Finished()))
                del self._active_tasks[uid]

        return itertools.chain(rejected, output, self._apply_cancellations())

    def _apply_cancellations(
        self,
    ) -> list[tuple[TaskId, Cancelled]]:
        if not self._cancelled_tasks:
            return []

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        uids_to_cancel: list[int] = []
        results: list[tuple[TaskId, Cancelled]] = []

        for uid, active in list(self._active_tasks.items()):
            task = active.task
            if task.task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(uid)
                results.append((task.task_id, Cancelled()))
                record_runner_phase(
                    "cancel_observed",
                    event="batch_cancel_applied",
                    task_id=task.task_id,
                    command_id=str(task.command_id),
                    include_memory=True,
                )
                self._record_decode_span(task.task_id)
                del self._active_tasks[uid]

        if uids_to_cancel:
            self._mlx_gen.cancel(uids_to_cancel)

        already_cancelled = {tid for tid, _ in results}
        for tid in self._cancelled_tasks:
            if tid != CANCEL_ALL_TASKS and tid not in already_cancelled:
                results.append((tid, Cancelled()))

        self._cancelled_tasks.clear()
        return results

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        record_trace_marker(
            "error",
            self.device_rank,
            task_id=task.task_id,
            tags=["error"],
            attrs={"message": str(e)},
        )
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _start_task(self, task: TextGeneration) -> int:
        _check_for_debug_prompts(task.task_params)
        with runner_phase(
            "prompt_build",
            detail="batch_apply_chat_template",
            task_id=task.task_id,
            command_id=str(task.command_id),
            attrs={
                "image_count": task.task_params.image_count,
                "input_messages": len(task.task_params.input),
            },
        ), trace(
            "prompt_build",
            self.device_rank,
            "prompt",
            task_id=task.task_id,
        ):
            prompt = apply_chat_template(
                self.tokenizer, task.task_params, model_card=self.model_card
            )
        if not self._logged_first_request_shape:
            log_request_shape(
                "first-live-request",
                task.task_params,
                prompt,
                extra={
                    "command_id": str(task.command_id),
                    "device_rank": self.device_rank,
                    "generator": "batch",
                    "task_id": str(task.task_id),
                },
            )
            self._logged_first_request_shape = True

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)

                self.agree_on_tasks()

        return self._mlx_gen.submit(
            task_id=task.task_id,
            task_params=task.task_params,
            prompt=prompt,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            trace_rank=self.device_rank,
        )

    def close(self) -> None:
        self._mlx_gen.close()
        del self.model, self.tokenizer, self.group

    def _record_decode_span(self, task_id: TaskId) -> None:
        start_us = self._decode_started_us.pop(task_id, None)
        self._first_token_seen.discard(task_id)
        if start_us is None:
            return
        record_shared_span(
            [task_id],
            name="decode",
            start_us=start_us,
            duration_us=max(now_us() - start_us, 0),
            rank=self.device_rank,
            category="decode",
        )


GenerationMode = Literal["text_batch", "vision_sequential"]


@dataclass(eq=False)
class DualModeGenerator(InferenceGenerator):
    """Schedule text through batching and image requests through reference decode.

    Native VLM families need request-local multimodal position and model state
    that MLX-LM's batch primitive cannot yet carry per sequence. Forcing the
    whole model instance onto ``SequentialGenerator`` preserves image
    correctness but also serializes ordinary text. This coordinator keeps the
    two proven engines mutually exclusive over one shared model:

    * consecutive text-only tasks form a batch cohort;
    * one image-bearing task runs through reference generation;
    * FIFO mode boundaries prevent either modality from starving the other.

    Task and cancellation agreement happens before a cohort is delegated, so
    every distributed rank changes mode in the same order. The child generators
    retain their existing per-mode agreement and cancellation behavior.
    """

    text_generator: BatchGenerator
    vision_generator: SequentialGenerator
    group: mx.distributed.Group | None
    cancel_receiver: MpReceiver[TaskId]
    text_cancel_sender: MpSender[TaskId]
    vision_cancel_sender: MpSender[TaskId]

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _active_mode: GenerationMode | None = field(default=None, init=False)
    _active_task_ids: set[TaskId] = field(default_factory=set, init=False)
    _task_modes: dict[TaskId, GenerationMode] = field(
        default_factory=dict, init=False
    )
    _completed_task_ids: set[TaskId] = field(default_factory=set, init=False)
    _step_count: int = field(default=0, init=False)

    @property
    def batches(self) -> bool:
        """Return the instance capability; task-local truth uses ``batches_for``."""
        return True

    def batches_for(self, task_id: TaskId) -> bool:
        """Return whether this task used the text batch path."""
        return self._task_modes.get(task_id) == "text_batch"

    def admission_concurrency(self, task_id: TaskId) -> int:
        """Return task-local admission width from the selected child engine."""
        if self._task_modes.get(task_id) == "text_batch":
            return self.text_generator.admission_concurrency(task_id)
        return 1

    def warmup(self) -> None:
        """Warm the shared model once through the reference-safe path."""
        self.vision_generator.warmup()
        self.text_generator.check_for_cancel_every = (
            self.vision_generator.check_for_cancel_every
        )

    def submit(self, task: TextGeneration) -> None:
        """Queue one task for distributed agreement and modality scheduling."""
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, ToolCallResponse | GenerationResponse | Cancelled | Finished]
    ]:
        """Advance the active cohort without overlapping the two generators."""
        self._step_count += 1
        self._prune_completed_metadata()

        outer_results: list[tuple[TaskId, Cancelled]] = []
        if self._should_coordinate():
            outer_results.extend(self._agree_on_cancellations())
            self._agree_on_tasks()

        self._dispatch_ready_tasks()
        if self._active_mode is None:
            return outer_results

        delegate: InferenceGenerator = (
            self.text_generator
            if self._active_mode == "text_batch"
            else self.vision_generator
        )
        delegated_results = list(delegate.step())
        for task_id, result in delegated_results:
            if not isinstance(result, (Cancelled, Finished)):
                continue
            self._active_task_ids.discard(task_id)
            self._all_tasks.pop(task_id, None)
            self._completed_task_ids.add(task_id)

        if not self._active_task_ids:
            record_runner_phase(
                "idle",
                event="dual_mode_cohort_complete",
                attrs={"mode": self._active_mode},
            )
            self._active_mode = None

        return itertools.chain(outer_results, delegated_results)

    def close(self) -> None:
        """Close both child engines after their shared model is idle."""
        self.vision_generator.close()
        self.text_generator.close()
        self.text_cancel_sender.close()
        self.vision_cancel_sender.close()

    def _should_coordinate(self) -> bool:
        """Coordinate promptly when idle and periodically while decoding.

        Single-rank operation has no collective cost, so it accepts new work on
        every step. Distributed reference generation already polls for new work
        periodically; using the same cadence avoids adding a collective to every
        vision token while keeping mode changes rank-symmetric.
        """
        if self.group is None or not self._active_task_ids:
            return True
        interval = max(
            1,
            min(
                self.text_generator.check_for_cancel_every,
                self.vision_generator.check_for_cancel_every,
            ),
        )
        return self._step_count % interval == 0

    def _agree_on_tasks(self) -> None:
        agreed, different = mx_all_gather_tasks(
            self._maybe_queue,
            self.group,
            preserve_rank_zero_order=True,
        )
        queued_ids = {task.task_id for task in self._queue}
        for task in agreed:
            if (
                task.task_id not in queued_ids
                and task.task_id not in self._cancelled_tasks
            ):
                self._queue.append(task)
                queued_ids.add(task.task_id)
        self._maybe_queue = [
            task for task in different if task.task_id not in self._cancelled_tasks
        ]
        if agreed or different:
            record_runner_phase(
                "task_agreement",
                event="dual_mode_tasks_agreed",
                attrs={
                    "agreed_count": len(agreed),
                    "different_count": len(different),
                    "queued_count": len(self._queue),
                },
            )

    def _agree_on_cancellations(self) -> list[tuple[TaskId, Cancelled]]:
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            task = self._all_tasks.get(task_id)
            if task is not None:
                self._maybe_cancel.append(task)

        cancel_all = mx_any(has_cancel_all, self.group)
        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._maybe_cancel = list(different)
        agreed_ids = {task.task_id for task in agreed}
        if cancel_all:
            agreed_ids.update(self._all_tasks)
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)
        self._cancelled_tasks.update(agreed_ids)

        if not agreed_ids:
            return []

        active_ids = agreed_ids & self._active_task_ids
        if active_ids:
            sender = (
                self.text_cancel_sender
                if self._active_mode == "text_batch"
                else self.vision_cancel_sender
            )
            if cancel_all:
                sender.send(CANCEL_ALL_TASKS)
            else:
                for task_id in active_ids:
                    sender.send(task_id)

        queued_ids = agreed_ids - self._active_task_ids
        if not queued_ids:
            return []

        self._queue = deque(
            task for task in self._queue if task.task_id not in queued_ids
        )
        self._maybe_queue = [
            task for task in self._maybe_queue if task.task_id not in queued_ids
        ]
        for task_id in queued_ids:
            self._all_tasks.pop(task_id, None)
            self._task_modes.pop(task_id, None)
        record_runner_phase(
            "cancel_observed",
            event="dual_mode_queued_cancellations",
            attrs={"task_ids": [str(task_id) for task_id in sorted(queued_ids)]},
        )
        return [(task_id, Cancelled()) for task_id in sorted(queued_ids)]

    def _dispatch_ready_tasks(self) -> None:
        if self._active_mode is None:
            if not self._queue:
                return
            self._active_mode = self._mode_for(self._queue[0])
            record_runner_phase(
                "task_submission",
                event="dual_mode_cohort_started",
                attrs={"mode": self._active_mode, "queued_count": len(self._queue)},
            )

        if self._active_mode == "vision_sequential":
            if self._active_task_ids or not self._queue:
                return
            task = self._queue[0]
            if self._mode_for(task) != "vision_sequential":
                return
            self._queue.popleft()
            self._assign(task, "vision_sequential")
            return

        # Do not place overflow into BatchGenerator's private waiting queues.
        # The coordinator owns not-yet-admitted work so it can cancel that work
        # exactly once without a child later admitting an already-finalized task.
        while (
            self._queue
            and self._mode_for(self._queue[0]) == "text_batch"
            and len(self._active_task_ids) < SKULK_MAX_CONCURRENT_REQUESTS
        ):
            self._assign(self._queue.popleft(), "text_batch")

    def _assign(self, task: TextGeneration, mode: GenerationMode) -> None:
        self._active_task_ids.add(task.task_id)
        self._task_modes[task.task_id] = mode
        if mode == "text_batch":
            self.text_generator.submit(task)
        else:
            self.vision_generator.submit(task)

    @staticmethod
    def _mode_for(task: TextGeneration) -> GenerationMode:
        # Runner dispatch receives fully assembled new and cached images. The
        # concrete payload list is therefore the authoritative modality signal;
        # ``image_count`` alone excludes cached-image reuse.
        return "vision_sequential" if task.task_params.images else "text_batch"

    def _prune_completed_metadata(self) -> None:
        for task_id in self._completed_task_ids:
            self._task_modes.pop(task_id, None)
            self._cancelled_tasks.discard(task_id)
        self._completed_task_ids.clear()
