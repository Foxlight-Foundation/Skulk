"""Request-aware scheduling between MLX text batching and reference vision."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from skulk.utils.channels import MpReceiver, mp_channel
from skulk.worker.runner.llm_inference.batch_generator import (
    Cancelled,
    DualModeGenerator,
    Finished,
    InferenceGenerator,
)


@dataclass(eq=False)
class _FakeGenerator(InferenceGenerator):
    batching: bool
    submitted: list[TextGeneration] = field(default_factory=list)
    completed: deque[TaskId] = field(default_factory=deque)
    step_calls: int = 0
    warmup_calls: int = 0
    close_calls: int = 0
    check_for_cancel_every: int = 50
    _cancelled_tasks: set[TaskId] = field(default_factory=set)

    @property
    def batches(self) -> bool:
        return self.batching

    def admission_concurrency(self, task_id: TaskId) -> int:
        for index, task in enumerate(self.submitted, start=1):
            if task.task_id == task_id:
                return index
        return 1

    def warmup(self) -> None:
        self.warmup_calls += 1

    def submit(self, task: TextGeneration) -> None:
        self.submitted.append(task)

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, ToolCallResponse | GenerationResponse | Cancelled | Finished]
    ]:
        self.step_calls += 1
        output = [(task_id, Finished()) for task_id in self.completed]
        self.completed.clear()
        return output

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class _FakeCancelPipe:
    pending: list[TaskId] = field(default_factory=list)

    def send(self, task_id: TaskId) -> None:
        self.pending.append(task_id)

    def collect(self) -> list[TaskId]:
        pending = list(self.pending)
        self.pending.clear()
        return pending


def _task(name: str, *, vision: bool = False) -> TextGeneration:
    task_id = TaskId(str(uuid5(NAMESPACE_URL, f"dual-mode:{name}")))
    return TextGeneration(
        task_id=task_id,
        command_id=CommandId(f"command-{name}"),
        instance_id=InstanceId("instance"),
        task_params=TextGenerationTaskParams(
            model=ModelId("local/test-vlm"),
            input=[InputMessage(role="user", content="describe the input")],
            images=["aW1hZ2U="] if vision else [],
            image_count=1 if vision else 0,
            stream=True,
        ),
    )


def _dual() -> tuple[
    DualModeGenerator,
    _FakeGenerator,
    _FakeGenerator,
    _FakeCancelPipe,
    MpReceiver[TaskId],
    MpReceiver[TaskId],
]:
    cancel_pipe = _FakeCancelPipe()
    text_cancel_sender, text_cancel_receiver = mp_channel[TaskId]()
    vision_cancel_sender, vision_cancel_receiver = mp_channel[TaskId]()
    text = _FakeGenerator(batching=True)
    vision = _FakeGenerator(batching=False)
    dual = DualModeGenerator(
        text_generator=text,  # type: ignore[arg-type]
        vision_generator=vision,  # type: ignore[arg-type]
        group=None,
        cancel_receiver=cast(MpReceiver[TaskId], cast(object, cancel_pipe)),
        text_cancel_sender=text_cancel_sender,
        vision_cancel_sender=vision_cancel_sender,
    )
    return (
        dual,
        text,
        vision,
        cancel_pipe,
        text_cancel_receiver,
        vision_cancel_receiver,
    )


def test_dual_mode_preserves_fifo_cohorts_without_engine_overlap() -> None:
    dual, text, vision, _, _, _ = _dual()
    text_one = _task("text-one")
    text_two = _task("text-two")
    image = _task("image", vision=True)
    text_three = _task("text-three")
    for task in (text_one, text_two, image, text_three):
        dual.submit(task)

    assert list(dual.step()) == []
    assert [task.task_id for task in text.submitted] == [
        text_one.task_id,
        text_two.task_id,
    ]
    assert vision.submitted == []
    assert dual.batches_for(text_one.task_id) is True
    assert dual.admission_concurrency(text_two.task_id) == 2

    text.completed.extend((text_one.task_id, text_two.task_id))
    assert [task_id for task_id, _ in dual.step()] == [
        text_one.task_id,
        text_two.task_id,
    ]
    # Terminal stats are stamped after step() returns, so task-local provenance
    # must remain available until the following scheduler step.
    assert dual.batches_for(text_one.task_id) is True
    assert vision.step_calls == 0

    assert list(dual.step()) == []
    assert [task.task_id for task in vision.submitted] == [image.task_id]
    assert [task.task_id for task in text.submitted] == [
        text_one.task_id,
        text_two.task_id,
    ]
    assert dual.batches_for(image.task_id) is False

    vision.completed.append(image.task_id)
    assert [task_id for task_id, _ in dual.step()] == [image.task_id]

    assert list(dual.step()) == []
    assert [task.task_id for task in text.submitted][-1] == text_three.task_id


def test_dual_mode_cancels_queued_other_modality_without_switching() -> None:
    dual, text, vision, cancel_sender, _, _ = _dual()
    text_task = _task("text")
    image_task = _task("image", vision=True)
    dual.submit(text_task)
    dual.submit(image_task)

    assert list(dual.step()) == []
    cancel_sender.send(image_task.task_id)

    results = list(dual.step())
    assert len(results) == 1
    assert results[0][0] == image_task.task_id
    assert isinstance(results[0][1], Cancelled)
    assert vision.submitted == []
    assert [task.task_id for task in text.submitted] == [text_task.task_id]


def test_dual_mode_keeps_text_overflow_owned_and_cancellable() -> None:
    dual, text, vision, cancel_sender, text_cancel_receiver, _ = _dual()
    tasks = [_task(f"text-{index}") for index in range(10)]
    for task in tasks:
        dual.submit(task)

    assert list(dual.step()) == []
    assert [task.task_id for task in text.submitted] == [
        task.task_id for task in tasks[:8]
    ]

    cancel_sender.send(tasks[8].task_id)
    results = list(dual.step())

    assert [task_id for task_id, _ in results] == [tasks[8].task_id]
    assert isinstance(results[0][1], Cancelled)
    assert text_cancel_receiver.collect() == []
    assert [task.task_id for task in text.submitted] == [
        task.task_id for task in tasks[:8]
    ]
    assert vision.submitted == []

    text.completed.extend(task.task_id for task in tasks[:8])
    assert len(list(dual.step())) == 8
    assert list(dual.step()) == []
    assert [task.task_id for task in text.submitted] == [
        *(task.task_id for task in tasks[:8]),
        tasks[9].task_id,
    ]


def test_dual_mode_forwards_active_cancellation_to_selected_engine() -> None:
    dual, text, vision, cancel_sender, text_cancel_receiver, vision_cancel_receiver = (
        _dual()
    )
    text_task = _task("active-text")
    dual.submit(text_task)
    assert list(dual.step()) == []

    cancel_sender.send(text_task.task_id)
    assert list(dual.step()) == []

    assert text_cancel_receiver.receive() == text_task.task_id
    assert vision_cancel_receiver.collect() == []
    assert [task.task_id for task in text.submitted] == [text_task.task_id]
    assert vision.submitted == []


def test_dual_mode_forwards_cancel_all_and_drops_queued_work() -> None:
    dual, text, vision, cancel_sender, text_cancel_receiver, vision_cancel_receiver = (
        _dual()
    )
    text_task = _task("active-text")
    image_task = _task("queued-image", vision=True)
    dual.submit(text_task)
    dual.submit(image_task)
    assert list(dual.step()) == []

    cancel_sender.send(CANCEL_ALL_TASKS)
    results = list(dual.step())

    assert text_cancel_receiver.receive() == CANCEL_ALL_TASKS
    assert vision_cancel_receiver.collect() == []
    assert [task_id for task_id, _ in results] == [image_task.task_id]
    assert isinstance(results[0][1], Cancelled)
    assert [task.task_id for task in text.submitted] == [text_task.task_id]
    assert vision.submitted == []


def test_dual_mode_warms_shared_model_once_and_closes_both_engines() -> None:
    dual, text, vision, _, _, _ = _dual()

    assert dual.group_size() == 1
    dual.warmup()
    assert vision.warmup_calls == 1
    assert text.warmup_calls == 0

    dual.close()
    assert vision.close_calls == 1
    assert text.close_calls == 1
