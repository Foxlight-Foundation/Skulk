from collections.abc import Callable
from typing import cast

from exo.api.types import ImageEditsTaskParams
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId
from exo.worker import main as worker_main_module


def _summarize_worker_task(task: object) -> str:
    module_dict = cast(dict[str, object], worker_main_module.__dict__)
    return cast(Callable[[object], str], module_dict["_summarize_worker_task"])(task)


def test_summarize_worker_task_redacts_text_generation_payloads() -> None:
    task = TextGeneration(
        task_id=TaskId("task-1"),
        command_id=CommandId("command-1"),
        instance_id=InstanceId("instance-1"),
        task_params=TextGenerationTaskParams(
            model=ModelId("model-a"),
            input=[InputMessage(role="user", content="secret prompt")],
            stream=True,
        ),
    )

    summary = _summarize_worker_task(task)

    assert "secret prompt" not in summary
    assert "input_messages=1" in summary
    assert "model='model-a'" in summary


def test_summarize_worker_task_redacts_image_edit_payloads() -> None:
    from exo.shared.types.tasks import ImageEdits

    task = ImageEdits(
        task_id=TaskId("task-2"),
        command_id=CommandId("command-2"),
        instance_id=InstanceId("instance-2"),
        task_params=ImageEditsTaskParams(
            image_data="A" * 128,
            total_input_chunks=3,
            prompt="remove background",
            model="image-model",
        ),
    )

    summary = _summarize_worker_task(task)

    assert "remove background" not in summary
    assert "AAAA" not in summary
    assert "has_inline_image_data=True" in summary
    assert "total_input_chunks=3" in summary
