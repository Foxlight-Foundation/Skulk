"""The worker's chunked image-edit reassembly must preserve owner_node (#310).

When an image edit arrives chunked, the worker reassembles the image and injects
it into the task before dispatching to the runner. That reconstruction must carry
``owner_node`` through, or the rank-0 supervisor stamps ``owner_node=None`` and
the Zenoh data plane routes the output to a key no node subscribes to, silently
dropping the response (#279 Phase 2).
"""

from skulk.api.types import ImageEditsTaskParams
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import ImageEdits
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.main import (
    _inject_assembled_image_edit,  # pyright: ignore[reportPrivateUsage]
)


def _task(owner: NodeId | None) -> ImageEdits:
    return ImageEdits(
        command_id=CommandId("cmd-1"),
        owner_node=owner,
        instance_id=InstanceId("instance-1"),
        task_params=ImageEditsTaskParams(
            prompt="make it blue",
            model=ModelId("mlx-community/test"),
            image_data="",
            total_input_chunks=3,
        ),
    )


def test_assembly_preserves_owner_node_and_injects_image() -> None:
    task = _task(NodeId("api-node-7"))
    assembled = "ASSEMBLED_IMAGE_BYTES"

    result = _inject_assembled_image_edit(task, assembled)

    # The owning API node survives reassembly (the #310 regression).
    assert result.owner_node == NodeId("api-node-7")
    # The reassembled image is injected, other params untouched.
    assert result.task_params.image_data == assembled
    assert result.task_params.prompt == "make it blue"
    assert result.task_params.total_input_chunks == 3
    # Task identity is preserved.
    assert result.command_id == task.command_id
    assert result.instance_id == task.instance_id


def test_assembly_keeps_owner_node_none_when_unset() -> None:
    # The gossipsub path leaves owner_node None; reassembly must not invent one.
    result = _inject_assembled_image_edit(_task(None), "img")
    assert result.owner_node is None
    assert result.task_params.image_data == "img"
