"""Native image requests fail bounded contract violations before dispatch."""

import pytest
from fastapi import HTTPException

from skulk.api.main import API
from skulk.api.tests.test_text_image_transport import (
    _build_api,  # pyright: ignore[reportPrivateUsage]
)
from skulk.api.types.api import AdvancedImageParams
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import NodeId
from skulk.shared.types.state import State
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.runner.stable_diffusion_cpp.tests.test_native_image_adapter import (
    _card,  # pyright: ignore[reportPrivateUsage]
)


def _native_api() -> API:
    card = _card(turbo=True)
    api, _, _, _ = _build_api()
    runner = RunnerId("native")
    node = NodeId("native-node")
    instance = MlxRingInstance(
        instance_id=InstanceId("native"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node: runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    world_size=1,
                    device_rank=0,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=52500,
    )
    api.state = State(instances={instance.instance_id: instance})
    return api


@pytest.mark.parametrize("count", [0, -1, 5])
def test_native_api_refuses_invalid_image_counts(count: int) -> None:
    with pytest.raises(HTTPException) as error:
        _native_api()._validate_image_request(  # pyright: ignore[reportPrivateUsage]
            ModelId("owner/model"), "auto", count, None
        )
    assert error.value.status_code == 400


@pytest.mark.parametrize("size", ["1025x1024", "4096x4096"])
def test_native_api_refuses_canvas_before_gpu_work(size: str) -> None:
    with pytest.raises(HTTPException) as error:
        _native_api()._validate_image_request(ModelId("owner/model"), size, 1, None)  # pyright: ignore[reportPrivateUsage]
    assert error.value.status_code == 400


def test_native_api_binds_reference_sampling_prompt_and_seed_limits() -> None:
    api = _native_api()
    assert api._validate_image_request(ModelId("owner/model"), "2048x2048", 1, None, 2)  # pyright: ignore[reportPrivateUsage]
    for advanced, references, prompt in [
        (AdvancedImageParams(seed=1 << 63), 0, "a fox"),
        (AdvancedImageParams(num_inference_steps=9), 0, "a fox"),
        (None, 3, "a fox"),
        (None, 0, "x" * 32_769),
    ]:
        with pytest.raises(HTTPException) as error:
            api._validate_image_request(  # pyright: ignore[reportPrivateUsage]
                ModelId("owner/model"), "auto", 1, advanced, references, prompt=prompt
            )
        assert error.value.status_code == 400
