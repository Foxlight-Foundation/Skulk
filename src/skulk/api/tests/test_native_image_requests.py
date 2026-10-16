# pyright: reportPrivateUsage=false
"""Native image requests fail bounded contract violations before dispatch."""

import pytest
from fastapi import HTTPException

from skulk.api.main import API
from skulk.api.tests.test_text_image_transport import (
    _build_api,
)
from skulk.api.types.api import AdvancedImageParams
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.common import NodeId
from skulk.shared.types.state import State
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.runner.stable_diffusion_cpp.tests.test_native_image_adapter import (
    _card,
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
        _native_api()._validate_image_request(
            ModelId("owner/model"), "auto", count, None
        )
    assert error.value.status_code == 400


@pytest.mark.parametrize("size", ["1025x1024", "4096x4096"])
def test_native_api_refuses_canvas_before_gpu_work(size: str) -> None:
    with pytest.raises(HTTPException) as error:
        _native_api()._validate_image_request(ModelId("owner/model"), size, 1, None)
    assert error.value.status_code == 400


def test_native_api_binds_reference_sampling_prompt_and_seed_limits() -> None:
    api = _native_api()
    assert api._validate_image_request(ModelId("owner/model"), "2048x2048", 1, None, 2)
    for advanced, references, prompt in [
        (AdvancedImageParams(seed=1 << 63), 0, "a fox"),
        (AdvancedImageParams(num_inference_steps=9), 0, "a fox"),
        (None, 3, "a fox"),
        (None, 0, "x" * 32_769),
    ]:
        with pytest.raises(HTTPException) as error:
            api._validate_image_request(
                ModelId("owner/model"), "auto", 1, advanced, references, prompt=prompt
            )
        assert error.value.status_code == 400


@pytest.mark.parametrize("operation", ["generate", "edit"])
async def test_terminal_only_stream_routes_to_sse(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both real handlers honor stream=true without requesting costly previews."""
    import io
    from collections.abc import AsyncGenerator

    from fastapi import Request, UploadFile
    from fastapi.responses import StreamingResponse

    from skulk.api.types.api import ImageEditsTaskParams, ImageGenerationTaskParams
    from skulk.shared.types.commands import ImageEdits

    api = _native_api()
    request = Request({"type": "http", "method": "POST", "path": "/v1/images", "headers": []})
    async def validate(model: ModelId) -> ModelId:
        return model
    async def send(_command: object) -> None:
        return None
    async def stream(**_kwargs: object) -> AsyncGenerator[str, None]:
        yield "data: [DONE]\n\n"
    async def collect(**_kwargs: object) -> None:
        raise AssertionError("terminal-only streaming must not collect a JSON body")
    monkeypatch.setattr(api, "_validate_image_model", validate)
    monkeypatch.setattr(api, "_send", send)
    monkeypatch.setattr(api, "_generate_image_stream", stream)
    monkeypatch.setattr(api, "_collect_image_generation", collect)
    if operation == "generate":
        response = await api.image_generations(request, ImageGenerationTaskParams(model="owner/model", prompt="fox", stream=True, partial_images=0))
    else:
        async def edits(**kwargs: object) -> ImageEdits:
            assert kwargs["stream"] is True and kwargs["partial_images"] == 0
            return ImageEdits(owner_node=api.node_id, task_params=ImageEditsTaskParams(model="owner/model", prompt="fox", stream=True, partial_images=0))
        monkeypatch.setattr(api, "_send_image_edits_command", edits)
        response = await api.image_edits(request, image=UploadFile(io.BytesIO(b"reference")), images=None, prompt="fox", model="owner/model", n=1, size="auto", response_format="b64_json", input_fidelity="low", stream="true", partial_images="0", quality="medium", output_format="png", advanced_params=None)
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert [item async for item in response.body_iterator] == ["data: [DONE]\n\n"]


def test_api_rejects_sdxl_references_beyond_the_runner_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broader intrinsic profile cannot authorize silently ignored media."""
    from skulk.worker.runner.stable_diffusion_cpp.tests.test_native_image_adapter import (
        _sdxl_card,
    )

    api = _native_api()
    def card_for_instance(_instance: object):
        return _sdxl_card()
    monkeypatch.setattr(api, "_model_card_for_instance", card_for_instance)
    with pytest.raises(HTTPException) as error:
        api._validate_image_request(ModelId("owner/model"), "512x512", 1, None, 2)
    assert error.value.status_code == 400
    assert "initialization image" in str(error.value.detail)
