# pyright: reportPrivateUsage=false
"""Image qualification must bind one exact instance in both HTTP wire modes."""

from collections.abc import AsyncGenerator

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.main import API
from skulk.api.tests.test_instance_chat import _ALLOWED, _MODEL, _app
from skulk.api.types.api import ImageData, ImageGenerationResponse
from skulk.shared.models.model_cards import ModelId, ModelTask
from skulk.shared.types.commands import ImageGeneration
from skulk.shared.types.common import NodeId
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.runner.stable_diffusion_cpp.tests.test_native_image_adapter import (
    _card,
)


def _image_app(monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, API, list[ImageGeneration]]:
    """Exercise the real adapter, card admission and command construction."""
    _, api, _ = _app(monkeypatch)
    api.node_id = NodeId("api")
    instances = {}
    for name, instance in api.state.instances.items():
        card = _card(turbo=True).model_copy(update={"model_id": _MODEL})
        if name == InstanceId("foreign"):
            assert card.image is not None
            card = card.model_copy(update={"image": card.image.model_copy(update={"dimension_multiple": 64})})
        assignments = instance.shard_assignments.model_copy(update={
            "runner_to_shard": {
                runner: shard.model_copy(update={"model_card": card})
                for runner, shard in instance.shard_assignments.runner_to_shard.items()
            }
        })
        instances[name] = instance.model_copy(update={"shard_assignments": assignments})
    api.state = api.state.model_copy(update={"instances": instances})
    dispatched: list[ImageGeneration] = []

    async def validate(model: ModelId) -> ModelId:
        return model

    async def send(command: ImageGeneration) -> None:
        dispatched.append(command)

    async def collect(**_kwargs: object) -> ImageGenerationResponse:
        return ImageGenerationResponse(data=[ImageData(b64_json="rendered-image")])

    async def stream(**_kwargs: object) -> AsyncGenerator[str, None]:
        yield 'data: {"b64_json":"rendered-image"}\n\n'

    monkeypatch.setattr(api, "_validate_image_model", validate)
    monkeypatch.setattr(api, "_send", send)
    monkeypatch.setattr(api, "_collect_image_generation", collect)
    monkeypatch.setattr(api, "_generate_image_stream", stream)
    app = FastAPI()
    app.post("/v1/instances/{instance_id}/images/generations", response_model=None)(api.instance_image_generations)
    return app, api, dispatched


@pytest.mark.parametrize("stream", [False, True])
async def test_scoped_image_generation_uses_selected_card_and_replica(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    """The other replica's incompatible canvas cannot influence the selected card."""
    app, _, sent = _image_app(monkeypatch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/instances/allowed/images/generations", json={
            "model": _MODEL, "prompt": "a blue square", "size": "96x96", "stream": stream,
            "allowed_node_install_ids": [str(_ALLOWED)], "advanced_params": {"seed": 42},
        })
    assert response.status_code == 200
    assert response.headers["X-Skulk-Instance-Id"] == "allowed"
    assert "rendered-image" in response.text
    assert len(sent) == 1 and sent[0].target_instance_id == InstanceId("allowed")
    assert sent[0].task_params.model == _MODEL


@pytest.mark.parametrize(("instance", "model", "scope", "status"), [
    ("foreign", _MODEL, [str(_ALLOWED)], 409),
    ("absent", _MODEL, [str(_ALLOWED)], 404),
    ("allowed", "org/other", [str(_ALLOWED)], 409),
    ("allowed", _MODEL, [], 422),
    ("allowed", _MODEL, ["invalid"], 422),
])
async def test_scoped_image_refusal_dispatches_nothing(monkeypatch: pytest.MonkeyPatch, instance: str, model: str, scope: list[str], status: int) -> None:
    """Invalid identities and foreign replicas fail before any GPU work."""
    app, _, sent = _image_app(monkeypatch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/v1/instances/{instance}/images/generations", json={
            "model": model, "prompt": "a blue square", "allowed_node_install_ids": scope,
        })
    assert response.status_code == status and not sent


async def test_scoped_image_request_rejects_missing_rank_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """An instance with unprovable installation scope cannot receive inference."""
    app, api, sent = _image_app(monkeypatch)
    api._telemetry_view.node_identities.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/instances/allowed/images/generations", json={
            "model": _MODEL, "prompt": "blue", "allowed_node_install_ids": [str(_ALLOWED)],
        })
    assert response.status_code == 409 and not sent


async def test_scoped_image_request_requires_an_image_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """A text-only instance is rejected even when the alias and scope match."""
    app, api, sent = _image_app(monkeypatch)
    instance = api.state.instances[InstanceId("allowed")]
    assignments = instance.shard_assignments.model_copy(update={"runner_to_shard": {
        runner: shard.model_copy(update={"model_card": shard.model_card.model_copy(update={"tasks": [ModelTask.TextGeneration]})})
        for runner, shard in instance.shard_assignments.runner_to_shard.items()
    }})
    api.state = api.state.model_copy(update={"instances": {**api.state.instances, instance.instance_id: instance.model_copy(update={"shard_assignments": assignments})}})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/instances/allowed/images/generations", json={
            "model": _MODEL, "prompt": "blue", "allowed_node_install_ids": [str(_ALLOWED)],
        })
    assert response.status_code == 409 and not sent
