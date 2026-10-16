# pyright: reportPrivateUsage=false
"""Exact-instance chat must preserve placement scope and never select a replica."""

from collections.abc import AsyncGenerator
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.main import API
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.chunks import TokenChunk
from skulk.shared.types.commands import TextGeneration
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeIdentity
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata

_ALLOWED = UUID("11111111-1111-4111-8111-111111111111")
_FOREIGN = UUID("22222222-2222-4222-8222-222222222222")
_MODEL = ModelId("org/reviewer")


def _instance(name: str) -> MlxRingInstance:
    """Build a ready single-rank replica with the same exact model alias."""
    card = ModelCard(
        model_id=_MODEL,
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    runner = RunnerId(name)
    return MlxRingInstance(
        instance_id=InstanceId(name),
        shard_assignments=ShardAssignments(
            model_id=_MODEL,
            node_to_runner={NodeId(name): runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
        ),
        hosts_by_node={NodeId(name): []},
        ephemeral_port=52416,
    )


def _app(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, API, list[InstanceId | None]]:
    """Use the real HTTP adapter and dispatch seam with two competing replicas."""
    api = object.__new__(API)
    api.state = State(
        instances={
            InstanceId(name): _instance(name) for name in ("foreign", "allowed")
        },
        runners={RunnerId(name): RunnerReady() for name in ("foreign", "allowed")},
    )
    api._extensions = None
    api._telemetry_view = TelemetryView()
    api._telemetry_view.node_identities = {
        NodeId("foreign"): NodeIdentity(node_install_id=_FOREIGN),
        NodeId("allowed"): NodeIdentity(node_install_id=_ALLOWED),
    }
    dispatched: list[InstanceId | None] = []

    async def dispatch(
        params: TextGenerationTaskParams,
        *,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        dispatched.append(target_instance_id)
        return TextGeneration(
            task_params=params, command_id=CommandId("scoped-command")
        )

    async def chunks(
        *_args: object, **_kwargs: object
    ) -> AsyncGenerator[TokenChunk, None]:
        yield TokenChunk(
            model=_MODEL,
            text="scoped reply",
            token_id=1,
            usage=None,
            finish_reason="stop",
        )

    monkeypatch.setattr(api, "_send_text_generation_with_images", dispatch)
    monkeypatch.setattr(api, "_tapped_text_stream", chunks)
    app = FastAPI()
    app.post("/v1/instances/{instance_id}/chat/completions")(
        api.instance_chat_completions_route
    )
    return app, api, dispatched


@pytest.mark.parametrize("stream", [False, True])
async def test_scoped_http_chat_dispatches_only_the_selected_replica(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """Both wire modes bind dispatch and response evidence to the approved replica."""
    app, _, dispatched = _app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/instances/allowed/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "allowed_node_install_ids": [str(_ALLOWED)],
                "stream": stream,
            },
        )
    assert response.status_code == 200
    assert response.headers["X-Skulk-Instance-Id"] == "allowed"
    assert "scoped reply" in response.text
    assert dispatched == [InstanceId("allowed")]


@pytest.mark.parametrize(
    ("instance", "model", "installations", "status"),
    [
        ("foreign", _MODEL, [str(_ALLOWED)], 409),
        ("absent", _MODEL, [str(_ALLOWED)], 404),
        ("allowed", "org/other", [str(_ALLOWED)], 409),
        ("allowed", _MODEL, [], 422),
    ],
)
async def test_scoped_http_refusal_never_dispatches_to_another_replica(
    monkeypatch: pytest.MonkeyPatch,
    instance: str,
    model: str,
    installations: list[str],
    status: int,
) -> None:
    """No missing target, wrong model, or invalid boundary can trigger fallback."""
    app, _, dispatched = _app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/v1/instances/{instance}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "allowed_node_install_ids": installations,
            },
        )
    assert response.status_code == status
    assert not dispatched


async def test_missing_installation_identity_fails_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node without stable identity cannot satisfy the request's allowlist."""
    app, api, dispatched = _app(monkeypatch)
    api._telemetry_view.node_identities.clear()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/instances/allowed/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "allowed_node_install_ids": [str(_ALLOWED)],
            },
        )
    assert response.status_code == 409
    assert not dispatched


async def test_every_assigned_rank_must_be_in_the_installation_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checking one approved rank cannot admit a second unapproved rank."""
    app, api, dispatched = _app(monkeypatch)
    instance = api.state.instances[InstanceId("allowed")]
    assignments = instance.shard_assignments.model_copy(
        update={
            "node_to_runner": {
                NodeId("allowed"): RunnerId("allowed"),
                NodeId("foreign"): RunnerId("foreign"),
            },
        }
    )
    api.state = api.state.model_copy(
        update={
            "instances": {
                **api.state.instances,
                InstanceId("allowed"): instance.model_copy(
                    update={"shard_assignments": assignments}
                ),
            }
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/instances/allowed/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "allowed_node_install_ids": [str(_ALLOWED)],
            },
        )
    assert response.status_code == 409
    assert not dispatched


async def test_disappearing_instance_is_not_replaced_with_an_available_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target lost between admission and conversion must fail without dispatch."""
    app, api, dispatched = _app(monkeypatch)

    async def resolve(model: ModelId) -> ModelId:
        api.state = api.state.model_copy(
            update={
                "instances": {
                    key: value
                    for key, value in api.state.instances.items()
                    if key != InstanceId("allowed")
                }
            }
        )
        return model

    monkeypatch.setattr(api, "_resolve_and_validate_text_model", resolve)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/instances/allowed/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "allowed_node_install_ids": [str(_ALLOWED)],
            },
        )
    assert response.status_code == 404
    assert not dispatched
