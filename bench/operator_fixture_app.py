"""Generated canonical responses behind the real operator authorization boundary.

This module never constructs a Skulk Node, discovers peers, loads a model, or
reads a model store. Responses are deliberately synthetic, not observed app
workload profiles. It is imported only by the opt-in benchmark fixture.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast, final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from skulk.api.operator_auth import create_operator_auth_router
from skulk.api.operator_gateway import OperatorGatewayAuthorization
from skulk.operator.pairing import OperatorPairingService

_NODE = "synthetic-node"
_CHAT = "fixture/generated-chat"
_SPEECH = "fixture/generated-speech"
_REQUEST_LIMIT = 65536


def generated_responses() -> dict[str, dict[str, object]]:
    """Return fresh, deterministic canonical read bodies with no real user data."""

    models: list[dict[str, object]] = [
        {
            "id": _CHAT,
            "name": "Synthetic chat (no model loaded)",
            "tasks": ["text-generation"],
            "placement": {"compatible_backends": ["mlx"]},
        },
        {
            "id": _SPEECH,
            "name": "Synthetic silence (no model loaded)",
            "tasks": ["text-to-speech"],
            "placement": {"compatible_backends": ["mlx_audio"]},
            "resolved_capabilities": {
                "supports_speech_synthesis": True,
                "supports_audio_output": True,
                "default_audio_response_format": "pcm",
                "audio_response_formats": ["pcm"],
            },
            "audio": {
                "kind": "tts",
                "supports_streaming": True,
                "supports_voice_listing": True,
                "default_response_format": "pcm",
                "response_formats": ["pcm"],
                "default_voice": "silence",
                "voices": ["silence"],
                "sample_rates": [24000],
            },
        },
    ]
    instances: dict[str, object] = {}
    runners: dict[str, object] = {}
    for index, model in enumerate(models):
        runner = f"synthetic-runner-{index}"
        instance = f"synthetic-instance-{index}"
        instances[instance] = {
            "MlxRingInstance": {
                "instanceId": instance,
                "shardAssignments": {
                    "modelId": model["id"],
                    "nodeToRunner": {_NODE: runner},
                    "runnerToShard": {
                        runner: {
                            "PipelineShardMetadata": {
                                "modelCard": {
                                    "placement": {
                                        "compatibleBackends": [
                                            "mlx" if index == 0 else "mlx_audio"
                                        ]
                                    }
                                }
                            }
                        }
                    },
                },
            }
        }
        runners[runner] = {"RunnerReady": {}}
    return {
        "/state": {
            "lastEventAppliedIdx": 1,
            "topology": {"nodes": [_NODE], "connections": {}},
            "instances": instances,
            "runners": runners,
            "downloads": {},
            "nodeIdentities": {
                _NODE: {
                    "nodeInstallId": "00000000-0000-4000-8000-000000000001",
                    "friendlyName": "Synthetic node (not hardware)",
                    "modelId": "Generated fixture",
                }
            },
            "nodeMemory": {
                _NODE: {
                    "ramTotal": {"inBytes": 16 * 1024**3},
                    "ramAvailable": {"inBytes": 12 * 1024**3},
                }
            },
            "nodeDisk": {
                _NODE: {
                    "total": {"inBytes": 64 * 1024**3},
                    "available": {"inBytes": 60 * 1024**3},
                }
            },
            "nodeSystem": {_NODE: {"gpuUsage": 0}},
            "nodeResources": {
                _NODE: {"participation": "full", "backends": ["mlx", "mlx_audio"]}
            },
            "nodeCapabilities": {_NODE: ["tts"]},
            "nodeHealth": {_NODE: {"level": "healthy"}},
        },
        "/v1/models": {"data": models},
        "/store/registry": {
            "cache_inventory": {
                "state": "current",
                "observed_nodes": 1,
                "expected_nodes": 1,
                "store_nodes": [_NODE],
            },
            "entries": [
                {"model_id": model["id"], "total_bytes": 0} for model in models
            ],
        },
        "/store/storage": {"nodeId": _NODE, "stagedModels": []},
        "/store/downloads": {"downloads": []},
        "/v1/audio/voices": {
            "data": [
                {
                    "id": "silence",
                    "name": "Synthetic silence",
                    "model": _SPEECH,
                    "kind": "builtin",
                    "preferred_languages": [],
                }
            ]
        },
    }


async def _generated_chat() -> AsyncIterator[bytes]:
    # Stable output deliberately does not echo the caller's prompt. Timing is
    # a fixture setting, not an observed model performance claim.
    for text in ("Synthetic ", "fixture ", "response. ", "No model was run."):
        await asyncio.sleep(0.1)
        delta = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
        yield f"data: {json.dumps(delta)}\n\n".encode()
    yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    yield b"data: [DONE]\n\n"


async def _generated_speech() -> AsyncIterator[bytes]:
    for _ in range(30):
        await asyncio.sleep(0.1)
        yield bytes(4800)  # 100 ms, mono signed PCM16, 24 kHz.


async def _never_tailnet_peer(_peer: str) -> bool:
    return False


@final
class _BoundedRequestBody:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = cast(bytes, message.get("body", b""))
            if len(body) + len(chunk) > _REQUEST_LIMIT:
                await JSONResponse(
                    {"detail": "fixture request limit"}, status_code=413
                )(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            payload = bytes(body)
            body.clear()
            return {"type": "http.request", "body": payload, "more_body": False}

        await self._app(scope, bounded_receive, send)


def create_fixture_app(service: OperatorPairingService) -> ASGIApp:
    """Build synthetic reads and streams using real pairing, refresh and scopes.

    All authority persistence belongs to the explicitly supplied service. No
    new authority is created here, and no live cluster or inference API is used.
    """

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(
        create_operator_auth_router(service, tailnet_peer_verifier=_never_tailnet_peer)
    )
    responses = generated_responses()

    async def read(request: Request) -> JSONResponse:
        return JSONResponse(responses[request.url.path])

    async def chat(request: Request) -> Response:
        async for _chunk in request.stream():
            pass
        return StreamingResponse(_generated_chat(), media_type="text/event-stream")

    async def speech(request: Request) -> Response:
        async for _chunk in request.stream():
            pass
        return StreamingResponse(
            _generated_speech(),
            media_type="audio/pcm",
            headers={
                "x-audio-sample-rate": "24000",
                "x-audio-channels": "1",
                "x-audio-sample-format": "s16le",
            },
        )

    for path in responses:
        app.add_api_route(
            path,
            read,
            methods=["GET"],
            tags=["Synthetic qualification"],
            summary="Read generated fixture data",
            description="Deterministic test data; no live cluster or model store is read.",
        )
    for path, endpoint in (
        ("/v1/chat/completions", chat),
        ("/v1/audio/speech", speech),
    ):
        app.add_api_route(
            path,
            endpoint,
            methods=["POST"],
            tags=["Synthetic qualification"],
            summary="Stream generated fixture output",
            description="Ignores bounded input and streams fixed synthetic output, without inference.",
        )
    # Bound the body before parsing; FastAPI otherwise converts a receive
    # exception during JSON parsing into a generic 400.
    app.add_middleware(_BoundedRequestBody)
    return OperatorGatewayAuthorization(app, service)
