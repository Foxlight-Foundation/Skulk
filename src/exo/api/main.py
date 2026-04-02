import base64
import contextlib
import hashlib
import json
import os
import random
import socket
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast
from uuid import uuid4

import anyio
import yaml
from anyio import BrokenResourceError
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger

from exo.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    fetch_image_url,
    generate_chat_stream,
)
from exo.api.adapters.claude import (
    claude_request_to_text_generation,
    collect_claude_response,
    generate_claude_stream,
)
from exo.api.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
    generate_ollama_chat_stream,
    generate_ollama_generate_stream,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from exo.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from exo.api.keepalive import with_sse_keepalive
from exo.api.types import (
    AddCustomModelParams,
    AdvancedImageParams,
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    BenchImageGenerationResponse,
    BenchImageGenerationTaskParams,
    CancelCommandResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteDownloadResponse,
    DeleteInstanceResponse,
    DeleteTracesRequest,
    DeleteTracesResponse,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    GenerationStats,
    HuggingFaceSearchResult,
    ImageData,
    ImageEditsTaskParams,
    ImageGenerationResponse,
    ImageGenerationStats,
    ImageGenerationTaskParams,
    ImageListItem,
    ImageListResponse,
    ImageSize,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    PurgeStagingRequest,
    PurgeStagingResponse,
    StartDownloadParams,
    StartDownloadResponse,
    ToolCall,
    TraceCategoryStats,
    TraceEventResponse,
    TraceListItem,
    TraceListResponse,
    TraceRankStats,
    TraceResponse,
    TraceStatsResponse,
    normalize_image_size,
)
from exo.api.types.claude_api import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
)
from exo.api.types.ollama_api import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelDetails,
    OllamaModelTag,
    OllamaPsModel,
    OllamaPsResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from exo.api.types.openai_responses import (
    ResponsesRequest,
    ResponsesResponse,
)
from exo.master.image_store import ImageStore
from exo.master.placement import place_instance as get_instance_placements
from exo.shared.apply import apply
from exo.shared.constants import (
    DASHBOARD_DIR,
    EXO_CACHE_HOME,
    EXO_EVENT_LOG_DIR,
    EXO_IMAGE_CACHE_DIR,
    EXO_MAX_CHUNK_SIZE,
    EXO_TRACING_CACHE_DIR,
)
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    get_card,
    get_model_cards,
)
from exo.shared.tracing import TraceEvent, compute_stats, export_trace, load_trace_file
from exo.shared.types.chunks import (
    EmbeddingChunk,
    ErrorChunk,
    ImageChunk,
    InputImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.commands import (
    AddCustomModelCard,
    Command,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteDownload,
    DeleteInstance,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    SendInputChunk,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TextEmbedding,
    TextGeneration,
)
from exo.shared.types.common import CommandId, Id, NodeId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    TracesMerged,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.power_sampler import PowerSampler
from exo.utils.task_group import TaskGroup

if TYPE_CHECKING:
    from exo.store.config import ExoConfig
    from exo.store.model_store_client import ModelStoreClient

_API_EVENT_LOG_DIR = EXO_EVENT_LOG_DIR / "api"
ONBOARDING_COMPLETE_FILE = EXO_CACHE_HOME / "onboarding_complete"

API_TAGS_METADATA = [
    {
        "name": "Compatibility APIs",
        "description": "Endpoints that let existing SDKs and tools talk to Skulk using OpenAI, Claude, or Ollama-style request formats.",
    },
    {
        "name": "Models",
        "description": "Model discovery, search, listing, and custom model-card management.",
    },
    {
        "name": "Instances",
        "description": "Placement previews, launch flows, instance lookup, and lifecycle management for running models.",
    },
    {
        "name": "Downloads",
        "description": "Low-level node download control and staging-cache management.",
    },
    {
        "name": "Store",
        "description": "Shared model-store health, registry inspection, download workflows, deletion, and optimization.",
    },
    {
        "name": "Config",
        "description": "Cluster configuration, safe filesystem browsing, and node identity helpers used by the dashboard.",
    },
    {
        "name": "State & Tracing",
        "description": "Cluster state, event log access, trace inspection/export, and onboarding helpers.",
    },
    {
        "name": "Images",
        "description": "Image generation, editing, retrieval, and benchmarking endpoints.",
    },
]


def _format_to_content_type(image_format: Literal["png", "jpeg", "webp"] | None) -> str:
    return f"image/{image_format or 'png'}"


def _ensure_seed(params: AdvancedImageParams | None) -> AdvancedImageParams:
    """Ensure advanced params has a seed set for distributed consistency."""
    if params is None:
        return AdvancedImageParams(seed=random.randint(0, 2**32 - 1))
    if params.seed is None:
        return params.model_copy(update={"seed": random.randint(0, 2**32 - 1)})
    return params


def _create_fastapi_app() -> FastAPI:
    return FastAPI(
        title="Skulk API",
        summary="Distributed inference, placement, store, and compatibility APIs for Skulk.",
        description=(
            "Skulk exposes OpenAI-compatible inference APIs, cluster placement controls, "
            "model-store workflows, configuration endpoints, and debugging endpoints. "
            "Most text-generation requests require the target model to already be placed "
            "and running on the node or cluster."
        ),
        version="0.1.0",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_tags=API_TAGS_METADATA,
    )


def _json_request_body(schema: dict[str, object]) -> dict[str, object]:
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": schema,
                }
            },
        }
    }


class API:
    def __init__(
        self,
        node_id: NodeId,
        *,
        port: int,
        event_receiver: Receiver[IndexedEvent],
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
        exo_config: "ExoConfig | None" = None,
        store_client: "ModelStoreClient | None" = None,
        enable_event_log: bool = True,
        mount_dashboard: bool = True,
    ) -> None:
        self.state = State()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR) if enable_event_log else None
        self._system_id = SystemId()
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.event_receiver = event_receiver
        self.election_receiver = election_receiver
        self.node_id: NodeId = node_id
        self.last_completed_election: int = 0
        self.port = port
        self._exo_config = exo_config
        self._store_client = store_client
        self._config_path = Path("exo.yaml")
        self._model_optimizer: "ModelOptimizer | None" = None
        # Initialize optimizer if store path is available
        if exo_config and exo_config.model_store and exo_config.model_store.enabled:
            from exo.store.model_optimizer import ModelOptimizer

            self._model_optimizer = ModelOptimizer(
                store_path=Path(exo_config.model_store.store_path)
            )

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = _create_fastapi_app()

        @self.app.middleware("http")
        async def _log_requests(  # pyright: ignore[reportUnusedFunction]
            request: Request,
            call_next: Callable[[Request], Awaitable[StreamingResponse]],
        ) -> StreamingResponse:
            logger.debug(f"API request: {request.method} {request.url.path}")
            return await call_next(request)

        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()

        if mount_dashboard:
            self.app.mount(
                "/",
                StaticFiles(
                    directory=DASHBOARD_DIR,
                    html=True,
                ),
                name="dashboard",
            )

        self._text_generation_queues: dict[
            CommandId,
            Sender[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk],
        ] = {}
        self._image_generation_queues: dict[
            CommandId, Sender[ImageChunk | ErrorChunk]
        ] = {}
        self._embedding_queues: dict[
            CommandId, Sender[EmbeddingChunk | ErrorChunk]
        ] = {}
        self._image_store = ImageStore(EXO_IMAGE_CACHE_DIR)
        self._tg: TaskGroup = TaskGroup()

    def reset(self, result_clock: int, event_receiver: Receiver[IndexedEvent]):
        logger.info("Resetting API State")
        if self._event_log is not None:
            self._event_log.close()
            self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self.state = State()
        self._system_id = SystemId()
        self._text_generation_queues = {}
        self._image_generation_queues = {}
        self._embedding_queues = {}
        self.unpause(result_clock)
        self.event_receiver.close()
        self.event_receiver = event_receiver
        self._tg.start_soon(self._apply_state)

    def unpause(self, result_clock: int):
        logger.info("Unpausing API")
        self.last_completed_election = result_clock
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_exception_handlers(self) -> None:
        self.app.exception_handler(HTTPException)(self.http_exception_handler)

    async def http_exception_handler(
        self, _: Request, exc: HTTPException
    ) -> JSONResponse:
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self.app.get(
            "/node_id",
            tags=["State & Tracing"],
            summary="Get this API node's ID",
        )(lambda: self.node_id)
        self.app.post(
            "/instance",
            tags=["Instances"],
            summary="Create an instance from a fully specified placement",
            description="Create an instance from an already computed placement object when you want exact control instead of Skulk picking the placement for you.",
        )(self.create_instance)
        self.app.post(
            "/place_instance",
            tags=["Instances"],
            summary="Quick-launch a model placement",
            description=(
                "Place and launch a model with Skulk choosing a valid concrete placement "
                "from the requested sharding, instance metadata, and minimum-node constraints."
            ),
        )(self.place_instance)
        self.app.get(
            "/instance/placement",
            tags=["Instances"],
            summary="Compute a concrete placement for one requested combination",
            description="Return the exact instance shape Skulk would create for one requested model, sharding mode, instance metadata, and node-count combination.",
        )(self.get_placement)
        self.app.get(
            "/instance/previews",
            tags=["Instances"],
            summary="Preview valid placements for a model",
            description=(
                "Return candidate placements for a model before launch. This is the best first "
                "step when you want to see what Skulk can place on the current node or cluster."
            ),
        )(self.get_placement_previews)
        self.app.get(
            "/instance/{instance_id}",
            tags=["Instances"],
            summary="Get one running instance",
        )(self.get_instance)
        self.app.delete(
            "/instance/{instance_id}",
            tags=["Instances"],
            summary="Delete a running instance",
        )(self.delete_instance)
        self.app.get(
            "/models",
            tags=["Models"],
            summary="List known models",
            description="Return known model cards, including metadata Skulk uses for placement and compatibility decisions.",
        )(self.get_models)
        self.app.get(
            "/v1/models",
            tags=["Models"],
            summary="List known models",
            description="OpenAI-style model listing endpoint backed by Skulk's model catalog rather than only currently running instances.",
        )(self.get_models)
        self.app.post(
            "/models/add",
            tags=["Models"],
            summary="Fetch and add a custom model card",
            description="Add a custom model card to Skulk's model catalog so it becomes searchable and launchable through the API or dashboard.",
        )(self.add_custom_model)
        self.app.delete(
            "/models/custom/{model_id:path}",
            tags=["Models"],
            summary="Delete a custom model card",
            description="Remove a previously added custom model card from Skulk's local catalog.",
        )(self.delete_custom_model)
        self.app.get(
            "/models/search",
            tags=["Models"],
            summary="Search Hugging Face for models",
            description=(
                "Search for models to add or launch. Skulk prefers MLX-friendly results first, "
                "then falls back to broader Hugging Face search results."
            ),
        )(self.search_models)
        self.app.post(
            "/v1/chat/completions",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="OpenAI Chat Completions-compatible text generation",
            description=(
                "Generate text with an OpenAI Chat Completions-compatible payload. The requested "
                "model must already be placed and running or Skulk will return a not-found error."
            ),
        )(self.chat_completions)
        self.app.post(
            "/v1/embeddings",
            tags=["Compatibility APIs"],
            summary="Generate embeddings",
        )(self.embeddings)
        self.app.post(
            "/bench/chat/completions",
            tags=["Compatibility APIs"],
            summary="Benchmark chat completions",
        )(self.bench_chat_completions)
        self.app.post(
            "/v1/images/generations",
            response_model=None,
            tags=["Images"],
            summary="Generate images",
        )(self.image_generations)
        self.app.post(
            "/bench/images/generations",
            tags=["Images"],
            summary="Benchmark image generation",
        )(self.bench_image_generations)
        self.app.post(
            "/v1/images/edits",
            response_model=None,
            tags=["Images"],
            summary="Edit images",
        )(self.image_edits)
        self.app.post(
            "/bench/images/edits",
            tags=["Images"],
            summary="Benchmark image editing",
        )(self.bench_image_edits)
        self.app.get("/images", tags=["Images"], summary="List stored images")(
            self.list_images
        )
        self.app.get(
            "/images/{image_id}", tags=["Images"], summary="Fetch one stored image"
        )(self.get_image)
        self.app.post(
            "/v1/messages",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Anthropic Claude Messages-compatible endpoint",
            description=(
                "Claude Messages-compatible text generation endpoint. As with chat completions, "
                "the target model must already be placed and ready."
            ),
        )(self.claude_messages)
        self.app.post(
            "/v1/responses",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="OpenAI Responses-compatible endpoint",
            description=(
                "OpenAI Responses-compatible endpoint for text generation and reasoning-style "
                "workflows backed by a placed Skulk model."
            ),
        )(self.openai_responses)
        self.app.post(
            "/v1/cancel/{command_id}",
            tags=["Compatibility APIs"],
            summary="Cancel an active text or image command",
            description="Request cancellation for an in-flight text or image generation command by its command ID.",
        )(self.cancel_command)

        # Ollama API
        self.app.head(
            "/ollama/", tags=["Compatibility APIs"], summary="Ollama version check"
        )(self.ollama_version)
        self.app.head(
            "/ollama/api/version",
            tags=["Compatibility APIs"],
            summary="Ollama version check",
        )(self.ollama_version)
        self.app.post(
            "/ollama/api/chat",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama chat",
            description="Ollama-compatible chat endpoint backed by Skulk model placement and routing.",
            openapi_extra=_json_request_body(OllamaChatRequest.model_json_schema()),
        )(self.ollama_chat)
        self.app.post(
            "/ollama/api/api/chat",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama chat alias",
        )(self.ollama_chat)
        self.app.post(
            "/ollama/api/v1/chat",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama chat alias",
        )(self.ollama_chat)
        self.app.post(
            "/ollama/api/generate",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama generate",
            description="Ollama-compatible prompt-completion endpoint backed by a placed Skulk model.",
            openapi_extra=_json_request_body(OllamaGenerateRequest.model_json_schema()),
        )(self.ollama_generate)
        self.app.get(
            "/ollama/api/tags",
            tags=["Compatibility APIs"],
            summary="List Ollama models",
        )(self.ollama_tags)
        self.app.get(
            "/ollama/api/api/tags",
            tags=["Compatibility APIs"],
            summary="List Ollama models alias",
        )(self.ollama_tags)
        self.app.get(
            "/ollama/api/v1/tags",
            tags=["Compatibility APIs"],
            summary="List Ollama models alias",
        )(self.ollama_tags)
        self.app.post(
            "/ollama/api/show",
            tags=["Compatibility APIs"],
            summary="Show Ollama model details",
        )(self.ollama_show)
        self.app.get(
            "/ollama/api/ps",
            tags=["Compatibility APIs"],
            summary="List running Ollama models",
        )(self.ollama_ps)
        self.app.get(
            "/ollama/api/version",
            tags=["Compatibility APIs"],
            summary="Get Ollama API version",
        )(self.ollama_version)

        self.app.get(
            "/state",
            tags=["State & Tracing"],
            summary="Get cluster state",
            description="Return the current cluster state as seen by this API node, including topology, instances, and node capabilities.",
        )(lambda: self.state)
        self.app.get(
            "/events",
            tags=["State & Tracing"],
            summary="Get stored event log",
            description="Stream or return the API-side event log used for debugging state transitions and cluster behavior.",
        )(self.stream_events)
        self.app.post(
            "/download/start",
            tags=["Downloads"],
            summary="Start a node download",
            description="Start a low-level node download for a specific shard on a specific node.",
        )(self.start_download)
        self.app.delete(
            "/download/{node_id}/{model_id:path}",
            tags=["Downloads"],
            summary="Delete a node download",
            description="Delete or cancel a download associated with a given node and model.",
        )(self.delete_download)
        self.app.get(
            "/v1/traces",
            tags=["State & Tracing"],
            summary="List saved traces",
            description="List saved trace files that can be inspected for debugging and performance analysis.",
        )(self.list_traces)
        self.app.post(
            "/v1/traces/delete",
            tags=["State & Tracing"],
            summary="Delete saved traces",
            description="Delete one or more saved trace artifacts by task ID.",
        )(self.delete_traces)
        self.app.get(
            "/v1/traces/{task_id}",
            tags=["State & Tracing"],
            summary="Get one saved trace",
            description="Return the structured trace events for one saved task trace.",
        )(self.get_trace)
        self.app.get(
            "/v1/traces/{task_id}/stats",
            tags=["State & Tracing"],
            summary="Get one trace summary",
            description="Return aggregated timing statistics for one saved trace.",
        )(self.get_trace_stats)
        self.app.get(
            "/v1/traces/{task_id}/raw",
            tags=["State & Tracing"],
            summary="Download raw trace JSON",
            description="Download the raw Chrome trace JSON artifact for one task.",
        )(self.get_trace_raw)
        self.app.get(
            "/onboarding",
            tags=["State & Tracing"],
            summary="Get onboarding completion status",
            description="Return whether the dashboard onboarding flow has been completed on this node.",
        )(self.get_onboarding)
        self.app.post(
            "/onboarding",
            tags=["State & Tracing"],
            summary="Mark onboarding complete",
            description="Mark the local dashboard onboarding flow as complete.",
        )(self.complete_onboarding)

        # Config & store endpoints
        self.app.get(
            "/config",
            tags=["Config"],
            summary="Get cluster config",
            description="Return the current cluster-wide config with sensitive fields such as the HF token removed.",
        )(self.get_config)
        self.app.put(
            "/config",
            tags=["Config"],
            summary="Update cluster config",
            description=(
                "Update cluster-wide config. Some changes apply to future launches immediately, "
                "while model-store location changes still require a restart."
            ),
        )(self.update_config)
        self.app.get(
            "/store/health",
            tags=["Store"],
            summary="Get model-store health",
            description="Check whether the configured shared model store is enabled and reachable.",
        )(self.get_store_health)
        self.app.get(
            "/store/registry",
            tags=["Store"],
            summary="Get model-store registry",
            description="List models and metadata known to the shared store registry.",
        )(self.get_store_registry)
        self.app.get(
            "/store/downloads",
            tags=["Store"],
            summary="List active store downloads",
            description="List in-progress downloads being managed by the shared model store.",
        )(self.get_store_downloads)
        self.app.delete(
            "/store/models/{model_id:path}",
            tags=["Store"],
            summary="Delete a model from the store",
            description="Delete a model and its shared-store artifacts from the configured model store.",
        )(self.delete_store_model)
        self.app.post(
            "/store/models/{model_id:path}/download",
            tags=["Store"],
            summary="Request a store download",
            description="Ask the shared model store to download and register a model by model ID.",
        )(self.request_store_download)
        self.app.get(
            "/store/models/{model_id:path}/download/status",
            tags=["Store"],
            summary="Get store download status",
            description="Return current status for a shared-store download request for one model.",
        )(self.get_store_download_status)
        self.app.post(
            "/store/purge-staging",
            tags=["Downloads"],
            summary="Purge staging caches",
            description="Broadcast a staging-cache purge request to nodes, optionally scoped to one model ID.",
        )(self.purge_staging_caches)
        self.app.post(
            "/store/models/{model_id:path}/optimize",
            tags=["Store"],
            summary="Start model optimization",
            description=(
                "Start an optimization job for a model already present in the shared store. "
                "Use this for workflows such as OptiQ conversion or alternate artifact generation."
            ),
        )(self.optimize_model)
        self.app.post(
            "/admin/restart",
            tags=["Admin"],
            summary="Restart this node",
            description=(
                "Gracefully restart the exo process on this node. "
                "Active inference is interrupted, GPU memory is fully released, "
                "and the node rejoins the cluster automatically on startup."
            ),
        )(self.restart_node)
        self.app.get(
            "/store/models/{model_id:path}/optimize/status",
            tags=["Store"],
            summary="Get model optimization status",
            description="Return the current status of an optimization job for one shared-store model.",
        )(self.get_optimize_status)
        self.app.get(
            "/filesystem/browse",
            tags=["Config"],
            summary="Browse filesystem roots for config selection",
            description="Browse a safe subset of filesystem roots so the dashboard can help users choose store and staging paths.",
        )(self.browse_filesystem)
        self.app.get(
            "/node/identity",
            tags=["Config"],
            summary="Get node identity and preferred IP",
            description="Return the node ID, hostname, and preferred LAN IP address that the dashboard uses during setup.",
        )(self.get_node_identity)

    async def place_instance(self, payload: PlaceInstanceParams):
        command = PlaceInstance(
            model_card=await ModelCard.load(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=command.model_card,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        model_card = await ModelCard.load(instance.shard_assignments.model_id)
        required_memory = model_card.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory.in_gb:.1f}GB, Available: {available_memory.in_gb:.1f}GB",
            )

        command = CreateInstance(
            instance=instance,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=model_card,
        )

    async def get_placement(
        self,
        model_id: ModelId,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
    ) -> Instance:
        model_card = await ModelCard.load(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_card=model_card,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                node_memory=self.state.node_memory,
                node_network=self.state.node_network,
                topology=self.state.topology,
                current_instances=self.state.instances,
                download_status=self.state.downloads,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self.state.instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self,
        model_id: ModelId,
        node_ids: Annotated[list[NodeId] | None, Query()] = None,
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        required_nodes = set(node_ids) if node_ids else None

        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        try:
            model_card = await ModelCard.load(model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model card: {exc}"
            ) from exc
        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                instance_combinations.extend(
                    [
                        (sharding, instance_meta, i)
                        for i in range(
                            1, len(list(self.state.topology.list_nodes())) + 1
                        )
                    ]
                )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        for sharding, instance_meta, min_nodes in instance_combinations:
            try:
                placements = get_instance_placements(
                    PlaceInstance(
                        model_card=model_card,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        min_nodes=min_nodes,
                    ),
                    node_memory=self.state.node_memory,
                    node_network=self.state.node_network,
                    topology=self.state.topology,
                    current_instances=self.state.instances,
                    required_nodes=required_nodes,
                    download_status=self.state.downloads,
                )
            except ValueError as exc:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error=str(exc),
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            current_ids = set(self.state.instances.keys())
            new_instances = [
                instance
                for instance_id, instance in placements.items()
                if instance_id not in current_ids
            ]

            if len(new_instances) != 1:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error="Expected exactly one new instance from placement",
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            instance = new_instances[0]
            shard_assignments = instance.shard_assignments
            placement_node_ids = list(shard_assignments.node_to_runner.keys())

            memory_delta_by_node: dict[str, int] = {}
            if placement_node_ids:
                total_bytes = model_card.storage_size.in_bytes
                per_node = total_bytes // len(placement_node_ids)
                remainder = total_bytes % len(placement_node_ids)
                for index, node_id in enumerate(sorted(placement_node_ids, key=str)):
                    extra = 1 if index < remainder else 0
                    memory_delta_by_node[str(node_id)] = per_node + extra

            if (
                model_card.model_id,
                sharding,
                instance_meta,
                len(placement_node_ids),
            ) not in seen:
                previews.append(
                    PlacementPreview(
                        model_id=model_card.model_id,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        instance=instance,
                        memory_delta_by_node=memory_delta_by_node or None,
                        error=None,
                    )
                )
            seen.add(
                (
                    model_card.model_id,
                    sharding,
                    instance_meta,
                    len(placement_node_ids),
                )
            )

        return PlacementPreviewResponse(previews=previews)

    def get_instance(self, instance_id: InstanceId) -> Instance:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self.state.instances[instance_id]

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def cancel_command(self, command_id: CommandId) -> CancelCommandResponse:
        """Cancel an active command by closing its stream and notifying workers."""
        sender = (
            self._text_generation_queues.get(command_id)
            or self._image_generation_queues.get(command_id)
            or self._embedding_queues.get(command_id)
        )
        if sender is None:
            raise HTTPException(
                status_code=404,
                detail="Command not found or already completed",
            )

        await self._send(TaskCancelled(cancelled_command_id=command_id))
        sender.close()

        return CancelCommandResponse(
            message="Command cancelled.",
            command_id=command_id,
        )

    async def _token_chunk_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Yield chunks for a given command until completion.

        This is the internal low-level stream used by all API adapters.
        """
        try:
            self._text_generation_queues[command_id], recv = channel[
                TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
            ]()

            with recv as token_chunks:
                async for chunk in token_chunks:
                    yield chunk
                    if isinstance(chunk, PrefillProgressChunk):
                        continue
                    if chunk.finish_reason is not None:
                        break

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._text_generation_queues:
                del self._text_generation_queues[command_id]

    async def _collect_text_generation_with_stats(
        self, command_id: CommandId
    ) -> BenchChatCompletionResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model: ModelId | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)

            async for chunk in self._token_chunk_stream(command_id):
                if isinstance(chunk, PrefillProgressChunk):
                    continue

                if chunk.finish_reason == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=chunk.error_message or "Internal server error",
                    )

                if model is None:
                    model = chunk.model

                if isinstance(chunk, TokenChunk):
                    text_parts.append(chunk.text)

                if isinstance(chunk, ToolCallChunk):
                    tool_calls.extend(
                        ToolCall(
                            id=str(uuid4()),
                            index=i,
                            function=tool,
                        )
                        for i, tool in enumerate(chunk.tool_calls)
                    )

                stats = chunk.stats or stats

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            tg.cancel_scope.cancel()

        combined_text = "".join(text_parts)
        assert model is not None

        return BenchChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            generation_stats=stats,
            power_usage=sampler.result(),
        )

    async def _trigger_notify_user_to_download_model(self, model_id: ModelId) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    _sent_image_hashes: set[str] = set()

    async def _send_text_generation_with_images(
        self, task_params: TextGenerationTaskParams
    ) -> TextGeneration:
        images = task_params.images
        if not images:
            command = TextGeneration(task_params=task_params)
            await self._send(command)
            return command

        hashes = [hashlib.sha256(img.encode("ascii")).hexdigest() for img in images]

        cached_hashes: dict[int, str] = {}
        new_images: list[tuple[int, str]] = []
        for idx, (img, h) in enumerate(zip(images, hashes, strict=True)):
            if h in self._sent_image_hashes:
                cached_hashes[idx] = h
            else:
                self._sent_image_hashes.add(h)
                new_images.append((idx, img))

        if not new_images:
            task_params = task_params.model_copy(
                update={"images": [], "image_hashes": cached_hashes}
            )
            command = TextGeneration(task_params=task_params)
            await self._send(command)
            return command

        all_chunks: list[tuple[int, str]] = []
        for img_idx, img_data in new_images:
            for i in range(0, len(img_data), EXO_MAX_CHUNK_SIZE):
                all_chunks.append((img_idx, img_data[i : i + EXO_MAX_CHUNK_SIZE]))

        task_params = task_params.model_copy(
            update={
                "images": [],
                "image_hashes": cached_hashes,
                "total_input_chunks": len(all_chunks),
                "image_count": len(new_images),
            }
        )
        command = TextGeneration(task_params=task_params)

        for global_idx, (img_idx, chunk_data) in enumerate(all_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=task_params.model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=global_idx,
                        total_chunks=len(all_chunks),
                        image_index=img_idx,
                    )
                )
            )

        await self._send(command)
        return command

    async def chat_completions(
        self, payload: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """OpenAI Chat Completions API - adapter."""
        task_params = await chat_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def bench_chat_completions(
        self, payload: BenchChatCompletionRequest
    ) -> BenchChatCompletionResponse:
        task_params = await chat_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        task_params = task_params.model_copy(update={"stream": False, "bench": True})

        command = await self._send_text_generation_with_images(task_params)

        return await self._collect_text_generation_with_stats(command.command_id)

    async def _resolve_and_validate_text_model(self, model_id: ModelId) -> ModelId:
        """Validate a text model exists and return the resolved model ID.

        Raises HTTPException 404 if no instance is found for the model.
        """
        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(model_id)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {model_id}",
            )
        return model_id

    async def _validate_image_model(self, model: ModelId) -> ModelId:
        """Validate model exists and return resolved model ID.

        Raises HTTPException 404 if no instance is found for the model.
        """
        model_card = await ModelCard.load(model)
        resolved_model = model_card.model_id
        if not any(
            instance.shard_assignments.model_id == resolved_model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved_model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {resolved_model}"
            )
        return resolved_model

    async def _validate_embedding_model(self, model_id: ModelId) -> ModelId:
        """Validate an embedding model exists and is the right type.

        Raises HTTPException 404 if no instance, 400 if not an embedding model.
        """
        from exo.shared.models.model_cards import ModelTask

        model_card = await ModelCard.load(model_id)
        resolved = model_card.model_id
        if ModelTask.TextEmbedding not in model_card.tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Model {resolved} is not an embedding model",
            )
        if not any(
            instance.shard_assignments.model_id == resolved
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {resolved}",
            )
        return resolved

    def stream_events(self) -> StreamingResponse:
        def _generate_json_array(events: Iterable[Event]) -> Iterable[str]:
            yield "["
            first = True
            for event in events:
                if not first:
                    yield ","
                first = False
                yield event.model_dump_json()
            yield "]"

        return StreamingResponse(
            _generate_json_array(
                [] if self._event_log is None else self._event_log.read_all()
            ),
            media_type="application/json",
        )

    async def get_image(self, image_id: str) -> FileResponse:
        stored = self._image_store.get(Id(image_id))
        if stored is None:
            raise HTTPException(status_code=404, detail="Image not found or expired")
        return FileResponse(path=stored.file_path, media_type=stored.content_type)

    async def list_images(self, request: Request) -> ImageListResponse:
        """List all stored images."""
        stored_images = self._image_store.list_images()
        return ImageListResponse(
            data=[
                ImageListItem(
                    image_id=img.image_id,
                    url=self._build_image_url(request, img.image_id),
                    content_type=img.content_type,
                    expires_at=img.expires_at,
                )
                for img in stored_images
            ]
        )

    def _build_image_url(self, request: Request, image_id: Id) -> str:
        host = request.headers.get("host", f"localhost:{self.port}")
        scheme = "https" if request.url.scheme == "https" else "http"
        return f"{scheme}://{host}/v1/images/{image_id}"

    async def image_generations(
        self, request: Request, payload: ImageGenerationTaskParams
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image generation requests.

        When stream=True and partial_images > 0, returns a StreamingResponse
        with SSE-formatted events for partial and final images.
        """
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        # Check if streaming is requested
        if payload.stream and payload.partial_images and payload.partial_images > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=payload.n or 1,
                    response_format=payload.response_format or "b64_json",
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all image chunks
        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _generate_image_stream(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream of partial and final images."""
        # Track chunks: {(image_index, is_partial): {chunk_index: data}}
        image_chunks: dict[tuple[int, bool], dict[int, str]] = {}
        image_total_chunks: dict[tuple[int, bool], int] = {}
        image_metadata: dict[tuple[int, bool], tuple[int | None, int | None]] = {}
        images_complete = 0

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            with recv as chunks:
                async for chunk in chunks:
                    if chunk.finish_reason == "error":
                        error_response = ErrorResponse(
                            error=ErrorInfo(
                                message=chunk.error_message or "Internal server error",
                                type="InternalServerError",
                                code=500,
                            )
                        )
                        yield f"data: {error_response.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    key = (chunk.image_index, chunk.is_partial)

                    if key not in image_chunks:
                        image_chunks[key] = {}
                        image_total_chunks[key] = chunk.total_chunks
                        image_metadata[key] = (
                            chunk.partial_index,
                            chunk.total_partials,
                        )

                    image_chunks[key][chunk.chunk_index] = chunk.data

                    # Check if this image is complete
                    if len(image_chunks[key]) == image_total_chunks[key]:
                        full_data = "".join(
                            image_chunks[key][i] for i in range(len(image_chunks[key]))
                        )

                        partial_idx, total_partials = image_metadata[key]

                        if chunk.is_partial:
                            # Yield partial image event (always use b64_json for partials)
                            event_data = {
                                "type": "partial",
                                "image_index": chunk.image_index,
                                "partial_index": partial_idx,
                                "total_partials": total_partials,
                                "format": str(chunk.format),
                                "data": {
                                    "b64_json": full_data
                                    if response_format == "b64_json"
                                    else None,
                                },
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        else:
                            # Final image
                            if response_format == "url":
                                image_bytes = base64.b64decode(full_data)
                                content_type = _format_to_content_type(chunk.format)
                                stored = self._image_store.store(
                                    image_bytes, content_type
                                )
                                url = self._build_image_url(request, stored.image_id)
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"url": url},
                                }
                            else:
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"b64_json": full_data},
                                }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            images_complete += 1

                            if images_complete >= num_images:
                                yield "data: [DONE]\n\n"
                                break

                        # Clean up completed image chunks
                        del image_chunks[key]
                        del image_total_chunks[key]
                        del image_metadata[key]

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_chunks(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
        capture_stats: bool = False,
    ) -> tuple[list[ImageData], ImageGenerationStats | None]:
        """Collect image chunks and optionally capture stats."""
        # Track chunks per image: {image_index: {chunk_index: data}}
        # Only track non-partial (final) images
        image_chunks: dict[int, dict[int, str]] = {}
        image_total_chunks: dict[int, int] = {}
        image_formats: dict[int, Literal["png", "jpeg", "webp"] | None] = {}
        images_complete = 0
        stats: ImageGenerationStats | None = None

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            while images_complete < num_images:
                with recv as chunks:
                    async for chunk in chunks:
                        if chunk.finish_reason == "error":
                            raise HTTPException(
                                status_code=500,
                                detail=chunk.error_message or "Internal server error",
                            )

                        if chunk.is_partial:
                            continue

                        if chunk.image_index not in image_chunks:
                            image_chunks[chunk.image_index] = {}
                            image_total_chunks[chunk.image_index] = chunk.total_chunks
                            image_formats[chunk.image_index] = chunk.format

                        image_chunks[chunk.image_index][chunk.chunk_index] = chunk.data

                        if capture_stats and chunk.stats is not None:
                            stats = chunk.stats

                        if (
                            len(image_chunks[chunk.image_index])
                            == image_total_chunks[chunk.image_index]
                        ):
                            images_complete += 1

                        if images_complete >= num_images:
                            break

            images: list[ImageData] = []
            for image_idx in range(num_images):
                chunks_dict = image_chunks[image_idx]
                full_data = "".join(chunks_dict[i] for i in range(len(chunks_dict)))
                if response_format == "url" and request is not None:
                    image_bytes = base64.b64decode(full_data)
                    content_type = _format_to_content_type(image_formats.get(image_idx))
                    stored = self._image_store.store(image_bytes, content_type)
                    url = self._build_image_url(request, stored.image_id)
                    images.append(ImageData(b64_json=None, url=url))
                else:
                    images.append(
                        ImageData(
                            b64_json=full_data
                            if response_format == "b64_json"
                            else None,
                            url=None,
                        )
                    )

            return (images, stats if capture_stats else None)
        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_generation(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> ImageGenerationResponse:
        """Collect all image chunks (non-streaming) and return a single response."""
        images, _ = await self._collect_image_chunks(
            request, command_id, num_images, response_format, capture_stats=False
        )
        return ImageGenerationResponse(data=images)

    async def _collect_image_generation_with_stats(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> BenchImageGenerationResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        images: list[ImageData] = []
        stats: ImageGenerationStats | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)
            images, stats = await self._collect_image_chunks(
                request, command_id, num_images, response_format, capture_stats=True
            )
            tg.cancel_scope.cancel()
        return BenchImageGenerationResponse(
            data=images, generation_stats=stats, power_usage=sampler.result()
        )

    async def bench_image_generations(
        self, request: Request, payload: BenchImageGenerationTaskParams
    ) -> BenchImageGenerationResponse:
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "stream": False,
                "partial_images": 0,
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _send_image_edits_command(
        self,
        image: UploadFile,
        prompt: str,
        model: ModelId,
        n: int,
        size: ImageSize,
        response_format: Literal["url", "b64_json"],
        input_fidelity: Literal["low", "high"],
        stream: bool,
        partial_images: int,
        bench: bool,
        quality: Literal["high", "medium", "low"],
        output_format: Literal["png", "jpeg", "webp"],
        advanced_params: AdvancedImageParams | None,
    ) -> ImageEdits:
        """Prepare and send an image edits command with chunked image upload."""
        resolved_model = await self._validate_image_model(model)
        advanced_params = _ensure_seed(advanced_params)

        image_content = await image.read()
        image_data = base64.b64encode(image_content).decode("utf-8")

        image_strength = 0.7 if input_fidelity == "high" else 0.3

        data_chunks = [
            image_data[i : i + EXO_MAX_CHUNK_SIZE]
            for i in range(0, len(image_data), EXO_MAX_CHUNK_SIZE)
        ]
        total_chunks = len(data_chunks)

        command = ImageEdits(
            task_params=ImageEditsTaskParams(
                image_data="",
                total_input_chunks=total_chunks,
                prompt=prompt,
                model=resolved_model,
                n=n,
                size=size,
                response_format=response_format,
                image_strength=image_strength,
                stream=stream,
                partial_images=partial_images,
                bench=bench,
                quality=quality,
                output_format=output_format,
                advanced_params=advanced_params,
            ),
        )

        logger.info(
            f"Sending input image: {len(image_data)} bytes in {total_chunks} chunks"
        )
        for chunk_index, chunk_data in enumerate(data_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=resolved_model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                    )
                )
            )

        await self._send(command)
        return command

    async def image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        stream: str = Form("false"),
        partial_images: str = Form("0"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image editing requests (img2img)."""
        # Parse string form values to proper types
        stream_bool = stream.lower() in ("true", "1", "yes")
        partial_images_int = int(partial_images) if partial_images.isdigit() else 0

        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=stream_bool,
            partial_images=partial_images_int,
            bench=False,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        if stream_bool and partial_images_int > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=n,
                    response_format=response_format,
                ),
                media_type="text/event-stream",
            )

        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def bench_image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> BenchImageGenerationResponse:
        """Handle benchmark image editing requests with generation stats."""
        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=False,
            partial_images=0,
            bench=True,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def claude_messages(
        self, payload: ClaudeMessagesRequest
    ) -> ClaudeMessagesResponse | StreamingResponse:
        """Claude Messages API - adapter."""
        task_params = claude_request_to_text_generation(payload)
        if task_params.images:
            resolved_images: list[str] = []
            for img in task_params.images:
                if img.startswith(("http://", "https://")):
                    resolved_images.append(await fetch_image_url(img))
                else:
                    resolved_images.append(img)
            task_params = task_params.model_copy(update={"images": resolved_images})
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_claude_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_claude_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def openai_responses(
        self, payload: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        """OpenAI Responses API."""
        task_params = await responses_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(task_params.model)
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_responses_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            return StreamingResponse(
                collect_responses_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def _ollama_root(self) -> JSONResponse:
        """Respond to HEAD / from Ollama CLI connectivity checks."""
        return JSONResponse(content="Ollama is running")

    async def ollama_chat(
        self, request: Request
    ) -> OllamaChatResponse | StreamingResponse:
        """Ollama Chat API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaChatRequest.model_validate_json(body)
        task_params = ollama_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_chat_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_generate(
        self, request: Request
    ) -> OllamaGenerateResponse | StreamingResponse:
        """Ollama Generate API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaGenerateRequest.model_validate_json(body)
        task_params = ollama_generate_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_generate_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_generate_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_tags(self) -> OllamaTagsResponse:
        """Returns list of models in Ollama tags format. We return the downloaded ones only."""

        def none_if_empty(value: str) -> str | None:
            return value or None

        downloaded_model_ids: set[str] = set()
        for node_downloads in self.state.downloads.values():
            for dl in node_downloads:
                if isinstance(dl, DownloadCompleted):
                    downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)

        cards = [
            c for c in await get_model_cards() if c.model_id in downloaded_model_ids
        ]

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return OllamaTagsResponse(
            models=[
                OllamaModelTag(
                    name=str(card.model_id),
                    model=str(card.model_id),
                    modified_at=now,
                    size=card.storage_size.in_bytes,
                    digest="sha256:000000000000",
                    details=OllamaModelDetails(
                        family=none_if_empty(card.family),
                        quantization_level=none_if_empty(card.quantization),
                    ),
                )
                for card in cards
            ]
        )

    async def ollama_show(self, request: Request) -> OllamaShowResponse:
        """Returns model information in Ollama show format."""
        body = await request.body()
        payload = OllamaShowRequest.model_validate_json(body)
        model_name = payload.name or payload.model
        if not model_name:
            raise HTTPException(status_code=400, detail="name or model is required")
        try:
            card = await ModelCard.load(ModelId(model_name))
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_name}"
            ) from exc

        return OllamaShowResponse(
            modelfile=f"FROM {card.model_id}",
            template="{{ .Prompt }}",
            details=OllamaModelDetails(
                family=card.family or None,
                quantization_level=card.quantization or None,
            ),
        )

    async def ollama_ps(self) -> OllamaPsResponse:
        """Returns list of running models (active instances)."""
        models: list[OllamaPsModel] = []
        seen: set[str] = set()
        for instance in self.state.instances.values():
            model_id = str(instance.shard_assignments.model_id)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                OllamaPsModel(
                    name=model_id,
                    model=model_id,
                    size=0,
                )
            )
        return OllamaPsResponse(models=models)

    async def ollama_version(self) -> dict[str, str]:
        """Returns version information for Ollama API compatibility."""
        return {"version": "exo v1.0"}

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for memory in self.state.node_memory.values():
            total_available += memory.ram_available

        return total_available

    @staticmethod
    def _model_tags(card: "ModelCard") -> list[str]:
        """Derive display tags from model metadata."""
        tags: list[str] = []
        model_id_lower = str(card.model_id).lower()
        quant_lower = card.quantization.lower()
        # OptiQ mixed-precision models
        if "optiq" in model_id_lower or "optiq" in quant_lower:
            tags.append("optiq")
        # Thinking capability
        if "thinking" in card.capabilities:
            tags.append("thinking")
        # Tensor parallel support
        if card.supports_tensor:
            tags.append("tensor")
        # Embedding models
        if "embedding" in card.capabilities:
            tags.append("embedding")
        return tags

    async def get_models(self, status: str | None = Query(default=None)) -> ModelList:
        """Returns list of available models, optionally filtered by being downloaded."""
        cards = await get_model_cards()

        if status == "downloaded":
            downloaded_model_ids: set[str] = set()
            for node_downloads in self.state.downloads.values():
                for dl in node_downloads:
                    if isinstance(dl, DownloadCompleted):
                        downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)
            cards = [c for c in cards if c.model_id in downloaded_model_ids]

        return ModelList(
            data=[
                ModelListModel(
                    id=card.model_id,
                    hugging_face_id=card.model_id,
                    name=card.model_id.short(),
                    description="",
                    tags=self._model_tags(card),
                    storage_size_megabytes=card.storage_size.in_mb,
                    supports_tensor=card.supports_tensor,
                    tasks=[task.value for task in card.tasks],
                    is_custom=card.is_custom,
                    family=card.family,
                    quantization=card.quantization,
                    base_model=card.base_model,
                    capabilities=card.capabilities,
                    context_length=card.context_length,
                )
                for card in cards
            ]
        )

    async def add_custom_model(self, payload: AddCustomModelParams) -> ModelListModel:
        """Fetch a model from HuggingFace and save as a custom model card, then sync across the cluster."""
        try:
            card = await ModelCard.fetch_from_hf(payload.model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch model: {exc}"
            ) from exc

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=AddCustomModelCard(model_card=card),
            )
        )

        return ModelListModel(
            id=card.model_id,
            hugging_face_id=card.model_id,
            name=card.model_id.short(),
            description="",
            tags=[],
            storage_size_megabytes=int(card.storage_size.in_mb),
            supports_tensor=card.supports_tensor,
            tasks=[task.value for task in card.tasks],
            is_custom=True,
            context_length=card.context_length,
        )

    async def delete_custom_model(self, model_id: ModelId) -> JSONResponse:
        """Delete a user-added custom model card and sync deletion across the cluster."""
        card = get_card(model_id)
        if card is None or not card.is_custom:
            raise HTTPException(status_code=404, detail="Custom model card not found")

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=DeleteCustomModelCard(model_id=model_id),
            )
        )

        return JSONResponse(
            {"message": "Model card deleted", "model_id": str(model_id)}
        )

    async def search_models(
        self, query: str = "", limit: int = 20
    ) -> list[HuggingFaceSearchResult]:
        """Search HuggingFace Hub — tries mlx-community first, falls back to all of HuggingFace."""
        from huggingface_hub import ModelInfo, list_models

        def _to_results(models: Iterable[ModelInfo]) -> list[HuggingFaceSearchResult]:
            return [
                HuggingFaceSearchResult(
                    id=m.id,
                    author=m.author or "",
                    downloads=m.downloads or 0,
                    likes=m.likes or 0,
                    last_modified=str(m.last_modified or ""),
                    tags=list(m.tags or []),
                )
                for m in models
            ]

        # Search mlx-community first
        mlx_results = _to_results(
            list_models(
                search=query or None,
                author="mlx-community",
                sort="downloads",
                limit=limit,
            )
        )
        if mlx_results:
            return mlx_results

        # Fall back to searching all of HuggingFace
        return _to_results(
            list_models(
                search=query or None,
                sort="downloads",
                limit=limit,
            )
        )

    async def run(self):
        shutdown_ev = anyio.Event()

        try:
            async with self._tg as tg:
                logger.info("Starting API")
                tg.start_soon(self._apply_state)
                tg.start_soon(self._pause_on_new_election)
                tg.start_soon(self._cleanup_expired_images)
                print_startup_banner(self.port)
                tg.start_soon(self.run_api, shutdown_ev)
                try:
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        shutdown_ev.set()
        finally:
            if self._event_log is not None:
                self._event_log.close()
            self.command_sender.close()
            self.event_receiver.close()

    async def run_api(self, ev: anyio.Event):
        cfg = Config()
        cfg.bind = [f"0.0.0.0:{self.port}"]
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger
        with anyio.CancelScope(shield=True):
            await serve(
                cast(ASGIFramework, self.app),
                cfg,
                shutdown_trigger=ev.wait,
            )

    async def _apply_state(self):
        with self.event_receiver as events:
            async for i_event in events:
                if self._event_log is not None:
                    self._event_log.append(i_event.event)
                self.state = apply(self.state, i_event)
                event = i_event.event

                if isinstance(event, ChunkGenerated):
                    if queue := self._image_generation_queues.get(
                        event.command_id, None
                    ):
                        assert isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except BrokenResourceError:
                            self._image_generation_queues.pop(event.command_id, None)
                    if queue := self._text_generation_queues.get(
                        event.command_id, None
                    ):
                        assert not isinstance(event.chunk, (ImageChunk, EmbeddingChunk))
                        try:
                            await queue.send(event.chunk)
                        except BrokenResourceError:
                            self._text_generation_queues.pop(event.command_id, None)
                    if queue := self._embedding_queues.get(event.command_id, None):
                        assert isinstance(event.chunk, (EmbeddingChunk, ErrorChunk))
                        try:
                            await queue.send(event.chunk)
                        except BrokenResourceError:
                            self._embedding_queues.pop(event.command_id, None)
                if isinstance(event, TracesMerged):
                    self._save_merged_trace(event)

    def _save_merged_trace(self, event: TracesMerged) -> None:
        traces = [
            TraceEvent(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in event.traces
        ]
        output_path = EXO_TRACING_CACHE_DIR / f"trace_{event.task_id}.json"
        export_trace(traces, output_path)
        logger.debug(f"Saved merged trace to {output_path}")

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _cleanup_expired_images(self):
        """Periodically clean up expired images from the store."""
        cleanup_interval_seconds = 300  # 5 minutes
        while True:
            await anyio.sleep(cleanup_interval_seconds)
            removed = self._image_store.cleanup_expired()
            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired images")

    async def _send(self, command: Command):
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self._system_id, command=command)
        )

    async def _send_download(self, command: DownloadCommand):
        await self.download_command_sender.send(
            ForwarderDownloadCommand(origin=self._system_id, command=command)
        )

    async def start_download(
        self, payload: StartDownloadParams
    ) -> StartDownloadResponse:
        command = StartDownload(
            target_node_id=payload.target_node_id,
            shard_metadata=payload.shard_metadata,
        )
        await self._send_download(command)
        return StartDownloadResponse(command_id=command.command_id)

    async def delete_download(
        self, node_id: NodeId, model_id: ModelId
    ) -> DeleteDownloadResponse:
        command = DeleteDownload(
            target_node_id=node_id,
            model_id=ModelId(model_id),
        )
        await self._send_download(command)
        return DeleteDownloadResponse(command_id=command.command_id)

    async def purge_staging_caches(
        self, payload: PurgeStagingRequest
    ) -> PurgeStagingResponse:
        from exo.shared.types.commands import PurgeStagingCache

        command = PurgeStagingCache(
            model_id=ModelId(payload.model_id) if payload.model_id else None,
        )
        await self._send_download(command)
        model_suffix = f" for model {payload.model_id}" if payload.model_id else ""
        return PurgeStagingResponse(
            command_id=command.command_id,
            message=f"Purge staging command broadcast to all nodes{model_suffix}",
        )

    @staticmethod
    def _get_trace_path(task_id: str) -> Path:
        trace_path = EXO_TRACING_CACHE_DIR / f"trace_{task_id}.json"
        if not trace_path.resolve().is_relative_to(EXO_TRACING_CACHE_DIR.resolve()):
            raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
        return trace_path

    async def list_traces(self) -> TraceListResponse:
        traces: list[TraceListItem] = []

        for trace_file in sorted(
            EXO_TRACING_CACHE_DIR.glob("trace_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            # Extract task_id from filename (trace_{task_id}.json)
            task_id = trace_file.stem.removeprefix("trace_")
            stat = trace_file.stat()
            created_at = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            traces.append(
                TraceListItem(
                    task_id=task_id,
                    created_at=created_at,
                    file_size=stat.st_size,
                )
            )

        return TraceListResponse(traces=traces)

    async def get_trace(self, task_id: str) -> TraceResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)

        return TraceResponse(
            task_id=task_id,
            traces=[
                TraceEventResponse(
                    name=event.name,
                    start_us=event.start_us,
                    duration_us=event.duration_us,
                    rank=event.rank,
                    category=event.category,
                )
                for event in trace_events
            ],
        )

    async def get_trace_stats(self, task_id: str) -> TraceStatsResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)
        stats = compute_stats(trace_events)

        return TraceStatsResponse(
            task_id=task_id,
            total_wall_time_us=stats.total_wall_time_us,
            by_category={
                category: TraceCategoryStats(
                    total_us=cat_stats.total_us,
                    count=cat_stats.count,
                    min_us=cat_stats.min_us,
                    max_us=cat_stats.max_us,
                    avg_us=cat_stats.avg_us,
                )
                for category, cat_stats in stats.by_category.items()
            },
            by_rank={
                rank: TraceRankStats(
                    by_category={
                        category: TraceCategoryStats(
                            total_us=cat_stats.total_us,
                            count=cat_stats.count,
                            min_us=cat_stats.min_us,
                            max_us=cat_stats.max_us,
                            avg_us=cat_stats.avg_us,
                        )
                        for category, cat_stats in rank_stats.items()
                    }
                )
                for rank, rank_stats in stats.by_rank.items()
            },
        )

    async def get_trace_raw(self, task_id: str) -> FileResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        return FileResponse(
            path=trace_path,
            media_type="application/json",
            filename=f"trace_{task_id}.json",
        )

    async def delete_traces(self, request: DeleteTracesRequest) -> DeleteTracesResponse:
        deleted: list[str] = []
        not_found: list[str] = []
        for task_id in request.task_ids:
            trace_path = self._get_trace_path(task_id)
            if trace_path.exists():
                trace_path.unlink()
                deleted.append(task_id)
            else:
                not_found.append(task_id)
        return DeleteTracesResponse(deleted=deleted, not_found=not_found)

    async def get_onboarding(self) -> JSONResponse:
        return JSONResponse({"completed": ONBOARDING_COMPLETE_FILE.exists()})

    async def complete_onboarding(self) -> JSONResponse:
        ONBOARDING_COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ONBOARDING_COMPLETE_FILE.write_text("true")
        return JSONResponse({"completed": True})

    # ------------------------------------------------------------------
    # Config & Store endpoints
    # ------------------------------------------------------------------

    async def get_config(self) -> JSONResponse:
        if not self._config_path.exists():
            return JSONResponse(
                {
                    "config": {},
                    "configPath": str(self._config_path),
                    "fileExists": False,
                    "effective": {
                        "kv_cache_backend": os.environ.get(
                            "EXO_KV_CACHE_BACKEND", "default"
                        ),
                    },
                }
            )
        with self._config_path.open() as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        # Remove sensitive fields — token is managed separately
        safe_raw = dict(raw) if raw else {}
        has_hf_token = bool(safe_raw.pop("hf_token", None))
        return JSONResponse(
            {
                "config": safe_raw,
                "configPath": str(self._config_path),
                "fileExists": True,
                "effective": {
                    "kv_cache_backend": os.environ.get(
                        "EXO_KV_CACHE_BACKEND", "default"
                    ),
                    "has_hf_token": has_hf_token or "HF_TOKEN" in os.environ,
                },
            }
        )

    async def update_config(self, request: Request) -> JSONResponse:
        body = await request.json()
        config_data = body.get("config", body)
        # Preserve existing hf_token if not provided in this update
        # (GET /config strips it for security, so saves won't have it)
        if "hf_token" not in config_data and self._config_path.exists():
            try:
                with self._config_path.open() as f:
                    existing = yaml.safe_load(f) or {}
                if "hf_token" in existing:
                    config_data["hf_token"] = existing["hf_token"]
            except Exception:
                pass
        # Validate by attempting to parse with Pydantic
        from exo.store.config import ExoConfig

        try:
            ExoConfig.model_validate(config_data)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        config_yaml = yaml.safe_dump(
            config_data, default_flow_style=False, sort_keys=False
        )
        # Write locally
        with self._config_path.open("w") as f:
            f.write(config_yaml)
        # Broadcast to all nodes via the download commands topic (gossipsub)
        from exo.shared.types.commands import SyncConfig

        await self._send_download(SyncConfig(config_yaml=config_yaml))
        # Apply inference config to env var immediately so next model launch uses it.
        # Don't overwrite if user provided the env var at launch.
        inference = config_data.get("inference")
        if (
            isinstance(inference, dict)
            and "kv_cache_backend" in inference
            and not os.environ.get("_EXO_KV_BACKEND_USER_SET")
        ):
            os.environ["EXO_KV_CACHE_BACKEND"] = str(inference["kv_cache_backend"])
        # Apply HF token immediately
        hf_token = config_data.get("hf_token")
        if hf_token and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = str(hf_token)
        # model_store changes still require restart; inference-only changes don't
        has_store_changes = "model_store" in config_data
        return JSONResponse(
            {
                "success": True,
                "message": "Config saved and synced to cluster."
                + (
                    " KV cache backend takes effect on next model launch."
                    if inference
                    else ""
                )
                + (
                    " Restart required for model store changes."
                    if has_store_changes
                    else ""
                ),
                "requiresRestart": has_store_changes,
            }
        )

    async def get_store_health(self) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        health = await self._store_client.health_check()
        if health is None:
            raise HTTPException(status_code=503, detail="Store unreachable")
        return JSONResponse(
            {
                "storePath": health.store_path,
                "freeBytes": health.free_bytes,
                "totalBytes": health.total_bytes,
                "usedBytes": health.used_bytes,
            }
        )

    async def get_store_registry(self) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        entries = await self._store_client.fetch_registry()
        return JSONResponse({"entries": entries})

    _ALLOWED_BROWSE_ROOTS = ["/Volumes", "/home", "/mnt", "/tmp", "/opt"]

    async def browse_filesystem(self, path: str = "/Volumes") -> JSONResponse:
        resolved = Path(path).resolve()
        if not any(
            resolved.is_relative_to(root) for root in self._ALLOWED_BROWSE_ROOTS
        ):
            raise HTTPException(status_code=400, detail="Path outside allowed roots")
        if not resolved.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        try:
            dirs = sorted(
                [
                    {"name": d.name, "path": str(d)}
                    for d in resolved.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d["name"],
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Permission denied") from exc
        return JSONResponse({"path": str(resolved), "directories": dirs})

    async def get_node_identity(self) -> JSONResponse:
        hostname = socket.gethostname()
        ip = hostname
        network = self.state.node_network.get(self.node_id)
        if network:
            # Prefer IPv4 LAN addresses — skip loopback, IPv6, and link-local
            for iface in network.interfaces:
                addr = iface.ip_address
                if (
                    addr
                    and not addr.startswith("127.")
                    and ":" not in addr  # skip IPv6
                    and not addr.startswith("169.254.")  # skip link-local
                ):
                    ip = addr
                    break
        return JSONResponse(
            {
                "nodeId": str(self.node_id),
                "hostname": hostname,
                "ipAddress": ip,
            }
        )

    async def get_store_downloads(self) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        downloads = await self._store_client.list_active_downloads()
        return JSONResponse({"downloads": downloads})

    async def request_store_download(self, model_id: str) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        result = await self._store_client.request_store_download(model_id)
        return JSONResponse(result)

    async def get_store_download_status(self, model_id: str) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        result = await self._store_client.get_store_download_status(model_id)
        return JSONResponse(result)

    async def delete_store_model(self, model_id: str) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        deleted = await self._store_client.delete_store_model(model_id)
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Model {model_id} not in store"
            )
        return JSONResponse({"modelId": model_id, "deleted": True})

    async def optimize_model(self, model_id: str, request: Request) -> JSONResponse:
        """Start an OptiQ mixed-precision optimization for a model."""
        if self._model_optimizer is None:
            raise HTTPException(
                status_code=503,
                detail="Model optimizer not available (store not configured)",
            )
        try:
            body = await request.json()
        except Exception:
            body = {}
        target_bpw = float(body.get("target_bpw", 4.5))
        candidate_bits = body.get("candidate_bits", [4, 8])
        try:
            await self._model_optimizer.optimize(
                model_id=model_id,
                target_bpw=target_bpw,
                candidate_bits=candidate_bits,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            {
                "modelId": model_id,
                "status": "started",
                "targetBpw": target_bpw,
            }
        )

    async def get_optimize_status(self, model_id: str) -> JSONResponse:
        """Get optimization status for a model."""
        if self._model_optimizer is None:
            raise HTTPException(status_code=503, detail="Model optimizer not available")
        job = self._model_optimizer.get_status(model_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No optimization job found")
        return JSONResponse(
            {
                "modelId": job.model_id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "resultPath": job.result_path,
                "achievedBpw": job.achieved_bpw,
                "estimatedSizeMb": job.estimated_size_mb,
                "error": job.error,
            }
        )

    async def embeddings(self, request: EmbeddingRequest) -> JSONResponse:
        """OpenAI-compatible embeddings endpoint — routes through cluster pipeline."""
        from exo.shared.types.embedding import TextEmbeddingTaskParams

        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")

        if request.dimensions is not None:
            raise HTTPException(
                status_code=400,
                detail="The `dimensions` parameter is not supported; embeddings are returned at the model's native dimensionality.",
            )

        model_id = await self._validate_embedding_model(ModelId(request.model))
        command = TextEmbedding(
            task_params=TextEmbeddingTaskParams(
                model=model_id,
                input_texts=texts,
                encoding_format=request.encoding_format,
            )
        )
        command_id = command.command_id

        try:
            self._embedding_queues[command_id], recv = channel[
                EmbeddingChunk | ErrorChunk
            ]()

            await self._send(command)

            with recv as chunks:
                async for chunk in chunks:
                    if isinstance(chunk, ErrorChunk):
                        raise HTTPException(
                            status_code=500,
                            detail=f"Embedding failed: {chunk.error_message}",
                        )
                    import base64
                    import struct

                    embeddings_data: list[list[float] | str] = []
                    for emb in chunk.embeddings:
                        if request.encoding_format == "base64":
                            b64 = base64.b64encode(
                                struct.pack(f"<{len(emb)}f", *emb)
                            ).decode("ascii")
                            embeddings_data.append(b64)
                        else:
                            embeddings_data.append(emb)

                    response = EmbeddingResponse(
                        data=[
                            EmbeddingObject(index=idx, embedding=emb_data)
                            for idx, emb_data in enumerate(embeddings_data)
                        ],
                        model=model_id,
                        usage=EmbeddingUsage(
                            prompt_tokens=chunk.token_count,
                            total_tokens=chunk.token_count,
                        ),
                    )
                    return JSONResponse(response.model_dump())

            raise HTTPException(
                status_code=500, detail="No embedding response received"
            )

        except anyio.get_cancelled_exc_class():
            cancel_command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=cancel_command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._embedding_queues:
                del self._embedding_queues[command_id]

    async def restart_node(self) -> JSONResponse:
        """Restart the exo process on this node.

        Spawns a new exo process and then exits the current one. The OS
        reclaims all GPU memory when the old process exits, and the new
        process takes over the port after a brief delay."""
        import subprocess
        import sys
        import threading

        logger.info("Node restart requested via API — spawning replacement and exiting")

        def _do_restart() -> None:
            import time

            time.sleep(1)  # Allow the JSON response to flush
            # Spawn a new exo process that will take over once we exit
            subprocess.Popen(
                [sys.executable, *sys.argv],
                start_new_session=True,
            )
            time.sleep(0.5)  # Brief pause to let the new process start binding
            import os

            os._exit(0)  # Hard exit — releases all GPU/Metal memory

        threading.Thread(target=_do_restart, daemon=True).start()
        return JSONResponse({"status": "restarting", "node_id": str(self.node_id)})
