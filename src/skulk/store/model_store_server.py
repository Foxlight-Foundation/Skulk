"""
ModelStoreServer — lightweight aiohttp HTTP file server for the store host.

Role in the system
------------------
``ModelStoreServer`` runs only on the **store host** node.  It exposes the
contents of the :class:`~skulk.store.model_store.ModelStore` over HTTP so that
worker nodes can discover available models and pull their assigned shards.

The server is started as a long-running async task in :func:`skulk.main.Node.run`
alongside all other node components.  It is never started on non-store-host
nodes.

Design decisions
----------------
* **aiohttp** is used for the HTTP layer (already a transitive dependency via
  Skulk's download stack) rather than adding FastAPI/uvicorn overhead.
* **Range request support** (HTTP 206 Partial Content) enables resumable
  transfers: if a worker's staging is interrupted, it resumes from the byte
  offset where it stopped rather than restarting the file from scratch.
* **Chunked streaming** (8 MB chunks) keeps memory usage bounded even for
  20+ GB model files.
* **Path traversal protection**: all requested file paths are resolved and
  checked to be within the model directory before any I/O is performed.
* The server does **not** require authentication — it is intended for
  trusted LAN use only.  Do not expose port 12415 to untrusted networks.

HTTP API
--------
All endpoints return JSON unless noted.

``GET /health``
    Liveness check.  Returns store path, and disk usage (free/used/total
    bytes).  Useful for monitoring and for clients to verify connectivity
    before attempting a staging operation.

``GET /registry``
    Full store index as a JSON array of
    :class:`~skulk.store.model_store.StoreModelEntry` objects.  Useful for
    management scripts and future dashboard integration. The internal
    ``recover_installed_cards=true`` query first adopts complete adjacent
    sidecars under the store publication lock.

``GET /models``
    JSON array of model ID strings currently in the store.

``GET /models/{model_id}/files``
    JSON array of file paths (relative to the model directory) for the
    given *model_id*.  ``model_id`` may contain ``/`` encoded as ``%2F``.
    Returns 404 if the model is not in the store.

``GET /models/{model_id}/{path}``
    File content.  Supports the ``Range: bytes=<start>-<end>`` header for
    partial/resumable downloads.  Returns 200 (full file) or 206 (partial).
    Returns 404 if the model or file is not found.

Example requests::

    curl http://mac-studio-1:12415/health
    curl http://mac-studio-1:12415/models
    curl http://mac-studio-1:12415/models/mlx-community%2FQwen3-30B-A3B-4bit/files
    curl -H "Range: bytes=0-8388607" \\
         http://mac-studio-1:12415/models/mlx-community%2FQwen3-30B-A3B-4bit/config.json
"""

from __future__ import annotations

import ipaddress
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from typing import cast, final

import aiofiles
import aiofiles.os as aios
import aiohttp.web as web
from loguru import logger

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    get_all_model_cards,
    get_current_registry_card,
    get_current_registry_cards,
    get_registry_card_by_id,
)
from skulk.shared.models.remote_code_approval import require_remote_code_approval
from skulk.store.config import DEFAULT_MODEL_STORE_PORT
from skulk.store.installed_cards import (
    InstalledArtifactRole,
    InstalledCardRecord,
    companion_artifact_role,
)
from skulk.store.model_store import ModelStore

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB per streaming chunk
_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PROXY_FORWARDING_HEADERS = frozenset(
    {"Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Real-IP"}
)


def _require_loopback_peer_import(
    remote: str | None,
    headers: Mapping[str, str],
) -> None:
    """Reject peer-import mutations not originating directly on loopback."""

    if any(header in headers for header in _PROXY_FORWARDING_HEADERS):
        raise web.HTTPForbidden(
            reason="peer imports reject proxy forwarding headers"
        )
    try:
        remote_address = ipaddress.ip_address(remote or "")
    except ValueError as error:
        raise web.HTTPForbidden(reason="peer imports are loopback-only") from error
    if not remote_address.is_loopback:
        raise web.HTTPForbidden(reason="peer imports are loopback-only")


def _immutable_registry_card_payload(card: ModelCard) -> dict[str, object]:
    """Return card fields covered by one immutable registry identity.

    Snapshot membership and provenance are mutable catalog metadata rather than
    part of the card envelope's identity. Every executable and artifact field
    remains in the comparison so a peer cannot alter runtime behavior while
    retaining a trusted card ID.
    """

    return cast(
        "dict[str, object]",
        card.model_dump(
            mode="json",
            exclude={"registry_snapshot_id", "registry_provenance"},
        ),
    )


async def _require_trusted_registry_peer_record(
    record: InstalledCardRecord,
) -> None:
    """Bind a peer's verified claim to this host's TUF-verified card truth.

    Locally retained legacy and custom records honestly make no signed-byte
    claim. A record labelled ``registry_verified`` is stronger: before the
    canonical store accepts it, the local host independently resolves its
    immutable card ID and validates the complete card plus artifact role,
    repository, revision, selected file, alias, and companion ownership.
    """

    if record.verification != "registry_verified":
        return
    registry_card_id = record.model_card.registry_card_id
    if registry_card_id is None:
        raise web.HTTPConflict(reason="registry-verified import omits its card id")
    await get_registry_card_by_id(
        registry_card_id,
        refresh_on_miss=True,
    )
    trusted_card = next(
        (
            card
            for card in get_current_registry_cards()
            if card.registry_card_id == registry_card_id
        ),
        None,
    )
    if trusted_card is None:
        raise web.HTTPConflict(
            reason="store host cannot verify the peer's immutable registry card"
        )
    if _immutable_registry_card_payload(record.model_card) != (
        _immutable_registry_card_payload(trusted_card)
    ):
        raise web.HTTPConflict(
            reason="peer card disagrees with the TUF-verified registry card"
        )

    expected_repository: str
    expected_revision: str | None
    expected_file: str | None
    if record.artifact_role == "base":
        expected_repository = str(trusted_card.artifact_repository)
        expected_revision = trusted_card.source_revision
        expected_file = trusted_card.gguf_file
        identity_matches = (
            record.artifact_model_id == str(trusted_card.model_id)
            and record.installed_identity == registry_card_id
            and record.owner_model_id is None
            and record.owner_card_id is None
        )
    else:
        expected_repository = record.artifact_repository
        try:
            declared_role = companion_artifact_role(
                trusted_card,
                expected_repository,
            )
        except ValueError as error:
            raise web.HTTPConflict(reason=str(error)) from error
        runtime = trusted_card.runtime
        expected_revision = None
        expected_file = None
        if record.artifact_role == "vision_weights" and trusted_card.vision is not None:
            expected_revision = trusted_card.vision.weights_revision
        elif runtime is not None:
            if record.artifact_role == "mtp_sidecar":
                expected_revision = runtime.mtp_sidecar_revision
            elif record.artifact_role == "assistant":
                expected_revision = runtime.assistant_model_revision
            elif record.artifact_role == "served_draft":
                expected_revision = runtime.served_spec_draft_revision
                expected_file = runtime.served_spec_draft_file
            elif record.artifact_role == "vllm_draft":
                expected_revision = runtime.vllm_spec_draft_revision
        identity_matches = (
            declared_role == record.artifact_role
            and record.artifact_model_id == expected_repository
            and record.owner_model_id == str(trusted_card.model_id)
            and record.owner_card_id == registry_card_id
        )
    if not identity_matches or (
        record.artifact_repository != expected_repository
        or record.artifact_revision != expected_revision
        or record.artifact_file != expected_file
    ):
        raise web.HTTPConflict(
            reason="peer artifact identity disagrees with the immutable registry card"
        )


_REGISTRY_CARD_ID_PATTERN = re.compile(r"^card_[a-z2-7]{52}$")
_ARTIFACT_ROLES: frozenset[str] = frozenset(
    {
        "base",
        "vision_weights",
        "mtp_sidecar",
        "assistant",
        "served_draft",
        "vllm_draft",
    }
)


def _sanitize_model_id(model_id: str) -> str:
    """URL-decode ``%2F`` in a model_id path segment.

    aiohttp leaves percent-encoded slashes encoded in ``match_info``, so a
    model ID like ``mlx-community/Qwen3`` arrives as
    ``mlx-community%2FQwen3``.  This helper normalises it back.
    """
    return model_id.replace("%2F", "/")


def _requested_source_revision(request: web.Request) -> str | None:
    """Parse and validate an optional immutable source revision query value."""

    source_revision = request.query.get("source_revision")
    if source_revision is None:
        return None
    if _SOURCE_REVISION_PATTERN.fullmatch(source_revision) is None:
        raise web.HTTPBadRequest(reason="source_revision must be a 40-character commit")
    return source_revision


def _require_store_revision(
    store: ModelStore, model_id: str, source_revision: str | None
) -> None:
    """Reject reads when the canonical store entry is a different revision."""

    if not store.entry_matches_revision(model_id, source_revision):
        raise web.HTTPNotFound(
            reason=(
                f"Model {model_id} is not available at source revision "
                f"{source_revision or 'main'}"
            )
        )


@final
class ModelStoreServer:
    """aiohttp HTTP server that serves model files from the store host.

    Lifecycle::

        server = ModelStoreServer(store, port=DEFAULT_MODEL_STORE_PORT)
        await server.start()   # called once on startup
        # ... runs until shutdown ...
        await server.stop()    # called on graceful shutdown

    In practice, :func:`start` is passed directly to the node's
    :class:`~skulk.utils.task_group.TaskGroup` and :func:`stop` is not called
    explicitly — the task is cancelled when the task group shuts down.
    """

    def __init__(
        self,
        store: ModelStore,
        host: str = "0.0.0.0",
        port: int = DEFAULT_MODEL_STORE_PORT,
    ) -> None:
        """
        Args:
            store: The :class:`~skulk.store.model_store.ModelStore` instance
                to serve files from.
            host: Bind address.  Defaults to ``"0.0.0.0"`` (all interfaces).
                Set to ``"127.0.0.1"`` for localhost-only access (e.g. tests).
            port: TCP port to listen on.  Must match ``store_port`` in
                ``skulk.yaml`` so that worker nodes can reach this server.
        """
        self._store = store
        self._host = host
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/registry", self._handle_registry)
        self._app.router.add_get("/models", self._handle_models)
        self._app.router.add_get("/models/{model_id}/files", self._handle_model_files)
        self._app.router.add_post(
            "/models/{model_id}/download", self._handle_download_request
        )
        self._app.router.add_delete(
            "/models/{model_id}/download", self._handle_download_cancellation
        )
        self._app.router.add_get(
            "/models/{model_id}/download/status", self._handle_download_status
        )
        self._app.router.add_get("/downloads", self._handle_list_downloads)
        self._app.router.add_post("/imports", self._handle_peer_import)
        self._app.router.add_delete("/models/{model_id}", self._handle_delete_model)
        self._app.router.add_get("/models/{model_id}/{path:.*}", self._handle_file)

    def refresh_recovered_generations(
        self,
        current_registry_cards: Iterable[ModelCard],
    ) -> None:
        """Apply post-TUF generation selection to the authoritative store."""

        self._store.refresh_recovered_generations(current_registry_cards)

    async def start(self) -> None:
        """Start the HTTP server.  Blocks until the server is stopped."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info(
            f"ModelStoreServer listening on {self._host}:{self._port} "
            f"(store: {self._store.store_path})"
        )

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            logger.info("ModelStoreServer stopped")

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, _request: web.Request) -> web.Response:
        """``GET /health`` — liveness check with disk usage stats."""
        store_path = self._store.store_path
        disk = shutil.disk_usage(store_path)
        return web.json_response(
            {
                "store_path": str(store_path),
                "free_bytes": disk.free,
                "total_bytes": disk.total,
                "used_bytes": disk.used,
            }
        )

    async def _handle_registry(self, _request: web.Request) -> web.Response:
        """``GET /registry`` — full store index."""
        if _request.query.get("recover_installed_cards") == "true":
            await self._store.recover_installed_cards_serialized()
        models = self._store.list_models()
        return web.json_response([m.model_dump() for m in models])

    async def _handle_models(self, _request: web.Request) -> web.Response:
        """``GET /models`` — list of model IDs in the store."""
        models = self._store.list_models()
        return web.json_response([m.model_id for m in models])

    async def _handle_peer_import(self, request: web.Request) -> web.Response:
        """``POST /imports`` — pull and commit one capability-bound peer artifact."""

        _require_loopback_peer_import(request.remote, request.headers)
        try:
            raw = cast("object", json.loads(await request.text()))
            if not isinstance(raw, dict):
                raise ValueError("request must be an object")
            body = cast("dict[str, object]", raw)
            record = InstalledCardRecord.model_validate(
                body.get("record"), strict=False
            )
            source_base_url = body.get("source_base_url")
            capability_token = body.get("capability_token")
            target_node_id = body.get("target_node_id")
            if not all(
                isinstance(value, str) and value
                for value in (
                    source_base_url,
                    capability_token,
                    target_node_id,
                )
            ):
                raise ValueError("source, capability and target are required")
        except (ValueError, TypeError) as error:
            raise web.HTTPBadRequest(reason=str(error)) from error
        await _require_trusted_registry_peer_record(record)
        entry = await self._store.import_peer_artifact(
            record,
            source_base_url=cast("str", source_base_url),
            capability_token=cast("str", capability_token),
            target_node_id=cast("str", target_node_id),
        )
        return web.json_response(entry.model_dump(mode="json"))

    async def _handle_model_files(self, request: web.Request) -> web.Response:
        """``GET /models/{model_id}/files`` — file list for a model."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
        source_revision = _requested_source_revision(request)
        _require_store_revision(self._store, model_id, source_revision)
        model_path = self._store.get_store_path(model_id)
        if model_path is None:
            raise web.HTTPNotFound(reason=f"Model not in store: {model_id}")
        # A stale vision-GGUF entry missing its mmproj projector is reported as
        # NOT available so the worker's ensure_shard path (which stages directly
        # on a 200 here) falls through to a download request, which re-fetches the
        # projector and re-registers a complete entry (#346). Without this, the
        # worker stages the incomplete entry and the runner crashes on load.
        if await self._store.vision_entry_missing_projector(model_id):
            raise web.HTTPNotFound(
                reason=(
                    f"Model in store but missing its vision projector "
                    f"(will re-download): {model_id}"
                )
            )
        files = [
            str(p.relative_to(model_path)) for p in model_path.rglob("*") if p.is_file()
        ]
        return web.json_response(files)

    async def _handle_file(self, request: web.Request) -> web.StreamResponse:
        """``GET /models/{model_id}/{path}`` — file content with Range support."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
        source_revision = _requested_source_revision(request)
        _require_store_revision(self._store, model_id, source_revision)
        file_rel = request.match_info["path"]

        model_path = self._store.get_store_path(model_id)
        if model_path is None:
            raise web.HTTPNotFound(reason=f"Model not in store: {model_id}")

        # Security: resolve symlinks and reject any path outside the model dir.
        # Use is_relative_to() rather than a startswith() string check — the
        # latter is unsafe when one model directory name is a prefix of another
        # (e.g. "...-4bit" vs "...-4bit-awq": "/store/model-2/f" starts with
        # "/store/model" even though it escapes the intended boundary).
        resolved = (model_path / file_rel).resolve()
        if not resolved.is_relative_to(model_path.resolve()):
            raise web.HTTPBadRequest(reason="Path traversal attempt rejected")

        if not (await aios.path.exists(resolved)) or not (
            await aios.path.isfile(resolved)
        ):
            raise web.HTTPNotFound(reason=f"File not found: {file_rel}")

        file_size = (await aios.stat(resolved)).st_size
        range_header = request.headers.get("Range")

        start = 0
        end = file_size - 1
        status = 200

        if range_header is not None:
            status = 206
            try:
                range_spec = range_header.removeprefix("bytes=")
                start_str, _, end_str = range_spec.partition("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
            except ValueError as exc:
                raise web.HTTPBadRequest(reason="Invalid Range header") from exc

            if start < 0 or end >= file_size or start > end:
                raise web.HTTPRequestRangeNotSatisfiable(
                    headers={"Content-Range": f"bytes */{file_size}"}
                )

        length = end - start + 1
        headers: dict[str, str] = {
            "Content-Length": str(length),
            "Accept-Ranges": "bytes",
            "Content-Type": "application/octet-stream",
        }
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

        response = web.StreamResponse(status=status, headers=headers)
        await response.prepare(request)

        async with aiofiles.open(resolved, "rb") as f:
            await f.seek(start)
            remaining = length
            while remaining > 0:
                to_read = min(_CHUNK_SIZE, remaining)
                chunk = await f.read(to_read)
                if not chunk:
                    break
                await response.write(chunk)
                remaining -= len(chunk)

        await response.write_eof()
        return response

    async def _handle_list_downloads(self, request: web.Request) -> web.Response:
        """``GET /downloads`` — list active store-side downloads."""
        downloads = self._store.list_active_downloads()
        return web.json_response(
            [
                {
                    "modelId": d.model_id,
                    "sourceRevision": d.source_revision,
                    "status": d.status,
                    "progress": d.progress,
                    "error": d.error,
                }
                for d in downloads
            ]
        )

    async def _handle_delete_model(self, request: web.Request) -> web.Response:
        """``DELETE /models/{model_id}`` — remove model from store."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
        deleted = await self._store.delete_model_serialized(model_id)
        if not deleted:
            raise web.HTTPNotFound(reason=f"Model {model_id} not in store")
        return web.json_response({"modelId": model_id, "deleted": True})

    async def _handle_download_request(self, request: web.Request) -> web.Response:
        """``POST /models/{model_id}/download``: request store-side HF download.

        Optional JSON body ``{"gguf_file": "<path>"}`` pins which GGUF quant's
        shard group the store fetches, honoring a card's custom pin (#344).
        Optional ``{"extra_gguf_files": ["<path>", ...]}`` names same-repo
        companion GGUFs (a served-engine draft bundled with the base) to
        co-fetch with the base quant. Optional ``source_revision`` pins every
        repository read to a full immutable Hugging Face commit. Optional
        ``source_repository`` separates an alias used as store identity from
        the upstream repository containing its bytes. The body is optional: a
        request without it (or a non-JSON body) falls back to the default quant
        preference, no companions, and mutable ``main``.
        """
        model_id = _sanitize_model_id(request.match_info["model_id"])
        pinned_gguf: str | None = None
        extra_pinned_gguf: list[str] | None = None
        source_revision: str | None = None
        source_repository: str | None = None
        registry_card_id: str | None = None
        owner_model_id: str | None = None
        owner_registry_card_id: str | None = None
        artifact_role: InstalledArtifactRole = "base"
        if request.can_read_body:
            try:
                body: object = cast("object", await request.json())
            except Exception:  # noqa: BLE001 - tolerate empty / non-JSON body
                body = None
            if isinstance(body, dict):
                body_dict = cast("dict[str, object]", body)
                raw = body_dict.get("gguf_file")
                if isinstance(raw, str) and raw:
                    pinned_gguf = raw
                raw_extra = body_dict.get("extra_gguf_files")
                if isinstance(raw_extra, list):
                    extras = [
                        item
                        for item in cast("list[object]", raw_extra)
                        if isinstance(item, str) and item
                    ]
                    if extras:
                        extra_pinned_gguf = extras
                raw_revision = body_dict.get("source_revision")
                if isinstance(raw_revision, str) and raw_revision:
                    if _SOURCE_REVISION_PATTERN.fullmatch(raw_revision) is None:
                        raise web.HTTPBadRequest(
                            reason=("source_revision must be a 40-character commit")
                        )
                    source_revision = raw_revision
                raw_repository = body_dict.get("source_repository")
                if isinstance(raw_repository, str) and raw_repository:
                    if len(raw_repository) > 512 or "/" not in raw_repository:
                        raise web.HTTPBadRequest(
                            reason="source_repository must be a repository identifier"
                        )
                    source_repository = raw_repository
                raw_card_id = body_dict.get("registry_card_id")
                if isinstance(raw_card_id, str) and raw_card_id:
                    if _REGISTRY_CARD_ID_PATTERN.fullmatch(raw_card_id) is None:
                        raise web.HTTPBadRequest(
                            reason="registry_card_id must be an immutable card id"
                        )
                    registry_card_id = raw_card_id
                raw_owner_model_id = body_dict.get("owner_model_id")
                if isinstance(raw_owner_model_id, str) and raw_owner_model_id:
                    if len(raw_owner_model_id) > 512 or "/" not in raw_owner_model_id:
                        raise web.HTTPBadRequest(
                            reason="owner_model_id must be a model identifier"
                        )
                    owner_model_id = raw_owner_model_id
                raw_owner_card_id = body_dict.get("owner_registry_card_id")
                if isinstance(raw_owner_card_id, str) and raw_owner_card_id:
                    if _REGISTRY_CARD_ID_PATTERN.fullmatch(raw_owner_card_id) is None:
                        raise web.HTTPBadRequest(
                            reason="owner_registry_card_id must be an immutable card id"
                        )
                    owner_registry_card_id = raw_owner_card_id
                raw_role = body_dict.get("artifact_role")
                if isinstance(raw_role, str):
                    if raw_role not in _ARTIFACT_ROLES:
                        raise web.HTTPBadRequest(reason="invalid artifact_role")
                    artifact_role = cast("InstalledArtifactRole", raw_role)
        verified_card = await self._require_remote_code_download_approval(
            model_id,
            registry_card_id,
            source_repository=source_repository,
            source_revision=source_revision,
            pinned_gguf=pinned_gguf,
            owner_model_id=owner_model_id,
            owner_registry_card_id=owner_registry_card_id,
            artifact_role=artifact_role,
        )
        try:
            status = await self._store.request_download(
                model_id,
                pinned_gguf=pinned_gguf,
                extra_pinned_gguf=extra_pinned_gguf,
                source_revision=source_revision,
                source_repository=source_repository,
                model_card=verified_card,
                artifact_role=artifact_role,
                owner_model_id=owner_model_id,
                owner_card_id=owner_registry_card_id,
            )
        except ValueError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        return web.json_response(
            {
                "modelId": status.model_id,
                "sourceRevision": status.source_revision,
                "status": status.status,
                "progress": status.progress,
            }
        )

    @staticmethod
    async def _require_remote_code_download_approval(
        model_id: str,
        registry_card_id: str | None,
        *,
        source_repository: str | None,
        source_revision: str | None,
        pinned_gguf: str | None,
        owner_model_id: str | None = None,
        owner_registry_card_id: str | None = None,
        artifact_role: InstalledArtifactRole = "base",
    ) -> ModelCard | None:
        """Verify artifact identity and return the full card retained locally."""
        cards = await get_all_model_cards()
        if artifact_role != "base":
            if owner_model_id is None:
                raise web.HTTPConflict(
                    reason="companion downloads require an owning model"
                )
            if owner_registry_card_id is not None:
                card = next(
                    (
                        candidate
                        for candidate in cards
                        if candidate.registry_card_id == owner_registry_card_id
                        and str(candidate.model_id) == owner_model_id
                    ),
                    None,
                )
                if card is None:
                    card = await get_registry_card_by_id(
                        owner_registry_card_id,
                        refresh_on_miss=True,
                    )
            else:
                card = next(
                    (
                        candidate
                        for candidate in cards
                        if (
                            candidate.is_custom
                            or candidate.registry_card_id is None
                        )
                        and str(candidate.model_id) == owner_model_id
                    ),
                    None,
                )
            if card is None:
                raise web.HTTPConflict(
                    reason="store host cannot verify the companion's owning card"
                )
            repository = source_repository or model_id
            try:
                declared_role = companion_artifact_role(card, repository)
            except ValueError as error:
                raise web.HTTPConflict(reason=str(error)) from error
            if declared_role != artifact_role:
                raise web.HTTPConflict(
                    reason="companion role disagrees with the owning card"
                )
            expected_revision: str | None = None
            expected_file: str | None = None
            if artifact_role == "vision_weights" and card.vision is not None:
                expected_revision = card.vision.weights_revision
            elif card.runtime is not None:
                runtime = card.runtime
                if artifact_role == "mtp_sidecar":
                    expected_revision = runtime.mtp_sidecar_revision
                elif artifact_role == "assistant":
                    expected_revision = runtime.assistant_model_revision
                elif artifact_role == "served_draft":
                    expected_revision = runtime.served_spec_draft_revision
                    expected_file = runtime.served_spec_draft_file
                elif artifact_role == "vllm_draft":
                    expected_revision = runtime.vllm_spec_draft_revision
            if source_revision != expected_revision or pinned_gguf != expected_file:
                raise web.HTTPConflict(
                    reason="companion request disagrees with immutable owning card"
                )
            try:
                require_remote_code_approval(card)
            except PermissionError as error:
                raise web.HTTPForbidden(reason=str(error)) from error
            return card
        card: ModelCard | None
        if registry_card_id is not None:
            card = next(
                (
                    candidate
                    for candidate in cards
                    if candidate.registry_card_id == registry_card_id
                ),
                None,
            )
            if card is None:
                card = await get_registry_card_by_id(
                    registry_card_id,
                    refresh_on_miss=True,
                )
        else:
            card = get_current_registry_card(ModelId(model_id)) or next(
                (
                    candidate
                    for candidate in cards
                    if str(candidate.model_id) == model_id
                ),
                None,
            )
        if card is None:
            raise web.HTTPConflict(
                reason="store host cannot verify the requested registry card"
            )
        artifact_repository = source_repository or model_id
        if (
            str(card.model_id) != model_id
            or str(card.artifact_repository) != artifact_repository
            or card.source_revision != source_revision
            or card.gguf_file != pinned_gguf
        ):
            raise web.HTTPConflict(
                reason="store request disagrees with immutable registry artifact"
            )
        try:
            require_remote_code_approval(card)
        except PermissionError as error:
            raise web.HTTPForbidden(reason=str(error)) from error
        return card

    async def _handle_download_status(self, request: web.Request) -> web.Response:
        """``GET /models/{model_id}/download/status`` — poll download progress."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
        status = self._store.get_download_status(model_id)
        if status is None:
            raise web.HTTPNotFound(reason=f"No download in progress for {model_id}")
        return web.json_response(
            {
                "modelId": status.model_id,
                "sourceRevision": status.source_revision,
                "status": status.status,
                "progress": status.progress,
                "error": status.error,
            }
        )

    async def _handle_download_cancellation(self, request: web.Request) -> web.Response:
        """``DELETE /models/{model_id}/download`` — cancel one store transfer."""

        model_id = _sanitize_model_id(request.match_info["model_id"])
        status = await self._store.cancel_download(model_id)
        if status is None:
            raise web.HTTPConflict(reason=f"No active download for {model_id}")
        return web.json_response(
            {
                "modelId": status.model_id,
                "sourceRevision": status.source_revision,
                "status": status.status,
                "progress": status.progress,
                "cancelled": True,
            }
        )
