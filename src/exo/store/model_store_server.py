"""
ModelStoreServer — lightweight aiohttp HTTP file server for the store host.

Role in the system
------------------
``ModelStoreServer`` runs only on the **store host** node.  It exposes the
contents of the :class:`~exo.store.model_store.ModelStore` over HTTP so that
worker nodes can discover available models and pull their assigned shards.

The server is started as a long-running async task in :func:`exo.main.Node.run`
alongside all other node components.  It is never started on non-store-host
nodes.

Design decisions
----------------
* **aiohttp** is used for the HTTP layer (already a transitive dependency via
  EXO's download stack) rather than adding FastAPI/uvicorn overhead.
* **Range request support** (HTTP 206 Partial Content) enables resumable
  transfers: if a worker's staging is interrupted, it resumes from the byte
  offset where it stopped rather than restarting the file from scratch.
* **Chunked streaming** (8 MB chunks) keeps memory usage bounded even for
  20+ GB model files.
* **Path traversal protection**: all requested file paths are resolved and
  checked to be within the model directory before any I/O is performed.
* The server does **not** require authentication — it is intended for
  trusted LAN use only.  Do not expose port 58080 to untrusted networks.

HTTP API
--------
All endpoints return JSON unless noted.

``GET /health``
    Liveness check.  Returns store path, and disk usage (free/used/total
    bytes).  Useful for monitoring and for clients to verify connectivity
    before attempting a staging operation.

``GET /registry``
    Full store index as a JSON array of
    :class:`~exo.store.model_store.StoreModelEntry` objects.  Useful for
    management scripts and future dashboard integration.

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

    curl http://mac-studio-1:58080/health
    curl http://mac-studio-1:58080/models
    curl http://mac-studio-1:58080/models/mlx-community%2FQwen3-30B-A3B-4bit/files
    curl -H "Range: bytes=0-8388607" \\
         http://mac-studio-1:58080/models/mlx-community%2FQwen3-30B-A3B-4bit/config.json
"""
from __future__ import annotations

import shutil
from typing import final

import aiofiles
import aiofiles.os as aios
import aiohttp.web as web
from loguru import logger

from exo.store.model_store import ModelStore

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB per streaming chunk


def _sanitize_model_id(model_id: str) -> str:
    """URL-decode ``%2F`` in a model_id path segment.

    aiohttp leaves percent-encoded slashes encoded in ``match_info``, so a
    model ID like ``mlx-community/Qwen3`` arrives as
    ``mlx-community%2FQwen3``.  This helper normalises it back.
    """
    return model_id.replace("%2F", "/")


@final
class ModelStoreServer:
    """aiohttp HTTP server that serves model files from the store host.

    Lifecycle::

        server = ModelStoreServer(store, port=58080)
        await server.start()   # called once on startup
        # ... runs until shutdown ...
        await server.stop()    # called on graceful shutdown

    In practice, :func:`start` is passed directly to the node's
    :class:`~exo.utils.task_group.TaskGroup` and :func:`stop` is not called
    explicitly — the task is cancelled when the task group shuts down.
    """

    def __init__(
        self,
        store: ModelStore,
        host: str = "0.0.0.0",
        port: int = 58080,
    ) -> None:
        """
        Args:
            store: The :class:`~exo.store.model_store.ModelStore` instance
                to serve files from.
            host: Bind address.  Defaults to ``"0.0.0.0"`` (all interfaces).
                Set to ``"127.0.0.1"`` for localhost-only access (e.g. tests).
            port: TCP port to listen on.  Must match ``store_port`` in
                ``exo.yaml`` so that worker nodes can reach this server.
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
        self._app.router.add_post("/models/{model_id}/download", self._handle_download_request)
        self._app.router.add_get("/models/{model_id}/download/status", self._handle_download_status)
        self._app.router.add_get("/models/{model_id}/{path:.*}", self._handle_file)

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
        models = self._store.list_models()
        return web.json_response([m.model_dump() for m in models])

    async def _handle_models(self, _request: web.Request) -> web.Response:
        """``GET /models`` — list of model IDs in the store."""
        models = self._store.list_models()
        return web.json_response([m.model_id for m in models])

    async def _handle_model_files(self, request: web.Request) -> web.Response:
        """``GET /models/{model_id}/files`` — file list for a model."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
        model_path = self._store.get_store_path(model_id)
        if model_path is None:
            raise web.HTTPNotFound(reason=f"Model not in store: {model_id}")
        files = [
            str(p.relative_to(model_path))
            for p in model_path.rglob("*")
            if p.is_file()
        ]
        return web.json_response(files)

    async def _handle_file(self, request: web.Request) -> web.StreamResponse:
        """``GET /models/{model_id}/{path}`` — file content with Range support."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
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

    async def _handle_download_request(self, request: web.Request) -> web.Response:
        """``POST /models/{model_id}/download`` — request store-side HF download."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
        status = await self._store.request_download(model_id)
        return web.json_response({
            "modelId": status.model_id,
            "status": status.status,
            "progress": status.progress,
        })

    async def _handle_download_status(self, request: web.Request) -> web.Response:
        """``GET /models/{model_id}/download/status`` — poll download progress."""
        model_id = _sanitize_model_id(request.match_info["model_id"])
        status = self._store.get_download_status(model_id)
        if status is None:
            raise web.HTTPNotFound(reason=f"No download in progress for {model_id}")
        return web.json_response({
            "modelId": status.model_id,
            "status": status.status,
            "progress": status.progress,
            "error": status.error,
        })
