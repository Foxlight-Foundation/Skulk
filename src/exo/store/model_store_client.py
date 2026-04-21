# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""
ModelStoreClient — HTTP client for staging model shards from the store host.
ModelStoreDownloader — ShardDownloader wrapper that intercepts ensure_shard().

Overview
--------
This module contains two closely related classes:

``ModelStoreClient``
    Knows how to talk to a :class:`~exo.store.model_store_server.ModelStoreServer`
    and copy model files to a node-local staging directory.  Used by both
    the :class:`ModelStoreDownloader` (download path) and the
    :class:`~exo.worker.main.Worker` (eviction path).

``ModelStoreDownloader``
    A :class:`~exo.download.shard_downloader.ShardDownloader` implementation
    that wraps EXO's standard ``ResumableShardDownloader`` and intercepts
    ``ensure_shard()`` to stage from the store instead of HuggingFace.  If
    the model is not yet in the store and ``allow_hf_fallback`` is ``True``,
    it falls through to the inner downloader transparently.

How staging works
-----------------
1. At inference time, the worker's ``DownloadCoordinator`` calls
   ``ensure_shard(shard)`` on the downloader.
2. ``ModelStoreDownloader.ensure_shard()`` asks the store client whether the
   model is available (HTTP ``GET /models/{id}/files``).
3. **Model in store**: ``stage_shard()`` downloads all model files to the
   node-local staging directory (``~/.exo/staging/<org>--<model>/`` by
   default).  Files are downloaded with HTTP ``Range`` resume support — if the
   process is interrupted, the next call picks up where it left off.
4. **Model not in store + HF fallback enabled**: delegates to the inner
   ``ResumableShardDownloader``, which downloads from HuggingFace as normal.
5. **Model not in store + HF fallback disabled**: raises
   :class:`ModelNotInStoreError` (use for air-gapped clusters).
6. MLX receives the local staging path and loads from the filesystem.  It
   has no knowledge of the store.

Store host optimisation
-----------------------
When this node IS the store host (``local_store_path`` is set in the
constructor), ``stage_shard()`` uses ``shutil.copy2()`` for a local
filesystem copy instead of making an HTTP round-trip over loopback.  If
``node_cache_path`` in ``exo.yaml`` is set to the same path as ``store_path``
for the store host, ``shutil.copy2()`` still runs but is effectively a no-op
(same inode, copy skipped by size check).

Eviction
--------
When an inference instance is deactivated (``Shutdown`` task received by the
:class:`~exo.worker.main.Worker`), the worker calls
``ModelStoreClient.evict_shard()`` to remove the staged files from the
node-local cache.  The canonical copy in the store is **never touched**.
Eviction is skipped if ``cleanup_on_deactivate`` is ``False`` for the node
(useful when staging is on fast local NVMe and warm cache is preferred).

Resume protocol
---------------
Each file is downloaded to a ``<filename>.partial`` temporary file.  On
completion the partial is atomically renamed to the final filename.  On
retry the client checks for an existing partial and sends a
``Range: bytes=<size>-`` header to resume from that offset.  If the final
file already exists (from a previous complete run), the file is skipped.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable
from datetime import timedelta
from pathlib import Path
from typing import AsyncIterator, Callable, final
from urllib.parse import quote

import aiofiles
import aiofiles.os as aios
import aiohttp
from loguru import logger

from exo.download.download_utils import create_http_session
from exo.download.shard_downloader import ShardDownloader
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import RepoDownloadProgress
from exo.shared.types.worker.shards import ShardMetadata
from exo.store.config import StagingNodeConfig

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB per read/write chunk
_CONNECT_TIMEOUT = 10.0  # seconds — abort if store host unreachable
_READ_TIMEOUT = 120.0  # seconds — abort if no data for 2 minutes


class ModelNotInStoreError(Exception):
    """Raised when a model is not in the store and HF fallback is disabled.

    This is an expected, handleable error — callers should catch it and
    present a meaningful message to the user (e.g. "model not available
    in offline mode").
    """


def _sanitize_model_id(model_id: str) -> str:
    """Convert a model ID to a safe directory name (matches EXO convention).

    ``"mlx-community/Qwen3-30B-A3B-4bit"`` → ``"mlx-community--Qwen3-30B-A3B-4bit"``

    This mirrors the sanitization used by EXO's own ``~/.exo/models/`` cache
    so that the staging layout is consistent with what users already expect.
    """
    return str(model_id).replace("/", "--")


def _staging_dir(node_cache_path: str, model_id: str) -> Path:
    """Resolve the staging directory for *model_id* on this node."""
    return Path(node_cache_path).expanduser() / _sanitize_model_id(model_id)


def _make_store_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


@final
class StoreHealthInfo:
    """Parsed response from ``GET /health`` on the store server.

    Attributes:
        store_path: Path of the model store root on the store host.
        free_bytes: Available disk space in bytes.
        total_bytes: Total disk capacity in bytes.
        used_bytes: Used disk space in bytes.
    """

    def __init__(
        self, store_path: str, free_bytes: int, total_bytes: int, used_bytes: int
    ) -> None:
        self.store_path = store_path
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.used_bytes = used_bytes


@final
class ModelStoreClient:
    """HTTP client for staging model files from the store host.

    Every non-store-host node holds one instance of this class.  The store
    host also holds one instance (with ``local_store_path`` set) so that the
    staging path is unified — the store host just uses ``shutil.copy2()``
    instead of HTTP.

    Args:
        store_host: Hostname or IP of the store host node.
        store_port: Port of the :class:`~exo.store.model_store_server.ModelStoreServer`.
        local_store_path: Set to the store's root path when this node IS the
            store host.  Causes ``stage_shard()`` to use local file copies
            instead of HTTP.
    """

    def __init__(
        self,
        store_host: str,
        store_port: int = 58080,
        local_store_path: Path | None = None,
    ) -> None:
        self._store_host = store_host
        self._store_port = store_port
        self._local_store_path = local_store_path

    @property
    def local_store_path(self) -> Path | None:
        """The local store path, or ``None`` if this is not the store host."""
        return self._local_store_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health_check(self) -> StoreHealthInfo | None:
        """Ping the store server and return its health info.

        Returns ``None`` if the server is unreachable or returns an error.
        Intended for startup checks and dashboard status polling.
        """
        url = _make_store_url(self._store_host, self._store_port, "/health")
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.get(url) as resp,
            ):
                if resp.status != 200:
                    return None
                data: object = await resp.json()
                if not isinstance(data, dict):
                    return None
                return StoreHealthInfo(
                    store_path=str(data.get("store_path", "")),
                    free_bytes=int(data.get("free_bytes", 0)),
                    total_bytes=int(data.get("total_bytes", 0)),
                    used_bytes=int(data.get("used_bytes", 0)),
                )
        except Exception as exc:
            logger.debug(f"ModelStoreClient: health_check failed: {exc}")
            return None

    async def is_model_available(self, model_id: str) -> bool:
        """Return ``True`` if *model_id* is available in the store.

        Uses ``GET /models/{model_id}/files`` — a 200 response means the
        model is present; 404 means it is not.  Any network error is treated
        as unavailable (returns ``False``).

        Args:
            model_id: HuggingFace-style model ID.
        """
        url = _make_store_url(
            self._store_host,
            self._store_port,
            f"/models/{quote(model_id, safe='')}/files",
        )
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.get(url) as resp,
            ):
                return resp.status == 200
        except Exception:
            return False

    async def stage_shard(
        self,
        model_id: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> Path:
        """Copy all model files for *model_id* into *dest_path*.

        Dispatches to :meth:`_stage_local` (``shutil.copy2``) on the store
        host, or :meth:`_stage_http` (HTTP with resume support) on workers.

        Args:
            model_id: HuggingFace-style model ID.
            dest_path: Directory to write staged files into.  Created if it
                does not exist.
            on_progress: Optional async callback ``(bytes_done, total_bytes)``
                called after each file is staged.

        Returns:
            *dest_path* (unchanged) after all files have been staged.

        Raises:
            :class:`ModelNotInStoreError`: If the model is not found in the
                store (should only happen if the store index is stale).
        """
        await aios.makedirs(dest_path, exist_ok=True)

        if self._local_store_path is not None:
            return await self._stage_local(model_id, dest_path, on_progress)
        return await self._stage_http(model_id, dest_path, on_progress)

    async def evict_shard(self, model_id: str, cache_path: Path) -> None:
        """Remove staged files for *model_id* from *cache_path*.

        The canonical copy in the store is **never touched**.  If no staged
        files exist for this model, the method returns silently.

        Args:
            model_id: Model whose staged files to remove.
            cache_path: Root of the node-local staging directory (the value
                of ``staging.node_cache_path`` for this node).
        """
        staged_dir = cache_path / _sanitize_model_id(model_id)
        if not staged_dir.exists():
            logger.debug(
                f"ModelStoreClient: evict_shard — no staged files for {model_id}"
            )
            return
        try:
            shutil.rmtree(staged_dir)
            logger.info(
                f"ModelStoreClient: evicted staged shard for {model_id} from {staged_dir}"
            )
        except Exception as exc:
            logger.warning(
                f"ModelStoreClient: evict_shard failed for {model_id}: {exc}"
            )

    async def list_models(self) -> list[str]:
        """Return the list of model IDs available in the store.

        Returns an empty list on any error (including network failures).
        """
        url = _make_store_url(self._store_host, self._store_port, "/models")
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.get(url) as resp,
            ):
                if resp.status != 200:
                    return []
                data: object = await resp.json()
                if not isinstance(data, list):
                    return []
                return [str(item) for item in data if isinstance(item, str)]
        except Exception as exc:
            logger.debug(f"ModelStoreClient: list_models failed: {exc}")
            return []

    async def fetch_registry(self) -> list[dict[str, object]]:
        """Fetch the full store registry from the store server.

        Returns a list of registry entry dicts, or an empty list on error.
        """
        url = _make_store_url(self._store_host, self._store_port, "/registry")
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.get(url) as resp,
            ):
                if resp.status != 200:
                    return []
                data: object = await resp.json()
                if not isinstance(data, list):
                    return []
                return [entry for entry in data if isinstance(entry, dict)]
        except Exception as exc:
            logger.debug(f"ModelStoreClient: fetch_registry failed: {exc}")
            return []

    async def list_active_downloads(self) -> list[dict[str, object]]:
        """Fetch the list of active store-side downloads."""
        url = _make_store_url(self._store_host, self._store_port, "/downloads")
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.get(url) as resp,
            ):
                if resp.status != 200:
                    return []
                data: object = await resp.json()
                if not isinstance(data, list):
                    return []
                return [d for d in data if isinstance(d, dict)]
        except Exception as exc:
            logger.debug(f"ModelStoreClient: list_active_downloads failed: {exc}")
            return []

    async def request_store_download(self, model_id: str) -> dict[str, object]:
        """Request the store host start downloading a model. Non-blocking."""
        url = _make_store_url(
            self._store_host,
            self._store_port,
            f"/models/{quote(model_id, safe='')}/download",
        )
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.post(url) as resp,
            ):
                if resp.status not in (200, 201):
                    return {"status": "error", "error": f"HTTP {resp.status}"}
                data: object = await resp.json()
                return data if isinstance(data, dict) else {"status": "unknown"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def get_store_download_status(self, model_id: str) -> dict[str, object]:
        """Poll store-side download status for a model."""
        url = _make_store_url(
            self._store_host,
            self._store_port,
            f"/models/{quote(model_id, safe='')}/download/status",
        )
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.get(url) as resp,
            ):
                if resp.status != 200:
                    return {"status": "not_found"}
                data: object = await resp.json()
                return data if isinstance(data, dict) else {"status": "unknown"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def delete_store_model(self, model_id: str) -> bool:
        """Delete a model from the store registry and disk.

        Returns ``True`` if deleted, ``False`` if not found.
        """
        url = _make_store_url(
            self._store_host,
            self._store_port,
            f"/models/{quote(model_id, safe='')}",
        )
        try:
            async with (
                create_http_session(timeout_profile="short") as session,
                session.delete(url) as resp,
            ):
                return resp.status == 200
        except Exception as exc:
            logger.warning(f"ModelStoreClient: delete_store_model failed: {exc}")
            return False

    async def request_and_wait_for_download(
        self,
        model_id: str,
        on_progress: Callable[[float], Awaitable[None]] | None = None,
        timeout: float = 7200,
        poll_interval: float = 5.0,
    ) -> bool:
        """Request the store host download a model from HuggingFace, then wait.

        Posts to ``/models/{id}/download`` to start the download, then polls
        ``/models/{id}/download/status`` until complete or failed.

        Args:
            model_id: HuggingFace model ID.
            on_progress: Called with progress (0.0-1.0) on each poll.
            timeout: Maximum wait time in seconds.
            poll_interval: Seconds between status polls.

        Returns:
            ``True`` if download completed successfully.

        Raises:
            RuntimeError: If the download failed on the store host.
            TimeoutError: If the download didn't complete within *timeout*.
        """
        import asyncio as _asyncio

        encoded_id = quote(model_id, safe="")

        # Request download
        url = _make_store_url(
            self._store_host, self._store_port, f"/models/{encoded_id}/download"
        )
        async with (
            create_http_session(timeout_profile="short") as session,
            session.post(url) as resp,
        ):
            if resp.status not in (200, 201):
                raise RuntimeError(f"Store download request failed: HTTP {resp.status}")
            data: object = await resp.json()
            if isinstance(data, dict) and data.get("status") == "complete":
                return True

        # Poll for completion
        status_url = _make_store_url(
            self._store_host, self._store_port, f"/models/{encoded_id}/download/status"
        )
        elapsed = 0.0
        while elapsed < timeout:
            await _asyncio.sleep(poll_interval)
            elapsed += poll_interval
            try:
                async with (
                    create_http_session(timeout_profile="short") as session,
                    session.get(status_url) as resp,
                ):
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    if not isinstance(data, dict):
                        continue
                    status = data.get("status", "")
                    progress = float(data.get("progress", 0.0))
                    if on_progress is not None:
                        await on_progress(progress)
                    if status == "complete":
                        return True
                    if status == "failed":
                        raise RuntimeError(
                            f"Store download of {model_id} failed: {data.get('error', 'unknown')}"
                        )
            except RuntimeError:
                raise
            except Exception as exc:
                logger.debug(f"ModelStoreClient: download status poll failed: {exc}")

        raise TimeoutError(f"Store download of {model_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # Local copy path (store host → same filesystem)
    # ------------------------------------------------------------------

    async def _stage_local(
        self,
        model_id: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None,
    ) -> Path:
        """Stage by local file copy (store host only)."""
        assert self._local_store_path is not None
        source_dir = self._local_store_path / _sanitize_model_id(model_id)
        if not source_dir.exists():
            raise ModelNotInStoreError(
                f"Model {model_id} not found in local store at {source_dir}"
            )

        files = [p for p in source_dir.rglob("*") if p.is_file()]
        total_bytes = sum(p.stat().st_size for p in files)
        staged_bytes = 0

        for src_file in files:
            rel = src_file.relative_to(source_dir)
            dst_file = dest_path / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            # Skip copy if destination already matches source size.
            # Run on a thread to avoid blocking the async event loop
            # during multi-GB safetensor copies.
            if (
                not dst_file.exists()
                or dst_file.stat().st_size != src_file.stat().st_size
            ):
                await asyncio.to_thread(shutil.copy2, src_file, dst_file)
            staged_bytes += src_file.stat().st_size
            if on_progress is not None:
                await on_progress(staged_bytes, total_bytes)

        logger.info(
            f"ModelStoreClient: staged {model_id} locally to {dest_path} "
            f"({total_bytes:,} bytes)"
        )
        return dest_path

    # ------------------------------------------------------------------
    # HTTP staging path (worker → store host)
    # ------------------------------------------------------------------

    async def _fetch_file_list(self, model_id: str) -> list[str]:
        """Fetch the list of files for *model_id* from the store server."""
        url = _make_store_url(
            self._store_host,
            self._store_port,
            f"/models/{quote(model_id, safe='')}/files",
        )
        async with (
            create_http_session(timeout_profile="short") as session,
            session.get(url) as resp,
        ):
            if resp.status == 404:
                raise ModelNotInStoreError(f"Model {model_id} not found in store")
            if resp.status != 200:
                raise RuntimeError(
                    f"ModelStoreClient: /models/{model_id}/files returned {resp.status}"
                )
            data: object = await resp.json()
            if not isinstance(data, list):
                raise RuntimeError(
                    "ModelStoreClient: unexpected file list response type"
                )
            return [str(item) for item in data if isinstance(item, str)]

    async def _download_store_file(
        self,
        model_id: str,
        file_path: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None,
        total_bytes_offset: int,
        grand_total: int,
    ) -> int:
        """Download a single file from the store to ``dest_path / file_path``.

        Uses a ``<filename>.partial`` temporary file and HTTP ``Range`` headers
        to support resuming interrupted downloads.  The partial file is
        atomically renamed on completion.

        Returns:
            Number of bytes written (for progress accumulation).
        """
        target = dest_path / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = dest_path / f"{file_path}.partial"

        # Already fully downloaded — skip
        if target.exists():
            return target.stat().st_size

        # Resume from partial if it exists
        resume_from = partial.stat().st_size if partial.exists() else 0

        headers: dict[str, str] = {}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            logger.debug(
                f"ModelStoreClient: resuming {file_path} from byte {resume_from:,}"
            )

        url = _make_store_url(
            self._store_host,
            self._store_port,
            f"/models/{quote(model_id, safe='')}/{file_path}",
        )
        timeout = aiohttp.ClientTimeout(
            total=3600,
            connect=_CONNECT_TIMEOUT,
            sock_read=_READ_TIMEOUT,
        )
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(url, headers=headers) as resp,
        ):
            if resp.status == 404:
                raise ModelNotInStoreError(
                    f"File not found in store: {model_id}/{file_path}"
                )
            # 416 Range Not Satisfiable means the partial file already
            # contains all bytes (process died after writing but before
            # the atomic rename).  Just promote it to final.
            if resp.status == 416 and resume_from > 0:
                await aios.rename(partial, target)
                return resume_from
            if resp.status not in (200, 206):
                raise RuntimeError(
                    f"ModelStoreClient: GET {url} returned {resp.status}"
                )

            n_read = resume_from
            async with aiofiles.open(partial, "ab" if resume_from > 0 else "wb") as f:
                async for chunk in resp.content.iter_chunked(_CHUNK_SIZE):
                    await f.write(chunk)
                    n_read += len(chunk)
                    if on_progress is not None:
                        await on_progress(total_bytes_offset + n_read, grand_total)

        # Atomic rename: partial → final
        await aios.rename(partial, target)
        return n_read

    async def _stage_http(
        self,
        model_id: str,
        dest_path: Path,
        on_progress: Callable[[int, int], Awaitable[None]] | None,
    ) -> Path:
        """Stage all files for *model_id* over HTTP."""
        file_list = await self._fetch_file_list(model_id)

        # Compute progress baseline from already-staged files
        grand_total = 0
        staged_offset = 0
        for file_path in file_list:
            target = dest_path / file_path
            if target.exists():
                size = target.stat().st_size
                grand_total += size
                staged_offset += size

        bytes_done = staged_offset
        for file_path in file_list:
            if (dest_path / file_path).exists():
                continue
            file_bytes = await self._download_store_file(
                model_id,
                file_path,
                dest_path,
                on_progress,
                total_bytes_offset=bytes_done,
                grand_total=max(grand_total, 1),
            )
            bytes_done += file_bytes

        logger.info(
            f"ModelStoreClient: staged {model_id} to {dest_path} ({bytes_done:,} bytes)"
        )
        return dest_path


# ---------------------------------------------------------------------------
# ModelStoreDownloader — ShardDownloader wrapper
# ---------------------------------------------------------------------------


@final
class ModelStoreDownloader(ShardDownloader):
    """ShardDownloader that intercepts ``ensure_shard()`` to stage from the store.

    This class wraps EXO's standard inner downloader (normally a
    ``SingletonShardDownloader`` wrapping a ``ResumableShardDownloader``) and
    overrides the download path:

    * **Model in store** → stage from store, return local staging path.
    * **Model not in store + HF fallback enabled** → delegate to inner
      downloader (standard HuggingFace download).
    * **Model not in store + HF fallback disabled** → raise
      :class:`ModelNotInStoreError`.

    The inner downloader is never bypassed for ``get_shard_download_status*``
    queries — those are forwarded directly so the download coordinator's
    progress tracking continues to work.

    Progress callbacks registered via ``on_progress()`` are forwarded to the
    inner downloader so that HuggingFace downloads still report progress.
    Store-staged downloads emit a synthetic ``in_progress`` → ``complete``
    progress pair.

    Wrapping pattern (mirrors how EXO wraps its own downloaders)::

        base = exo_shard_downloader(offline=args.offline)
        downloader = ModelStoreDownloader(
            inner=base,
            store_client=store_client,
            staging_config=staging_cfg,
            allow_hf_fallback=ms.download.allow_hf_fallback,
        )
        # Pass downloader to DownloadCoordinator as normal
    """

    def __init__(
        self,
        inner: ShardDownloader,
        store_client: ModelStoreClient,
        staging_config: StagingNodeConfig,
        allow_hf_fallback: bool = True,
    ) -> None:
        """
        Args:
            inner: The wrapped downloader used for HF fallback and status queries.
            store_client: Client for the store server.
            staging_config: Node-local staging configuration (path, cleanup policy).
            allow_hf_fallback: When ``True``, models not in the store are
                downloaded from HuggingFace via *inner*.  When ``False``,
                :class:`ModelNotInStoreError` is raised instead.
        """
        self._inner = inner
        self._store_client = store_client
        self._staging_config = staging_config
        self._allow_hf_fallback = allow_hf_fallback
        self._on_progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        """Register a progress callback (forwarded to the inner downloader)."""
        self._on_progress_callbacks.append(callback)
        self._inner.on_progress(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        """Ensure the shard is available locally, staging from the store if needed.

        Resolution order:

        1. If staging is disabled, delegate to inner downloader unconditionally.
        2. Check if model is available in the store.
        3. If yes: stage to ``node_cache_path`` and return the staging path.
        4. If no and ``allow_hf_fallback``: delegate to inner downloader.
        5. If no and not ``allow_hf_fallback``: raise :class:`ModelNotInStoreError`.

        Args:
            shard: Metadata for the shard to ensure is available.
            config_only: If ``True``, only fetch config files (forwarded to
                inner downloader for HF path; store staging always fetches all
                files).

        Returns:
            Absolute path to the local directory containing the model files.
        """
        model_id = str(shard.model_card.model_id)

        if not self._staging_config.enabled:
            # When staging is disabled but the store client has a local store
            # path (i.e. this is the store host), serve directly from the
            # canonical store directory instead of re-downloading from HF.
            if self._store_client.local_store_path is not None:
                direct_path = self._store_client.local_store_path / _sanitize_model_id(
                    model_id
                )
                if direct_path.exists() and any(direct_path.iterdir()):
                    logger.info(
                        f"ModelStoreDownloader: staging disabled — loading {model_id} directly from store at {direct_path}"
                    )
                    await self._emit_progress(shard, status="complete")
                    return direct_path
            return await self._inner.ensure_shard(shard, config_only)

        # Fast path: if the model is already staged locally, skip the
        # HTTP availability probe.  This keeps inference working when the store
        # server is temporarily unreachable and avoids an unnecessary round-trip.
        dest_path = _staging_dir(self._staging_config.node_cache_path, model_id)
        if dest_path.exists() and any(dest_path.iterdir()):
            logger.info(
                f"ModelStoreDownloader: {model_id} already staged at {dest_path} — skipping availability probe"
            )
            await self._emit_progress(shard, status="complete")
            return dest_path

        available = await self._store_client.is_model_available(model_id)

        if available:
            logger.info(
                f"ModelStoreDownloader: staging {model_id} from store → {dest_path}"
            )
            try:
                path = await self._store_client.stage_shard(
                    model_id,
                    dest_path,
                    on_progress=lambda downloaded, total: self._emit_progress(
                        shard,
                        status="in_progress",
                        downloaded_bytes=downloaded,
                        total_bytes=total,
                    ),
                )
                await self._emit_progress(shard, status="complete")
                return path
            except ModelNotInStoreError:
                # Store index was stale — the file was reported present but
                # could not be staged.  Fall through to the HF path.
                logger.warning(
                    f"ModelStoreDownloader: store reported {model_id} available "
                    "but staging failed — falling back to HuggingFace"
                )

        if self._allow_hf_fallback:
            # Ask the store host to download from HuggingFace on our behalf.
            # Workers never download from HF directly — the store is the
            # single source of truth for model files.
            logger.info(
                f"ModelStoreDownloader: {model_id} not in store — "
                "requesting store host to download from HuggingFace"
            )
            await self._emit_progress(shard, status="in_progress")
            try:
                await self._store_client.request_and_wait_for_download(
                    model_id,
                    on_progress=lambda _p: self._emit_progress(
                        shard, status="in_progress"
                    ),
                )
            except (RuntimeError, TimeoutError) as exc:
                raise ModelNotInStoreError(
                    f"Store host failed to download {model_id}: {exc}"
                ) from exc
            # Model now in store — stage it
            path = await self._store_client.stage_shard(
                model_id,
                dest_path,
                on_progress=lambda downloaded, total: self._emit_progress(
                    shard,
                    status="in_progress",
                    downloaded_bytes=downloaded,
                    total_bytes=total,
                ),
            )
            await self._emit_progress(shard, status="complete")
            return path

        raise ModelNotInStoreError(
            f"Model {model_id} is not in the store and HuggingFace fallback is disabled"
        )

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        """Forward to the inner downloader's status stream."""
        async for item in self._inner.get_shard_download_status():
            yield item

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        """Forward to the inner downloader's per-shard status query."""
        return await self._inner.get_shard_download_status_for_shard(shard)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit_progress(
        self,
        shard: ShardMetadata,
        status: str,
        downloaded_bytes: int = 0,
        total_bytes: int | None = None,
    ) -> None:
        """Emit a synthetic download progress event to all registered callbacks.

        This keeps the download coordinator's progress tracking coherent for
        store-staged models (which don't go through the normal HF download
        pipeline that would emit these events naturally).
        """
        total_memory = (
            Memory.from_bytes(total_bytes)
            if total_bytes is not None
            else shard.model_card.storage_size
        )
        downloaded_memory = Memory.from_bytes(downloaded_bytes)
        progress = RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="store",
            shard=shard,
            completed_files=1 if status == "complete" else 0,
            total_files=1,
            downloaded=downloaded_memory if status == "in_progress" else total_memory,
            downloaded_this_session=(
                downloaded_memory if status == "in_progress" else total_memory
            ),
            total=total_memory,
            overall_speed=0.0,
            overall_eta=timedelta(seconds=0),
            status="in_progress" if status == "in_progress" else "complete",
        )
        for cb in self._on_progress_callbacks:
            await cb(shard, progress)
