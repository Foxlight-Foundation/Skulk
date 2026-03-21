# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
ModelStore — centralized model registry and path resolution for the store host.

Role in the system
------------------
``ModelStore`` runs only on the **store host** node (the node whose hostname
or node_id matches ``model_store.store_host`` in ``exo.yaml``).  Worker nodes
never instantiate this class — they interact with the store exclusively via
:class:`~exo.store.model_store_client.ModelStoreClient` over HTTP.

Responsibilities
----------------
* Maintain a persistent JSON registry (``{store_path}/registry.json``) that
  maps HuggingFace model IDs to store metadata (path, file list, size,
  timestamp).
* Provide path resolution so :class:`~exo.store.model_store_server.ModelStoreServer`
  can serve files without scanning the filesystem on every request.
* Expose ``register_model()`` so external tools (and, in a future phase, the
  automatic HF-download hook) can add new models to the store.

Registry format
---------------
``registry.json`` is a plain JSON object::

    {
      "mlx-community/Qwen3-30B-A3B-4bit": {
        "model_id": "mlx-community/Qwen3-30B-A3B-4bit",
        "store_path": "mlx-community--Qwen3-30B-A3B-4bit",
        "files": ["config.json", "model-00001-of-00008.safetensors", ...],
        "downloaded_at": "2026-03-20T14:32:00+00:00",
        "total_bytes": 21474836480
      },
      ...
    }

Directory layout on the store host::

    <store_path>/
      registry.json
      mlx-community--Qwen3-30B-A3B-4bit/
        config.json
        tokenizer.json
        model-00001-of-00008.safetensors
        ...
      mlx-community--Llama-3.1-8B-Instruct-4bit/
        ...

The ``/`` → ``--`` sanitization in directory names matches the convention
used by EXO's own ``~/.exo/models/`` cache.

Thread safety
-------------
All registry I/O is synchronous.  In async contexts where blocking I/O
would be a concern, callers should wrap in ``anyio.to_thread.run_sync``.
In practice, registry reads happen once per request and writes happen only
when a new model is registered, so this is not a hot path.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, final

import aiofiles.os as aios
from loguru import logger
from pydantic import BaseModel, ConfigDict


@final
class StoreModelEntry(BaseModel):
    """Metadata for a single model in the store registry.

    This is the value type stored in ``registry.json``.  It is intentionally
    minimal — the registry is an index, not a full catalogue.  Richer
    metadata (quantization, parameter count, etc.) lives in the model card
    accessible via the HuggingFace model ID.

    Attributes:
        model_id: HuggingFace-style model identifier,
            e.g. ``"mlx-community/Qwen3-30B-A3B-4bit"``.
        store_path: Path of the model directory **relative to** the store
            root, e.g. ``"mlx-community--Qwen3-30B-A3B-4bit"``.
        files: List of file paths relative to the model directory.
        downloaded_at: ISO 8601 UTC timestamp of when the model was
            registered in the store.
        total_bytes: Sum of all file sizes at registration time.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str
    store_path: str
    files: list[str]
    downloaded_at: str
    total_bytes: int


@dataclass
class StoreDownloadStatus:
    """Tracks the progress of a store-side HuggingFace download."""
    model_id: str
    status: Literal["pending", "downloading", "complete", "failed"] = "pending"
    progress: float = 0.0
    error: str | None = None


@final
class ModelStore:
    """Manages the model registry on the store host node.

    Instantiated once per process on the store host.  Worker nodes do not
    use this class directly.

    The registry is a JSON file at ``{store_path}/registry.json``.
    It maps ``model_id → StoreModelEntry``.

    Example usage (store-host management script)::

        from pathlib import Path
        from exo.store.model_store import ModelStore

        store = ModelStore(Path("/Volumes/ModelStore/models"))
        model_path = Path("/Volumes/ModelStore/models/mlx-community--Qwen3-30B-A3B-4bit")
        files = [str(p.relative_to(model_path)) for p in model_path.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in model_path.rglob("*") if p.is_file())
        store.register_model("mlx-community/Qwen3-30B-A3B-4bit", model_path, files, total)
    """

    def __init__(self, store_path: Path) -> None:
        """
        Args:
            store_path: Absolute path to the model store root directory on the
                store host.  This directory must be readable by the exo process
                and writable for registry updates.
        """
        self._store_path = store_path
        self._registry_path = store_path / "registry.json"
        self._active_downloads: dict[str, StoreDownloadStatus] = {}
        self._download_lock = asyncio.Lock()

    @property
    def store_path(self) -> Path:
        """Absolute path to the model store root directory."""
        return self._store_path

    def is_in_store(self, model_id: str) -> bool:
        """Return ``True`` if *model_id* is in the registry **and** its
        directory exists on disk.

        Both conditions must be true: a registry entry whose directory has
        been deleted returns ``False``.

        Args:
            model_id: HuggingFace-style model ID.
        """
        return self.get_store_path(model_id) is not None

    def get_store_path(self, model_id: str) -> Path | None:
        """Return the absolute path to *model_id*'s directory, or ``None``.

        Returns ``None`` if the model is not in the registry, or if the
        registered directory no longer exists on disk.

        Args:
            model_id: HuggingFace-style model ID.
        """
        registry = self._read_registry()
        entry = registry.get(model_id)
        if entry is None:
            return None
        model_path = self._store_path / entry.store_path
        if not model_path.exists():
            return None
        return model_path

    def list_models(self) -> list[StoreModelEntry]:
        """Return all :class:`StoreModelEntry` objects currently in the registry."""
        return list(self._read_registry().values())

    def delete_model(self, model_id: str) -> bool:
        """Remove *model_id* from the registry and delete its files from disk.

        Returns ``True`` if the model was found and deleted, ``False`` if not
        in the registry.
        """
        import shutil

        registry = self._read_registry()
        entry = registry.pop(model_id, None)
        if entry is None:
            return False
        # Remove files from disk
        model_path = self._store_path / entry.store_path
        if model_path.exists():
            shutil.rmtree(model_path, ignore_errors=True)
            logger.info(f"ModelStore: deleted {model_id} from {model_path}")
        # Update registry
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(
            json.dumps(
                {k: v.model_dump() for k, v in registry.items()},
                indent=2,
            )
        )
        return True

    def register_model(
        self,
        model_id: str,
        model_path: Path,
        files: list[str],
        total_bytes: int,
    ) -> None:
        """Add or update *model_id* in the registry.

        If an entry already exists for this model it is overwritten (idempotent).

        Args:
            model_id: HuggingFace-style model ID,
                e.g. ``"mlx-community/Qwen3-30B-A3B-4bit"``.
            model_path: Absolute path to the model directory on the store host.
                Must be inside ``store_path``.
            files: List of file paths relative to *model_path*.
            total_bytes: Sum of file sizes in bytes.
        """
        relative_path = str(model_path.relative_to(self._store_path))
        entry = StoreModelEntry(
            model_id=model_id,
            store_path=relative_path,
            files=files,
            downloaded_at=datetime.now(tz=timezone.utc).isoformat(),
            total_bytes=total_bytes,
        )
        self._write_registry_entry(entry)
        logger.info(
            f"ModelStore: registered {model_id} at {relative_path} "
            f"({total_bytes:,} bytes, {len(files)} files)"
        )

    def list_files_for_model(self, model_id: str) -> list[str] | None:
        """Return the file list for *model_id* from the registry, or ``None``.

        ``None`` means the model is not in the registry (equivalent to
        ``is_in_store() == False``).

        Args:
            model_id: HuggingFace-style model ID.
        """
        registry = self._read_registry()
        entry = registry.get(model_id)
        if entry is None:
            return None
        return entry.files

    # ------------------------------------------------------------------
    # Registry I/O (synchronous)
    # ------------------------------------------------------------------

    def _read_registry(self) -> dict[str, StoreModelEntry]:
        """Read and parse ``registry.json``.  Returns empty dict on any error."""
        if not self._registry_path.exists():
            return {}
        try:
            data: object = json.loads(self._registry_path.read_text())
            if not isinstance(data, dict):
                logger.warning("ModelStore: registry.json is not a dict — resetting")
                return {}
            return {
                k: StoreModelEntry.model_validate(v)
                for k, v in data.items()
                if isinstance(k, str)
            }
        except Exception as exc:
            logger.warning(f"ModelStore: failed to read registry: {exc}")
            return {}

    def _write_registry_entry(self, entry: StoreModelEntry) -> None:
        """Atomically upsert *entry* in ``registry.json``."""
        self._store_path.mkdir(parents=True, exist_ok=True)
        registry = self._read_registry()
        registry[entry.model_id] = entry
        self._registry_path.write_text(
            json.dumps(
                {k: v.model_dump() for k, v in registry.items()},
                indent=2,
            )
        )

    # ------------------------------------------------------------------
    # Store-side HuggingFace downloads
    # ------------------------------------------------------------------

    async def request_download(self, model_id: str) -> StoreDownloadStatus:
        """Request that the store download a model from HuggingFace.

        Deduplicates: if the model is already downloading, returns the
        existing status.  If already in the store, returns "complete".
        """
        async with self._download_lock:
            existing = self._active_downloads.get(model_id)
            if existing is not None:
                # Allow retrying failed downloads
                if existing.status == "failed":
                    del self._active_downloads[model_id]
                else:
                    return existing
            if self.is_in_store(model_id):
                return StoreDownloadStatus(model_id=model_id, status="complete", progress=1.0)
            status = StoreDownloadStatus(model_id=model_id, status="pending")
            self._active_downloads[model_id] = status
        asyncio.create_task(self._do_download(model_id))
        return status

    def get_download_status(self, model_id: str) -> StoreDownloadStatus | None:
        """Return the download status for *model_id*, or None."""
        if model_id in self._active_downloads:
            return self._active_downloads[model_id]
        if self.is_in_store(model_id):
            return StoreDownloadStatus(model_id=model_id, status="complete", progress=1.0)
        return None

    def list_active_downloads(self) -> list[StoreDownloadStatus]:
        """Return all in-progress or pending downloads."""
        return [s for s in self._active_downloads.values() if s.status in ("pending", "downloading")]

    async def _do_download(self, model_id: str) -> None:
        """Download a model from HuggingFace into the store and register it."""
        from exo.download.download_utils import (
            download_file_with_retry,
            fetch_file_list_with_cache,
        )
        from exo.shared.models.model_cards import ModelId

        status = self._active_downloads[model_id]
        status.status = "downloading"
        sanitized = model_id.replace("/", "--")
        target_dir = self._store_path / sanitized

        try:
            await aios.makedirs(str(target_dir), exist_ok=True)

            file_list = await fetch_file_list_with_cache(
                ModelId(model_id), "main", recursive=True
            )
            total_bytes = sum(f.size or 0 for f in file_list)
            downloaded_bytes = 0

            for f in file_list:
                file_size = f.size or 0

                def make_progress_cb(fsize: int):
                    def cb(curr: int, total: int, is_renamed: bool) -> None:
                        nonlocal downloaded_bytes
                        status.progress = (downloaded_bytes + curr) / max(total_bytes, 1)
                    return cb

                await download_file_with_retry(
                    ModelId(model_id),
                    "main",
                    f.path,
                    target_dir,
                    make_progress_cb(file_size),
                )
                downloaded_bytes += file_size
                status.progress = downloaded_bytes / max(total_bytes, 1)

            # Register in the store
            files = [str(p.relative_to(target_dir)) for p in target_dir.rglob("*") if p.is_file()]
            total = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
            self.register_model(model_id, target_dir, files, total)

            status.status = "complete"
            status.progress = 1.0
            logger.info(f"ModelStore: downloaded {model_id} from HuggingFace ({total:,} bytes)")

        except Exception as exc:
            status.status = "failed"
            status.error = str(exc)
            logger.error(f"ModelStore: download of {model_id} failed: {exc}")
