# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
ModelStore — centralized model registry and path resolution for the store host.

Role in the system
------------------
``ModelStore`` runs only on the **store host** node (the node whose hostname
or node_id matches ``model_store.store_host`` in ``skulk.yaml``).  Worker nodes
never instantiate this class — they interact with the store exclusively via
:class:`~skulk.store.model_store_client.ModelStoreClient` over HTTP.

Responsibilities
----------------
* Maintain a persistent JSON registry (``{store_path}/registry.json``) that
  maps HuggingFace model IDs to store metadata (path, file list, size,
  timestamp).
* Provide path resolution so :class:`~skulk.store.model_store_server.ModelStoreServer`
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
used for HuggingFace cache directories.

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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, final

import aiofiles.os as aios
from loguru import logger
from pydantic import BaseModel, ConfigDict

from skulk.shared.types.worker.downloads import FileListEntry


def select_store_gguf_download_files(
    file_list: list[FileListEntry],
    pinned_gguf: str | None = None,
) -> list[FileListEntry]:
    """Filter a repo's file list to what the store host should download (#339).

    A multi-quant GGUF repo ships every quantization (and sometimes the original
    full-precision weights under ``original/`` plus ``metal/`` artifacts). The
    store host should fetch exactly what the direct-HuggingFace path fetches:
    the preferred quant's shard group plus ``config.json``, and nothing else.
    This mirrors ``download_utils.resolve_allow_patterns`` for a GGUF card
    (``[*gguf_allow_patterns(gguf_file), "config.json"]``), so a store-routed
    download is no larger than a direct one. A non-GGUF repo (no ``.gguf`` LM
    weights, excluding ``mmproj`` projectors) is returned unchanged.

    Args:
        file_list: The full recursive repo file list.
        pinned_gguf: The card's pinned GGUF file (``ModelCard.gguf_file``), when
            the requester carries one. The store fetches *that* quant's shard
            group, honoring a custom/non-default pin (#344). When ``None`` (or
            the pin is absent from the repo) the store falls back to the default
            quant preference, matching the prior behavior.

    Returns:
        The subset to download. Identical to the input for non-GGUF repos.
    """
    from fnmatch import fnmatch

    from skulk.shared.models.model_cards import (
        gguf_allow_patterns,
        select_preferred_gguf,
    )

    gguf_weights = [
        entry
        for entry in file_list
        if entry.path.endswith(".gguf") and "mmproj" not in entry.path.lower()
    ]
    if not gguf_weights:
        return file_list  # not a GGUF repo (or vision-only projector): unchanged

    # Honor a card's pinned quant when it names a file actually in the repo;
    # otherwise fall back to the default preference (#344). The pin is matched by
    # exact repo-relative path so a stale/typo'd pin can't silently select the
    # wrong file -- it just degrades to the default.
    pinned_paths = {entry.path for entry in gguf_weights}
    if pinned_gguf and pinned_gguf in pinned_paths:
        selected = pinned_gguf
    else:
        if pinned_gguf:
            logger.warning(
                f"ModelStore: pinned GGUF {pinned_gguf!r} not found in repo; "
                "falling back to the default quant preference"
            )
        selected = select_preferred_gguf(
            [(entry.path, entry.size or 0) for entry in gguf_weights]
        )
    # Match the direct-HF GGUF allow-list exactly: the selected shard group
    # (``gguf_allow_patterns`` returns either the single file or the
    # ``<base>-*-of-*.gguf`` glob) plus ``config.json``. Patterns and paths are
    # both repo-relative, the same basis HuggingFace's ``allow_patterns`` matches
    # on, so a GGUF in a subdirectory aligns with the direct path. Every other
    # file (other quants, original/* full-precision weights, metal/*, tokenizer,
    # README) is dropped, just as the direct path drops them.
    keep_patterns = [*gguf_allow_patterns(selected), "config.json"]

    def _keep(entry: FileListEntry) -> bool:
        # Keep the selected quant's shard group + config.json, plus the
        # multimodal projector matched case-insensitively: ``has_gguf_projector``
        # (used to detect a vision repo and to verify completeness) is
        # case-insensitive, so the selection MUST be too, or an uppercase
        # ``MMPROJ-*.gguf`` would be detected-but-never-selected and the
        # registration guard would fail every retry.
        if has_gguf_projector([entry.path]):
            return True
        return any(fnmatch(entry.path, pattern) for pattern in keep_patterns)

    return [entry for entry in file_list if _keep(entry)]


def has_gguf_projector(paths: Iterable[str]) -> bool:
    """Whether any path is a multimodal projector GGUF (``*mmproj*.gguf``).

    A vision GGUF ships its projector as a separate ``mmproj`` file alongside the
    LM weights; the store uses this both to detect that a repo is a vision GGUF
    (from its full file list) and to verify the projector actually landed before
    registering the model (#346).

    Matches the same convention as the card resolver (``gguf_repo_has_projector``)
    and the runner (``find_mmproj_file``): a case-sensitive ``.gguf`` extension
    (HF GGUF files are always lowercase-extension) with a case-insensitive
    ``mmproj`` in the **basename**. The runner identifies the projector by
    basename, so matching on the basename here (not anywhere in the path) keeps
    all three aligned and avoids misclassifying a non-projector GGUF that merely
    sits under a directory whose name contains ``mmproj``.
    """
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".gguf") and "mmproj" in name.lower():
            return True
    return False


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
    # Whether the upstream repo ships a multimodal projector, recorded at
    # registration so the availability hot path can decide a vision GGUF's
    # completeness without an HF repo-list probe (#346). ``None`` on entries
    # written before this field existed (legacy): those fall back to a one-time
    # HF probe until they are re-registered.
    repo_has_projector: bool | None = None


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
        from skulk.store.model_store import ModelStore

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
                store host.  This directory must be readable by the Skulk process
                and writable for registry updates.
        """
        self._store_path = store_path
        self._registry_path = store_path / "registry.json"
        self._active_downloads: dict[str, StoreDownloadStatus] = {}
        self._download_lock = asyncio.Lock()
        self._download_tasks: set[asyncio.Task[None]] = set()

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
        """Return all :class:`StoreModelEntry` objects currently in the registry
        whose directories still exist on disk.

        Entries whose model directory has been removed are silently excluded.
        """
        registry = self._read_registry()
        return [
            entry
            for entry in registry.values()
            if (self._store_path / entry.store_path).exists()
        ]

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
        repo_has_projector: bool | None = None,
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
            repo_has_projector=repo_has_projector,
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

    async def vision_entry_missing_projector(self, model_id: str) -> bool:
        """True when an in-store GGUF entry is a vision model missing its projector.

        Recovers stale entries registered before the projector was retained
        (#346): such an entry lists only the LM quant, so staging it produces an
        unloadable vision model. We detect that here so a re-download request is
        not short-circuited as "already complete": the re-download skips the
        already-present weights (size-matched) and fetches only the missing
        projector, then re-registers a complete entry. Returns ``False`` when the
        model is absent, already has a projector, is not a vision GGUF, or the
        repo listing cannot be fetched (offline): we only force the redownload on
        positive evidence of an incomplete vision entry.
        """
        entry = self._read_registry().get(model_id)
        if entry is None or has_gguf_projector(entry.files):
            return False
        # Steady state: the registration guard records whether the upstream repo
        # ships a projector, so the answer is local: no HF probe on this hot
        # path (it backs the worker's store-availability check). A vision GGUF
        # missing its projector is stale; a text model (no projector upstream) is
        # complete.
        if entry.repo_has_projector is not None:
            return entry.repo_has_projector
        # Legacy entry (written before the flag existed): fall back to a one-time
        # (cached) HF probe, restricted to GGUF entries so an MLX/safetensors
        # model is never probed. After it re-registers, the flag above takes over.
        if not any(name.endswith(".gguf") for name in entry.files):
            return False
        from skulk.download.download_utils import fetch_file_list_with_cache
        from skulk.shared.models.model_cards import ModelId

        try:
            repo_files = await fetch_file_list_with_cache(
                ModelId(model_id), "main", recursive=True
            )
        except Exception as exc:
            logger.debug(
                f"ModelStore: could not check projector completeness for "
                f"{model_id} ({exc}); leaving the existing entry as-is"
            )
            return False
        return has_gguf_projector(f.path for f in repo_files)

    async def request_download(
        self, model_id: str, pinned_gguf: str | None = None
    ) -> StoreDownloadStatus:
        """Request that the store download a model from HuggingFace.

        Deduplicates: if the model is already downloading, returns the
        existing status.  If already in the store, returns "complete", unless it
        is a vision GGUF whose stored entry is missing its ``mmproj`` projector
        (a stale pre-#346 entry), in which case it re-downloads to recover the
        projector instead of staging an unloadable model.

        ``pinned_gguf`` is the requester's card-pinned GGUF file, forwarded so a
        GGUF repo fetches that quant's shard group rather than the default (#344).
        """
        # Checked outside the lock: it may do a (cached) repo file-list fetch, and
        # holding the download lock across network I/O would serialize unrelated
        # requests.
        missing_projector = await self.vision_entry_missing_projector(model_id)
        async with self._download_lock:
            existing = self._active_downloads.get(model_id)
            if existing is not None:
                if existing.status == "failed":
                    del self._active_downloads[model_id]
                else:
                    return existing
            if self.is_in_store(model_id) and not missing_projector:
                return StoreDownloadStatus(
                    model_id=model_id, status="complete", progress=1.0
                )
            if missing_projector:
                logger.warning(
                    f"ModelStore: {model_id} is in the store but its entry is "
                    "missing the mmproj projector for a vision GGUF; "
                    "re-downloading to recover it (existing weights are reused)."
                )
            status = StoreDownloadStatus(model_id=model_id, status="pending")
            self._active_downloads[model_id] = status
        task = asyncio.create_task(self._do_download(model_id, pinned_gguf))
        self._download_tasks.add(task)
        task.add_done_callback(self._download_tasks.discard)
        return status

    def get_download_status(self, model_id: str) -> StoreDownloadStatus | None:
        """Return the download status for *model_id*, or None."""
        if model_id in self._active_downloads:
            return self._active_downloads[model_id]
        if self.is_in_store(model_id):
            return StoreDownloadStatus(
                model_id=model_id, status="complete", progress=1.0
            )
        return None

    def list_active_downloads(self) -> list[StoreDownloadStatus]:
        """Return all in-progress or pending downloads."""
        return [
            s
            for s in self._active_downloads.values()
            if s.status in ("pending", "downloading")
        ]

    async def _do_download(
        self, model_id: str, pinned_gguf: str | None = None
    ) -> None:
        """Download a model from HuggingFace into the store and register it.

        ``pinned_gguf`` (the requester's ``ModelCard.gguf_file``) selects which
        GGUF quant's shard group to fetch, honoring a custom pin (#344).
        """
        from skulk.download.download_utils import (
            download_file_with_retry,
            fetch_file_list_with_cache,
        )
        from skulk.shared.models.model_cards import ModelId

        status = self._active_downloads[model_id]
        status.status = "downloading"
        sanitized = model_id.replace("/", "--")
        target_dir = self._store_path / sanitized
        logger.info(
            f"ModelStore: downloading {model_id} from HuggingFace to {target_dir}"
        )

        try:
            await aios.makedirs(str(target_dir), exist_ok=True)

            repo_file_list = await fetch_file_list_with_cache(
                ModelId(model_id), "main", recursive=True
            )
            # A vision GGUF (LLaVA/Qwen-VL/Gemma-VLM style) ships its multimodal
            # projector as a separate ``*mmproj*.gguf`` alongside the LM weights;
            # the llama.cpp runner cannot load the model without it. Detect that
            # from the full repo listing so we can verify the projector actually
            # lands before registering (see the post-download guard below): a
            # store entry that omits the projector stages an unloadable vision
            # model, which surfaces only as a runner crash at load time (#346).
            repo_ships_projector = has_gguf_projector(
                f.path for f in repo_file_list
            )
            # For a GGUF repo, fetch only the preferred quant's shard group plus
            # config.json (dropping other quants, original/*, metal/*, etc.),
            # mirroring the direct-HF selective download (#339). The projector
            # glob is retained by ``gguf_allow_patterns``, so a vision GGUF keeps
            # its ``*mmproj*.gguf``.
            selected_files = select_store_gguf_download_files(
                repo_file_list, pinned_gguf
            )
            if len(selected_files) != len(repo_file_list):
                logger.info(
                    f"ModelStore: {model_id} is a GGUF repo; downloading "
                    f"{len(selected_files)}/{len(repo_file_list)} files (selected "
                    "quant's shard group + config.json only)"
                )
            file_list = selected_files
            total_bytes = sum(f.size or 0 for f in file_list)
            downloaded_bytes = 0

            for f in file_list:
                file_size = f.size or 0

                def make_progress_cb(fsize: int):
                    def cb(curr: int, total: int, is_renamed: bool) -> None:
                        nonlocal downloaded_bytes
                        status.progress = (downloaded_bytes + curr) / max(  # noqa: B023
                            total_bytes, 1
                        )

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
            files = [
                str(p.relative_to(target_dir))
                for p in target_dir.rglob("*")
                if p.is_file()
            ]
            # Vision-projector completeness guard (#346): a vision GGUF whose repo
            # ships an ``mmproj`` projector MUST land its projector, or staging it
            # to a worker produces an unloadable model that only fails as a runner
            # crash at load time. Refuse to register an incomplete vision model so
            # the failure is a loud, fixable download error here (re-running the
            # download re-fetches the projector, which the selective allow-list
            # already retains) instead of a confusing crash on a remote node.
            if repo_ships_projector and not has_gguf_projector(files):
                raise RuntimeError(
                    f"{model_id} is a vision GGUF whose repo ships an mmproj "
                    "projector, but none landed in the store download; refusing to "
                    "register an unusable vision model. Re-run the download to "
                    "re-fetch the projector."
                )
            total = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
            self.register_model(
                model_id, target_dir, files, total, repo_has_projector=repo_ships_projector
            )

            status.status = "complete"
            status.progress = 1.0
            logger.info(
                f"ModelStore: downloaded {model_id} from HuggingFace ({total:,} bytes)"
            )

        except Exception as exc:
            status.status = "failed"
            status.error = str(exc)
            logger.error(f"ModelStore: download of {model_id} failed: {exc}")
