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
``registry.json`` is a versioned, rebuildable index::

    {
      "schema_version": 1,
      "entries": {
        "mlx-community/Qwen3-30B-A3B-4bit": {
          "model_id": "mlx-community/Qwen3-30B-A3B-4bit",
          "store_path": "mlx-community--Qwen3-30B-A3B-4bit",
          "files": ["config.json", "model-00001-of-00008.safetensors", ...],
          "downloaded_at": "2026-03-20T14:32:00+00:00",
          "total_bytes": 21474836480
        }
      }
    }

Legacy plain mappings are migrated on the next write. Complete artifact
sidecars rebuild the index after corruption or loss.

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
import contextlib
import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, final

import aiofiles.os as aios
import aiohttp
from loguru import logger
from pydantic import BaseModel, ConfigDict

from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store.installed_cards import (
    INSTALLED_CARD_RELATIVE_PATH,
    InstalledArtifactRole,
    InstalledCardRecord,
    build_file_manifest,
    build_installed_card_record,
    manifest_sha256,
    read_installed_card,
    verify_installed_card,
    write_installed_card,
)
from skulk.store.staging_eviction import MINIMUM_STAGING_FREE_DISK_BYTES

if TYPE_CHECKING:
    from skulk.shared.models.model_cards import ModelCard

_SOURCE_REVISION_MARKER = ".skulk-source-revision"
_SOURCE_REVISION_STAGING_MARKER = ".skulk-source-revision-staging"
_RECONCILIATION_TOMBSTONES_RELATIVE_PATH = (
    Path(".skulk") / "reconciliation-tombstones.json"
)


def _remaining_store_download_bytes(
    target_directory: Path,
    file_list: list[FileListEntry],
) -> int:
    """Return physical file bytes still needed for one canonical transfer."""

    remaining_bytes = 0
    for file_entry in file_list:
        if file_entry.size is None:
            raise ModelStoreCapacityError(
                "Cannot verify canonical model-store disk capacity because "
                "the selected manifest contains a file with unknown size. "
                "Retry after repository metadata is available."
            )
        expected_size = file_entry.size
        target = target_directory / file_entry.path
        partial = target_directory / f"{file_entry.path}.partial"
        target_stat = target.stat() if target.is_file() else None
        target_bytes = target_stat.st_size if target_stat is not None else 0
        if target_bytes == expected_size:
            continue
        resumable_bytes = partial.stat().st_size if partial.is_file() else 0
        # A mismatched single-link target is removed before download, returning
        # its current blocks to this filesystem. When staging hardlinked the
        # canonical file, unlinking only its store name leaves those blocks
        # owned by the staged link, so none of its bytes can be credited.
        reclaimable_target_bytes = (
            target_bytes if target_stat is not None and target_stat.st_nlink == 1 else 0
        )
        # A partial is grown in place, so its existing bytes remain reusable.
        remaining_bytes += max(
            0,
            expected_size - reclaimable_target_bytes - resumable_bytes,
        )
    return remaining_bytes


def _sha256_path(path: Path) -> str:
    """Hash one artifact path without loading multi-gigabyte files into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_store_gguf_download_files(
    file_list: list[FileListEntry],
    pinned_gguf: str | None = None,
    extra_pinned_gguf: list[str] | None = None,
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
        extra_pinned_gguf: Additional repo-relative GGUF files in the SAME repo
            that must be co-fetched with the base quant, e.g. a served-engine
            draft GGUF bundled alongside the base (``served_spec_draft_repo`` ==
            the base repo). Each is kept as its own shard group on top of the
            base selection so a single store download lands both. Names absent
            from the repo are skipped with a warning.

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
    # Co-fetch any same-repo companion GGUF the requester pinned (e.g. a
    # served-engine draft bundled with the base): keep each one's own shard
    # group so a single store download lands the base AND the companion. A pin
    # that doesn't name a file in this repo is dropped with a warning, matching
    # how a stale base pin degrades above (the served runner surfaces a genuinely
    # absent draft loudly at launch via _draft_model_args).
    for extra in extra_pinned_gguf or []:
        if extra in pinned_paths:
            keep_patterns.extend(gguf_allow_patterns(extra))
        elif extra:
            logger.warning(
                f"ModelStore: extra pinned GGUF {extra!r} not found in repo; "
                "skipping (it will not be co-fetched)"
            )

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


def _resolve_store_child_path(store_root: Path, registered_path: str) -> Path | None:
    """Resolve a registry path only when it remains below the store root."""

    resolved_root = store_root.resolve()
    candidate = (resolved_root / registered_path).resolve()
    if candidate == resolved_root or not candidate.is_relative_to(resolved_root):
        return None
    return candidate


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
    source_revision: str | None = None
    """Immutable Hugging Face commit that produced this entry, when pinned."""
    source_repository: str | None = None
    """Upstream repository that supplied the bytes; ``None`` means ``model_id``
    for entries written before repository-aware artifact identity."""
    # Whether the upstream repo ships a multimodal projector, recorded at
    # registration so the availability hot path can decide a vision GGUF's
    # completeness without an HF repo-list probe (#346). ``None`` on entries
    # written before this field existed (legacy): those fall back to a one-time
    # HF probe until they are re-registered.
    repo_has_projector: bool | None = None
    installed_card: InstalledCardRecord | None = None
    """Complete local card and manifest identity for air-gapped operation."""


@final
class StoreRegistryIndex(BaseModel):
    """Schema-versioned rebuildable index for canonical store artifacts."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    entries: dict[str, StoreModelEntry]


@final
class StoreReconciliationTombstones(BaseModel):
    """Durable aliases that automatic cache reconciliation must not restore."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    deleted_aliases: dict[str, str]
    """Model aliases mapped to the UTC time their operator deletion committed."""


def read_reconciliation_tombstones(store_path: Path) -> frozenset[str]:
    """Return aliases suppressed from automatic peer-cache reconciliation.

    A malformed tombstone file raises instead of silently treating every alias
    as eligible for import. Losing deletion intent is less safe than pausing
    reconciliation until the operator repairs the small metadata file.

    Args:
        store_path: Canonical model-store root.

    Returns:
        The aliases retained as operator-deletion tombstones.

    Raises:
        ValueError: If the durable tombstone file is malformed.
        OSError: If the file exists but cannot be read.
    """

    path = store_path / _RECONCILIATION_TOMBSTONES_RELATIVE_PATH
    if not path.exists():
        return frozenset()
    index = StoreReconciliationTombstones.model_validate_json(
        path.read_bytes(), strict=False
    )
    return frozenset(index.deleted_aliases)


@dataclass
class StoreDownloadStatus:
    """Tracks the progress of a store-side HuggingFace download."""

    model_id: str
    source_revision: str | None = None
    source_repository: str | None = None
    pinned_gguf: str | None = None
    extra_pinned_gguf: tuple[str, ...] = ()
    model_card: ModelCard | None = None
    artifact_role: InstalledArtifactRole = "base"
    owner_model_id: str | None = None
    owner_card_id: str | None = None
    status: Literal["pending", "downloading", "complete", "failed", "cancelled"] = (
        "pending"
    )
    progress: float = 0.0
    error: str | None = None


class ModelStoreCapacityError(RuntimeError):
    """Raised before a canonical store transfer that would consume headroom."""


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
        self._reconciliation_tombstones_path = (
            store_path / _RECONCILIATION_TOMBSTONES_RELATIVE_PATH
        )
        self._legacy_registry_backup_path = (
            store_path / "registry.pre-installed-cards.json"
        )
        self._active_downloads: dict[str, StoreDownloadStatus] = {}
        self._download_lock = asyncio.Lock()
        self._download_transfer_lock = asyncio.Lock()
        self._download_tasks: set[asyncio.Task[None]] = set()
        self._download_tasks_by_model: dict[str, asyncio.Task[None]] = {}
        self._rebuild_registry_from_sidecars()

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
        model_path = _resolve_store_child_path(self._store_path, entry.store_path)
        if model_path is None:
            logger.warning(
                f"ModelStore: ignoring unsafe registry path for {model_id}: "
                f"{entry.store_path!r}"
            )
            return None
        if not model_path.exists():
            return None
        return model_path

    def get_entry(self, model_id: str) -> StoreModelEntry | None:
        """Return one registered model entry, or ``None`` when unavailable."""

        entry = self._read_registry().get(model_id)
        if entry is None:
            return None
        model_path = _resolve_store_child_path(self._store_path, entry.store_path)
        if model_path is None or not model_path.exists():
            return None
        return entry

    def entry_matches_revision(
        self, model_id: str, source_revision: str | None
    ) -> bool:
        """Return whether the canonical entry matches the requested revision."""

        entry = self.get_entry(model_id)
        return entry is not None and entry.source_revision == source_revision

    def entry_matches_artifact(
        self,
        model_id: str,
        source_revision: str | None,
        source_repository: str | None,
    ) -> bool:
        """Return whether the canonical entry matches the complete byte source."""
        entry = self.get_entry(model_id)
        if entry is None or entry.source_revision != source_revision:
            return False
        registered_repository = entry.source_repository or entry.model_id
        requested_repository = source_repository or model_id
        return registered_repository == requested_repository

    def list_models(self) -> list[StoreModelEntry]:
        """Return all :class:`StoreModelEntry` objects currently in the registry
        whose directories still exist on disk.

        Entries whose model directory has been removed are silently excluded.
        """
        registry = self._read_registry()
        return [
            entry
            for entry in registry.values()
            if (
                (
                    model_path := _resolve_store_child_path(
                        self._store_path, entry.store_path
                    )
                )
                is not None
                and model_path.exists()
            )
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
        # Persist deletion intent before removing either the canonical bytes or
        # the index entry. A node that missed the best-effort staged eviction may
        # otherwise advertise its retained copy and cause reconciliation to
        # resurrect the model immediately after this method reports success.
        self._record_reconciliation_tombstone(model_id)
        # Remove files from disk
        model_path = _resolve_store_child_path(self._store_path, entry.store_path)
        if model_path is None:
            logger.warning(
                f"ModelStore: removed unsafe registry entry for {model_id} "
                f"without deleting {entry.store_path!r}"
            )
        elif model_path.exists():
            shutil.rmtree(model_path, ignore_errors=True)
            logger.info(f"ModelStore: deleted {model_id} from {model_path}")
        # Update registry
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._write_registry(registry)
        # Drop a *terminal* cached download status for the deleted model. Leaving
        # a stale "complete" here makes a later request_download short-circuit and
        # never re-fetch the model we just removed (the registry entry and files
        # are gone but the in-memory status would still say complete). Only clear
        # complete/failed/cancelled entries: an in-flight (pending/downloading) entry is
        # held by a live _do_download task, and popping it would crash that task
        # (it reads self._active_downloads[model_id]). request_download also
        # re-checks is_in_store as a backstop, so a surviving in-flight entry that
        # later completes against a since-deleted model is still self-correcting.
        cached = self._active_downloads.get(model_id)
        if cached is not None and cached.status in (
            "complete",
            "failed",
            "cancelled",
        ):
            del self._active_downloads[model_id]
        return True

    async def delete_model_serialized(self, model_id: str) -> bool:
        """Delete one alias while excluding downloads and peer imports.

        Args:
            model_id: Store alias selected for operator deletion.

        Returns:
            ``True`` when the alias existed and was deleted.

        Side effects:
            Writes the durable reconciliation tombstone and removes canonical
            bytes while holding the same publication lock as model transfers.
        """

        async with self._download_transfer_lock:
            return await asyncio.to_thread(self.delete_model, model_id)

    def register_model(
        self,
        model_id: str,
        model_path: Path,
        files: list[str],
        total_bytes: int,
        repo_has_projector: bool | None = None,
        source_revision: str | None = None,
        source_repository: str | None = None,
        installed_card: InstalledCardRecord | None = None,
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
            repo_has_projector: Whether the source repository contains a GGUF
                vision projector, or ``None`` when not probed.
            source_revision: Full immutable Hugging Face commit represented by
                this entry, or ``None`` for mutable ``main``. Registration
                writes the matching filesystem marker before publishing the
                registry entry so revision-aware runners resolve the same path.
            source_repository: Upstream Hugging Face repository that supplied
                the bytes, or ``None`` when it is identical to ``model_id``.
            installed_card: Full local card and byte manifest retained beside
                the artifact for registry-independent restart.
        """
        from skulk.download.download_utils import write_source_revision_marker

        resolved_root = self._store_path.resolve()
        resolved_model_path = model_path.resolve()
        if (
            resolved_model_path == resolved_root
            or not resolved_model_path.is_relative_to(resolved_root)
        ):
            raise ValueError(f"Model path must be contained by the store: {model_path}")
        relative_path = str(resolved_model_path.relative_to(resolved_root))
        effective_files = list(files)
        if installed_card is not None:
            write_installed_card(resolved_model_path, installed_card)
            sidecar_name = INSTALLED_CARD_RELATIVE_PATH.as_posix()
            if sidecar_name not in effective_files:
                effective_files.append(sidecar_name)
        entry = StoreModelEntry(
            model_id=model_id,
            store_path=relative_path,
            files=effective_files,
            downloaded_at=datetime.now(tz=timezone.utc).isoformat(),
            total_bytes=total_bytes,
            repo_has_projector=repo_has_projector,
            source_revision=source_revision,
            source_repository=source_repository,
            installed_card=installed_card,
        )
        write_source_revision_marker(resolved_model_path, source_revision)
        self._write_registry_entry(entry)
        logger.info(
            f"ModelStore: registered {model_id} at {relative_path} "
            f"({total_bytes:,} bytes, {len(effective_files)} files)"
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
            if "schema_version" in data or "entries" in data:
                return StoreRegistryIndex.model_validate(data, strict=False).entries
            # V0 was an unversioned mapping. It remains readable and migrates
            # atomically the next time any entry changes.
            return {
                key: StoreModelEntry.model_validate(value, strict=False)
                for key, value in data.items()
                if isinstance(key, str)
            }
        except Exception as exc:
            logger.warning(f"ModelStore: failed to read registry: {exc}")
            return {}

    def _write_registry_entry(self, entry: StoreModelEntry) -> None:
        """Atomically upsert *entry* in ``registry.json``."""
        self._store_path.mkdir(parents=True, exist_ok=True)
        registry = self._read_registry()
        registry[entry.model_id] = entry
        self._write_registry(registry)

    def _write_registry(self, registry: dict[str, StoreModelEntry]) -> None:
        """Atomically replace the versioned rebuildable registry index."""

        self._store_path.mkdir(parents=True, exist_ok=True)
        if (
            self._registry_path.is_file()
            and not self._legacy_registry_backup_path.exists()
        ):
            try:
                existing_bytes = self._registry_path.read_bytes()
                existing_payload: object = json.loads(existing_bytes)
                if isinstance(existing_payload, dict) and not (
                    "schema_version" in existing_payload
                    or "entries" in existing_payload
                ):
                    backup_temporary = self._legacy_registry_backup_path.with_name(
                        f".{self._legacy_registry_backup_path.name}.tmp"
                    )
                    backup_temporary.write_bytes(existing_bytes)
                    backup_temporary.replace(self._legacy_registry_backup_path)
            except (OSError, ValueError):
                # A corrupt index is recovered from sidecars; there is no valid
                # legacy index to preserve as the migration backup.
                pass
        temporary = self._registry_path.with_name(".registry.json.tmp")
        temporary.write_text(
            StoreRegistryIndex(entries=registry).model_dump_json(indent=2)
        )
        temporary.replace(self._registry_path)

    def _read_reconciliation_tombstone_index(
        self,
    ) -> StoreReconciliationTombstones:
        """Read durable automatic-import suppression metadata."""

        if not self._reconciliation_tombstones_path.exists():
            return StoreReconciliationTombstones(deleted_aliases={})
        return StoreReconciliationTombstones.model_validate_json(
            self._reconciliation_tombstones_path.read_bytes(),
            strict=False,
        )

    def _write_reconciliation_tombstone_index(
        self,
        index: StoreReconciliationTombstones,
    ) -> None:
        """Atomically replace durable automatic-import suppression metadata."""

        self._reconciliation_tombstones_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        temporary = self._reconciliation_tombstones_path.with_name(
            f".{self._reconciliation_tombstones_path.name}.tmp"
        )
        temporary.write_text(index.model_dump_json(indent=2))
        temporary.replace(self._reconciliation_tombstones_path)

    def _record_reconciliation_tombstone(self, model_id: str) -> None:
        """Persist one operator-deleted alias before its bytes are removed."""

        index = self._read_reconciliation_tombstone_index()
        deleted_aliases = dict(index.deleted_aliases)
        deleted_aliases[model_id] = datetime.now(tz=timezone.utc).isoformat()
        self._write_reconciliation_tombstone_index(
            StoreReconciliationTombstones(deleted_aliases=deleted_aliases)
        )

    def _clear_reconciliation_tombstone(self, model_id: str) -> None:
        """Allow a successful explicit download to reintroduce one alias."""

        index = self._read_reconciliation_tombstone_index()
        if model_id not in index.deleted_aliases:
            return
        deleted_aliases = dict(index.deleted_aliases)
        del deleted_aliases[model_id]
        self._write_reconciliation_tombstone_index(
            StoreReconciliationTombstones(deleted_aliases=deleted_aliases)
        )

    def refresh_recovered_generations(
        self,
        current_registry_cards: Iterable[ModelCard],
    ) -> None:
        """Re-select recovered generations after signed catalog discovery.

        Store construction intentionally recovers installed sidecars before any
        registry access. Once TUF discovery completes, this second pass makes
        current signed identity outrank a preserved prior index selection.
        """

        current_registry_ids = {
            str(card.model_id): card.registry_card_id
            for card in current_registry_cards
            if card.registry_card_id is not None
        }
        self._rebuild_registry_from_sidecars(current_registry_ids)

    def _rebuild_registry_from_sidecars(
        self,
        current_registry_ids: dict[str, str] | None = None,
    ) -> None:
        """Recover missing index entries from artifact-owned sidecars.

        The index is intentionally rebuildable: a torn or removed
        ``registry.json`` must not make complete air-gapped artifacts disappear.
        Existing entries win unless a valid sidecar provides the same model id,
        in which case the sidecar refreshes the installed-card payload.
        """

        if not self._store_path.is_dir():
            return
        registry = self._read_registry()
        changed = False
        for model_id, entry in tuple(registry.items()):
            retained = entry.installed_card
            if retained is None:
                continue
            model_directory = _resolve_store_child_path(
                self._store_path, entry.store_path
            )
            try:
                adjacent = (
                    read_installed_card(model_directory)
                    if model_directory is not None
                    else None
                )
            except (OSError, ValueError):
                adjacent = None
            if (
                model_directory is None
                or adjacent != retained
                or not verify_installed_card(model_directory, retained)
            ):
                logger.warning(
                    f"ModelStore: removing incomplete installed index entry "
                    f"for {model_id}; reconciliation may recover a healthy replica"
                )
                del registry[model_id]
                changed = True
        recovered_by_model: dict[str, list[StoreModelEntry]] = {}
        for model_directory in self._store_path.iterdir():
            if not model_directory.is_dir() or model_directory.name.startswith("."):
                continue
            try:
                record = read_installed_card(model_directory)
            except (OSError, ValueError) as error:
                logger.warning(
                    f"ModelStore: ignoring invalid installed-card sidecar at "
                    f"{model_directory}: {error}"
                )
                continue
            if record is None:
                continue
            if not verify_installed_card(model_directory, record):
                logger.warning(
                    f"ModelStore: ignoring incomplete installed artifact at "
                    f"{model_directory} while rebuilding the registry"
                )
                continue
            files = [
                *(entry.path for entry in record.files),
                INSTALLED_CARD_RELATIVE_PATH.as_posix(),
            ]
            relative_path = str(
                model_directory.resolve().relative_to(self._store_path.resolve())
            )
            previous = registry.get(record.artifact_model_id)
            recovered = StoreModelEntry(
                model_id=record.artifact_model_id,
                store_path=relative_path,
                files=files,
                downloaded_at=(
                    previous.downloaded_at
                    if previous is not None
                    else record.captured_at.isoformat()
                ),
                total_bytes=sum(entry.size_bytes for entry in record.files),
                source_revision=record.artifact_revision,
                source_repository=record.artifact_repository,
                repo_has_projector=(
                    previous.repo_has_projector if previous is not None else None
                ),
                installed_card=record,
            )
            recovered_by_model.setdefault(record.artifact_model_id, []).append(
                recovered
            )
        for model_id, candidates in recovered_by_model.items():
            previous = registry.get(model_id)
            current_card_id = (
                current_registry_ids.get(model_id)
                if current_registry_ids is not None
                else None
            )
            previous_identity = (
                previous.installed_card.installed_identity
                if previous is not None and previous.installed_card is not None
                else None
            )
            previous_path = previous.store_path if previous is not None else None

            def recovery_rank(
                candidate: StoreModelEntry,
                previous_identity: str | None = previous_identity,
                previous_path: str | None = previous_path,
                current_card_id: str | None = current_card_id,
            ) -> tuple[bool, bool, bool, float, str]:
                installed = candidate.installed_card
                assert installed is not None
                matches_previous = (
                    installed.installed_identity == previous_identity
                    if previous_identity is not None
                    else candidate.store_path == previous_path
                )
                matches_current = (
                    current_card_id is not None
                    and installed.model_card.registry_card_id == current_card_id
                )
                return (
                    not matches_current if current_card_id is not None else False,
                    not matches_previous,
                    installed.verification != "registry_verified",
                    -installed.captured_at.timestamp(),
                    candidate.store_path,
                )

            recovered = min(candidates, key=recovery_rank)
            if recovered != previous:
                registry[model_id] = recovered
                changed = True
        if not changed:
            return
        self._write_registry(registry)

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
                ModelId(model_id), entry.source_revision or "main", recursive=True
            )
        except Exception as exc:
            logger.debug(
                f"ModelStore: could not check projector completeness for "
                f"{model_id} ({exc}); leaving the existing entry as-is"
            )
            return False
        return has_gguf_projector(f.path for f in repo_files)

    def entry_missing_files(self, model_id: str, required_files: list[str]) -> bool:
        """True when an in-store entry omits any of ``required_files``.

        Recovers a stale base-only store entry that predates a card declaring a
        same-repo companion (e.g. a served-engine draft GGUF bundled with the
        base): the entry lists only the base quant, so staging it omits the
        companion and the served runner's ``--model-draft`` resolution fails.
        Forcing a re-download fetches just the missing file (size-matched skips
        reuse present weights) and re-registers a complete entry. Returns
        ``False`` when the model is absent (the normal download path handles it)
        or no files are required.
        """
        if not required_files:
            return False
        entry = self._read_registry().get(model_id)
        if entry is None:
            return False
        registered = set(entry.files)
        return any(name not in registered for name in required_files)

    @staticmethod
    def _same_requested_card(
        existing: ModelCard | None,
        requested: ModelCard | None,
    ) -> bool:
        """Return whether concurrent requests carry the same full card truth."""

        if existing is None or requested is None:
            return existing is requested
        if (
            existing.registry_card_id is not None
            or requested.registry_card_id is not None
        ):
            return existing.registry_card_id == requested.registry_card_id
        return existing == requested

    def _ensure_installed_card(
        self,
        model_id: str,
        model_card: ModelCard,
        artifact_role: InstalledArtifactRole,
        owner_model_id: str | None,
        owner_card_id: str | None,
        artifact_file: str | None,
    ) -> None:
        """Materialize or refresh a sidecar without transferring model bytes."""

        entry = self.get_entry(model_id)
        if entry is None:
            raise ValueError(f"cannot attach a card to missing store model {model_id}")
        model_path = _resolve_store_child_path(self._store_path, entry.store_path)
        if model_path is None or not model_path.is_dir():
            raise ValueError(
                f"cannot attach a card to unavailable store model {model_id}"
            )
        record = build_installed_card_record(
            model_path,
            model_card,
            artifact_role=artifact_role,
            artifact_model_id=model_id,
            owner_model_id=owner_model_id,
            owner_card_id=owner_card_id,
            artifact_repository=entry.source_repository or entry.model_id,
            artifact_revision=entry.source_revision,
            artifact_file=artifact_file,
            file_manifest=(
                entry.installed_card.files
                if entry.installed_card is not None
                and entry.installed_card.artifact_repository
                == (entry.source_repository or entry.model_id)
                and entry.installed_card.artifact_revision == entry.source_revision
                and entry.installed_card.artifact_file == artifact_file
                and verify_installed_card(model_path, entry.installed_card)
                else None
            ),
        )
        refreshed = entry.model_copy(update={"installed_card": record})
        write_installed_card(model_path, record)
        self._write_registry_entry(refreshed)

    async def request_download(
        self,
        model_id: str,
        pinned_gguf: str | None = None,
        extra_pinned_gguf: list[str] | None = None,
        source_revision: str | None = None,
        source_repository: str | None = None,
        model_card: ModelCard | None = None,
        artifact_role: InstalledArtifactRole = "base",
        owner_model_id: str | None = None,
        owner_card_id: str | None = None,
    ) -> StoreDownloadStatus:
        """Request that the store download a model from HuggingFace.

        Deduplicates: if the model is already downloading, returns the
        existing status.  If already in the store, returns "complete", unless it
        is a vision GGUF whose stored entry is missing its ``mmproj`` projector
        (a stale pre-#346 entry) or its entry omits a requested same-repo
        companion GGUF, in which case it re-downloads to recover the missing
        files instead of staging an incomplete model.

        ``pinned_gguf`` is the requester's card-pinned GGUF file, forwarded so a
        GGUF repo fetches that quant's shard group rather than the default (#344).
        ``extra_pinned_gguf`` names same-repo companion GGUFs (a served-engine
        draft bundled with the base) to co-fetch with the base quant.
        ``source_revision`` pins repository metadata and bytes to a full
        immutable Hugging Face commit. A different registered revision is
        replaced only after the requested revision has downloaded and
        registered successfully.
        ``source_repository`` names the upstream byte source when ``model_id``
        is a distinct store and runtime alias.
        ``model_card`` is the independently verified complete card retained
        beside the finished artifact.
        Companion downloads retain that owning card together with their role
        and owning identities instead of synthesizing a bare model card.
        """
        requested_repository = source_repository or model_id
        requested_companions = tuple(sorted(extra_pinned_gguf or []))
        # Checked outside the lock: it may do a (cached) repo file-list fetch, and
        # holding the download lock across network I/O would serialize unrelated
        # requests.
        missing_projector = await self.vision_entry_missing_projector(model_id)
        missing_pinned_gguf = self.entry_missing_files(
            model_id,
            [pinned_gguf] if pinned_gguf is not None else [],
        )
        missing_companion = self.entry_missing_files(model_id, extra_pinned_gguf or [])
        async with self._download_lock:
            existing = self._active_downloads.get(model_id)
            if existing is not None:
                existing_repository = existing.source_repository or model_id
                if existing.status in ("pending", "downloading") and (
                    existing.source_revision != source_revision
                    or existing_repository != requested_repository
                    or existing.pinned_gguf != pinned_gguf
                    or existing.extra_pinned_gguf != requested_companions
                    or not self._same_requested_card(existing.model_card, model_card)
                    or existing.artifact_role != artifact_role
                    or existing.owner_model_id != owner_model_id
                    or existing.owner_card_id != owner_card_id
                ):
                    raise ValueError(
                        f"{model_id} is already downloading a different artifact "
                        f"selection from {existing_repository}@"
                        f"{existing.source_revision or 'main'}"
                    )
                # A failed entry retries. A cached-complete entry is stale when:
                #  - it is missing a newly-requested companion (or projector) --
                #    a prior base-only download in this process left a "complete"
                #    status, so returning it would skip the recovery re-download
                #    below (the registry-based missing checks would never run); or
                #  - the model is no longer actually in the store -- a store-delete
                #    (``delete_model``) or out-of-band file removal drops the
                #    registry entry and on-disk files but cannot reach this
                #    in-memory status, so trusting it would short-circuit the
                #    re-download and the model would never come back (staging then
                #    fails "not found in store"). ``is_in_store`` re-checks the
                #    registry + on-disk dir, so a cached "complete" that no longer
                #    reflects reality is dropped and re-fetched.
                # Drop a stale entry and fall through. An in-progress
                # (downloading/pending) entry is returned as-is to dedup
                # concurrent requests.
                stale_complete = existing.status == "complete" and (
                    missing_projector
                    or missing_pinned_gguf
                    or missing_companion
                    or existing.source_revision != source_revision
                    or existing_repository != requested_repository
                    or not self._same_requested_card(existing.model_card, model_card)
                    or existing.artifact_role != artifact_role
                    or existing.owner_model_id != owner_model_id
                    or existing.owner_card_id != owner_card_id
                    or not self.is_in_store(model_id)
                )
                if existing.status in ("failed", "cancelled") or stale_complete:
                    del self._active_downloads[model_id]
                else:
                    return existing
            artifact_mismatch = self.is_in_store(
                model_id
            ) and not self.entry_matches_artifact(
                model_id,
                source_revision,
                source_repository,
            )
            if (
                self.is_in_store(model_id)
                and not missing_projector
                and not missing_pinned_gguf
                and not missing_companion
                and not artifact_mismatch
            ):
                if model_card is not None:
                    await asyncio.to_thread(
                        self._ensure_installed_card,
                        model_id,
                        model_card,
                        artifact_role,
                        owner_model_id,
                        owner_card_id,
                        pinned_gguf,
                    )
                return StoreDownloadStatus(
                    model_id=model_id,
                    source_revision=source_revision,
                    source_repository=requested_repository,
                    status="complete",
                    progress=1.0,
                    model_card=model_card,
                    artifact_role=artifact_role,
                    owner_model_id=owner_model_id,
                    owner_card_id=owner_card_id,
                )
            if missing_projector:
                logger.warning(
                    f"ModelStore: {model_id} is in the store but its entry is "
                    "missing the mmproj projector for a vision GGUF; "
                    "re-downloading to recover it (existing weights are reused)."
                )
            if missing_pinned_gguf:
                logger.warning(
                    f"ModelStore: {model_id} is in the store but its entry is "
                    f"missing requested GGUF {pinned_gguf!r}; re-downloading to "
                    "recover it (existing weights are reused)."
                )
            if missing_companion:
                logger.warning(
                    f"ModelStore: {model_id} is in the store but its entry is "
                    f"missing a requested companion GGUF ({extra_pinned_gguf}); "
                    "re-downloading to recover it (existing weights are reused)."
                )
            if artifact_mismatch:
                current = self.get_entry(model_id)
                current_repository = (
                    current.source_repository or current.model_id
                    if current is not None
                    else "unknown"
                )
                current_revision = (
                    current.source_revision or "main"
                    if current is not None
                    else "unknown"
                )
                logger.warning(
                    f"ModelStore: {model_id} requests "
                    f"{requested_repository}@{source_revision or 'main'}, but the "
                    f"canonical entry identifies {current_repository}@"
                    f"{current_revision}; downloading a qualified replacement."
                )
            status = StoreDownloadStatus(
                model_id=model_id,
                source_revision=source_revision,
                source_repository=requested_repository,
                pinned_gguf=pinned_gguf,
                extra_pinned_gguf=requested_companions,
                model_card=model_card,
                artifact_role=artifact_role,
                owner_model_id=owner_model_id,
                owner_card_id=owner_card_id,
                status="pending",
            )
            self._active_downloads[model_id] = status
            task = asyncio.create_task(
                self._do_download(
                    model_id,
                    pinned_gguf,
                    extra_pinned_gguf,
                    source_revision,
                    source_repository,
                    model_card,
                    artifact_role,
                    owner_model_id,
                    owner_card_id,
                )
            )
            self._download_tasks.add(task)
            self._download_tasks_by_model[model_id] = task
            task.add_done_callback(self._download_tasks.discard)
        return status

    async def cancel_download(self, model_id: str) -> StoreDownloadStatus | None:
        """Cancel one pending or active canonical-store download.

        Partial files remain available for the resumable downloader. Repeating
        cancellation for an already-cancelled request is idempotent. ``None``
        means the model has no cancellable store transfer.

        Args:
            model_id: HuggingFace-style model identifier.

        Returns:
            The cancelled status, or ``None`` when no transfer can be cancelled.
        """

        async with self._download_lock:
            status = self._active_downloads.get(model_id)
            if status is not None and status.status == "cancelled":
                return status
            if status is None or status.status not in ("pending", "downloading"):
                return None
            task = self._download_tasks_by_model.get(model_id)
            if task is None:
                return None
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        final_status = self.get_download_status(model_id)
        return (
            final_status
            if final_status is not None and final_status.status == "cancelled"
            else None
        )

    def get_download_status(self, model_id: str) -> StoreDownloadStatus | None:
        """Return the download status for *model_id*, or None."""
        if model_id in self._active_downloads:
            return self._active_downloads[model_id]
        entry = self.get_entry(model_id)
        if entry is not None:
            return StoreDownloadStatus(
                model_id=model_id,
                source_revision=entry.source_revision,
                source_repository=entry.source_repository or entry.model_id,
                status="complete",
                progress=1.0,
            )
        return None

    def list_active_downloads(self) -> list[StoreDownloadStatus]:
        """Return all in-progress or pending downloads."""
        return [
            s
            for s in self._active_downloads.values()
            if s.status in ("pending", "downloading")
        ]

    async def import_peer_artifact(
        self,
        record: InstalledCardRecord,
        *,
        source_base_url: str,
        capability_token: str,
        target_node_id: str,
    ) -> StoreModelEntry:
        """Pull, verify and atomically publish one peer-cached artifact.

        The transfer resumes per file through HTTP ranges and shares the same
        publication lock as normal Hugging Face downloads.  Existing source
        caches are read-only from this operation and are never evicted.
        """

        from urllib.parse import quote, urlsplit

        parsed = urlsplit(source_base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("source_base_url must be an HTTP(S) node API")
        async with self._download_transfer_lock:
            tombstones = read_reconciliation_tombstones(self._store_path)
            if record.artifact_model_id in tombstones or (
                record.owner_model_id is not None
                and record.owner_model_id in tombstones
            ):
                raise ValueError(
                    f"peer import is suppressed by an operator-deletion tombstone "
                    f"for {record.owner_model_id or record.artifact_model_id}"
                )
            existing = self.get_entry(record.artifact_model_id)
            existing_path = (
                _resolve_store_child_path(self._store_path, existing.store_path)
                if existing is not None
                else None
            )
            if (
                existing is not None
                and existing.installed_card is not None
                and existing.installed_card.installed_identity
                == record.installed_identity
                and existing.installed_card.manifest_sha256 == record.manifest_sha256
                and existing_path is not None
                and verify_installed_card(existing_path, existing.installed_card)
            ):
                return existing
            import_root = self._store_path / ".imports"
            await aios.makedirs(str(import_root), exist_ok=True)
            temporary = import_root / f"{record.installed_identity}.partial"
            await aios.makedirs(str(temporary), exist_ok=True)
            verified_destinations: set[Path] = set()
            for entry in record.files:
                destination = temporary / entry.path
                if not (
                    destination.is_file()
                    and destination.stat().st_size == entry.size_bytes
                ):
                    continue
                digest = await asyncio.to_thread(_sha256_path, destination)
                if digest == entry.sha256:
                    verified_destinations.add(destination)
                    continue
                await aios.remove(str(destination))
            remaining = 0
            for entry in record.files:
                destination = temporary / entry.path
                if destination in verified_destinations:
                    continue
                partial = destination.with_name(f"{destination.name}.partial")
                partial_bytes = partial.stat().st_size if partial.is_file() else 0
                if partial_bytes > entry.size_bytes:
                    partial.unlink()
                    partial_bytes = 0
                elif partial_bytes == entry.size_bytes:
                    digest = await asyncio.to_thread(_sha256_path, partial)
                    if digest == entry.sha256:
                        await aios.makedirs(str(destination.parent), exist_ok=True)
                        partial.replace(destination)
                        verified_destinations.add(destination)
                        continue
                    partial.unlink()
                    partial_bytes = 0
                remaining += entry.size_bytes - partial_bytes
            if remaining > 0:
                free_bytes = await asyncio.to_thread(
                    lambda: shutil.disk_usage(temporary).free
                )
                required = remaining + MINIMUM_STAGING_FREE_DISK_BYTES
                if free_bytes < required:
                    raise ModelStoreCapacityError(
                        "Insufficient canonical model-store capacity for peer import"
                    )
            headers = {"X-Skulk-Store-Node": target_node_id}
            timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for manifest_entry in record.files:
                    destination = temporary / manifest_entry.path
                    await aios.makedirs(str(destination.parent), exist_ok=True)
                    partial = destination.with_name(f"{destination.name}.partial")
                    offset = partial.stat().st_size if partial.is_file() else 0
                    if destination in verified_destinations:
                        continue
                    if offset > manifest_entry.size_bytes:
                        partial.unlink()
                        offset = 0
                    elif offset == manifest_entry.size_bytes:
                        digest = await asyncio.to_thread(_sha256_path, partial)
                        if digest == manifest_entry.sha256:
                            partial.replace(destination)
                            continue
                        partial.unlink()
                        offset = 0
                    request_headers = dict(headers)
                    if offset:
                        request_headers["Range"] = f"bytes={offset}-"
                    relative = quote(manifest_entry.path, safe="/")
                    url = (
                        f"{source_base_url.rstrip('/')}/store/internal/exports/"
                        f"{capability_token}/{relative}"
                    )
                    async with session.get(url, headers=request_headers) as response:
                        if response.status not in ({206} if offset else {200, 206}):
                            raise RuntimeError(
                                f"peer export returned HTTP {response.status}"
                            )
                        mode = "ab" if offset else "wb"
                        with partial.open(mode) as output:
                            async for chunk in response.content.iter_chunked(
                                8 * 1024 * 1024
                            ):
                                output.write(chunk)
                    if partial.stat().st_size != manifest_entry.size_bytes:
                        raise RuntimeError(
                            f"peer export size mismatch for {manifest_entry.path}"
                        )
                    partial.replace(destination)
            imported_files = build_file_manifest(temporary)
            if manifest_sha256(imported_files) != record.manifest_sha256:
                raise RuntimeError("peer export manifest digest mismatch")
            final_name = (
                record.artifact_model_id.replace("/", "--")
                + "--installed-"
                + record.installed_identity[-16:]
            )
            final_path = self._store_path / final_name
            if final_path.exists():
                await asyncio.to_thread(shutil.rmtree, final_path)
            temporary.replace(final_path)
            write_installed_card(final_path, record)
            files = [entry.path for entry in record.files]
            self.register_model(
                record.artifact_model_id,
                final_path,
                files,
                sum(entry.size_bytes for entry in record.files),
                source_revision=record.artifact_revision,
                source_repository=record.artifact_repository,
                installed_card=record,
            )
            if existing is not None and existing.store_path != final_name:
                previous = _resolve_store_child_path(
                    self._store_path, existing.store_path
                )
                if previous is not None:
                    await asyncio.to_thread(shutil.rmtree, previous, True)
            imported = self.get_entry(record.artifact_model_id)
            if imported is None:
                raise RuntimeError("peer import committed without a registry entry")
            return imported

    async def _do_download(
        self,
        model_id: str,
        pinned_gguf: str | None = None,
        extra_pinned_gguf: list[str] | None = None,
        source_revision: str | None = None,
        source_repository: str | None = None,
        model_card: ModelCard | None = None,
        artifact_role: InstalledArtifactRole = "base",
        owner_model_id: str | None = None,
        owner_card_id: str | None = None,
    ) -> None:
        """Download a model from HuggingFace into the store and register it.

        ``pinned_gguf`` (the requester's ``ModelCard.gguf_file``) selects which
        GGUF quant's shard group to fetch, honoring a custom pin (#344).
        ``extra_pinned_gguf`` names same-repo companion GGUFs (a served-engine
        draft bundled with the base) co-fetched in the same store download.
        ``source_repository`` is the upstream repository when ``model_id`` is
        a distinct immutable-artifact alias used for store identity.
        """
        from skulk.download.download_utils import (
            download_file_with_retry,
            fetch_file_list_with_cache,
            write_source_revision_marker,
        )
        from skulk.shared.models.model_cards import ModelId

        status = self._active_downloads[model_id]
        revision = source_revision or "main"
        artifact_repository = source_repository or model_id
        sanitized = model_id.replace("/", "--")
        if source_revision is not None:
            sanitized = f"{sanitized}--revision-{source_revision}"
        if source_repository is not None:
            repository_digest = hashlib.sha256(source_repository.encode()).hexdigest()[
                :12
            ]
            sanitized = f"{sanitized}--source-{repository_digest}"
        target_dir = self._store_path / sanitized
        previous_entry = self.get_entry(model_id)
        transfer_lock_acquired = False
        logger.info(
            f"ModelStore: downloading {model_id} from HuggingFace repository "
            f"{artifact_repository} to {target_dir}"
        )

        try:
            if source_revision is None and (
                (target_dir / _SOURCE_REVISION_MARKER).exists()
                or (target_dir / _SOURCE_REVISION_STAGING_MARKER).exists()
            ):
                # Older shared-root staging placed pinned bytes in mutable-main's
                # normalized directory. Never let size-based resume reuse those
                # bytes after an operator removes the pin.
                logger.warning(
                    f"ModelStore: clearing revision-qualified staging residue "
                    f"before downloading mutable main for {model_id}"
                )
                await asyncio.to_thread(shutil.rmtree, target_dir)
            await aios.makedirs(str(target_dir), exist_ok=True)

            repo_file_list = await fetch_file_list_with_cache(
                ModelId(artifact_repository), revision, recursive=True
            )
            # A vision GGUF (LLaVA/Qwen-VL/Gemma-VLM style) ships its multimodal
            # projector as a separate ``*mmproj*.gguf`` alongside the LM weights;
            # the llama.cpp runner cannot load the model without it. Detect that
            # from the full repo listing so we can verify the projector actually
            # lands before registering (see the post-download guard below): a
            # store entry that omits the projector stages an unloadable vision
            # model, which surfaces only as a runner crash at load time (#346).
            repo_ships_projector = has_gguf_projector(f.path for f in repo_file_list)
            # For a GGUF repo, fetch only the preferred quant's shard group plus
            # config.json (dropping other quants, original/*, metal/*, etc.),
            # mirroring the direct-HF selective download (#339). The projector
            # glob is retained by ``gguf_allow_patterns``, so a vision GGUF keeps
            # its ``*mmproj*.gguf``.
            selected_files = select_store_gguf_download_files(
                repo_file_list, pinned_gguf, extra_pinned_gguf
            )
            if len(selected_files) != len(repo_file_list):
                logger.info(
                    f"ModelStore: {model_id} is a GGUF repo; downloading "
                    f"{len(selected_files)}/{len(repo_file_list)} files (selected "
                    "quant's shard group + config.json only)"
                )
            file_list = selected_files
            total_bytes = sum(f.size or 0 for f in file_list)
            # Store downloads share one canonical filesystem. Serialize exact
            # admission with transfer so concurrent requests cannot each spend
            # the same free bytes. The store is authoritative and therefore
            # never evicted automatically; an unsafe transfer fails closed.
            await self._download_transfer_lock.acquire()
            transfer_lock_acquired = True
            # Keep the public status pending while this request is queued
            # behind another canonical transfer. Clients deliberately exclude
            # pending time from their no-progress stall budget; switching only
            # after lock acquisition makes that status truthful.
            status.status = "downloading"
            additional_bytes = await asyncio.to_thread(
                _remaining_store_download_bytes,
                target_dir,
                file_list,
            )
            if additional_bytes > 0:
                free_bytes = await asyncio.to_thread(
                    lambda: shutil.disk_usage(target_dir).free
                )
                required_free_bytes = additional_bytes + MINIMUM_STAGING_FREE_DISK_BYTES
                if free_bytes < required_free_bytes:
                    raise ModelStoreCapacityError(
                        f"Insufficient canonical model-store disk capacity for "
                        f"{model_id}: need {required_free_bytes / 2**30:.1f} GiB "
                        "free for the remaining transfer and operating-system "
                        f"reserve, but only {free_bytes / 2**30:.1f} GiB is "
                        "available. Free disk space or move the model store."
                    )
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
                    ModelId(artifact_repository),
                    revision,
                    f.path,
                    target_dir,
                    make_progress_cb(file_size),
                )
                downloaded_bytes += file_size
                status.progress = downloaded_bytes / max(total_bytes, 1)

            await asyncio.to_thread(
                write_source_revision_marker, target_dir, source_revision
            )

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
            # Same-repo companion completeness guard: a requested companion GGUF
            # (served-engine draft) that the repo actually ships MUST land, or
            # staging produces a model the served runner can't launch (its
            # --model-draft path 404s). Refuse to register an incomplete entry so
            # the failure is a loud, fixable download error here rather than a
            # runner spawn failure on a remote node. A companion absent from the
            # repo is a card-config error surfaced at launch, not a download bug,
            # so it is not guarded here.
            repo_paths = {f.path for f in repo_file_list}
            missing_companions = [
                name
                for name in (extra_pinned_gguf or [])
                if name in repo_paths and name not in files
            ]
            if missing_companions:
                raise RuntimeError(
                    f"{model_id}: requested companion GGUF(s) {missing_companions} "
                    "are in the repo but did not land in the store download; "
                    "refusing to register an incomplete entry. Re-run the download."
                )
            total = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
            installed_card = (
                build_installed_card_record(
                    target_dir,
                    model_card,
                    artifact_role=artifact_role,
                    artifact_model_id=model_id,
                    owner_model_id=owner_model_id,
                    owner_card_id=owner_card_id,
                    artifact_repository=artifact_repository,
                    artifact_revision=source_revision,
                    artifact_file=pinned_gguf,
                )
                if model_card is not None
                else None
            )
            self.register_model(
                model_id,
                target_dir,
                files,
                total,
                repo_has_projector=repo_ships_projector,
                source_revision=source_revision,
                source_repository=artifact_repository,
                installed_card=installed_card,
            )
            # A successful explicit upstream download is the operator's durable
            # opt-in to reintroduce an alias that was previously deleted. Clear
            # suppression only after the replacement is complete and indexed;
            # failed transfers must leave stale peer replicas ineligible.
            self._clear_reconciliation_tombstone(model_id)
            if previous_entry is not None and previous_entry.store_path != sanitized:
                previous_path = _resolve_store_child_path(
                    self._store_path, previous_entry.store_path
                )
                if previous_path is None:
                    logger.warning(
                        f"ModelStore: refusing to delete unsafe previous registry "
                        f"path for {model_id}: {previous_entry.store_path!r}"
                    )
                else:
                    await asyncio.to_thread(shutil.rmtree, previous_path, True)

            status.status = "complete"
            status.progress = 1.0
            logger.info(
                f"ModelStore: downloaded {model_id} from HuggingFace ({total:,} bytes)"
            )

        except asyncio.CancelledError:
            status.status = "cancelled"
            status.error = None
            logger.info(f"ModelStore: cancelled download of {model_id}")
            raise
        except Exception as exc:
            status.status = "failed"
            # Always record the exception TYPE: several failure modes (notably a
            # TimeoutError) stringify to "", which produced an empty, useless
            # error field that hid the real cause of a stalled large download.
            detail = str(exc)
            status.error = (
                f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            )
            logger.exception(
                f"ModelStore: download of {model_id} failed ({type(exc).__name__})"
            )
        finally:
            if transfer_lock_acquired:
                self._download_transfer_lock.release()
            current_task = asyncio.current_task()
            if self._download_tasks_by_model.get(model_id) is current_task:
                del self._download_tasks_by_model[model_id]
