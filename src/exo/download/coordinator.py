import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from anyio import current_time
from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    map_repo_download_progress_to_download_progress_data,
    resolve_model_in_path,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.constants import EXO_MODELS_DIR, EXO_MODELS_PATH
from exo.shared.models.model_cards import ModelId, get_model_cards
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    PurgeStagingCache,
    StartDownload,
    SyncConfig,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    NodeDownloadProgress,
)
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.task_group import TaskGroup


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    event_sender: Sender[Event]
    offline: bool = False
    staging_cache_path: Path | None = None

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, anyio.CancelScope] = field(default_factory=dict)

    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shard_downloader.on_progress(self._download_progress_callback)

    def _model_dir(self, model_id: ModelId) -> str:
        return str(EXO_MODELS_DIR / model_id.normalize())

    async def _download_progress_callback(
        self, callback_shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        model_id = callback_shard.model_card.model_id
        throttle_interval_secs = 1.0

        if progress.status == "complete":
            completed = DownloadCompleted(
                shard_metadata=callback_shard,
                node_id=self.node_id,
                total=progress.total,
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            self._last_progress_time.pop(model_id, None)
        elif (
            progress.status == "in_progress"
            and current_time() - self._last_progress_time.get(model_id, 0.0)
            > throttle_interval_secs
        ):
            ongoing = DownloadOngoing(
                node_id=self.node_id,
                shard_metadata=callback_shard,
                download_progress=map_repo_download_progress_to_download_progress_data(
                    progress
                ),
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = ongoing
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=ongoing)
            )
            self._last_progress_time[model_id] = current_time()

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._emit_existing_download_progress)
        finally:
            self._stopped.set()

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _command_processor(self) -> None:
        with self.download_command_receiver as commands:
            async for cmd in commands:
                # Cluster-wide commands — every node handles these
                match cmd.command:
                    case SyncConfig(config_yaml=config_yaml):
                        await self._sync_config(config_yaml)
                        continue
                    case PurgeStagingCache(model_id=purge_model_id):
                        await self._purge_staging_cache(purge_model_id)
                        continue
                    case _:
                        pass  # Targeted commands handled below

                # Only process targeted commands for this node
                if cmd.command.target_node_id != self.node_id:
                    logger.debug(
                        f"DownloadCoordinator: ignoring command for {cmd.command.target_node_id} (we are {self.node_id})"
                    )
                    continue

                match cmd.command:
                    case StartDownload(shard_metadata=shard):
                        logger.info(
                            f"DownloadCoordinator: received StartDownload for {shard.model_card.model_id}"
                        )
                        await self._start_download(shard)
                    case DeleteDownload(model_id=model_id):
                        await self._delete_download(model_id)
                    case CancelDownload(model_id=model_id):
                        await self._cancel_download(model_id)

    async def _cancel_download(self, model_id: ModelId) -> None:
        if model_id in self.active_downloads and model_id in self.download_status:
            logger.info(f"Cancelling download for {model_id}")
            self.active_downloads[model_id].cancel()
            current_status = self.download_status[model_id]
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = pending
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )

    async def _sync_config(self, config_yaml: str) -> None:
        """Write received config YAML to the local exo.yaml file and
        apply runtime-effective settings (e.g., KV cache backend)."""
        config_path = Path("exo.yaml")
        try:
            config_path.write_text(config_yaml)
            logger.info(
                f"DownloadCoordinator: synced exo.yaml from cluster ({len(config_yaml)} bytes)"
            )
            # Apply inference config to env var so next runner spawn picks it up
            import yaml
            raw = yaml.safe_load(config_yaml)
            if raw and isinstance(raw, dict):
                inference = raw.get("inference")
                if isinstance(inference, dict) and "kv_cache_backend" in inference:
                    # Don't overwrite if user provided the env var at launch
                    if not os.environ.get("_EXO_KV_BACKEND_USER_SET"):
                        os.environ["EXO_KV_CACHE_BACKEND"] = str(inference["kv_cache_backend"])
                        logger.info(
                            f"DownloadCoordinator: updated EXO_KV_CACHE_BACKEND={inference['kv_cache_backend']}"
                        )
                    else:
                        logger.info(
                            f"DownloadCoordinator: skipping KV backend update (user env var override active)"
                        )
        except Exception as exc:
            logger.warning(f"DownloadCoordinator: failed to sync exo.yaml: {exc}")

    async def _purge_dir(self, path: Path, label: str) -> int:
        """Remove all model subdirectories from *path*. Returns count purged."""
        if not path.exists():
            return 0
        purged = 0
        for entry in path.iterdir():
            if entry.is_dir():
                logger.info(f"PurgeStagingCache: removing {entry} ({label})")
                await asyncio.to_thread(shutil.rmtree, entry, True)
                purged += 1
        return purged

    async def _purge_staging_cache(self, model_id: ModelId | None) -> None:
        """Remove staged and downloaded model files from the local node."""
        # Collect all directories to purge: staging cache + standard models dir
        purge_dirs: list[tuple[Path, str]] = []
        if self.staging_cache_path is not None:
            purge_dirs.append((self.staging_cache_path.expanduser(), "staging"))
        purge_dirs.append((EXO_MODELS_DIR, "models"))

        if model_id is not None:
            # Purge a specific model
            if model_id in self.active_downloads:
                self.active_downloads[model_id].cancel()
            sanitized = str(model_id).replace("/", "--")
            found = False
            for dir_path, label in purge_dirs:
                model_dir = dir_path / sanitized
                if model_dir.exists():
                    logger.info(f"PurgeStagingCache: removing {model_dir} ({label})")
                    await asyncio.to_thread(shutil.rmtree, model_dir, True)
                    found = True
            if not found:
                # Also try the normalized form used by EXO_MODELS_DIR
                norm_dir = EXO_MODELS_DIR / model_id.normalize()
                if norm_dir.exists():
                    logger.info(f"PurgeStagingCache: removing {norm_dir} (models)")
                    await asyncio.to_thread(shutil.rmtree, norm_dir, True)
                    found = True
            if found and model_id in self.download_status:
                current = self.download_status[model_id]
                pending = DownloadPending(
                    shard_metadata=current.shard_metadata,
                    node_id=self.node_id,
                    model_directory=self._model_dir(model_id),
                )
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=pending)
                )
                del self.download_status[model_id]
            elif not found:
                logger.info(f"PurgeStagingCache: model {model_id} not found")
        else:
            # Purge all models from all directories
            # Cancel all active downloads first
            for _mid, scope in list(self.active_downloads.items()):
                scope.cancel()

            total_purged = 0
            for dir_path, label in purge_dirs:
                total_purged += await self._purge_dir(dir_path, label)

            # Also clear the HF file list cache so stale entries don't linger
            hf_cache = EXO_MODELS_DIR / "caches"
            if hf_cache.exists():
                logger.info(
                    f"PurgeStagingCache: clearing file list cache at {hf_cache}"
                )
                await asyncio.to_thread(shutil.rmtree, hf_cache, True)

            # Reset all download statuses (including read-only entries —
            # their files have been deleted so they are no longer valid)
            for mid, status in list(self.download_status.items()):
                pending = DownloadPending(
                    shard_metadata=status.shard_metadata,
                    node_id=self.node_id,
                    model_directory=self._model_dir(mid),
                )
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=pending)
                )
                del self.download_status[mid]

            logger.info(f"PurgeStagingCache: purged {total_purged} model directories")

    async def _start_download(self, shard: ShardMetadata) -> None:
        model_id = shard.model_card.model_id

        # Check if already downloading, complete, or recently failed
        if model_id in self.download_status:
            status = self.download_status[model_id]
            if isinstance(status, (DownloadOngoing, DownloadCompleted, DownloadFailed)):
                logger.info(
                    f"DownloadCoordinator: {model_id} already {type(status).__name__}, re-emitting"
                )
                # Re-emit so the global state picks it up (the planner may
                # not have seen the original event, e.g. after election).
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=status)
                )
                return

        # Check EXO_MODELS_PATH for pre-downloaded models
        found_path = resolve_model_in_path(model_id)
        if found_path is not None:
            logger.info(
                f"DownloadCoordinator: Model {model_id} found in EXO_MODELS_PATH at {found_path}"
            )
            completed = DownloadCompleted(
                shard_metadata=shard,
                node_id=self.node_id,
                total=shard.model_card.storage_size,
                model_directory=str(found_path),
                read_only=True,
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            return

        # Emit pending status
        progress = DownloadPending(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._model_dir(model_id),
        )
        self.download_status[model_id] = progress
        await self.event_sender.send(NodeDownloadProgress(download_progress=progress))

        # Check initial status from downloader
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(shard)
        )

        if initial_progress.status == "complete":
            completed = DownloadCompleted(
                shard_metadata=shard,
                node_id=self.node_id,
                total=initial_progress.total,
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            return

        if self.offline:
            logger.warning(
                f"Offline mode: model {model_id} is not fully available locally, cannot download"
            )
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=f"Model files not found locally in offline mode: {model_id}",
                model_directory=self._model_dir(model_id),
            )
            self.download_status[model_id] = failed
            await self.event_sender.send(NodeDownloadProgress(download_progress=failed))
            return

        # Start actual download
        self._start_download_task(shard, initial_progress)

    def _start_download_task(
        self, shard: ShardMetadata, initial_progress: RepoDownloadProgress
    ) -> None:
        model_id = shard.model_card.model_id

        # Emit ongoing status
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=shard,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
            model_directory=self._model_dir(model_id),
        )
        self.download_status[model_id] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        async def download_wrapper(cancel_scope: anyio.CancelScope) -> None:
            try:
                path: Path | None = None
                with cancel_scope:
                    path = await self.shard_downloader.ensure_shard(shard)
                if path is not None:
                    # Correct the model_directory in case the downloader staged to a
                    # non-default location (e.g. the model store staging path rather
                    # than the standard EXO_MODELS_DIR).  The progress callback fired
                    # inside ensure_shard() always uses _model_dir(), so we override it
                    # here with the actual returned path.
                    actual_dir = str(path)
                    if actual_dir != self._model_dir(model_id):
                        existing = self.download_status.get(model_id)
                        if isinstance(existing, DownloadCompleted):
                            corrected = DownloadCompleted(
                                shard_metadata=existing.shard_metadata,
                                node_id=existing.node_id,
                                total=existing.total,
                                read_only=existing.read_only,
                                model_directory=actual_dir,
                            )
                            self.download_status[model_id] = corrected
                            await self.event_sender.send(
                                NodeDownloadProgress(download_progress=corrected)
                            )
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
                    model_directory=self._model_dir(model_id),
                )
                self.download_status[model_id] = failed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=failed)
                )
            except anyio.get_cancelled_exc_class():
                # ignore cancellation - let cleanup do its thing
                pass
            finally:
                self.active_downloads.pop(model_id, None)

        scope = anyio.CancelScope()
        self._tg.start_soon(download_wrapper, scope)
        self.active_downloads[model_id] = scope

    async def _delete_download(self, model_id: ModelId) -> None:
        # Protect read-only models (from EXO_MODELS_PATH) from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(
                    f"Refusing to delete read-only model {model_id} (from EXO_MODELS_PATH)"
                )
                return

        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        deleted = await delete_model(model_id)

        if deleted:
            logger.info(f"Successfully deleted model {model_id}")
        else:
            logger.warning(f"Model {model_id} was not found on disk")

        # Also remove the staging directory if the model was downloaded from the
        # store into a non-default location (e.g. ~/.exo/staging/<model>).
        # delete_model() only removes EXO_MODELS_DIR, so the staged copy would
        # otherwise survive and make the model reappear on the next activation.
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            if isinstance(current_status, DownloadCompleted):
                staging_dir = Path(current_status.model_directory)
                standard_dir = Path(self._model_dir(model_id))
                if staging_dir != standard_dir and staging_dir.exists():
                    logger.info(f"Deleting staged model files at {staging_dir}")
                    await asyncio.to_thread(shutil.rmtree, staging_dir, True)

        # Emit pending status to reset UI state, then remove from local tracking
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._model_dir(model_id),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
            del self.download_status[model_id]

    async def _emit_existing_download_progress(self) -> None:
        while True:
            try:
                logger.debug(
                    "DownloadCoordinator: Fetching and emitting existing download progress..."
                )
                async for (
                    _,
                    progress,
                ) in self.shard_downloader.get_shard_download_status():
                    model_id = progress.shard.model_card.model_id

                    logger.info(
                        f"DownloadCoordinator scan: {model_id} status={progress.status} "
                        f"downloaded={progress.downloaded.in_bytes} total={progress.total.in_bytes} "
                        f"completed_files={progress.completed_files}/{progress.total_files}"
                    )

                    # Active downloads emit progress via the callback — don't overwrite
                    if model_id in self.active_downloads:
                        continue

                    if progress.status == "complete":
                        status: DownloadProgress = DownloadCompleted(
                            node_id=self.node_id,
                            shard_metadata=progress.shard,
                            total=progress.total,
                            model_directory=self._model_dir(
                                progress.shard.model_card.model_id
                            ),
                        )
                    elif progress.status in ["in_progress", "not_started"]:
                        # Skip models with no local bytes — they were never
                        # downloaded and shouldn't appear in the downloads list.
                        if progress.downloaded.in_bytes == 0:
                            continue
                        if progress.downloaded_this_session.in_bytes == 0:
                            status = DownloadPending(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                model_directory=self._model_dir(
                                    progress.shard.model_card.model_id
                                ),
                                downloaded=progress.downloaded,
                                total=progress.total,
                            )
                        else:
                            status = DownloadOngoing(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                download_progress=map_repo_download_progress_to_download_progress_data(
                                    progress
                                ),
                                model_directory=self._model_dir(
                                    progress.shard.model_card.model_id
                                ),
                            )
                    else:
                        continue

                    self.download_status[progress.shard.model_card.model_id] = status
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=status)
                    )
                # Scan EXO_MODELS_PATH for pre-downloaded models
                import exo.shared.constants as _dbg_constants

                _search_paths = _dbg_constants.EXO_MODELS_PATH
                logger.info(f"DownloadCoordinator: EXO_MODELS_PATH={_search_paths}")
                if EXO_MODELS_PATH is not None:
                    for card in await get_model_cards():
                        mid = card.model_id
                        if mid in self.active_downloads:
                            continue
                        if isinstance(
                            self.download_status.get(mid),
                            (DownloadCompleted, DownloadOngoing, DownloadFailed),
                        ):
                            continue
                        found = resolve_model_in_path(mid)
                        if found is not None:
                            logger.info(
                                f"DownloadCoordinator: EXO_MODELS_PATH hit: {mid} -> {found}"
                            )
                            path_shard = PipelineShardMetadata(
                                model_card=card,
                                device_rank=0,
                                world_size=1,
                                start_layer=0,
                                end_layer=card.n_layers,
                                n_layers=card.n_layers,
                            )
                            path_completed: DownloadProgress = DownloadCompleted(
                                node_id=self.node_id,
                                shard_metadata=path_shard,
                                total=card.storage_size,
                                model_directory=str(found),
                                read_only=True,
                            )
                            self.download_status[mid] = path_completed
                            await self.event_sender.send(
                                NodeDownloadProgress(download_progress=path_completed)
                            )

                logger.debug(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
            except Exception as e:
                logger.error(
                    f"DownloadCoordinator: Error emitting existing download progress: {e}"
                )
            await anyio.sleep(60)
