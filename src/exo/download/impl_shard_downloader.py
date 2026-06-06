import asyncio
from asyncio import create_task
from collections.abc import Awaitable
from pathlib import Path
from typing import AsyncIterator, Callable

from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    companion_download_specs,
    download_shard,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    get_model_cards,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)


def exo_shard_downloader(
    max_parallel_downloads: int = 8, offline: bool = False
) -> ShardDownloader:
    return SingletonShardDownloader(
        ResumableShardDownloader(max_parallel_downloads, offline=offline)
    )


async def build_base_shard(model_id: ModelId) -> ShardMetadata:
    model_card = await ModelCard.load(model_id)
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_card.n_layers,
        n_layers=model_card.n_layers,
    )


async def build_full_shard(model_id: ModelId) -> PipelineShardMetadata:
    base_shard = await build_base_shard(model_id)
    return PipelineShardMetadata(
        model_card=base_shard.model_card,
        device_rank=base_shard.device_rank,
        world_size=base_shard.world_size,
        start_layer=base_shard.start_layer,
        end_layer=base_shard.n_layers,
        n_layers=base_shard.n_layers,
    )


class SingletonShardDownloader(ShardDownloader):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.active_downloads: dict[ShardMetadata, asyncio.Task[Path]] = {}

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.shard_downloader.on_progress(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        if shard not in self.active_downloads:
            self.active_downloads[shard] = asyncio.create_task(
                self.shard_downloader.ensure_shard(shard, config_only)
            )
        try:
            return await self.active_downloads[shard]
        finally:
            if shard in self.active_downloads and self.active_downloads[shard].done():
                del self.active_downloads[shard]

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async for path, status in self.shard_downloader.get_shard_download_status():
            yield path, status

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        return await self.shard_downloader.get_shard_download_status_for_shard(shard)


class ResumableShardDownloader(ShardDownloader):
    def __init__(self, max_parallel_downloads: int = 8, offline: bool = False):
        self.max_parallel_downloads = max_parallel_downloads
        self.offline = offline
        self.on_progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    async def on_progress_wrapper(
        self, shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        for callback in self.on_progress_callbacks:
            await callback(shard, progress)

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.on_progress_callbacks.append(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        allow_patterns = ["config.json"] if config_only else None

        # Companions download BEFORE the base on purpose: the base repo's
        # "complete" progress event becomes cluster-visible download state
        # the moment it fires, and the planner dispatches model loads off
        # that state — so it must mean "everything the model needs is
        # here", not "the base is here and the sidecar is on its way".
        # Criticality differs per companion: split vision weights are
        # load-bearing (their failure fails the base — a vision model
        # without them is broken), while MTP sidecars and assistants are
        # best-effort (the runtime degrades to run-without-speculation;
        # failures log loudly instead).
        if not config_only and not self.offline:
            for companion_shard, allow, required in companion_download_specs(
                shard.model_card
            ):
                try:
                    _, companion_progress = await download_shard(
                        companion_shard,
                        self.on_progress_wrapper,
                        max_parallel_downloads=self.max_parallel_downloads,
                        allow_patterns=allow,
                        skip_internet=self.offline,
                    )
                    # download_shard converts repo-level fetch failures
                    # (e.g. FileNotFoundError on the file list) into a
                    # not_started result instead of raising — a required
                    # companion must not slip through that hole.
                    if required and companion_progress.status != "complete":
                        raise RuntimeError(
                            f"Required companion repo "
                            f"{companion_shard.model_card.model_id} did not "
                            f"download (status="
                            f"{companion_progress.status!r})"
                        )
                except Exception as error:
                    if required:
                        # Split vision weights are load-bearing: a vision
                        # model without them is broken, not degraded.
                        raise
                    logger.warning(
                        f"Companion repo {companion_shard.model_card.model_id} "
                        f"for {shard.model_card.model_id} could not be fetched "
                        f"({error}); speculative decoding that depends on it "
                        "will be unavailable on this node."
                    )

        target_dir, _ = await download_shard(
            shard,
            self.on_progress_wrapper,
            max_parallel_downloads=self.max_parallel_downloads,
            allow_patterns=allow_patterns,
            skip_internet=self.offline,
        )

        return target_dir

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async def _status_for_model(
            model_id: ModelId,
        ) -> tuple[Path, RepoDownloadProgress]:
            """Helper coroutine that builds the shard for a model and gets its download status."""
            shard = await build_full_shard(model_id)
            return await download_shard(
                shard,
                self.on_progress_wrapper,
                skip_download=True,
                skip_internet=self.offline,
            )

        semaphore = asyncio.Semaphore(self.max_parallel_downloads)

        async def download_with_semaphore(
            model_card: ModelCard,
        ) -> tuple[Path, RepoDownloadProgress]:
            async with semaphore:
                return await _status_for_model(model_card.model_id)

        tasks = [
            create_task(download_with_semaphore(model_card))
            for model_card in await get_model_cards()
        ]

        for task in asyncio.as_completed(tasks):
            try:
                yield await task
            except Exception as e:
                logger.warning(f"Error downloading shard: {type(e).__name__}")

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        _, progress = await download_shard(
            shard,
            self.on_progress_wrapper,
            skip_download=True,
            skip_internet=self.offline,
        )
        # A base cached before its card declared companion repos
        # (mtp_sidecar_repo / assistant_model_repo) reports complete here and
        # the coordinator never calls ensure_shard — so the companion is
        # never fetched (phase-c spec gotcha, flagged on PR #185). Degrade
        # the reported status when a declared companion is missing on disk
        # so the download path runs and pulls it.
        # Never degrade in offline mode: the companion cannot be fetched
        # anyway, and load_mlx_items treats missing companions as optional —
        # degrading would turn a perfectly loadable cached base into
        # DownloadFailed on air-gapped nodes.
        if (
            progress.status == "complete"
            and not self.offline
            and self._missing_companion(shard)
        ):
            return progress.model_copy(update={"status": "in_progress"})
        return progress

    @staticmethod
    def _missing_companion(shard: ShardMetadata) -> bool:
        """True when the card declares a companion repo absent from disk."""
        from exo.download.download_utils import model_companions_present_on_disk

        return not model_companions_present_on_disk(shard.model_card)
