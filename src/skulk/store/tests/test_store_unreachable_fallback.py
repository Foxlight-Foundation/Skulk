# pyright: reportPrivateUsage=false
"""Direct-HF fallback when the store host is unreachable (#657)."""

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import (
    ModelNotInStoreError,
    ModelStoreClient,
    ModelStoreDownloader,
    StoreUnreachableError,
)

_MODEL_ID = "org/wan-member-model"


class _RecordingInnerDownloader(ShardDownloader):
    """Inner (direct Hugging Face) downloader that records engagement."""

    def __init__(self, dest: Path) -> None:
        self.dest = dest
        self.ensure_calls: list[tuple[str, bool]] = []

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        self.ensure_calls.append((str(shard.model_card.model_id), config_only))
        self.dest.mkdir(parents=True, exist_ok=True)
        return self.dest

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        pass

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:
            yield (Path("/unused"), cast(RepoDownloadProgress, object()))

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        raise AssertionError("status queries are not used in this test")


class _UnreachableStoreClient:
    """Store client whose every HTTP touch raises StoreUnreachableError."""

    def __init__(self) -> None:
        self.availability_checks = 0

    async def is_model_available(
        self, model_id: str, source_revision: str | None = None
    ) -> bool:
        self.availability_checks += 1
        raise StoreUnreachableError("connection timeout to store host")

    async def request_and_wait_for_download(self, *args: object, **kwargs: object):
        raise AssertionError(
            "an unreachable store must not receive download requests"
        )

    async def stage_shard(self, *args: object, **kwargs: object) -> Path:
        raise AssertionError("an unreachable store must not receive stage calls")


class _ReachableThenUnreachableClient:
    """Answers the probe, then drops off before the transfer completes."""

    async def is_model_available(
        self, model_id: str, source_revision: str | None = None
    ) -> bool:
        return True

    async def request_and_wait_for_download(self, *args: object, **kwargs: object):
        return True

    async def stage_shard(self, *args: object, **kwargs: object) -> Path:
        raise StoreUnreachableError("route dropped mid staging")


def _shard() -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId(_MODEL_ID),
            storage_size=Memory.from_bytes(8),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _downloader(
    store: object, inner: ShardDownloader, tmp_path: Path, *, allow_hf: bool = True
) -> ModelStoreDownloader:
    return ModelStoreDownloader(
        inner=inner,
        store_client=cast(ModelStoreClient, store),
        staging_config=StagingNodeConfig(
            enabled=True, node_cache_path=str(tmp_path / "staging")
        ),
        allow_hf_fallback=allow_hf,
    )


@pytest.mark.anyio
async def test_unreachable_store_falls_back_to_direct_hf(tmp_path: Path) -> None:
    """A store-unreachable node stages directly from Hugging Face.

    The remote-member shape (#657): the availability probe never reaches the
    store, so the node must engage its inner (pre-store) HF path instead of
    hammering the store-host download endpoint and starving the placement.
    """
    inner = _RecordingInnerDownloader(tmp_path / "hf")
    store = _UnreachableStoreClient()
    downloader = _downloader(store, inner, tmp_path)

    path = await downloader.ensure_shard(_shard())

    assert path == inner.dest
    assert store.availability_checks == 1
    assert inner.ensure_calls == [(_MODEL_ID, False)]


@pytest.mark.anyio
async def test_unreachable_store_with_fallback_disabled_names_unreachability(
    tmp_path: Path,
) -> None:
    """Air-gapped deployments get an error naming the actual problem.

    "Store unreachable" and "model not in store" demand different operator
    responses; the disabled-fallback error must not claim the latter.
    """
    inner = _RecordingInnerDownloader(tmp_path / "hf")
    downloader = _downloader(
        _UnreachableStoreClient(), inner, tmp_path, allow_hf=False
    )

    with pytest.raises(ModelNotInStoreError, match="unreachable"):
        await downloader.ensure_shard(_shard())
    assert inner.ensure_calls == []


@pytest.mark.anyio
async def test_store_dropping_mid_transfer_falls_back(tmp_path: Path) -> None:
    """Unreachability after a successful probe still reaches the fallback.

    A route that flaps out between the availability probe and staging must
    not strand the node: the mid-transfer StoreUnreachableError routes to
    the same direct-HF fallback as a failed probe.
    """
    inner = _RecordingInnerDownloader(tmp_path / "hf")
    downloader = _downloader(_ReachableThenUnreachableClient(), inner, tmp_path)

    path = await downloader.ensure_shard(_shard())

    assert path == inner.dest
    assert inner.ensure_calls == [(_MODEL_ID, False)]


@pytest.mark.anyio
async def test_probe_uses_small_budget_then_raises_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real availability probe exhausts exactly the 3-attempt budget.

    Exercises ModelStoreClient.is_model_available itself (not a stub):
    transport failures must consume _STORE_PROBE_RETRY_ATTEMPTS attempts,
    then raise StoreUnreachableError rather than returning False
    (PR #664 review).
    """
    import aiohttp

    from skulk.store import model_store_client as msc

    counter = {"attempts": 0}

    class _Session:
        async def __aenter__(self) -> "_Session":
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        def get(self, url: str) -> "_Session":
            return self

    class _Request(_Session):
        async def __aenter__(self) -> "_Session":
            counter["attempts"] += 1
            raise aiohttp.ClientConnectionError("no route to store")

    class _OuterSession(_Session):
        def get(self, url: str) -> _Request:
            return _Request()

    def _fake_session(timeout_profile: str = "short") -> _OuterSession:
        return _OuterSession()

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(msc, "create_http_session", _fake_session)
    monkeypatch.setattr(msc.asyncio, "sleep", _no_sleep)

    client = ModelStoreClient("unreachable-host")
    with pytest.raises(StoreUnreachableError):
        await client.is_model_available("org/some-model")
    assert counter["attempts"] == msc._STORE_PROBE_RETRY_ATTEMPTS


@pytest.mark.anyio
async def test_poll_dropout_raises_unreachable_not_stall_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store that stops answering status polls surfaces unreachability.

    The stall clock exists for a live store grinding through a huge file; a
    store that stops ANSWERING must not ride it for the multi-hour budget
    and then fail with a misleading "no progress" error (PR #664 review).
    """
    import aiohttp

    from skulk.store import model_store_client as msc

    poll_attempts = {"attempts": 0}

    class _Session:
        async def __aenter__(self) -> "_Session":
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

    class _PostRequest(_Session):
        """The download request itself succeeds (store was up)."""

        status = 200

        async def json(self) -> dict[str, object]:
            return {"status": "started"}

    class _GetRequest(_Session):
        async def __aenter__(self) -> "_Session":
            poll_attempts["attempts"] += 1
            raise aiohttp.ClientConnectionError("route dropped mid poll")

    class _OuterSession(_Session):
        def post(self, url: str, **kwargs: object) -> _PostRequest:
            return _PostRequest()

        def get(self, url: str) -> _GetRequest:
            return _GetRequest()

    def _fake_session(timeout_profile: str = "short") -> _OuterSession:
        return _OuterSession()

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(msc, "create_http_session", _fake_session)
    monkeypatch.setattr(msc.asyncio, "sleep", _no_sleep)

    client = ModelStoreClient("dropping-host")
    with pytest.raises(StoreUnreachableError, match="stopped answering"):
        await client.request_and_wait_for_download("org/some-model")
    assert poll_attempts["attempts"] == msc._STORE_POLL_UNREACHABLE_THRESHOLD
