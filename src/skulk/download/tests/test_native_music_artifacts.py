"""Native reader metadata must not authorize incomplete artifact transactions."""

import tomllib
from pathlib import Path

import pytest
from foxlight_model_preservation.models import PreservationError

from skulk.download import download_utils
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.installed_cards import build_installed_card_record
from skulk.store.model_store import ModelStore
from skulk.store.preservation_closure import companion_selections


def _card() -> ModelCard:
    source = Path(__file__).parents[2] / "shared/models/tests/ace-native-card.toml"
    return ModelCard.model_validate(tomllib.loads(source.read_text()))


def _shard(card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )


def test_native_companion_selection_cannot_silently_omit_both_components() -> None:
    """Refuse an incomplete transaction even though reading the card is valid."""
    with pytest.raises(ValueError, match="native music.*artifact"):
        download_utils.companion_download_specs(_card())


@pytest.mark.parametrize("required_only", [False, True])
def test_native_profile_cannot_be_inferred_download_complete(required_only: bool) -> None:
    """Cached and offline shortcuts cannot mistake primary bytes for full closure."""
    assert not download_utils.model_companions_present_on_disk(
        _card(), required_only=required_only
    )


async def test_native_primary_selection_is_not_a_complete_download() -> None:
    """Direct callers cannot resolve only three files and label the model complete."""
    with pytest.raises(ValueError, match="native music.*artifact"):
        await download_utils.resolve_allow_patterns(_shard(_card()))


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("config_only", [False, True])
async def test_native_downloader_refuses_before_cache_or_network_effects(
    offline: bool, config_only: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration-only and offline paths must also honor the unsupported layout."""
    cache = tmp_path / "not-created"
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", cache)
    downloader = ResumableShardDownloader(offline=offline)
    with pytest.raises(ValueError, match="native music.*artifact"):
        await downloader.ensure_shard(_shard(_card()), config_only=config_only)
    assert not cache.exists()


@pytest.mark.parametrize("skip_download", [False, True])
async def test_explicit_patterns_cannot_bypass_native_download_refusal(
    skip_download: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw shard request fails before creating directories, even with supplied paths."""
    cache = tmp_path / "not-created"
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", cache)
    downloader = ResumableShardDownloader()
    with pytest.raises(ValueError, match="native music.*artifact"):
        await download_utils.download_shard(
            _shard(_card()),
            downloader.on_progress_wrapper,
            skip_download=skip_download,
            allow_patterns=["config.json"],
            skip_internet=True,
        )
    assert not cache.exists()


def test_native_preservation_cannot_select_primary_only_closure() -> None:
    """Both export and recovery share this refusal before selecting any byte work."""
    with pytest.raises(PreservationError, match="native music.*artifact"):
        companion_selections(_card())


def test_native_installed_record_cannot_certify_primary_only_bytes(tmp_path: Path) -> None:
    """Record creation refuses before inspecting an uncreated artifact directory."""
    with pytest.raises(ValueError, match="native music.*artifact"):
        build_installed_card_record(tmp_path / "missing", _card())


async def test_native_store_download_is_rejected_before_admission(tmp_path: Path) -> None:
    """Direct store callers cannot bypass the shard downloader's layout refusal."""
    root = tmp_path / "not-created"
    store = ModelStore(root, recover_installed_cards=False)
    card = _card()
    with pytest.raises(ValueError, match="native music.*artifact"):
        await store.request_download(str(card.model_id), model_card=card)
    assert not store.list_active_downloads()
    assert not root.exists()
