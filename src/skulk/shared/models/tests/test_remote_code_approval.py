"""Tests for cluster-wide model repository-code trust enforcement."""

from pathlib import Path
from typing import cast

import pytest

import skulk.download.download_utils as download_utils_module
import skulk.shared.models.remote_code_approval as approval_module
from skulk.download.download_utils import download_shard
from skulk.download.shard_downloader import NoopShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelTask, VisionCardConfig
from skulk.shared.models.remote_code_approval import (
    approved_remote_code_identities,
    loopback_mutation_allowed,
    remote_code_execution_requires_approval,
    remote_code_is_automatically_trusted,
    remote_code_trust_identity,
    require_remote_code_approval,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import RepoDownloadProgress
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.config import ModelTrustConfig, SkulkConfig, StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient, ModelStoreDownloader


def _registry_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("org/model"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=True,
        registry_card_id=f"card_{'a' * 52}",
        registry_snapshot_id="snapshot_1_test",
        registry_provenance="agent",
    )


def _no_cluster_approvals(_config: SkulkConfig | None = None) -> frozenset[str]:
    """Return an explicitly empty injected cluster trust set."""
    return frozenset()


def test_cluster_config_exposes_approved_exact_identities() -> None:
    """Every node derives the same decisions from converged cluster config."""
    identity = f"card_{'a' * 52}"
    config = SkulkConfig(
        model_trust=ModelTrustConfig(
            approved_remote_code_identities=[identity, identity]
        )
    )

    assert approved_remote_code_identities(config) == frozenset({identity})
    assert config.model_trust is not None
    assert config.model_trust.approved_remote_code_identities == [identity]


def test_approval_is_immutable_card_scoped_for_the_cluster() -> None:
    """One exact card identity permits execution throughout the cluster."""
    card = _registry_card()
    trust_identity = remote_code_trust_identity(card)

    with pytest.raises(PermissionError, match=trust_identity):
        require_remote_code_approval(card, frozenset())

    require_remote_code_approval(card, frozenset({trust_identity}))


def test_unsigned_cards_require_content_scoped_cluster_approval() -> None:
    """Unsigned repository code cannot bypass the operator decision."""
    card = _registry_card().model_copy(
        update={
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
        }
    )

    trust_identity = remote_code_trust_identity(card)
    assert trust_identity.startswith("local_")
    with pytest.raises(PermissionError, match=trust_identity):
        require_remote_code_approval(card, frozenset())

    require_remote_code_approval(card, frozenset({trust_identity}))

    changed = card.model_copy(update={"source_revision": "c" * 40})
    assert remote_code_trust_identity(changed) != trust_identity
    with pytest.raises(PermissionError):
        require_remote_code_approval(changed, frozenset({trust_identity}))


def test_foxlight_signed_pinned_card_is_the_remote_code_trust_decision() -> None:
    """Curated signed provenance needs no redundant operator approval."""
    card = _registry_card().model_copy(update={"registry_provenance": "foxlight"})

    assert remote_code_is_automatically_trusted(card)
    assert not remote_code_execution_requires_approval(card)
    require_remote_code_approval(card, frozenset())


def test_custom_card_cannot_copy_foxlight_metadata_to_gain_automatic_trust() -> None:
    """Only cards actually installed from signed registry truth are automatic."""
    card = _registry_card().model_copy(
        update={"registry_provenance": "foxlight", "is_custom": True}
    )

    assert not remote_code_is_automatically_trusted(card)
    assert remote_code_execution_requires_approval(card)


@pytest.mark.parametrize(
    "update",
    [
        {"registry_snapshot_id": None},
        {"registry_provenance": "community"},
        {"source_revision": None},
    ],
)
def test_incomplete_or_non_foxlight_registry_evidence_requires_approval(
    update: dict[str, str | None],
) -> None:
    """Copied provenance or mutable artifacts never gain automatic trust."""
    card = _registry_card().model_copy(
        update={"registry_provenance": "foxlight", **update}
    )

    assert not remote_code_is_automatically_trusted(card)
    assert remote_code_execution_requires_approval(card)


def test_registry_vision_card_requires_approval_for_platform_loader() -> None:
    """A vision loader that enables remote code cannot bypass card approval."""
    card = _registry_card().model_copy(
        update={"trust_remote_code": False, "vision": VisionCardConfig()}
    )

    assert remote_code_execution_requires_approval(card)
    with pytest.raises(PermissionError, match=card.registry_card_id or ""):
        require_remote_code_approval(card, frozenset())


def test_approval_cannot_authorize_unpinned_processor_repository() -> None:
    """Approval cannot make mutable external processor code safe."""
    card = _registry_card().model_copy(
        update={
            "trust_remote_code": False,
            "vision": VisionCardConfig(processor_repo="org/processor"),
        }
    )

    with pytest.raises(PermissionError, match="unpinned vision processor"):
        require_remote_code_approval(
            card,
            frozenset({remote_code_trust_identity(card)}),
        )


@pytest.mark.parametrize(
    ("client_host", "origin", "allowed"),
    [
        ("127.0.0.1", None, True),
        ("::1", "https://[::1]:52415", True),
        ("127.0.0.1", "http://localhost:52415", True),
        ("192.0.2.10", None, False),
        ("127.0.0.1", "https://example.com", False),
        ("127.0.0.1", "null", False),
    ],
)
def test_operator_mutations_require_loopback_peer_and_browser_origin(
    client_host: str, origin: str | None, allowed: bool
) -> None:
    """Network peers and cross-origin pages cannot invoke loopback operations."""
    assert loopback_mutation_allowed(client_host, origin) is allowed


@pytest.mark.parametrize("client_host", ["127.0.0.1", "::1", "localhost"])
def test_operator_mutations_reject_forwarded_loopback_requests(
    client_host: str,
) -> None:
    """A reverse proxy's loopback socket cannot become local authority."""
    assert not loopback_mutation_allowed(
        client_host,
        None,
        forwarding_headers_present=True,
    )


@pytest.mark.asyncio
async def test_store_backed_download_fails_before_any_store_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common store wrapper enforces approval before staging or fetching."""

    class UntouchedStore:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"store must not be accessed before approval: {name}")

    monkeypatch.setattr(
        approval_module,
        "approved_remote_code_identities",
        _no_cluster_approvals,
    )
    card = _registry_card()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    downloader = ModelStoreDownloader(
        inner=NoopShardDownloader(),
        store_client=cast(ModelStoreClient, cast(object, UntouchedStore())),
        staging_config=StagingNodeConfig(
            enabled=True,
            node_cache_path=str(tmp_path / "staging"),
        ),
    )

    with pytest.raises(PermissionError, match=card.registry_card_id or ""):
        await downloader.ensure_shard(shard)


@pytest.mark.asyncio
async def test_status_only_probe_does_not_raise_for_unapproved_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approval gates transfers, not the coordinator's read-only status probe."""

    async def ignore_progress(
        _shard: ShardMetadata, _progress: RepoDownloadProgress
    ) -> None:
        return None

    monkeypatch.setattr(download_utils_module, "SKULK_MODELS_DIR", tmp_path)
    card = _registry_card()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )

    _path, progress = await download_shard(
        shard,
        ignore_progress,
        skip_download=True,
        skip_internet=True,
    )

    assert progress.status != "complete"
