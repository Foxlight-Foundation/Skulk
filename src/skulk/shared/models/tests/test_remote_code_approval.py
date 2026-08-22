"""Tests for cluster-wide model repository-code trust enforcement."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import skulk.download.download_utils as download_utils_module
from skulk.download.download_utils import download_shard
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
from skulk.store.config import ModelTrustConfig, SkulkConfig


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


def test_signed_agent_card_is_authorized_by_publication() -> None:
    """Signed publication authorizes repository code regardless of provenance."""
    card = _registry_card()

    assert remote_code_is_automatically_trusted(card)
    assert not remote_code_execution_requires_approval(card)
    require_remote_code_approval(card, frozenset())


@pytest.mark.asyncio
async def test_unknown_card_load_never_becomes_implicit_hub_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read or launch lookup cannot persist an attacker-selected Hub model."""
    model_id = ModelId("untrusted/unknown-model")
    fetch = AsyncMock()

    async def no_refresh() -> None:
        return None

    monkeypatch.setattr(
        "skulk.shared.models.model_cards._refresh_card_cache_if_due",
        no_refresh,
    )
    monkeypatch.setattr(ModelCard, "fetch_from_hf", fetch)
    with pytest.raises(ValueError, match="add it through POST /models/add"):
        await ModelCard.load(model_id)

    fetch.assert_not_awaited()


def test_explicitly_added_card_is_authorized_by_addition() -> None:
    """Persisting a custom card is the operator's execution decision."""
    card = _registry_card().model_copy(
        update={
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
            "is_custom": True,
        }
    )

    trust_identity = remote_code_trust_identity(card)
    assert trust_identity.startswith("local_")
    assert remote_code_is_automatically_trusted(card)
    require_remote_code_approval(card, frozenset({trust_identity}))

    changed = card.model_copy(update={"source_revision": "c" * 40})
    assert remote_code_trust_identity(changed) != trust_identity
    require_remote_code_approval(changed, frozenset())


def test_qualification_ownership_does_not_change_remote_code_identity() -> None:
    """The cleanup marker is platform state, not executable artifact identity."""
    card = _registry_card().model_copy(
        update={
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
            "is_custom": True,
        }
    )

    assert remote_code_trust_identity(
        card.model_copy(update={"qualification_only": True})
    ) == remote_code_trust_identity(card)


def test_foxlight_signed_pinned_card_is_authorized_by_publication() -> None:
    """Foxlight provenance remains authorized without special treatment."""
    card = _registry_card().model_copy(update={"registry_provenance": "foxlight"})

    assert remote_code_is_automatically_trusted(card)
    assert not remote_code_execution_requires_approval(card)
    require_remote_code_approval(card, frozenset())


def test_custom_card_authorization_does_not_depend_on_copied_provenance() -> None:
    """The explicit add action, not copied provenance, authorizes a custom card."""
    card = _registry_card().model_copy(
        update={"registry_provenance": "foxlight", "is_custom": True}
    )

    assert remote_code_is_automatically_trusted(card)
    assert not remote_code_execution_requires_approval(card)
    assert remote_code_trust_identity(card).startswith("local_")
    assert remote_code_trust_identity(card) != card.registry_card_id


def test_community_registry_card_is_authorized_by_publication() -> None:
    """Provenance records evidence quality rather than runtime permission."""
    card = _registry_card().model_copy(
        update={"registry_provenance": "community"}
    )

    assert remote_code_is_automatically_trusted(card)
    assert not remote_code_execution_requires_approval(card)
    require_remote_code_approval(card, frozenset())


@pytest.mark.parametrize("missing_field", ["registry_snapshot_id", "source_revision"])
def test_signed_repository_code_requires_immutable_publication_identity(
    missing_field: str,
) -> None:
    """A malformed signed card cannot authorize mutable executable content."""
    card = _registry_card().model_copy(update={missing_field: None})

    assert not remote_code_is_automatically_trusted(card)
    assert not remote_code_execution_requires_approval(card)
    with pytest.raises(PermissionError, match="immutable signed execution identity"):
        require_remote_code_approval(card, frozenset())


def test_registry_vision_card_is_authorized_by_publication() -> None:
    """Vision capability alone never creates a second approval ceremony."""
    card = _registry_card().model_copy(
        update={"trust_remote_code": False, "vision": VisionCardConfig()}
    )

    assert remote_code_is_automatically_trusted(card)
    assert not remote_code_execution_requires_approval(card)
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


def test_empty_legacy_allow_list_does_not_block_published_card() -> None:
    """Publication remains sufficient when the retired allow-list is empty."""
    card = _registry_card()

    require_remote_code_approval(card, frozenset())


@pytest.mark.asyncio
async def test_status_only_probe_does_not_recheck_publication_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only status probe does not re-evaluate card authorization."""

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
