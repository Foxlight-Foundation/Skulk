# pyright: reportPrivateUsage=false
"""Cluster model trust resolves immutable IDs from the complete catalog."""

from unittest.mock import AsyncMock

import pytest

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.shared.models.model_cards import ModelCard, ModelTask, VisionCardConfig
from skulk.shared.models.remote_code_approval import remote_code_trust_identity
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory


@pytest.mark.asyncio
async def test_image_card_approval_uses_unfiltered_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A headless/store node can approve a signed image card it does not list."""
    card_id = f"card_{'a' * 52}"
    card = ModelCard(
        model_id=ModelId("org/image-model"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
        vision=VisionCardConfig(model_type="test"),
        registry_card_id=card_id,
    )

    async def complete_catalog() -> list[ModelCard]:
        return [card]

    persist = AsyncMock()
    api = object.__new__(API)
    monkeypatch.setattr(api, "set_cluster_remote_code_approval", persist)
    monkeypatch.setattr(api_main, "get_all_model_cards", complete_catalog)

    result = await api.approve_remote_code(card_id)

    assert result.card_id == card_id
    assert result.approved_for_cluster
    persist.assert_awaited_once_with(card_id, approved=True)


def test_model_catalog_exposes_foxlight_automatic_remote_code_trust() -> None:
    """Operators can distinguish signed trust from explicit approval."""
    card = ModelCard(
        model_id=ModelId("org/model"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=True,
        registry_card_id=f"card_{'a' * 52}",
        registry_snapshot_id="snapshot_1_test",
        registry_provenance="foxlight",
    )

    entry = API._model_list_entry(card, frozenset())

    assert entry.remote_code_automatically_trusted
    assert not entry.remote_code_approval_required
    assert not entry.remote_code_approved_for_cluster


@pytest.mark.asyncio
async def test_custom_card_approval_uses_content_derived_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsigned cards have an approvable identity that changes with content."""
    card = ModelCard(
        model_id=ModelId("org/custom"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=True,
        is_custom=True,
    )

    async def complete_catalog() -> list[ModelCard]:
        return [card]

    persist = AsyncMock()
    api = object.__new__(API)
    monkeypatch.setattr(api, "set_cluster_remote_code_approval", persist)
    monkeypatch.setattr(api_main, "get_all_model_cards", complete_catalog)
    trust_identity = remote_code_trust_identity(card)

    result = await api.approve_remote_code(trust_identity)

    assert result.card_id == trust_identity
    assert result.approved_for_cluster
    persist.assert_awaited_once_with(trust_identity, approved=True)
