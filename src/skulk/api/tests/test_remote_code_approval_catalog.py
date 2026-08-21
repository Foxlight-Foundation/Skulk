# pyright: reportPrivateUsage=false
"""Cluster model trust resolves immutable IDs from the complete catalog."""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.api.operator_gateway import OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY
from skulk.api.types import AddExactCustomModelCardParams
from skulk.shared.models.model_cards import ModelCard, ModelTask, VisionCardConfig
from skulk.shared.models.remote_code_approval import remote_code_trust_identity
from skulk.shared.types.commands import AddCustomModelCard, ForwarderCommand
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.utils.channels import channel


def _loopback_operator_request() -> Request:
    """Return a direct-local request authorized for operator mutations."""
    return Request(
        {
            "type": "http",
            "headers": [],
            "client": ("127.0.0.1", 52415),
        }
    )


def _authenticated_gateway_request() -> Request:
    """Return a remote request already validated by the operator gateway."""
    return Request(
        {
            "type": "http",
            "headers": [],
            "client": ("198.51.100.10", 52415),
            OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY: True,
        }
    )


def _remote_bearer_request(token: str | None = None) -> Request:
    """Return one direct remote request with an optional bearer credential."""
    headers = [] if token is None else [(b"authorization", f"Bearer {token}".encode())]
    return Request(
        {
            "type": "http",
            "headers": headers,
            "client": ("198.51.100.10", 52415),
        }
    )


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

    result = await api.approve_remote_code(card_id, _loopback_operator_request())

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

    result = await api.approve_remote_code(
        trust_identity, _authenticated_gateway_request()
    )

    assert result.card_id == trust_identity
    assert result.approved_for_cluster
    persist.assert_awaited_once_with(trust_identity, approved=True)


@pytest.mark.asyncio
async def test_exact_custom_card_preserves_artifact_but_strips_registry_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-publication validation cannot impersonate signed registry trust."""
    registry_card_id = f"card_{'a' * 52}"
    supplied = ModelCard(
        model_id=ModelId("org/exact@q4_k_m"),
        source_repository=ModelId("org/exact"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1234),
        n_layers=2,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="exact-Q4_K_M.gguf",
        registry_card_id=registry_card_id,
        registry_snapshot_id="snapshot_unpublished",
        registry_provenance="foxlight",
    )
    sender, receiver = channel[ForwarderCommand]()
    api = object.__new__(API)
    object.__setattr__(api, "command_sender", sender)
    object.__setattr__(api, "_system_id", NodeId("test-system"))

    result = await api.add_exact_custom_model_card(
        AddExactCustomModelCardParams(model_card=supplied),
        _authenticated_gateway_request(),
    )

    forwarded = await receiver.receive()
    assert isinstance(forwarded.command, AddCustomModelCard)
    persisted = forwarded.command.model_card
    assert result.id == "org/exact@q4_k_m"
    assert persisted.is_custom
    assert persisted.source_revision == "b" * 40
    assert persisted.gguf_file == "exact-Q4_K_M.gguf"
    assert persisted.registry_card_id is None
    assert persisted.registry_snapshot_id is None
    assert persisted.registry_provenance is None


@pytest.mark.asyncio
async def test_qualification_token_controls_exact_card_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry credential installs and removes only an unsigned exact card."""
    token = "q" * 48
    monkeypatch.setenv("SKULK_EXACT_CARD_QUALIFICATION_TOKEN", token)
    supplied = ModelCard(
        model_id=ModelId("org/exact@q4_k_m"),
        source_repository=ModelId("org/exact"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1234),
        n_layers=2,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="exact-Q4_K_M.gguf",
    )
    sender, receiver = channel[ForwarderCommand]()
    api = object.__new__(API)
    object.__setattr__(api, "command_sender", sender)
    object.__setattr__(api, "_system_id", NodeId("test-system"))

    with pytest.raises(HTTPException, match="loopback"):
        API._require_operator_mutation(_remote_bearer_request(token))
    with pytest.raises(HTTPException, match="loopback"):
        await api.add_exact_custom_model_card(
            AddExactCustomModelCardParams(model_card=supplied),
            _remote_bearer_request("wrong"),
        )

    await api.add_exact_custom_model_card(
        AddExactCustomModelCardParams(model_card=supplied),
        _remote_bearer_request(token),
    )
    added = await receiver.receive()
    assert isinstance(added.command, AddCustomModelCard)

    def custom_card(_model_id: ModelId) -> ModelCard:
        return supplied.model_copy(update={"is_custom": True})

    monkeypatch.setattr(api_main, "get_card", custom_card)
    response = await api.delete_custom_model(
        supplied.model_id,
        _remote_bearer_request(token),
    )
    deleted = await receiver.receive()
    assert isinstance(deleted.command, api_main.DeleteCustomModelCard)
    assert response.status_code == 200
