# pyright: reportPrivateUsage=false
"""Remote-code approval resolves immutable IDs from the complete catalog."""

from typing import cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.api.types.api import AddCustomModelParams
from skulk.shared.models.model_cards import ModelCard, ModelTask, VisionCardConfig
from skulk.shared.models.remote_code_approval import RemoteCodeApprovalStore
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory


class _RecordingApprovalStore:
    """Minimal approval-store test double that records immutable IDs."""

    def __init__(self) -> None:
        self.approved: list[str] = []

    def approve(self, card_id: str) -> None:
        """Record one approved card ID."""
        self.approved.append(card_id)


@pytest.mark.asyncio
async def test_remote_client_cannot_add_custom_remote_code_card() -> None:
    """Custom-card generation shares the node-local approval boundary."""
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/models/add",
            "headers": [],
            "client": ("192.0.2.10", 12345),
            "scheme": "http",
            "server": ("192.0.2.20", 52415),
        }
    )

    with pytest.raises(HTTPException) as raised:
        await object.__new__(API).add_custom_model(
            AddCustomModelParams(model_id=ModelId("org/custom-model")),
            request,
        )

    assert raised.value.status_code == 403


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

    store = _RecordingApprovalStore()
    monkeypatch.setattr(api_main, "get_all_model_cards", complete_catalog)
    monkeypatch.setattr(
        api_main,
        "REMOTE_CODE_APPROVALS",
        cast(RemoteCodeApprovalStore, cast(object, store)),
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/models/remote-code-approvals/test",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
            "server": ("127.0.0.1", 52415),
        }
    )

    result = await object.__new__(API).approve_remote_code(card_id, request)

    assert result.card_id == card_id
    assert store.approved == [card_id]
