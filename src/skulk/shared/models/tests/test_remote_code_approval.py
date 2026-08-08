"""Tests for node-local remote-code approval enforcement."""

import stat
from pathlib import Path

import pytest

import skulk.shared.models.remote_code_approval as approval_module
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.models.remote_code_approval import (
    RemoteCodeApprovalStore,
    require_remote_code_approval,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory


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
    )


def test_approval_is_immutable_card_scoped_and_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approval persists by card hash and the local file is owner-only."""
    path = tmp_path / "approvals.json"
    store = RemoteCodeApprovalStore(path)
    monkeypatch.setattr(approval_module, "REMOTE_CODE_APPROVALS", store)
    card = _registry_card()

    with pytest.raises(PermissionError, match=card.registry_card_id or ""):
        require_remote_code_approval(card)

    assert card.registry_card_id is not None
    store.approve(card.registry_card_id)
    require_remote_code_approval(card)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    store.revoke(card.registry_card_id)
    with pytest.raises(PermissionError):
        require_remote_code_approval(card)


def test_local_cards_do_not_enter_registry_approval_policy() -> None:
    """Legacy bundled and custom cards retain their existing trust behavior."""
    card = _registry_card().model_copy(
        update={"registry_card_id": None, "registry_snapshot_id": None}
    )
    require_remote_code_approval(card)
