# pyright: reportPrivateUsage=false
"""Store-client download requests preserve a selected GGUF file."""

from pathlib import Path
from types import TracebackType
from typing import Self

import pytest
from aiohttp import web

import skulk.shared.models.remote_code_approval as approval_module
import skulk.store.model_store_server as model_store_server_module
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.models.remote_code_approval import RemoteCodeApprovalStore
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.store import model_store_client
from skulk.store.model_store_client import ModelStoreClient
from skulk.store.model_store_server import ModelStoreServer


class _Response:
    status = 200

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    async def json(self) -> object:
        return {"status": "pending"}


class _Session:
    def __init__(self) -> None:
        self.requests: list[tuple[str, object | None]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def post(self, url: str, *, json: object | None = None) -> _Response:
        self.requests.append((url, json))
        return _Response()

    def delete(self, url: str) -> _Response:
        self.requests.append((url, None))
        return _Response()


def _registry_card() -> ModelCard:
    """Build one signed card whose artifact can execute repository code."""
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


@pytest.mark.anyio
async def test_store_host_requires_its_own_remote_code_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker approval cannot authorize download on the canonical host."""
    card = _registry_card()

    async def cards() -> list[ModelCard]:
        return [card]

    approvals = RemoteCodeApprovalStore(tmp_path / "approvals.json")
    monkeypatch.setattr(model_store_server_module, "get_all_model_cards", cards)
    monkeypatch.setattr(approval_module, "REMOTE_CODE_APPROVALS", approvals)

    with pytest.raises(web.HTTPForbidden, match=card.registry_card_id or ""):
        await ModelStoreServer._require_remote_code_download_approval(
            str(card.model_id),
            card.registry_card_id,
            source_repository=None,
            source_revision=card.source_revision,
            pinned_gguf=None,
        )

    assert card.registry_card_id is not None
    approvals.approve(card.registry_card_id)
    await ModelStoreServer._require_remote_code_download_approval(
        str(card.model_id),
        card.registry_card_id,
        source_repository=None,
        source_revision=card.source_revision,
        pinned_gguf=None,
    )


@pytest.mark.anyio
async def test_store_host_omitted_card_id_selects_current_signed_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy callers may omit the immutable ID and still select current."""

    card = _registry_card()
    installed_old_card = card.model_copy(
        update={
            "source_revision": "c" * 40,
            "registry_card_id": f"card_{'b' * 52}",
        }
    )

    async def cards() -> list[ModelCard]:
        return [installed_old_card]

    def current_card(_model_id: ModelId) -> ModelCard:
        return card

    approvals = RemoteCodeApprovalStore(tmp_path / "approvals.json")
    approvals.approve(card.registry_card_id or "")
    monkeypatch.setattr(model_store_server_module, "get_all_model_cards", cards)
    monkeypatch.setattr(
        model_store_server_module,
        "get_current_registry_card",
        current_card,
    )
    monkeypatch.setattr(approval_module, "REMOTE_CODE_APPROVALS", approvals)

    retained = await ModelStoreServer._require_remote_code_download_approval(
        str(card.model_id),
        None,
        source_repository=str(card.artifact_repository),
        source_revision=card.source_revision,
        pinned_gguf=card.gguf_file,
    )

    assert retained == card


def test_peer_import_loopback_guard_rejects_forwarding_headers() -> None:
    model_store_server_module._require_loopback_peer_import("127.0.0.1", {})

    with pytest.raises(web.HTTPForbidden, match="proxy forwarding"):
        model_store_server_module._require_loopback_peer_import(
            "127.0.0.1",
            {"X-Forwarded-For": "203.0.113.10"},
        )
    with pytest.raises(web.HTTPForbidden, match="loopback-only"):
        model_store_server_module._require_loopback_peer_import(
            "203.0.113.10",
            {},
        )


@pytest.mark.anyio
async def test_store_host_rejects_unverifiable_registry_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller cannot bind another alias to an approved immutable card id."""

    async def cards() -> list[ModelCard]:
        return [_registry_card()]

    monkeypatch.setattr(model_store_server_module, "get_all_model_cards", cards)

    with pytest.raises(web.HTTPConflict, match="immutable registry artifact"):
        await ModelStoreServer._require_remote_code_download_approval(
            "org/different",
            f"card_{'a' * 52}",
            source_repository=None,
            source_revision="b" * 40,
            pinned_gguf=None,
        )


@pytest.mark.anyio
async def test_store_host_refreshes_once_for_new_registry_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent store/API refresh intervals do not reject a rollout."""
    card = _registry_card().model_copy(update={"trust_remote_code": False})
    refreshes: list[str] = []

    async def stale_cards() -> list[ModelCard]:
        return []

    async def refreshed_card(
        card_id: str,
        *,
        refresh_on_miss: bool = False,
    ) -> ModelCard | None:
        assert refresh_on_miss
        refreshes.append(card_id)
        return card

    monkeypatch.setattr(model_store_server_module, "get_all_model_cards", stale_cards)
    monkeypatch.setattr(
        model_store_server_module,
        "get_registry_card_by_id",
        refreshed_card,
    )

    await ModelStoreServer._require_remote_code_download_approval(
        str(card.model_id),
        card.registry_card_id,
        source_repository=None,
        source_revision=card.source_revision,
        pinned_gguf=None,
    )

    assert refreshes == [card.registry_card_id]


@pytest.mark.anyio
async def test_store_host_binds_approval_to_exact_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approved card ID cannot authorize bytes from another revision."""
    card = _registry_card()

    async def cards() -> list[ModelCard]:
        return [card]

    monkeypatch.setattr(model_store_server_module, "get_all_model_cards", cards)

    with pytest.raises(web.HTTPConflict, match="immutable registry artifact"):
        await ModelStoreServer._require_remote_code_download_approval(
            str(card.model_id),
            card.registry_card_id,
            source_repository=None,
            source_revision="c" * 40,
            pinned_gguf=None,
        )


@pytest.mark.anyio
async def test_store_host_binds_companion_to_full_owning_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A companion repository retains and is authorized by its owning card."""
    payload = _registry_card().model_dump(mode="json")
    payload["trust_remote_code"] = False
    payload["runtime"] = {
        "mtp_sidecar_repo": "org/model-mtp",
        "mtp_sidecar_revision": "c" * 40,
        "mtp_heads": True,
    }
    owner = ModelCard.model_validate(payload)

    async def cards() -> list[ModelCard]:
        return [owner]

    monkeypatch.setattr(model_store_server_module, "get_all_model_cards", cards)

    retained = await ModelStoreServer._require_remote_code_download_approval(
        "org/model-mtp",
        None,
        source_repository="org/model-mtp",
        source_revision="c" * 40,
        pinned_gguf=None,
        owner_model_id=str(owner.model_id),
        owner_registry_card_id=owner.registry_card_id,
        artifact_role="mtp_sidecar",
    )

    assert retained == owner


@pytest.mark.anyio
async def test_store_host_accepts_bundled_companion_owner_without_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundled fallback cards authorize their declared companion offline."""
    payload = _registry_card().model_dump(mode="json")
    payload.update(
        {
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "trust_remote_code": False,
            "runtime": {
                "mtp_sidecar_repo": "org/model-mtp",
                "mtp_sidecar_revision": "c" * 40,
                "mtp_heads": True,
            },
        }
    )
    owner = ModelCard.model_validate(payload)

    async def cards() -> list[ModelCard]:
        return [owner]

    monkeypatch.setattr(model_store_server_module, "get_all_model_cards", cards)

    retained = await ModelStoreServer._require_remote_code_download_approval(
        "org/model-mtp",
        None,
        source_repository="org/model-mtp",
        source_revision="c" * 40,
        pinned_gguf=None,
        owner_model_id=str(owner.model_id),
        owner_registry_card_id=None,
        artifact_role="mtp_sidecar",
    )

    assert retained == owner


@pytest.mark.anyio
async def test_store_client_sends_requested_gguf_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()

    def fake_session_factory(**_kwargs: object) -> _Session:
        return session

    monkeypatch.setattr(
        model_store_client,
        "create_http_session",
        fake_session_factory,
    )
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    result = await client.request_store_download(
        "org/model",
        gguf_file="model-IQ3_XXS.gguf",
        source_revision="0123456789abcdef0123456789abcdef01234567",
    )

    assert result == {"status": "pending"}
    assert session.requests == [
        (
            "http://store.local:58080/models/org%2Fmodel/download",
            {
                "gguf_file": "model-IQ3_XXS.gguf",
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
            },
        )
    ]


@pytest.mark.anyio
async def test_store_client_sends_signed_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual store requests preserve the card identity bound to approval."""
    session = _Session()

    def fake_session_factory(**_kwargs: object) -> _Session:
        return session

    monkeypatch.setattr(model_store_client, "create_http_session", fake_session_factory)
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    await client.request_store_download(
        "org/model@q4",
        gguf_file="model-Q4.gguf",
        source_revision="a" * 40,
        source_repository="org/model",
        registry_card_id=f"card_{'b' * 52}",
    )

    assert session.requests == [
        (
            "http://store.local:58080/models/org%2Fmodel%40q4/download",
            {
                "gguf_file": "model-Q4.gguf",
                "source_revision": "a" * 40,
                "source_repository": "org/model",
                "registry_card_id": f"card_{'b' * 52}",
            },
        )
    ]


@pytest.mark.anyio
async def test_store_client_cancels_the_canonical_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()

    def fake_session_factory(**_kwargs: object) -> _Session:
        return session

    monkeypatch.setattr(
        model_store_client,
        "create_http_session",
        fake_session_factory,
    )
    client = ModelStoreClient(store_host="store.local", store_port=58080)

    cancelled = await client.cancel_store_download("org/model")

    assert cancelled
    assert session.requests == [
        ("http://store.local:58080/models/org%2Fmodel/download", None)
    ]
