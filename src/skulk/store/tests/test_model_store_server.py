# pyright: reportPrivateUsage=false
"""HTTP serialization contracts for the authoritative model-store server."""

import json
from pathlib import Path
from typing import cast

import pytest
from aiohttp.test_utils import make_mocked_request

from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import build_installed_card_record
from skulk.store.model_store import ModelStore
from skulk.store.model_store_server import ModelStoreServer


@pytest.mark.anyio
async def test_registry_serializes_installed_card_timestamp(tmp_path: Path) -> None:
    """Installed-card datetimes remain valid JSON after reconciliation."""

    store_root = tmp_path / "store"
    artifact = store_root / "org--model"
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"weights")
    card = ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_bytes(7),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    installed_card = build_installed_card_record(artifact, card)
    store = ModelStore(store_root)
    store.register_model(
        "org/model",
        artifact,
        ["model.safetensors"],
        7,
        installed_card=installed_card,
    )
    server = ModelStoreServer(store, host="127.0.0.1", port=0)

    response = await server._handle_registry(make_mocked_request("GET", "/registry"))
    assert response.text is not None
    payload = cast("list[dict[str, object]]", json.loads(response.text))
    installed_card_payload = cast(
        "dict[str, object]", payload[0]["installed_card"]
    )
    captured_at = installed_card_payload["captured_at"]

    assert isinstance(captured_at, str)
    assert captured_at.endswith("Z")
