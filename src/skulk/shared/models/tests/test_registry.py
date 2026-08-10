# pyright: reportPrivateUsage=false
"""Tests for signed registry loading and artifact identity separation."""

import json
from pathlib import Path
from typing import cast

import pytest

import skulk.download.download_utils as download_utils
import skulk.shared.models.model_cards as model_cards_module
import skulk.shared.models.registry as registry_module
from skulk.shared.models.model_cards import ModelCard, ModelTask, registry_model_cards
from skulk.shared.models.registry import (
    RegistryCatalog,
    RegistryUnavailableError,
    TufRegistryClient,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata


def _catalog_payload() -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "snapshot_id": "snapshot_1_test",
            "generated_at": "2026-08-08T12:00:00Z",
            "published_by": "validator@example.com",
            "note": "test",
            "card_metadata": {
                f"card_{'a' * 52}": {"provenance": "foxlight"}
            },
            "cards": [
                {
                    "schema_version": 1,
                    "card_id": f"card_{'a' * 52}",
                    "alias": "org/multi-gguf@q4-k-m",
                    "model_ref": "org/multi-gguf@q4-k-m",
                    "artifact": {
                        "repository": "org/multi-gguf",
                        "revision": "b" * 40,
                        "selected_file": "model-Q4_K_M.gguf",
                        "format": "gguf",
                        "quantization": "Q4_K_M",
                    },
                    "card": {
                        "model_id": "org/multi-gguf",
                        "source_revision": "b" * 40,
                        "storage_size": {"in_bytes": 1024},
                        "n_layers": 4,
                        "hidden_size": 64,
                        "supports_tensor": False,
                        "tasks": ["TextGeneration"],
                        "gguf_file": "model-Q4_K_M.gguf",
                        "quantization": "Q4_K_M",
                        "placement": {"compatible_backends": ["llama_server"]},
                    },
                }
            ],
        }
    ).encode()


def test_registry_alias_is_separate_from_artifact_repository() -> None:
    """Two quants can use distinct runtime ids while sharing one Hub repo."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)

    card = registry_model_cards(catalog)[0]

    assert str(card.model_id) == "org/multi-gguf@q4-k-m"
    assert str(card.artifact_repository) == "org/multi-gguf"
    assert card.gguf_file == "model-Q4_K_M.gguf"
    assert card.registry_snapshot_id == "snapshot_1_test"
    assert card.registry_provenance == "foxlight"


@pytest.mark.parametrize("alias", [".", "..", "org/..", "../model", "org\\model"])
def test_registry_rejects_path_like_aliases(alias: str) -> None:
    """Signed aliases can never address a staging root or its parent."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    cards[0]["alias"] = alias

    with pytest.raises(ValueError, match="safe repository identifier"):
        RegistryCatalog.model_validate(payload, strict=False)


def test_registry_forces_signed_cards_to_non_custom() -> None:
    """Signed payload content cannot acquire local override semantics."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload["is_custom"] = True
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    assert not registry_model_cards(catalog)[0].is_custom


def test_registry_rejects_unpinned_separate_processor_repository() -> None:
    """A signed card cannot approve code that remains mutable upstream."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload["vision"] = {
        "model_type": "test_vlm",
        "processor_repo": "org/processor",
    }
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="processor_revision"):
        registry_model_cards(catalog)


def test_registry_rejects_catalog_metadata_outside_card_identity() -> None:
    """Every published card has exactly one signed provenance record."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    payload["card_metadata"] = {}
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="metadata does not match"):
        registry_model_cards(catalog)


def test_offline_mode_disables_registry_network_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Air-gapped nodes use bundled cards without contacting public TUF."""
    monkeypatch.delenv("SKULK_TESTS", raising=False)
    monkeypatch.setattr(model_cards_module, "SKULK_MODEL_REGISTRY_ENABLED", True)
    monkeypatch.setattr(model_cards_module, "SKULK_OFFLINE", True)

    assert not model_cards_module._registry_enabled()


def test_client_uses_hash_bound_last_known_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified catalog survives an outage but not local byte tampering."""
    payload_path = tmp_path / "downloaded.json"
    payload_path.write_bytes(_catalog_payload())
    embedded_root = tmp_path / "embedded-root.json"
    embedded_root.write_text("{}")
    cached_root = tmp_path / "cache/metadata/root.json"
    cached_root.parent.mkdir(parents=True)
    cached_root.write_text('{"attacker":"preseeded"}')
    observed_trusted_roots: list[bytes] = []

    class WorkingUpdater:
        def __init__(self, **kwargs: object) -> None:
            observed_trusted_roots.append(cast("bytes", kwargs["bootstrap"]))

        def refresh(self) -> None:
            pass

        def get_targetinfo(self, _: str) -> object:
            return object()

        def download_target(self, _: object) -> str:
            return str(payload_path)

    monkeypatch.setattr(registry_module, "Updater", WorkingUpdater)
    client = TufRegistryClient(
        base_url="https://registry.example/",
        cache_dir=tmp_path / "cache",
        embedded_root=embedded_root,
        timeout_seconds=1,
        max_stale_days=30,
    )
    assert client.load_catalog().snapshot_id == "snapshot_1_test"
    assert observed_trusted_roots == [embedded_root.read_bytes()]

    class FailingUpdater(WorkingUpdater):
        def refresh(self) -> None:
            raise OSError("offline")

    monkeypatch.setattr(registry_module, "Updater", FailingUpdater)
    assert client.load_catalog().snapshot_id == "snapshot_1_test"

    (tmp_path / "cache/last-known-good-catalog.json").write_bytes(b"tampered")
    with pytest.raises(RegistryUnavailableError):
        client.load_catalog()


def test_embedded_roots_match_release_resources() -> None:
    """Package and frozen-app trust anchors cannot drift independently."""
    package_root = registry_module.EMBEDDED_REGISTRY_ROOT.read_bytes()
    release_root = (
        Path(__file__).parents[5] / "resources/model_registry/root.json"
    ).read_bytes()
    assert package_root == release_root


@pytest.mark.asyncio
async def test_failed_refresh_removes_previous_registry_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired in-memory catalog cannot outlive the configured LKG bound."""
    registry_card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    bundled_card = registry_card.model_copy(
        update={
            "model_id": ModelId("org/bundled"),
            "source_repository": None,
            "registry_card_id": None,
            "registry_snapshot_id": None,
        }
    )

    class FailingClient:
        def load_catalog(self) -> RegistryCatalog:
            raise OSError("registry offline and LKG expired")

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[registry_card.model_id] = registry_card
    model_cards_module._card_cache[bundled_card.model_id] = bundled_card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_registry_client", FailingClient())
    try:
        await model_cards_module._load_cards_from_registry()
        assert registry_card.model_id not in model_cards_module._card_cache
        assert model_cards_module._card_cache[bundled_card.model_id] == bundled_card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_successful_refresh_excludes_unlisted_bundled_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed snapshot can revoke a card that the distribution once bundled."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    registry_card = registry_model_cards(catalog)[0]
    bundled_card = registry_card.model_copy(
        update={
            "model_id": ModelId("org/revoked-bundled"),
            "source_repository": None,
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
        }
    )
    custom_card = bundled_card.model_copy(
        update={"model_id": ModelId("org/custom"), "is_custom": True}
    )

    class WorkingClient:
        def load_catalog(self) -> RegistryCatalog:
            return catalog

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[bundled_card.model_id] = bundled_card
    model_cards_module._card_cache[custom_card.model_id] = custom_card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_registry_client", WorkingClient())
    try:
        assert await model_cards_module._load_cards_from_registry()
        assert bundled_card.model_id not in model_cards_module._card_cache
        assert model_cards_module._card_cache[registry_card.model_id] == registry_card
        assert model_cards_module._card_cache[custom_card.model_id] == custom_card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_downloader_fetches_source_repository_under_alias_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network origin uses artifact truth while local state keeps the alias."""
    observed: list[ModelId] = []

    async def fake_models_dir() -> Path:
        return tmp_path

    async def fake_file_list(
        model_id: ModelId, *_: object, **__: object
    ) -> list[object]:
        observed.append(model_id)
        return []

    async def ignore_progress(*_: object) -> None:
        pass

    monkeypatch.setattr(download_utils, "ensure_models_dir", fake_models_dir)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", fake_file_list)
    card = ModelCard(
        model_id=ModelId("org/multi@q4"),
        source_repository=ModelId("org/multi"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )

    path, _ = await download_utils.download_shard(
        shard,
        ignore_progress,
        skip_download=True,
        skip_internet=True,
        allow_patterns=["config.json"],
    )

    assert observed == [ModelId("org/multi")]
    assert path.name == "org--multi@q4"
