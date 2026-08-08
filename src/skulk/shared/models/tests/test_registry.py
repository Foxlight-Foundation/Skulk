"""Tests for signed registry loading and artifact identity separation."""

import json
from pathlib import Path

import pytest

import skulk.download.download_utils as download_utils
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
                        "placement": {
                            "compatible_backends": ["llama_server"]
                        },
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


def test_client_uses_hash_bound_last_known_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified catalog survives an outage but not local byte tampering."""
    payload_path = tmp_path / "downloaded.json"
    payload_path.write_bytes(_catalog_payload())
    embedded_root = tmp_path / "embedded-root.json"
    embedded_root.write_text("{}")

    class WorkingUpdater:
        def __init__(self, **_: object) -> None:
            pass

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
    monkeypatch.setattr(
        download_utils, "fetch_file_list_with_cache", fake_file_list
    )
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
