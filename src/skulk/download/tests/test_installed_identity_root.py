"""A completed nested GGUF must publish signed identity at its repository root."""

import base64
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import skulk.store.installed_cards as installed_cards_module
from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.installed_cards import (
    read_installed_card,
    require_registry_installed_artifact,
)


@pytest.mark.parametrize("stale_nested_sidecar", [False, True])
async def test_nested_gguf_download_records_identity_at_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_nested_sidecar: bool
) -> None:
    """The runner's trust check accepts a freshly downloaded nested bundle."""

    revision = "a" * 40
    relative_path = "nested/weights/model.gguf"
    weight_bytes = b"synthetic weights"
    digest = hashlib.sha256(weight_bytes).hexdigest()
    manifest = [
        {
            "relative_path": relative_path,
            "size_bytes": len(weight_bytes),
            "object_id": f"sha256:{digest}",
        }
    ]
    bundle_digest = hashlib.sha256(
        json.dumps(
            manifest, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).digest()
    bundle_id = "bundle_" + base64.b32encode(bundle_digest).decode().rstrip("=").lower()
    card = ModelCard.model_validate(
        {
            "model_id": "org/nested-gguf",
            "storage_size": Memory.from_bytes(len(weight_bytes)),
            "n_layers": 1,
            "hidden_size": 1,
            "supports_tensor": False,
            "tasks": [ModelTask.TextGeneration],
            "source_revision": revision,
            "gguf_file": relative_path,
            "registry_card_id": f"card_{'a' * 52}",
            "artifact_bundle": {
                "bundle_id": bundle_id,
                "root": "nested/weights",
                "download_size": len(weight_bytes),
                "files": [
                    {
                        "path": relative_path,
                        "size_bytes": len(weight_bytes),
                        "object_id": f"sha256:{digest}",
                    }
                ],
            },
        }
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    model_root = tmp_path / card.model_id.normalize()
    selected_file = model_root / relative_path
    selected_file.parent.mkdir(parents=True)
    selected_file.write_bytes(weight_bytes)
    if stale_nested_sidecar:
        stale_record = selected_file.parent / ".skulk/installed-card.json"
        stale_record.parent.mkdir()
        stale_record.write_text("{}")
    (model_root / ".skulk-source-revision").write_text(f"{revision}\n")
    progress = RepoDownloadProgress(
        repo_id=str(card.model_id),
        repo_revision=revision,
        shard=shard,
        completed_files=1,
        total_files=1,
        downloaded=card.storage_size,
        downloaded_this_session=card.storage_size,
        total=card.storage_size,
        overall_speed=0,
        overall_eta=timedelta(0),
        status="complete",
    )
    monkeypatch.setattr(installed_cards_module, "SKULK_MODELS_DIR", tmp_path)
    downloader = ResumableShardDownloader()
    with patch.object(
        downloader,
        "_download_with_capacity",
        new=AsyncMock(return_value=(selected_file, progress)),
    ):
        assert await downloader.ensure_shard(shard) == selected_file

    root_record = read_installed_card(model_root)
    assert root_record is not None
    assert [entry.path for entry in root_record.files] == [relative_path]
    require_registry_installed_artifact(model_root, card)
