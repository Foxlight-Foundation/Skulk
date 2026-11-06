# pyright: reportPrivateUsage=false
"""Bundle-scoped artifacts are complete by their installed manifest."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Never

import pytest

import skulk.download.download_utils as download_utils
import skulk.shared.constants as constants
from skulk.download.download_utils import (
    build_model_path,
    is_model_directory_complete,
    resolve_model_in_path,
)
from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store import installed_cards
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
    write_installed_card_with_fallback,
)
from skulk.store.model_store import StoreModelEntry, StoreRegistryIndex

REVISION = "a" * 40
FILES = {
    "diffusion_models/transformer.safetensors": b"transformer-bytes",
    "text_encoders/encoder.safetensors": b"encoder",
    "vae/video_vae.safetensors": b"video-vae",
    "vae/audio_vae.safetensors": b"audio-vae",
}


def _bundle_id(files: list[dict[str, str | int]]) -> str:
    document = [
        {"relative_path": item["path"], "size_bytes": item["size_bytes"], "object_id": item["object_id"]}
        for item in sorted(files, key=lambda item: str(item["path"]))
    ]
    digest = hashlib.sha256(
        json.dumps(document, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).digest()
    return "bundle_" + base64.b32encode(digest).decode().rstrip("=").lower()


def _card(root: str | None = None) -> ModelCard:
    files = [
        {"path": path, "size_bytes": len(payload), "object_id": "sha256:" + hashlib.sha256(payload).hexdigest()}
        for path, payload in FILES.items()
    ]
    if root is not None:
        files = [{**item, "path": f"{root}/{item['path']}"} for item in files]
    return ModelCard.model_validate(
        {
            "model_id": "org/bundle-model",
            "source_repository": "org/bundle-source",
            "source_revision": REVISION,
            "storage_size": Memory.from_bytes(sum(len(v) for v in FILES.values())),
            "n_layers": 1,
            "hidden_size": 1,
            "supports_tensor": False,
            "tasks": [ModelTask.TextGeneration],
            "trust_remote_code": False,
            "artifact_bundle": {
                "bundle_id": _bundle_id(files),
                "root": root,
                "download_size": sum(len(v) for v in FILES.values()),
                "files": files,
            },
        }
    )


def _install(root: Path, card: ModelCard) -> Path:
    artifact = root / card.model_id.normalize()
    bundle_root = card.artifact_bundle.root if card.artifact_bundle is not None else None
    for path, payload in FILES.items():
        target = artifact / (f"{bundle_root}/{path}" if bundle_root else path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (artifact / ".skulk-source-revision").write_text(f"{REVISION}\n")
    write_installed_card(artifact, build_installed_card_record(artifact, card))
    return artifact


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("bundle_root", [None, "payload/loader"])
async def test_offline_download_status_recognizes_installed_bundle_without_hub_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: bool,
    bundle_root: str | None,
) -> None:
    """Canonical bundle custody is sufficient without a shard index or Hub listing."""
    card = _card(bundle_root)
    if not alias:
        card = card.model_copy(update={"source_repository": None})
    _install(tmp_path, card)
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", None)
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    progress = await ResumableShardDownloader(offline=True).get_shard_download_status_for_shard(shard)
    assert progress.status == "complete"
    assert progress.completed_files == len(FILES)
    assert progress.downloaded == card.storage_size


@pytest.mark.parametrize("damage", ["missing", "short", "no_manifest", "revision"])
async def test_offline_bundle_status_refuses_incomplete_or_other_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Local discovery cannot turn bare or incomplete tensors into a ready download."""
    card = _card().model_copy(update={"source_repository": None})
    artifact = _install(tmp_path, card)
    if damage == "missing":
        (artifact / "vae/audio_vae.safetensors").unlink()
    elif damage == "short":
        (artifact / "vae/audio_vae.safetensors").write_bytes(b"short")
    elif damage == "no_manifest":
        installed_cards.installed_card_path(artifact).unlink()
    else:
        (artifact / ".skulk-source-revision").write_text("b" * 40)
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", None)
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    progress = await ResumableShardDownloader(offline=True).get_shard_download_status_for_shard(shard)
    assert progress.status != "complete"


@pytest.mark.parametrize("change", ["path", "size", "digest"])
@pytest.mark.parametrize("revision", [REVISION, "b" * 40])
@pytest.mark.parametrize("offline", [False, True])
async def test_offline_status_handles_changed_bundle_selection_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    revision: str,
    offline: bool,
) -> None:
    """A previous generation's inventory is not the newly requested artifact."""
    previous = _card().model_copy(update={"source_repository": None})
    artifact = _install(tmp_path, previous)
    bundle = previous.artifact_bundle
    assert bundle is not None
    files = list(bundle.files)
    first = files[0]
    if change == "path":
        files[0] = first.model_copy(update={"path": "other-transformer.safetensors"})
        (artifact / "other-transformer.safetensors").write_bytes(bytes(first.size_bytes))
    elif change == "size":
        files[0] = first.model_copy(update={"size_bytes": first.size_bytes + 1})
    else:
        files[0] = first.model_copy(update={"object_id": "sha256:" + "f" * 64})
    size = sum(item.size_bytes for item in files)
    selected = previous.model_copy(
        update={
            "source_revision": revision,
            "storage_size": Memory.from_bytes(size),
            "artifact_bundle": bundle.model_copy(
                update={
                    "bundle_id": _bundle_id([
                        {"path": item.path, "size_bytes": item.size_bytes, "object_id": item.object_id or ""}
                        for item in files
                    ]),
                    "download_size": size,
                    "files": tuple(files),
                }
            ),
        }
    )
    selected = ModelCard.model_validate_json(selected.model_dump_json())
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", None)

    async def unavailable(*_args: object, **_kwargs: object) -> Never:
        raise ConnectionError("upstream metadata is unavailable")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_retry", unavailable)
    shard = PipelineShardMetadata(
        model_card=selected,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    progress = await ResumableShardDownloader(offline=offline).get_shard_download_status_for_shard(shard)
    assert progress.status != "complete"


def test_bundle_directory_is_complete_by_its_installed_manifest(tmp_path: Path) -> None:
    card = _card()
    artifact = _install(tmp_path, card)
    assert is_model_directory_complete(artifact)
    # A missing or short file breaks completeness; the manifest is the truth.
    (artifact / "vae/audio_vae.safetensors").write_bytes(b"audio")
    assert not is_model_directory_complete(artifact)
    (artifact / "vae/audio_vae.safetensors").unlink()
    assert not is_model_directory_complete(artifact)


def test_bundle_directory_without_a_sidecar_is_not_complete(tmp_path: Path) -> None:
    artifact = tmp_path / "org--bundle-model"
    for path, payload in FILES.items():
        (artifact / path).parent.mkdir(parents=True, exist_ok=True)
        (artifact / path).write_bytes(payload)
    assert not is_model_directory_complete(artifact)


def test_resolver_finds_a_staged_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card = _card()
    artifact = _install(tmp_path, card)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(constants, "SKULK_MODELS_DIR", tmp_path / "unused")
    assert resolve_model_in_path(ModelId(card.model_id), REVISION) == artifact
    assert build_model_path(ModelId(card.model_id), REVISION) == artifact
    assert resolve_model_in_path(ModelId(card.model_id), "b" * 40) is None


def _indexed_bundle(store: Path, card: ModelCard) -> Path:
    """Publish real installed metadata at the Store's bundle-qualified location."""
    artifact = _install(store, card)
    bundle = card.artifact_bundle
    assert bundle is not None
    canonical = artifact.with_name(
        f"{card.model_id.normalize()}--revision-{REVISION}"
        f"--bundle-{bundle.bundle_id.removeprefix('bundle_')[:16]}"
    )
    artifact.rename(canonical)
    entry = StoreModelEntry(
        model_id=str(card.model_id),
        store_path=canonical.name,
        source_revision=REVISION,
        source_repository=str(card.artifact_repository),
        files=[item.path for item in bundle.files],
        downloaded_at="2026-11-06T00:00:00+00:00",
        total_bytes=bundle.download_size,
    )
    (store / "registry.json").write_text(
        StoreRegistryIndex(entries={str(card.model_id): entry}).model_dump_json()
    )
    return canonical


@pytest.mark.parametrize("bundle_root", [None, "payload/loader"])
@pytest.mark.parametrize("signed_card", [False, True])
def test_card_resolver_loads_store_direct_bundle_without_a_staged_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle_root: str | None,
    signed_card: bool,
) -> None:
    """Staging-disabled downloads remain loadable by the exact requested card."""
    card = _card(bundle_root)
    if signed_card:
        card = card.model_copy(
            update={
                "registry_card_id": "card_" + "a" * 52,
                "registry_snapshot_id": "snapshot_1_current",
                "registry_provenance": "agent",
            }
        )
    store = tmp_path / "store"
    canonical = _indexed_bundle(store, card)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (store,))
    monkeypatch.setattr(download_utils, "SKULK_MODELS_DIR", tmp_path / "unused")
    assert not (store / card.model_id.normalize()).exists()
    assert resolve_model_in_path(
        card.model_id, REVISION, expected_card=card, artifact_root=bundle_root
    ) == canonical
    assert build_model_path(
        card.model_id, REVISION, bundle_root, expected_card=card
    ) == (canonical / bundle_root if bundle_root else canonical)


@pytest.mark.parametrize(
    "damage",
    ["outside", "missing", "card", "signed_card", "revision", "tombstone", "bad_tombstone", "index"],
)
def test_store_index_cannot_admit_an_unverified_or_deleted_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """An index hint cannot replace artifact, identity, containment or deletion checks."""
    card = _card()
    store = tmp_path / "store"
    canonical = _indexed_bundle(store, card)
    index_path = store / "registry.json"
    index = StoreRegistryIndex.model_validate_json(index_path.read_bytes())
    entry = index.entries[str(card.model_id)]
    if damage == "outside":
        outside = tmp_path / "outside"
        canonical.rename(outside)
        index_path.write_text(
            index.model_copy(
                update={"entries": {str(card.model_id): entry.model_copy(update={"store_path": "../outside"})}}
            ).model_dump_json()
        )
    elif damage == "missing":
        (canonical / "vae/audio_vae.safetensors").unlink()
    elif damage == "card":
        card = card.model_copy(update={"family": "different-policy"})
    elif damage == "signed_card":
        card = card.model_copy(update={"registry_card_id": "card_" + "b" * 52})
    elif damage == "revision":
        index_path.write_text(
            index.model_copy(
                update={"entries": {str(card.model_id): entry.model_copy(update={"source_revision": "b" * 40})}}
            ).model_dump_json()
        )
    elif damage in {"tombstone", "bad_tombstone"}:
        tombstone = store / ".skulk/reconciliation-tombstones.json"
        tombstone.parent.mkdir()
        tombstone.write_text(
            "{}" if damage == "bad_tombstone" else json.dumps(
                {"schema_version": 1, "deleted_aliases": {str(card.model_id): "now"}}
            )
        )
    else:
        index_path.write_text("{}")
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (store,))
    assert resolve_model_in_path(
        card.model_id, REVISION, expected_card=card
    ) is None


def test_bundle_entry_pointing_outside_the_artifact_is_not_complete(tmp_path: Path) -> None:
    card = _card()
    artifact = _install(tmp_path / "models", card)
    outside = tmp_path / "elsewhere.safetensors"
    target = artifact / "vae/audio_vae.safetensors"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    assert not is_model_directory_complete(artifact)
    # A directory entry of the right name is not a file either.
    target.unlink()
    target.mkdir()
    assert not is_model_directory_complete(artifact)


def test_resolver_finds_a_rooted_bundle_by_its_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card = _card(root="comfy")
    artifact = _install(tmp_path, card)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(constants, "SKULK_MODELS_DIR", tmp_path / "unused")
    assert resolve_model_in_path(ModelId(card.model_id), REVISION, artifact_root="comfy") == artifact
    assert build_model_path(ModelId(card.model_id), REVISION, "comfy") == artifact / "comfy"


def test_manifest_outranks_a_safetensors_index_inside_the_bundle(tmp_path: Path) -> None:
    card = _card()
    artifact = _install(tmp_path, card)
    # An index that covers only the transformer shard would declare the
    # directory complete on its own; the manifest still sees the short VAE.
    (artifact / "diffusion_models" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"w": "transformer.safetensors"}})
    )
    assert is_model_directory_complete(artifact)
    (artifact / "vae/audio_vae.safetensors").write_bytes(b"short")
    assert not is_model_directory_complete(artifact)


def test_bundle_on_a_read_only_root_is_complete_by_its_detached_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _card()
    artifact = tmp_path / "models" / card.model_id.normalize()
    for path, payload in FILES.items():
        (artifact / path).parent.mkdir(parents=True, exist_ok=True)
        (artifact / path).write_bytes(payload)
    (artifact / ".skulk-source-revision").write_text(f"{REVISION}\n")
    record = build_installed_card_record(artifact, card)
    records = tmp_path / "records"
    artifact.chmod(0o555)
    try:
        written = write_installed_card_with_fallback(artifact, record, fallback_root=records)
        assert written.parent == records
        monkeypatch.setattr(installed_cards, "SKULK_INSTALLED_CARD_RECORDS_DIR", records)
        hashes = 0
        real_verify = installed_cards.verify_installed_card

        def counting_verify(*args: object, **kwargs: object) -> bool:
            nonlocal hashes
            hashes += 1
            return real_verify(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(installed_cards, "verify_installed_card", counting_verify)
        assert is_model_directory_complete(artifact)
        assert is_model_directory_complete(artifact)
        # The detached record is hash-verified once per unchanged artifact.
        assert hashes == 1
        monkeypatch.setattr(installed_cards, "SKULK_INSTALLED_CARD_RECORDS_DIR", tmp_path / "elsewhere")
        assert not is_model_directory_complete(artifact)
    finally:
        artifact.chmod(0o755)


def test_rooted_bundle_manifest_outranks_a_config_under_the_loader_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _card(root="comfy")
    artifact = _install(tmp_path, card)
    (artifact / "comfy" / "config.json").write_text("{}")
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(constants, "SKULK_MODELS_DIR", tmp_path / "unused")
    assert resolve_model_in_path(ModelId(card.model_id), REVISION, artifact_root="comfy") == artifact
    (artifact / "comfy/vae/audio_vae.safetensors").write_bytes(b"short")
    assert resolve_model_in_path(ModelId(card.model_id), REVISION, artifact_root="comfy") is None
