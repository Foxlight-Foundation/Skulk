# pyright: reportPrivateUsage=false
"""Bundle-scoped artifacts are complete by their installed manifest."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

import skulk.shared.constants as constants
from skulk.download.download_utils import (
    build_model_path,
    is_model_directory_complete,
    resolve_model_in_path,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)

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


def _card() -> ModelCard:
    files = [
        {"path": path, "size_bytes": len(payload), "object_id": "sha256:" + hashlib.sha256(payload).hexdigest()}
        for path, payload in FILES.items()
    ]
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
                "download_size": sum(len(v) for v in FILES.values()),
                "files": files,
            },
        }
    )


def _install(root: Path, card: ModelCard) -> Path:
    artifact = root / card.model_id.normalize()
    for path, payload in FILES.items():
        (artifact / path).parent.mkdir(parents=True, exist_ok=True)
        (artifact / path).write_bytes(payload)
    (artifact / ".skulk-source-revision").write_text(f"{REVISION}\n")
    write_installed_card(artifact, build_installed_card_record(artifact, card))
    return artifact


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
