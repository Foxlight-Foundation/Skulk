"""The server must receive the card's loader root for nested bundles."""

import tomllib
from pathlib import Path

import pytest

from skulk.download import download_utils
from skulk.shared.constants import RESOURCES_DIR
from skulk.shared.models.model_cards import ModelCard
from skulk.worker.runner.audio_cpp.runner import model_directory


def test_ace_step_server_uses_nested_loader_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A repository root would make ACE-Step select the wrong model variant."""
    card_path = (
        Path(RESOURCES_DIR)
        / "music_model_cards"
        / "audio-cpp--ACE-Step1.5-Turbo-BF16.toml"
    )
    card = ModelCard.model_validate(tomllib.loads(card_path.read_text()))
    loader_root = tmp_path / "ACE-Step1.5-GGUF" / "turbo"
    loader_root.mkdir(parents=True)
    artifact = loader_root / "ace-step-1.5-turbo-bf16.gguf"
    artifact.touch()

    def local_model_path(model_id: object, revision: object, root: object) -> Path:
        assert model_id == card.model_id
        assert revision == card.source_revision
        assert root == "ACE-Step1.5-GGUF/turbo"
        return loader_root

    monkeypatch.setattr(download_utils, "build_model_path", local_model_path)
    assert model_directory(card) == loader_root
    assert card.artifact_bundle is not None
    assert download_utils.resolve_artifact_file(
        loader_root,
        card.artifact_bundle.root,
        card.artifact_bundle.files[0].path,
    ) == artifact
