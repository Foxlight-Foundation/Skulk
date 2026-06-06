# pyright: reportPrivateUsage=false
"""Tests for the on-disk companion presence check.

This predicate gates the DownloadCoordinator's resolve-in-path shortcut: a
base model already on disk must NOT be reported download-complete when the
card declares an MTP sidecar or assistant that is missing — that shortcut
bypassed ensure_shard entirely and loaded staged models with speculative
decoding silently unavailable (codex review on #213).
"""

from pathlib import Path

import pytest

import exo.download.download_utils as download_utils_module
import exo.shared.constants as constants_module
from exo.download.download_utils import model_companions_present_on_disk
from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    RuntimeCapabilityCardConfig,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory

_SIDECAR_REPO = "FoxlightAI/test-9b-mtp"
_ASSISTANT_REPO = "mlx-community/test-12b-assistant-bf16"


def _card(runtime: RuntimeCapabilityCardConfig | None) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-org/test-base"),
        storage_size=Memory.from_bytes(0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        runtime=runtime,
    )


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(constants_module, "EXO_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils_module, "EXO_MODELS_DIR", tmp_path)
    return tmp_path


def test_bare_card_has_no_missing_companions(models_dir: Path) -> None:
    assert model_companions_present_on_disk(_card(None))


def test_declared_sidecar_missing_on_disk(models_dir: Path) -> None:
    card = _card(
        RuntimeCapabilityCardConfig(mtp_heads=True, mtp_sidecar_repo=_SIDECAR_REPO)
    )
    assert not model_companions_present_on_disk(card)


def test_declared_sidecar_present_on_disk(models_dir: Path) -> None:
    sidecar_dir = models_dir / "FoxlightAI--test-9b-mtp"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "mtp.safetensors").write_bytes(b"fake")
    card = _card(
        RuntimeCapabilityCardConfig(mtp_heads=True, mtp_sidecar_repo=_SIDECAR_REPO)
    )
    assert model_companions_present_on_disk(card)


def test_declared_assistant_missing_on_disk(models_dir: Path) -> None:
    card = _card(RuntimeCapabilityCardConfig(assistant_model_repo=_ASSISTANT_REPO))
    assert not model_companions_present_on_disk(card)


def test_declared_assistant_present_on_disk(models_dir: Path) -> None:
    assistant_dir = models_dir / "mlx-community--test-12b-assistant-bf16"
    assistant_dir.mkdir(parents=True)
    (assistant_dir / "config.json").write_text("{}")
    (assistant_dir / "model.safetensors").write_bytes(b"fake")
    card = _card(RuntimeCapabilityCardConfig(assistant_model_repo=_ASSISTANT_REPO))
    assert model_companions_present_on_disk(card)


def test_sidecar_without_mtp_heads_is_not_required(models_dir: Path) -> None:
    """A sidecar repo declared without mtp_heads is never loaded by the
    runner, so its absence must not block the path shortcut."""
    card = _card(
        RuntimeCapabilityCardConfig(mtp_heads=False, mtp_sidecar_repo=_SIDECAR_REPO)
    )
    assert model_companions_present_on_disk(card)
