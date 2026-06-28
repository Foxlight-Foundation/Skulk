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

import skulk.download.download_utils as download_utils_module
import skulk.shared.constants as constants_module
from skulk.download.download_utils import model_companions_present_on_disk
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    RuntimeCapabilityCardConfig,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory

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
    monkeypatch.setattr(constants_module, "SKULK_MODELS_PATH", (tmp_path,))
    monkeypatch.setattr(download_utils_module, "SKULK_MODELS_DIR", tmp_path)
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


def test_split_vision_repo_missing_blocks_completion(models_dir: Path) -> None:
    """A vision card with a separate weights repo absent from disk must not
    let the base short-circuit as complete (codex round 2 on #213)."""
    from skulk.shared.models.model_cards import VisionCardConfig

    card = ModelCard(
        model_id=ModelId("test-org/test-base"),
        storage_size=Memory.from_bytes(0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        vision=VisionCardConfig(
            image_token_id=1,
            model_type="test",
            weights_repo="test-org/test-base-vision",
        ),
    )
    assert not model_companions_present_on_disk(card)

    # Present via the single-file companion layout -> complete.
    vision_dir = models_dir / "test-org--test-base-vision"
    vision_dir.mkdir(parents=True)
    (vision_dir / "config.json").write_text("{}")
    (vision_dir / "model.safetensors").write_bytes(b"fake")
    assert model_companions_present_on_disk(card)


def test_same_repo_vision_is_not_a_companion(models_dir: Path) -> None:
    from skulk.shared.models.model_cards import VisionCardConfig

    card = ModelCard(
        model_id=ModelId("test-org/test-base"),
        storage_size=Memory.from_bytes(0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        vision=VisionCardConfig(
            image_token_id=1,
            model_type="test",
            weights_repo="test-org/test-base",
        ),
    )
    assert model_companions_present_on_disk(card)


def test_required_only_checks_vision_but_not_speculation(models_dir: Path) -> None:
    """Offline mode: optional companions never block (they can't arrive),
    but missing split-vision weights still do — a vision model without
    them is broken, not degraded."""
    from skulk.shared.models.model_cards import VisionCardConfig

    card = ModelCard(
        model_id=ModelId("test-org/test-base"),
        storage_size=Memory.from_bytes(0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        vision=VisionCardConfig(
            image_token_id=1,
            model_type="test",
            weights_repo="test-org/test-base-vision",
        ),
        runtime=RuntimeCapabilityCardConfig(
            mtp_heads=True, mtp_sidecar_repo=_SIDECAR_REPO
        ),
    )
    # Vision missing: blocked in both modes.
    assert not model_companions_present_on_disk(card, required_only=True)
    assert not model_companions_present_on_disk(card)

    vision_dir = models_dir / "test-org--test-base-vision"
    vision_dir.mkdir(parents=True)
    (vision_dir / "config.json").write_text("{}")
    (vision_dir / "model.safetensors").write_bytes(b"fake")

    # Vision present, sidecar missing: required-only passes, full check blocks.
    assert model_companions_present_on_disk(card, required_only=True)
    assert not model_companions_present_on_disk(card)


def test_vision_repo_found_in_models_dir(
    models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vision repo downloaded via HF lives in SKULK_MODELS_DIR, not on the
    SKULK_MODELS_PATH search path — the probe must find it there too, or
    cached bases degrade to in_progress forever (Copilot round 7)."""
    import skulk.shared.constants as constants_module
    from skulk.shared.models.model_cards import VisionCardConfig

    hf_dir = tmp_path / "hf_models"
    hf_dir.mkdir()
    monkeypatch.setattr(constants_module, "SKULK_MODELS_PATH", ())
    monkeypatch.setattr(download_utils_module, "SKULK_MODELS_DIR", hf_dir)

    vision_dir = hf_dir / "test-org--test-base-vision"
    vision_dir.mkdir(parents=True)
    (vision_dir / "config.json").write_text("{}")
    (vision_dir / "model.safetensors").write_bytes(b"fake")

    card = ModelCard(
        model_id=ModelId("test-org/test-base"),
        storage_size=Memory.from_bytes(0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        vision=VisionCardConfig(
            image_token_id=1,
            model_type="test",
            weights_repo="test-org/test-base-vision",
        ),
    )
    assert model_companions_present_on_disk(card)


def test_served_draft_missing_on_disk(models_dir: Path) -> None:
    # A card declaring a served draft GGUF whose file is absent must NOT report
    # companions present, so a staged base does not short-circuit the download
    # path and leave --model-draft unresolvable at llama-server launch.
    card = _card(
        RuntimeCapabilityCardConfig(
            served_spec_type="draft_mtp",
            served_spec_draft_repo="test-org/test-base",
            served_spec_draft_file="draft.gguf",
        )
    )
    assert not model_companions_present_on_disk(card)


def test_served_draft_present_on_disk(models_dir: Path) -> None:
    # Same-repo draft: the draft GGUF shares the base repo's directory.
    base_dir = models_dir / "test-org--test-base"
    base_dir.mkdir(parents=True)
    (base_dir / "draft.gguf").write_bytes(b"fake")
    card = _card(
        RuntimeCapabilityCardConfig(
            served_spec_type="draft_mtp",
            served_spec_draft_repo="test-org/test-base",
            served_spec_draft_file="draft.gguf",
        )
    )
    assert model_companions_present_on_disk(card)
