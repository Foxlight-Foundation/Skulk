# pyright: reportPrivateUsage=false
"""Direct-download cache identity tests for immutable source revisions."""

from pathlib import Path

import pytest

import skulk.shared.constants as constants
from skulk.download.download_utils import resolve_model_in_path
from skulk.shared.models.model_cards import ModelId


def test_unpinned_resolution_rejects_pinned_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mutable-main requests must not reuse bytes marked as a pinned revision."""

    model_id = ModelId("org/model")
    model_dir = tmp_path / model_id.normalize()
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"weights")
    marker = model_dir / ".skulk-source-revision"
    marker.write_text(f"{'0' * 40}\n")
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))

    assert resolve_model_in_path(model_id) is None

    marker.unlink()
    assert resolve_model_in_path(model_id) == model_dir
