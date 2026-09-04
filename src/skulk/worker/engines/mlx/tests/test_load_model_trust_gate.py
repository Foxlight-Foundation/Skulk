# pyright: reportPrivateUsage=false
"""The card's ``trust_remote_code`` must reach mlx-lm's loader (CVE-2026-5843).

mlx-lm executes a Python file named by the config's ``model_file`` key; the
upstream fix gates that behind an explicit ``trust_remote_code`` argument, and
the Foxlight fork now carries it. Skulk's wrapper must pass the flag through
to the mlx-lm path (and only that path) and must default it to ``False``, so
an unflagged card can never reach custom-architecture execution even if every
authorization layer above mis-classified it.
"""

from pathlib import Path
from typing import Any

import pytest

import skulk.worker.engines.mlx.utils_mlx as utils_mlx


class _RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model_path: Path, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append({"model_path": model_path, **kwargs})
        return object(), None


@pytest.fixture
def recording_loader(monkeypatch: pytest.MonkeyPatch) -> _RecordingLoader:
    loader = _RecordingLoader()
    monkeypatch.setattr(utils_mlx, "_mlx_lm_load_model", loader)
    return loader


def test_flag_defaults_to_false(recording_loader: _RecordingLoader) -> None:
    utils_mlx.load_model(Path("/models/m"))

    assert recording_loader.calls[0]["trust_remote_code"] is False


def test_flag_passes_through_when_set(recording_loader: _RecordingLoader) -> None:
    utils_mlx.load_model(Path("/models/m"), trust_remote_code=True)

    assert recording_loader.calls[0]["trust_remote_code"] is True


def test_other_loader_options_still_forward(
    recording_loader: _RecordingLoader,
) -> None:
    utils_mlx.load_model(Path("/models/m"), lazy=True, strict=False)

    call = recording_loader.calls[0]
    assert call["lazy"] is True
    assert call["strict"] is False
    assert call["trust_remote_code"] is False
