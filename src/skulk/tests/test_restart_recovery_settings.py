"""The Settings toggle that brings models back after a restart."""

from pathlib import Path

import pytest

import skulk.main as node_main


def test_the_toggle_is_read_from_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert node_main._restart_recovery_restores_placements()  # pyright: ignore[reportPrivateUsage]

    (tmp_path / "skulk.yaml").write_text("restart_recovery:\n  enabled: false\n")
    assert not node_main._restart_recovery_restores_placements()  # pyright: ignore[reportPrivateUsage]

    (tmp_path / "skulk.yaml").write_text("restart_recovery:\n  enabled: true\n")
    assert node_main._restart_recovery_restores_placements()  # pyright: ignore[reportPrivateUsage]


def test_unreadable_settings_keep_models_coming_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken skulk.yaml fails loudly elsewhere; a restart should not also drop models."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skulk.yaml").write_text("restart_recovery: [unclosed\n")

    assert node_main._restart_recovery_restores_placements()  # pyright: ignore[reportPrivateUsage]
