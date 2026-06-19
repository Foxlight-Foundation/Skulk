# pyright: reportPrivateUsage=false
"""Dashboard-asset resolution for headless/worker nodes (#333)."""

from pathlib import Path

import pytest

from skulk.utils import dashboard_path


def test_find_dashboard_optional_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A headless node with no built assets and no bundle: resolution must yield
    # None rather than raising, so importing constants does not fail on boot.
    monkeypatch.setattr(dashboard_path, "_find_react_dashboard_in_repo", lambda: None)
    monkeypatch.setattr(dashboard_path, "_find_dashboard_in_bundle", lambda: None)
    assert dashboard_path.find_dashboard_optional() is None


def test_find_dashboard_raises_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The strict variant still raises for callers that require the UI.
    monkeypatch.setattr(dashboard_path, "_find_react_dashboard_in_repo", lambda: None)
    monkeypatch.setattr(dashboard_path, "_find_dashboard_in_bundle", lambda: None)
    with pytest.raises(FileNotFoundError):
        dashboard_path.find_dashboard()


def test_find_dashboard_optional_returns_found_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dashboard_path, "_find_react_dashboard_in_repo", lambda: tmp_path)
    monkeypatch.setattr(dashboard_path, "_find_dashboard_in_bundle", lambda: None)
    assert dashboard_path.find_dashboard_optional() == tmp_path
    assert dashboard_path.find_dashboard() == tmp_path
