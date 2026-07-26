# Copyright 2026 Foxlight Foundation
"""Regression contracts for the documented supervised-service installers."""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    "relative_path",
    [
        "deployment/install/install-launchd.sh",
        "deployment/install/install-systemd.sh",
    ],
)
def test_service_installer_resolves_fresh_uv_location(relative_path: str) -> None:
    """A just-installed uv must be visible before the installer rejects PATH."""

    script = (_REPO_ROOT / relative_path).read_text()
    path_resolution = script.index('for candidate_dir in "$HOME/.local/bin"')
    uv_guard = script.index("if ! command -v uv")

    assert path_resolution < uv_guard
    assert "PATH=\"$candidate_dir:$PATH\"" in script
    assert "export PATH" in script[path_resolution:uv_guard]


def test_systemd_runner_oom_does_not_stop_skulk_node() -> None:
    """An OOM-killed runner child must not take down the API and model store."""

    unit = (_REPO_ROOT / "deployment/systemd/skulk.service").read_text()

    assert "OOMPolicy=continue" in unit
