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


def test_vector_startup_resolves_home_defaults_before_exec() -> None:
    """External Vector paths must not retain an unexpanded nested HOME token."""

    script = (
        _REPO_ROOT / "deployment/install/vector-startup.sh"
    ).read_text()
    config = (
        _REPO_ROOT / "deployment/logging/vector-external.yaml"
    ).read_text()

    data_dir_default = script.index("export SKULK_VECTOR_DATA_DIR=")
    log_file_default = script.index("export SKULK_LOG_FILE=")
    vector_exec = script.index('exec vector --config "$CONFIG"')

    assert data_dir_default < vector_exec
    assert log_file_default < vector_exec
    assert "data_dir: ${SKULK_VECTOR_DATA_DIR}" in config
    assert "- ${SKULK_LOG_FILE}" in config
    assert ":-${HOME}" not in config
