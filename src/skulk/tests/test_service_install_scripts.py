# Copyright 2026 Foxlight Foundation
"""Regression contracts for the documented supervised-service installers."""

import os
import subprocess
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


def test_supervised_startup_uses_bundled_npm_before_system_fallback() -> None:
    """A Linux service without host npm must still refresh its dashboard."""

    script = (_REPO_ROOT / "deployment/install/skulk-startup.sh").read_text()

    assert '"$REPO_ROOT/scripts/run_bundled_npm.py" "$@"' in script
    bundled_probe = script.index("if run_bundled_npm --version")
    bundled_install = script.index(
        "run_bundled_npm install --no-fund --no-audit", bundled_probe
    )
    bundled_build = script.index("run_bundled_npm run build", bundled_install)
    system_fallback = script.index("command -v npm", bundled_build)
    system_install = script.index(
        "npm install --no-fund --no-audit", system_fallback
    )

    assert bundled_probe < bundled_install < bundled_build < system_fallback
    assert system_fallback < system_install


def test_vector_startup_expands_home_defaults_before_vector(
    tmp_path: Path,
) -> None:
    """Vector must receive absolute user paths instead of literal shell syntax."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_vector = fake_bin / "vector"
    fake_vector.write_text(
        "#!/bin/sh\n"
        'printf "data=%s\\nlog=%s\\n" '
        '"$SKULK_VECTOR_DATA_DIR" "$SKULK_LOG_FILE"\n'
    )
    fake_vector.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "SKULK_ENV_FILE": str(tmp_path / "missing.env"),
        }
    )

    result = subprocess.run(
        [_REPO_ROOT / "deployment/install/vector-startup.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.splitlines() == [
        f"data={home}/.skulk/vector",
        f"log={home}/.skulk/logs/skulk.stdout.log",
    ]
