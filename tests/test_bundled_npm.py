"""Installer coverage for the bundled Node.js dashboard toolchain."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_bundled_npm import run_npm_probe


def test_bundled_npm_exposes_node_to_lifecycle_scripts(tmp_path: Path) -> None:
    """npm scripts must find Node when the host has no system installation."""

    package = {
        "name": "skulk-bundled-node-test",
        "private": True,
        "scripts": {"verify-node": "node --version && npm --version"},
    }
    (tmp_path / "package.json").write_text(json.dumps(package))
    root = Path(__file__).parents[1]
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    (empty_path / "sh").symlink_to("/bin/sh")
    environment = os.environ.copy()
    environment["PATH"] = str(empty_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_bundled_npm.py"),
            "run",
            "verify-node",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "v25." in completed.stdout


def test_bundled_npm_probe_retries_a_transient_launch_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A one-off Node launch failure must not strand a fresh installation."""

    statuses = iter((75, 0))
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def npm_runner(arguments: list[str]) -> int:
        calls.append(arguments)
        return next(statuses)

    status = run_npm_probe(
        ["--version"],
        npm_runner=npm_runner,
        sleep=sleeps.append,
    )

    assert status == 0
    assert calls == [["--version"], ["--version"]]
    assert sleeps == [1.0]
    assert "attempt 1/3 failed with exit 75; retrying" in capsys.readouterr().err


def test_bundled_npm_probe_retries_a_transient_launch_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A transient inability to create the Node process must be retried."""

    attempts = 0
    sleeps: list[float] = []

    def npm_runner(arguments: list[str]) -> int:
        nonlocal attempts
        assert arguments == ["--version"]
        attempts += 1
        if attempts == 1:
            raise OSError(11, "resource temporarily unavailable")
        return 0

    status = run_npm_probe(
        ["--version"],
        npm_runner=npm_runner,
        sleep=sleeps.append,
    )

    assert status == 0
    assert attempts == 2
    assert sleeps == [1.0]
    assert (
        "attempt 1/3 failed with launch error [Errno 11] resource temporarily "
        "unavailable; retrying"
        in capsys.readouterr().err
    )


def test_bundled_npm_probe_preserves_a_persistent_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All failed attempts still return failure with visible diagnostics."""

    calls: list[list[str]] = []
    sleeps: list[float] = []

    def npm_runner(arguments: list[str]) -> int:
        calls.append(arguments)
        return 126

    status = run_npm_probe(
        ["--version"],
        npm_runner=npm_runner,
        sleep=sleeps.append,
    )

    assert status == 126
    assert calls == [["--version"], ["--version"], ["--version"]]
    assert sleeps == [1.0, 1.0]
    error = capsys.readouterr().err
    assert "attempt 1/3 failed with exit 126; retrying" in error
    assert "attempt 3/3 failed with exit 126" in error


def test_bundled_npm_probe_preserves_a_persistent_launch_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A persistent launch exception is reported for every attempt and reraised."""

    attempts = 0
    sleeps: list[float] = []

    def npm_runner(arguments: list[str]) -> int:
        nonlocal attempts
        assert arguments == ["--version"]
        attempts += 1
        raise OSError(11, "resource temporarily unavailable")

    with pytest.raises(OSError, match="resource temporarily unavailable"):
        run_npm_probe(
            ["--version"],
            npm_runner=npm_runner,
            sleep=sleeps.append,
        )

    assert attempts == 3
    assert sleeps == [1.0, 1.0]
    error = capsys.readouterr().err
    assert "attempt 1/3 failed with launch error" in error
    assert "attempt 3/3 failed with launch error" in error
