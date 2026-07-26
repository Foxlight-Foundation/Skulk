"""Installer coverage for the bundled Node.js dashboard toolchain."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


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
