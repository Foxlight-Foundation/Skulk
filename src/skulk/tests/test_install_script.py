# Copyright 2026 Foxlight Foundation
"""Installer contract tests for commit-pinned release qualification."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from skulk.store.config import DEFAULT_MODEL_STORE_PORT


def _installer() -> str:
    return (Path(__file__).parents[3] / "install.sh").read_text()


def _readme() -> str:
    return (Path(__file__).parents[3] / "README.md").read_text()


def test_readme_shipping_command_remains_literal_installer_path() -> None:
    """The public shipping qualification must exercise the documented command."""

    literal_command = (
        "curl -fsSL "
        "https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh "
        "| bash"
    )
    assert literal_command in _readme()
    installer = _installer()
    assert literal_command in installer
    assert 'INSTALL_REF="${SKULK_INSTALL_REF:-main}"' in installer


def test_full_commit_ref_fetches_exact_object_and_detaches() -> None:
    """A full SHA must not pass through branch-only git clone semantics."""

    installer = _installer()
    assert '[[ "$INSTALL_REF" =~ ^[0-9a-fA-F]{40}$ ]]' in installer
    assert 'git -C "$INSTALL_DIR" fetch --depth 1 origin "$INSTALL_REF"' in installer
    assert 'git -C "$INSTALL_DIR" checkout --detach FETCH_HEAD' in installer
    assert 'RESOLVED_COMMIT="$(git rev-parse HEAD)"' in installer
    assert 'log "resolved ref $INSTALL_REF to commit $RESOLVED_COMMIT"' in installer


def test_generated_config_pins_safe_model_store_port() -> None:
    """The installer must materialize the same safe port as runtime defaults."""

    assert f"store_port: {DEFAULT_MODEL_STORE_PORT}" in _installer()


def test_bundled_node_probe_keeps_retry_diagnostics_visible() -> None:
    """Fresh-install Node probe failures must remain visible to operators."""

    installer = _installer()
    assert "elif run_bundled_npm --version; then" in installer
    assert "run_bundled_npm --version >/dev/null" not in installer



def _git(cwd: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "installer-test",
        "GIT_AUTHOR_EMAIL": "installer-test@example.invalid",
        "GIT_COMMITTER_NAME": "installer-test",
        "GIT_COMMITTER_EMAIL": "installer-test@example.invalid",
    }
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


@pytest.mark.parametrize("ref_kind", ["branch", "commit"])
def test_update_switches_past_a_lock_file_the_dashboard_build_rewrote(
    tmp_path: Path, ref_kind: str
) -> None:
    """Rerunning the installer upgrades an install whose build rewrote its lock file.

    The dashboard build's `npm install` rewrites dashboard-react/package-lock.json
    in the checkout. Upgrading a 1.5.1 source install to a release whose lock
    file differs then aborted at checkout for a pinned commit, and for a branch
    the failed fast-forward was swallowed, so the installer reported success on
    the old version.
    """

    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    lock = origin / "dashboard-react" / "package-lock.json"
    lock.parent.mkdir()
    lock.write_text('{"lockfileVersion": 3, "release": "old"}\n')
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "old release")
    install = tmp_path / "skulk"
    _git(tmp_path, "clone", "-q", str(origin), str(install))
    lock.write_text('{"lockfileVersion": 3, "release": "new"}\n')
    _git(origin, "commit", "-q", "-am", "new release")
    new_head = _git(origin, "rev-parse", "HEAD")
    installed_lock = install / "dashboard-react" / "package-lock.json"
    installed_lock.write_text('{"lockfileVersion": 3, "rewritten": true}\n')

    installer = _installer()
    start = installer.index("# --- fetch")
    end = installer.index('cd "$INSTALL_DIR"', start)
    reference = "main" if ref_kind == "branch" else new_head
    script = (
        "set -euo pipefail\n"
        "log() { printf '%s\\n' \"$*\"; }\n"
        "die() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
        f"INSTALL_DIR={shlex.quote(str(install))}\n"
        f"INSTALL_REF={shlex.quote(reference)}\n" + installer[start:end]
    )
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _git(install, "rev-parse", "HEAD") == new_head
    assert json.loads(installed_lock.read_text())["release"] == "new"
    assert "restoring dashboard-react/package-lock.json" in result.stdout


def _rust_fixture(
    root: Path, cargo_ready: bool | None, compiler_ready: bool | None
) -> Path:
    tools = root / "bin"
    tools.mkdir()
    cargo_directory = root / ".cargo" / "bin"
    cargo_directory.mkdir(parents=True)
    for name, ready in (("cargo", cargo_ready), ("rustc", compiler_ready)):
        template = root / (name + ".template")
        template.write_text(
            '#!/bin/sh\ntest -f "$SKULK_INSTALLER_TEST_ROOT/' + name + '.ready"\n'
        )
        template.chmod(0o755)
        if ready is not None:
            (cargo_directory / name).symlink_to(template)
        if ready:
            (root / (name + ".ready")).touch()
    # Simulate rustup's proxy-before-toolchain ordering without network access.
    installation = """set -eu
printf 'install\n' >> "$SKULK_INSTALLER_TEST_ROOT/installations"
/bin/ln -sf "$SKULK_INSTALLER_TEST_ROOT/cargo.template" "$SKULK_INSTALLER_TEST_ROOT/.cargo/bin/cargo"
/bin/ln -sf "$SKULK_INSTALLER_TEST_ROOT/rustc.template" "$SKULK_INSTALLER_TEST_ROOT/.cargo/bin/rustc"
test ! -f "$SKULK_INSTALLER_TEST_ROOT/interrupted" || exit 23
test ! -f "$SKULK_INSTALLER_TEST_ROOT/incomplete" || exit 0
printf ready > "$SKULK_INSTALLER_TEST_ROOT/cargo.ready"
printf ready > "$SKULK_INSTALLER_TEST_ROOT/rustc.ready"
"""
    download = tools / "curl"
    download.write_text("#!/bin/sh\nprintf '%s' " + shlex.quote(installation))
    download.chmod(0o755)
    (tools / "sh").symlink_to("/bin/sh")
    return tools


def _run_rust_prerequisites(
    root: Path, tools: Path
) -> subprocess.CompletedProcess[str]:
    installer = _installer()
    start = installer.index("# --- Rust toolchain")
    end = installer.index("# --- uv", start)
    # Redirect only the block's home lookup; never touch the operator's toolchain.
    block = installer[start:end].replace("$HOME", "$SKULK_INSTALLER_TEST_ROOT")
    script = (
        "set -euo pipefail\n"
        "log() { printf '%s\\n' \"$*\"; }\n"
        "die() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
        + block
        + '\nprintf done > "$SKULK_INSTALLER_TEST_ROOT/advanced"\n'
    )
    environment = os.environ.copy()
    environment.update(PATH=str(tools), SKULK_INSTALLER_TEST_ROOT=str(root))
    return subprocess.run(
        ["/bin/bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize(
    ("cargo_ready", "compiler_ready", "repair_expected"),
    [
        (None, None, True),
        (False, False, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_rust_setup_checks_usability_before_advancing(
    tmp_path: Path,
    cargo_ready: bool | None,
    compiler_ready: bool | None,
    repair_expected: bool,
) -> None:
    """Missing/partial tools recover, while a working toolchain is preserved."""
    tools = _rust_fixture(tmp_path, cargo_ready, compiler_ready)
    result = _run_rust_prerequisites(tmp_path, tools)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "advanced").is_file()
    assert (tmp_path / "cargo.ready").is_file()
    assert (tmp_path / "rustc.ready").is_file()
    assert (tmp_path / "installations").exists() is repair_expected


def test_interrupted_rust_setup_recovers_on_the_same_installation(
    tmp_path: Path,
) -> None:
    """A failed setup retains proxies and a retry repairs them without deletion."""
    tools = _rust_fixture(tmp_path, None, None)
    interruption = tmp_path / "interrupted"
    interruption.touch()
    failed = _run_rust_prerequisites(tmp_path, tools)
    assert failed.returncode != 0
    assert "Rust toolchain setup failed" in failed.stderr
    assert not (tmp_path / "advanced").exists()
    assert (tmp_path / ".cargo/bin/cargo").is_symlink()
    assert (tmp_path / ".cargo/bin/rustc").is_symlink()
    interruption.unlink()
    retried = _run_rust_prerequisites(tmp_path, tools)
    assert retried.returncode == 0, retried.stderr
    assert (tmp_path / "advanced").is_file()
    assert (tmp_path / "installations").read_text().splitlines() == [
        "install",
        "install",
    ]


def test_successful_rustup_exit_requires_working_tools(tmp_path: Path) -> None:
    """A nominally successful setup cannot hide a still-unusable compiler."""
    tools = _rust_fixture(tmp_path, False, False)
    (tmp_path / "incomplete").touch()
    result = _run_rust_prerequisites(tmp_path, tools)
    assert result.returncode != 0
    assert "Rust remains unavailable after setup" in result.stderr
    assert not (tmp_path / "advanced").exists()
