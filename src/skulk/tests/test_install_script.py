# Copyright 2026 Foxlight Foundation
"""Installer contract tests for commit-pinned release qualification."""

from pathlib import Path


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
