"""Tests for first-run dashboard browser behavior."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from skulk.utils import banner

_SSH_ENVIRONMENT_VARIABLES = ("SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY")


class _TerminalStream(io.StringIO):
    """In-memory stderr stream with configurable terminal behavior."""

    def __init__(self, *, is_terminal: bool) -> None:
        super().__init__()
        self._is_terminal = is_terminal

    def isatty(self) -> bool:
        """Return the configured terminal state."""

        return self._is_terminal


def _prepare_first_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    is_terminal: bool,
) -> tuple[Path, list[str]]:
    """Prepare an isolated first run and capture browser-open attempts."""

    marker = tmp_path / ".dashboard_opened"
    opened_urls: list[str] = []
    monkeypatch.setattr(banner, "_FIRST_RUN_MARKER", marker)
    monkeypatch.setattr(banner.webbrowser, "open", opened_urls.append)
    monkeypatch.setattr(
        banner.sys,
        "stderr",
        _TerminalStream(is_terminal=is_terminal),
    )
    monkeypatch.delenv("SKULK_RUNTIME_DIR", raising=False)
    for variable in _SSH_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    return marker, opened_urls


@pytest.mark.parametrize("ssh_variable", ["SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY"])
def test_should_not_open_dashboard_over_ssh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ssh_variable: str,
) -> None:
    marker, opened_urls = _prepare_first_run(
        monkeypatch,
        tmp_path,
        is_terminal=True,
    )
    monkeypatch.setenv(ssh_variable, "present")

    banner.print_startup_banner(52415)

    assert marker.exists()
    assert opened_urls == []


def test_should_not_open_dashboard_without_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker, opened_urls = _prepare_first_run(
        monkeypatch,
        tmp_path,
        is_terminal=False,
    )

    banner.print_startup_banner(52415)

    assert marker.exists()
    assert opened_urls == []


def test_should_not_open_dashboard_inside_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker, opened_urls = _prepare_first_run(
        monkeypatch,
        tmp_path,
        is_terminal=True,
    )
    monkeypatch.setenv("SKULK_RUNTIME_DIR", "/tmp/skulk-runtime")

    banner.print_startup_banner(52415)

    assert marker.exists()
    assert opened_urls == []


def test_should_open_dashboard_for_local_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker, opened_urls = _prepare_first_run(
        monkeypatch,
        tmp_path,
        is_terminal=True,
    )

    banner.print_startup_banner(52415)

    assert marker.exists()
    assert opened_urls == ["http://localhost:52415"]


def test_subsequent_run_never_opens_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / ".dashboard_opened"
    marker.touch()
    opened_urls: list[str] = []
    monkeypatch.setattr(banner, "_FIRST_RUN_MARKER", marker)
    monkeypatch.setattr(banner.webbrowser, "open", opened_urls.append)
    monkeypatch.setattr(banner.sys, "stderr", _TerminalStream(is_terminal=True))
    monkeypatch.delenv("SKULK_RUNTIME_DIR", raising=False)
    for variable in _SSH_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    banner.print_startup_banner(52415)

    assert opened_urls == []
