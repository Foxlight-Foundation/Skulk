# pyright: reportPrivateUsage=false
"""Tests for the disposable macOS Local Network probe app builder."""

from __future__ import annotations

import os
import plistlib
import stat
from pathlib import Path
from typing import cast

import pytest
from pytest import MonkeyPatch

from skulk.connectivity import local_network_probe_app
from skulk.connectivity.local_network_probe_app import (
    DEFAULT_BUNDLE_IDENTIFIER,
    DEFAULT_DISPLAY_NAME,
    DEFAULT_EXECUTABLE_NAME,
    LOCAL_NETWORK_USAGE_DESCRIPTION,
    build_macos_local_network_probe_app,
)


def test_build_macos_local_network_probe_app_writes_bundle(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    log_dir = tmp_path / "logs"
    output_app = tmp_path / "SkulkLocalNetworkProbe.app"

    result = build_macos_local_network_probe_app(
        output_app=output_app,
        repo_root=repo_root,
        log_dir=log_dir,
        launcher_kind="script",
        ad_hoc_sign=False,
    )

    info_plist = output_app / "Contents" / "Info.plist"
    executable = output_app / "Contents" / "MacOS" / DEFAULT_EXECUTABLE_NAME
    with info_plist.open("rb") as file:
        loaded_info = cast("object", plistlib.load(file))
    assert isinstance(loaded_info, dict)
    raw_info = cast("dict[object, object]", loaded_info)

    assert result.app_path == str(output_app.resolve())
    assert raw_info["CFBundleIdentifier"] == DEFAULT_BUNDLE_IDENTIFIER
    assert raw_info["CFBundleDisplayName"] == DEFAULT_DISPLAY_NAME
    assert raw_info["CFBundleExecutable"] == DEFAULT_EXECUTABLE_NAME
    assert raw_info["NSLocalNetworkUsageDescription"] == LOCAL_NETWORK_USAGE_DESCRIPTION
    assert executable.exists()
    assert executable.stat().st_mode & stat.S_IXUSR
    launcher = executable.read_text(encoding="utf-8")
    assert str(repo_root.resolve()) in launcher
    assert str(log_dir) in launcher
    assert "skulk-macos-local-network-probe --json" in launcher


def test_build_macos_local_network_probe_app_can_use_native_launcher(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output_app = tmp_path / "SkulkLocalNetworkProbe.app"

    def fake_compile(*, source_path: Path, executable_path: Path) -> tuple[bool, str | None]:
        assert source_path.exists()
        executable_path.write_text("native", encoding="utf-8")
        return True, None

    monkeypatch.setattr(
        local_network_probe_app,
        "_compile_native_launcher",
        fake_compile,
    )

    result = build_macos_local_network_probe_app(
        output_app=output_app,
        repo_root=repo_root,
        ad_hoc_sign=False,
    )

    launcher_source = output_app / "Contents" / "Resources" / "launcher.m"
    assert result.launcher_kind == "native"
    assert launcher_source.exists()
    assert "static const char *REPO_ROOT" in launcher_source.read_text(
        encoding="utf-8"
    )
    assert (
        output_app / "Contents" / "MacOS" / DEFAULT_EXECUTABLE_NAME
    ).read_text(encoding="utf-8") == "native"


def test_build_macos_local_network_probe_app_refuses_non_app_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must end with .app"):
        build_macos_local_network_probe_app(
            output_app=tmp_path / "SkulkLocalNetworkProbe",
            repo_root=tmp_path,
            launcher_kind="script",
            ad_hoc_sign=False,
        )


def test_build_macos_local_network_probe_app_refuses_existing_path(
    tmp_path: Path,
) -> None:
    output_app = tmp_path / "SkulkLocalNetworkProbe.app"
    output_app.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        build_macos_local_network_probe_app(
            output_app=output_app,
            repo_root=tmp_path,
            launcher_kind="script",
            ad_hoc_sign=False,
        )


def test_build_macos_local_network_probe_app_replaces_existing_path(
    tmp_path: Path,
) -> None:
    output_app = tmp_path / "SkulkLocalNetworkProbe.app"
    stale_file = output_app / "stale"
    output_app.mkdir()
    stale_file.write_text("old", encoding="utf-8")

    build_macos_local_network_probe_app(
        output_app=output_app,
        repo_root=tmp_path,
        launcher_kind="script",
        ad_hoc_sign=False,
        replace_existing=True,
    )

    assert not stale_file.exists()
    assert os.access(
        output_app / "Contents" / "MacOS" / DEFAULT_EXECUTABLE_NAME,
        os.X_OK,
    )
