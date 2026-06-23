# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
"""Tests for the macOS Local Network identity probe."""

from __future__ import annotations

import plistlib
from pathlib import Path

from pytest import MonkeyPatch

from skulk.connectivity import local_network_probe
from skulk.connectivity.local_network_probe import (
    LocalNetworkIdentityProbe,
    ProcessIdentity,
)


def test_app_bundle_for_executable_reads_info_plist(tmp_path: Path) -> None:
    app_root = tmp_path / "Skulk.app"
    executable = app_root / "Contents" / "MacOS" / "Skulk"
    info_plist = app_root / "Contents" / "Info.plist"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    with info_plist.open("wb") as file:
        plistlib.dump(
            {
                "CFBundleIdentifier": "foundation.foxlight.skulk",
                "CFBundleDisplayName": "Skulk",
                "CFBundleName": "Skulk",
                "CFBundleExecutable": "Skulk",
            },
            file,
        )

    bundle = local_network_probe._app_bundle_for_executable(str(executable))

    assert bundle is not None
    assert bundle.path == str(app_root)
    assert bundle.bundle_identifier == "foundation.foxlight.skulk"
    assert bundle.display_name == "Skulk"
    assert bundle.executable_name == "Skulk"


def test_app_bundle_for_executable_returns_none_without_app(tmp_path: Path) -> None:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    executable.write_text("", encoding="utf-8")

    assert local_network_probe._app_bundle_for_executable(str(executable)) is None


def test_format_probe_report_includes_status_and_process() -> None:
    report = LocalNetworkIdentityProbe(
        local_network_status="blocked",
        platform_system="Darwin",
        macos_version="26.0",
        hostname="kite1",
        process=ProcessIdentity(
            pid=123,
            ppid=1,
            name="python",
            executable="/usr/bin/python3",
            command_line=["python3", "-m", "skulk"],
            app_bundle=None,
        ),
        ancestors=[],
        notes=["Local Network access appears blocked."],
    )

    text = local_network_probe.format_local_network_identity_probe(report)

    assert "Local Network status: blocked" in text
    assert "pid=123" in text
    assert "python3" in text
    assert "Local Network access appears blocked." in text


def test_run_probe_returns_two_when_blocked_and_fail_enabled(
    monkeypatch: MonkeyPatch,
) -> None:
    report = LocalNetworkIdentityProbe(
        local_network_status="blocked",
        platform_system="Darwin",
        macos_version="26.0",
        hostname="kite1",
        process=ProcessIdentity(pid=123),
        ancestors=[],
        notes=[],
    )
    monkeypatch.setattr(
        local_network_probe,
        "collect_local_network_identity_probe",
        lambda max_ancestors=8: report,
    )

    assert local_network_probe.run_local_network_identity_probe(
        fail_on_blocked=True
    ) == 2


def test_run_probe_returns_zero_when_blocked_without_fail(
    monkeypatch: MonkeyPatch,
) -> None:
    report = LocalNetworkIdentityProbe(
        local_network_status="blocked",
        platform_system="Darwin",
        macos_version="26.0",
        hostname="kite1",
        process=ProcessIdentity(pid=123),
        ancestors=[],
        notes=[],
    )
    monkeypatch.setattr(
        local_network_probe,
        "collect_local_network_identity_probe",
        lambda max_ancestors=8: report,
    )

    assert local_network_probe.run_local_network_identity_probe() == 0


def test_friendly_executable_label_maps_python_variants() -> None:
    assert local_network_probe._friendly_executable_label("/x/python3.13") == "Python"
    assert local_network_probe._friendly_executable_label("/usr/bin/python") == "Python"
    assert local_network_probe._friendly_executable_label("/opt/uv") == "uv"
    assert local_network_probe._friendly_executable_label("/x/mylauncher") == "mylauncher"
    assert local_network_probe._friendly_executable_label(None) is None


def test_responsible_app_label_prefers_nearest_app_bundle(
    monkeypatch: MonkeyPatch,
) -> None:
    # A process chain where an ancestor lives in a .app bundle: that bundle's
    # display name is what macOS attributes the grant to.
    bundle = local_network_probe.AppBundleIdentity(
        path="/Applications/iTerm.app", display_name="iTerm2"
    )
    current = ProcessIdentity(pid=1, executable="/x/python3.13")
    ancestor = ProcessIdentity(pid=2, executable="/Applications/iTerm.app/Contents/MacOS/iTerm2", app_bundle=bundle)
    monkeypatch.setattr(local_network_probe.psutil, "Process", lambda: object())
    monkeypatch.setattr(local_network_probe, "_process_identity", lambda _p: current)
    monkeypatch.setattr(local_network_probe, "_ancestor_identities", lambda _p, _n: [ancestor])

    assert local_network_probe.responsible_app_label() == "iTerm2"


def test_responsible_app_label_falls_back_to_executable_when_no_bundle(
    monkeypatch: MonkeyPatch,
) -> None:
    current = ProcessIdentity(pid=1, executable="/opt/python3.13")
    monkeypatch.setattr(local_network_probe.psutil, "Process", lambda: object())
    monkeypatch.setattr(local_network_probe, "_process_identity", lambda _p: current)
    monkeypatch.setattr(local_network_probe, "_ancestor_identities", lambda _p, _n: [])

    assert local_network_probe.responsible_app_label() == "Python"
