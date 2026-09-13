"""Managed runtime and system-service selection use kernel and architecture only."""

import platform
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from skulk.extensions import runtime_artifacts
from skulk.extensions.service_registration import service_platform


@pytest.mark.parametrize(
    "release",
    [
        {"ID": "ubuntu", "VERSION_ID": "24.04"},
        {"ID": "ubuntu", "VERSION_ID": "26.04"},
        {"ID": "debian", "VERSION_ID": "13"},
        {"ID": "fedora", "VERSION_ID": "44"},
        {},
    ],
)
def test_linux_distribution_does_not_gate_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release: dict[str, str]
) -> None:
    """Actual host measurement and service setup admit later and non-Ubuntu Linux."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "freedesktop_os_release", lambda: release)
    source = tmp_path / "__init__.py"
    source.write_text("# synthetic core package\n")

    def specification(name: str) -> ModuleSpec:
        return ModuleSpec(name, loader=None, origin=str(source))

    monkeypatch.setattr(runtime_artifacts.importlib.util, "find_spec", specification)
    assert runtime_artifacts.measure_host().platform == "ubuntu-24.04-x86_64"
    assert service_platform() == "ubuntu-24.04-x86_64"


def test_incompatible_architecture_remains_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a distribution gate cannot select x86 wheels for an ARM host."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="platform"):
        runtime_artifacts.measure_host()
    with pytest.raises(ValueError, match="Linux x86_64"):
        service_platform()
