"""An artifact family is derived from the host, never enumerated in advance.

Two closed pairs of names, one in the runtime installer and one in the manager
service registration, meant a Grace Blackwell node on Linux aarch64 could not
install a capability runtime or register the service. These tests pin the
derived vocabulary, the names already written into signed releases, the
agreement between the two derivations, and the one property the old closed set
guarded: x86_64 artifacts are never taken for an aarch64 host.
"""

import platform
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from skulk.extensions import runtime_artifacts
from skulk.extensions.runtime_artifacts import canonical_platform, platform_matches
from skulk.extensions.service_registration import service_platform


def _linux(monkeypatch: pytest.MonkeyPatch, machine: str, libc: str = "glibc") -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(platform, "libc_ver", lambda: (libc, "2.39" if libc else ""))


def _synthetic_core(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "__init__.py"
    source.write_text("# synthetic core package\n")

    def specification(name: str) -> ModuleSpec:
        return ModuleSpec(name, loader=None, origin=str(source))

    monkeypatch.setattr(runtime_artifacts.importlib.util, "find_spec", specification)


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
    """A distribution name or version never appears in the family."""
    _linux(monkeypatch, "x86_64")
    monkeypatch.setattr(platform, "freedesktop_os_release", lambda: release)
    _synthetic_core(monkeypatch, tmp_path)
    assert runtime_artifacts.measure_host().platform == "linux-glibc-x86_64"
    assert service_platform() == "linux-glibc-x86_64"


def test_every_host_names_its_own_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No host is refused, and the installer and service registration agree."""
    _synthetic_core(monkeypatch, tmp_path)
    _linux(monkeypatch, "aarch64")
    assert runtime_artifacts.measure_host().platform == "linux-glibc-aarch64"
    assert service_platform() == "linux-glibc-aarch64"
    _linux(monkeypatch, "x86_64", libc="")
    assert runtime_artifacts.current_platform() == service_platform()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert runtime_artifacts.current_platform() == service_platform() == "macos-x86_64"


def test_signed_legacy_names_still_resolve() -> None:
    """Releases already signed with the old Ubuntu label keep installing."""
    assert canonical_platform("ubuntu-24.04-x86_64") == "linux-glibc-x86_64"
    assert platform_matches("ubuntu-24.04-x86_64", "linux-glibc-x86_64")
    assert platform_matches("ubuntu-24.04-x86_64", "linux-unknown-x86_64")
    assert platform_matches("linux-glibc-x86_64", "linux-unknown-x86_64")


def test_incompatible_architecture_remains_rejected() -> None:
    """Opening the family set cannot select x86_64 artifacts for an ARM host."""
    assert not platform_matches("linux-glibc-x86_64", "linux-glibc-aarch64")
    assert not platform_matches("ubuntu-24.04-x86_64", "linux-glibc-aarch64")
    assert not platform_matches("macos-arm64", "macos-x86_64")
