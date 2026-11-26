"""An artifact family is derived from the host, never enumerated in advance.

Two closed pairs of names, one in the runtime installer and one in the manager
service registration, meant a Grace Blackwell node on Linux aarch64 could not
install a capability runtime or register the service. These tests pin the
derived vocabulary, the names already written into signed releases, the
agreement between the two derivations, and the one property the old closed set
guarded: x86_64 artifacts are never taken for an aarch64 host.
"""

import os
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from skulk.extensions import runtime_artifacts
from skulk.extensions.runtime_artifacts import canonical_platform, platform_matches
from skulk.extensions.service_registration import service_platform


def _linux(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    libc: str = "glibc",
    systemd: bool = True,
    systemd_running: bool = True,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(platform, "libc_ver", lambda: (libc, "2.39" if libc else ""))
    real_exists = os.path.exists

    def exists(path: str) -> bool:
        if path == "/usr/bin/systemctl":
            return systemd
        if path == "/run/systemd/system":
            return systemd_running
        return real_exists(path)

    monkeypatch.setattr("skulk.extensions.service_registration.os.path.exists", exists)


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


@pytest.mark.parametrize(
    ("built_for", "family"),
    [("win-amd64", "windows-x86_64"), ("win-arm64", "windows-aarch64")],
)
def test_a_windows_host_has_a_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, built_for: str, family: str
) -> None:
    """The name describes the machine, as the interpreter was built for it.

    It is what lets a Windows node keep its own plugin service; it does not
    say that anything is published for one. ``platform.machine()`` is not
    asked: for an ordinary account it answers nothing in a process started
    with a fixed environment, which is how the plugin manager is started.
    """
    _synthetic_core(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sysconfig, "get_platform", lambda: built_for)
    monkeypatch.setattr(platform, "machine", lambda: "")
    assert runtime_artifacts.current_platform() == service_platform() == family
    assert runtime_artifacts.measure_host().platform == family
    # The name passes the family pattern every signed claim is held to.
    assert runtime_artifacts.QualifiedHost(family, "3.13.13", "1.5.2", "a" * 64)


@pytest.mark.parametrize("built_for", ["win32", "win-arm32"])
def test_a_windows_python_skulk_does_not_name_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, built_for: str
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sysconfig, "get_platform", lambda: built_for)
    with pytest.raises(ValueError, match="64-bit Python"):
        runtime_artifacts.current_platform()
    with pytest.raises(ValueError, match="64-bit Python"):
        service_platform()


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows machine name")
def test_a_windows_host_is_named_the_same_whatever_the_environment_holds() -> None:
    """A child with a fixed environment names the family this process names.

    That child is how the plugin manager runs and how a service copy is asked
    what it is; a different answer there fails every setup.
    """
    family = runtime_artifacts.current_platform()
    assert family in {"windows-x86_64", "windows-aarch64"}
    assert service_platform() == family
    system = os.environ.get("SYSTEMROOT", r"C:\Windows")
    child = subprocess.run(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-I",
            "-c",
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('r', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules['r'] = module\n"
            "spec.loader.exec_module(module)\n"
            "print(module.service_platform())\n",
            str(Path(runtime_artifacts.__file__).with_name("service_registration.py")),
        ],
        env={"SystemRoot": system, "PATH": str(Path(system) / "System32")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == family


def test_a_windows_family_matches_no_other_system_and_no_other_machine() -> None:
    """Naming Windows cannot select a macOS or Linux artifact for it."""
    assert platform_matches("windows-x86_64", "windows-x86_64")
    assert not platform_matches("windows-x86_64", "windows-aarch64")
    for published in ("macos-arm64", "macos-x86_64", "linux-glibc-x86_64"):
        assert not platform_matches(published, "windows-x86_64")
    # Not the old Ubuntu label either, which matches every Linux C library.
    assert not platform_matches("ubuntu-24.04-x86_64", "windows-x86_64")


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


def test_other_operating_systems_are_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Architecture and libc are open; the operating system is not.

    Skulk runs on macOS, Linux and Windows, and service registration knows
    launchd, systemd and Task Scheduler. Naming another system would claim a
    host nothing serves and refuse it later with a worse message.
    """
    monkeypatch.setattr(sys, "platform", "freebsd14")
    monkeypatch.setattr(platform, "machine", lambda: "amd64")
    with pytest.raises(ValueError, match="macOS and Linux"):
        runtime_artifacts.current_platform()
    with pytest.raises(ValueError, match="macOS and Linux"):
        service_platform()


def test_linux_without_systemd_is_refused_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration drives systemctl, so a host without it is refused up front."""
    _linux(monkeypatch, "aarch64", systemd=False)
    with pytest.raises(ValueError, match="systemd"):
        service_platform()
    # The binary alone does not prove a manager: a container can carry
    # systemctl while nothing answers it, and setup would fail at daemon-reload
    # after privileged preparation had begun.
    _linux(monkeypatch, "aarch64", systemd_running=False)
    with pytest.raises(ValueError, match="running systemd"):
        service_platform()
