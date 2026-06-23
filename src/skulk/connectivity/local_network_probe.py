"""Read-only macOS Local Network permission identity probe.

The probe records the process identity macOS is likely to attribute to a local
network access attempt, then runs Skulk's existing Local Network reachability
check. It is intentionally diagnostic-only: it does not alter privacy settings,
network services, Thunderbolt Bridge, RDMA state, or launchd configuration.
"""

from __future__ import annotations

import argparse
import platform
import plistlib
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast, final

import psutil
from pydantic import Field

from skulk.connectivity.local_network import (
    LocalNetworkStatus,
    check_local_network_access,
)
from skulk.utils.pydantic_ext import FrozenModel


@final
class AppBundleIdentity(FrozenModel):
    """Metadata read from an ancestor ``.app`` bundle, when present.

    Attributes:
        path: Absolute path to the discovered ``.app`` bundle.
        bundle_identifier: ``CFBundleIdentifier`` from ``Info.plist``.
        display_name: Human-facing ``CFBundleDisplayName`` from ``Info.plist``.
        bundle_name: ``CFBundleName`` from ``Info.plist``.
        executable_name: ``CFBundleExecutable`` from ``Info.plist``.
    """

    path: str = Field(description="Absolute path to the discovered .app bundle.")
    bundle_identifier: str | None = Field(
        default=None,
        description="CFBundleIdentifier from the bundle Info.plist.",
    )
    display_name: str | None = Field(
        default=None,
        description="CFBundleDisplayName from the bundle Info.plist.",
    )
    bundle_name: str | None = Field(
        default=None,
        description="CFBundleName from the bundle Info.plist.",
    )
    executable_name: str | None = Field(
        default=None,
        description="CFBundleExecutable from the bundle Info.plist.",
    )


@final
class ProcessIdentity(FrozenModel):
    """One process identity entry in the probe's process tree.

    Attributes:
        pid: Operating-system process identifier.
        ppid: Parent process identifier, when available.
        name: Process name reported by ``psutil``.
        executable: Absolute executable path reported by ``psutil``.
        command_line: Command-line vector reported by ``psutil``.
        app_bundle: Nearest ancestor ``.app`` bundle for ``executable``.
    """

    pid: int = Field(description="Operating-system process identifier.")
    ppid: int | None = Field(
        default=None,
        description="Parent process identifier, when available.",
    )
    name: str | None = Field(
        default=None,
        description="Process name reported by psutil.",
    )
    executable: str | None = Field(
        default=None,
        description="Absolute executable path reported by psutil.",
    )
    command_line: list[str] = Field(
        default_factory=list,
        description="Command-line vector reported by psutil.",
    )
    app_bundle: AppBundleIdentity | None = Field(
        default=None,
        description="Nearest ancestor .app bundle for executable.",
    )


@final
class LocalNetworkIdentityProbe(FrozenModel):
    """Read-only report for macOS Local Network permission attribution.

    Attributes:
        local_network_status: Result from Skulk's Local Network access check.
        platform_system: Operating system name from ``platform.system()``.
        macos_version: macOS version string on Darwin, otherwise ``None``.
        hostname: Local hostname.
        process: Identity for the current process.
        ancestors: Parent process chain, nearest parent first.
        notes: Human-facing interpretation hints.
    """

    local_network_status: LocalNetworkStatus = Field(
        description="Result from Skulk's Local Network access check.",
    )
    platform_system: str = Field(description="Operating system name.")
    macos_version: str | None = Field(
        default=None,
        description="macOS version string on Darwin, otherwise None.",
    )
    hostname: str = Field(description="Local hostname.")
    process: ProcessIdentity = Field(description="Identity for the current process.")
    ancestors: list[ProcessIdentity] = Field(
        description="Parent process chain, nearest parent first.",
    )
    notes: list[str] = Field(description="Human-facing interpretation hints.")


def _string_value(info: dict[object, object], key: str) -> str | None:
    value = info.get(key)
    return value if isinstance(value, str) and value else None


def _app_bundle_for_executable(executable: str | None) -> AppBundleIdentity | None:
    """Return metadata for the nearest containing ``.app`` bundle.

    Args:
        executable: Executable path to inspect.

    Returns:
        Bundle metadata when the executable lives inside a macOS app bundle;
        otherwise ``None``.
    """

    if executable is None:
        return None

    path = Path(executable).expanduser()
    for candidate in (path, *path.parents):
        if candidate.suffix != ".app":
            continue
        info_path = candidate / "Contents" / "Info.plist"
        info: dict[object, object] = {}
        if info_path.exists():
            try:
                with info_path.open("rb") as file:
                    raw_info = cast("object", plistlib.load(file))
                if isinstance(raw_info, dict):
                    info = cast("dict[object, object]", raw_info)
            except (OSError, plistlib.InvalidFileException):
                info = {}
        return AppBundleIdentity(
            path=str(candidate),
            bundle_identifier=_string_value(info, "CFBundleIdentifier"),
            display_name=_string_value(info, "CFBundleDisplayName"),
            bundle_name=_string_value(info, "CFBundleName"),
            executable_name=_string_value(info, "CFBundleExecutable"),
        )
    return None


def _process_identity(process: psutil.Process) -> ProcessIdentity:
    """Return a best-effort identity snapshot for ``process``.

    Args:
        process: Process to inspect.

    Returns:
        A strict, JSON-serialisable identity snapshot.
    """

    try:
        name = process.name()
    except (psutil.Error, OSError):
        name = None

    try:
        executable = process.exe() or None
    except (psutil.Error, OSError):
        executable = None

    try:
        command_line = process.cmdline()
    except (psutil.Error, OSError):
        command_line = []

    try:
        ppid = process.ppid()
    except (psutil.Error, OSError):
        ppid = None

    return ProcessIdentity(
        pid=process.pid,
        ppid=ppid,
        name=name,
        executable=executable,
        command_line=command_line,
        app_bundle=_app_bundle_for_executable(executable),
    )


def _ancestor_identities(process: psutil.Process, max_ancestors: int) -> list[ProcessIdentity]:
    ancestors: list[ProcessIdentity] = []
    parent = process.parent()
    while parent is not None and len(ancestors) < max_ancestors:
        ancestors.append(_process_identity(parent))
        parent = parent.parent()
    return ancestors


def _notes(
    *,
    local_network_status: LocalNetworkStatus,
    process: ProcessIdentity,
    ancestors: Sequence[ProcessIdentity],
) -> list[str]:
    notes = [
        "This probe is read-only except for one local TCP reachability attempt.",
        "It does not grant permissions or change Thunderbolt/RDMA configuration.",
    ]
    if local_network_status == "blocked":
        notes.append(
            "Local Network access appears blocked for this launch identity; "
            "macOS may show or require a grant for the process/app named above."
        )
    elif local_network_status == "ok":
        notes.append("Local Network access appears usable for this launch identity.")
    else:
        notes.append("Local Network status is inconclusive on this machine or OS.")

    if process.app_bundle is None:
        notes.append(
            "The current executable is not inside a .app bundle; prompts may name "
            "a launcher, terminal, Python, uv, or another runtime component."
        )

    app_ancestors = [item for item in ancestors if item.app_bundle is not None]
    if app_ancestors:
        notes.append(
            "At least one parent process belongs to a .app bundle; compare that "
            "bundle with the Local Network entry shown in System Settings."
        )

    return notes


_FRIENDLY_EXECUTABLE_LABELS = {"uv": "uv"}


def _friendly_executable_label(executable: str | None) -> str | None:
    """A human label for a bare executable (no .app), e.g. ``Python``."""
    if not executable:
        return None
    name = Path(executable).name
    if name.startswith("python"):
        # pythonNN / python3.13 / Python -> "Python" (how macOS surfaces it).
        return "Python"
    return _FRIENDLY_EXECUTABLE_LABELS.get(name, name)


def responsible_app_label(max_ancestors: int = 8) -> str | None:
    """Best guess at the app macOS attributes a Local Network grant to.

    macOS attributes local-network access to the "responsible" app in the launch
    chain, not necessarily the python binary: a GUI terminal/launcher holds the
    grant when present, otherwise the executable itself does. This returns the
    display name of the nearest ancestor ``.app`` bundle (the terminal that
    launched Skulk), or a friendly executable name (``"Python"`` for any
    ``pythonNN``) when there is no ``.app`` in the chain (SSH / launchd /
    headless), which is exactly the identity a user must enable in System
    Settings. ``None`` when the process tree cannot be read. Does NOT touch the
    network, so it is cheap and safe to call alongside the reachability check.
    """
    try:
        current = psutil.Process()
    except (psutil.Error, OSError):
        return None
    identity = _process_identity(current)
    for item in (identity, *_ancestor_identities(current, max(0, max_ancestors))):
        if item.app_bundle is not None:
            bundle = item.app_bundle
            return (
                bundle.display_name
                or bundle.bundle_name
                or Path(bundle.path).stem
            )
    return _friendly_executable_label(identity.executable)


def collect_local_network_identity_probe(
    max_ancestors: int = 8,
) -> LocalNetworkIdentityProbe:
    """Collect a read-only macOS Local Network identity probe.

    Args:
        max_ancestors: Maximum number of parent processes to include.

    Returns:
        A structured probe report.
    """

    current_process = psutil.Process()
    max_ancestors = max(0, max_ancestors)
    process = _process_identity(current_process)
    ancestors = _ancestor_identities(current_process, max_ancestors)
    local_network_status = check_local_network_access()
    system = platform.system()
    return LocalNetworkIdentityProbe(
        local_network_status=local_network_status,
        platform_system=system,
        macos_version=platform.mac_ver()[0] if system == "Darwin" else None,
        hostname=socket.gethostname(),
        process=process,
        ancestors=ancestors,
        notes=_notes(
            local_network_status=local_network_status,
            process=process,
            ancestors=ancestors,
        ),
    )


def _format_bundle(bundle: AppBundleIdentity | None) -> str:
    if bundle is None:
        return "none"
    labels = [
        f"path={bundle.path}",
        f"id={bundle.bundle_identifier or 'unknown'}",
        f"name={bundle.display_name or bundle.bundle_name or 'unknown'}",
    ]
    return ", ".join(labels)


def _format_process(identity: ProcessIdentity) -> str:
    executable = identity.executable or "unknown"
    name = identity.name or "unknown"
    return (
        f"pid={identity.pid} ppid={identity.ppid or 'unknown'} "
        f"name={name} executable={executable} app_bundle=({_format_bundle(identity.app_bundle)})"
    )


def format_local_network_identity_probe(report: LocalNetworkIdentityProbe) -> str:
    """Format a probe report for humans.

    Args:
        report: Structured probe report to format.

    Returns:
        Multi-line human-readable text.
    """

    lines = [
        "Skulk macOS Local Network identity probe",
        "",
        f"Local Network status: {report.local_network_status}",
        f"Platform: {report.platform_system}",
        f"macOS version: {report.macos_version or 'n/a'}",
        f"Hostname: {report.hostname}",
        "",
        "Current process:",
        f"  {_format_process(report.process)}",
    ]
    if report.process.command_line:
        lines.append(f"  argv: {report.process.command_line}")

    lines.append("")
    lines.append("Parent processes:")
    if report.ancestors:
        for index, ancestor in enumerate(report.ancestors, start=1):
            lines.append(f"  {index}. {_format_process(ancestor)}")
            if ancestor.command_line:
                lines.append(f"     argv: {ancestor.command_line}")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("Notes:")
    lines.extend(f"  - {note}" for note in report.notes)
    return "\n".join(lines)


def run_local_network_identity_probe(
    *,
    json_output: bool = False,
    fail_on_blocked: bool = False,
    max_ancestors: int = 8,
) -> int:
    """Run the probe, print the report, and return a process exit status.

    Args:
        json_output: Print JSON instead of human-readable text.
        fail_on_blocked: Return status 2 when Local Network access is blocked.
        max_ancestors: Maximum number of parent processes to include.

    Returns:
        ``0`` on a completed probe, or ``2`` when ``fail_on_blocked`` is true
        and the probe detects a Local Network denial.
    """

    report = collect_local_network_identity_probe(max_ancestors=max_ancestors)
    if json_output:
        print(report.model_dump_json(indent=2))
    else:
        print(format_local_network_identity_probe(report))
    return 2 if fail_on_blocked and report.local_network_status == "blocked" else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe as a standalone command-line utility.

    Args:
        argv: Optional argument vector, excluding program name.

    Returns:
        Process exit status.
    """

    parser = argparse.ArgumentParser(
        prog="skulk-macos-local-network-probe",
        description="Read-only macOS Local Network permission identity probe.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit with status 2 when Local Network access appears blocked.",
    )
    parser.add_argument(
        "--max-ancestors",
        type=int,
        default=8,
        help="Maximum number of parent processes to include.",
    )
    parsed_values = cast("dict[str, object]", vars(parser.parse_args(argv)))
    max_ancestors_value = parsed_values.get("max_ancestors", 8)
    max_ancestors = (
        max_ancestors_value if isinstance(max_ancestors_value, int) else 8
    )
    return run_local_network_identity_probe(
        json_output=parsed_values.get("json") is True,
        fail_on_blocked=parsed_values.get("fail_on_blocked") is True,
        max_ancestors=max_ancestors,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
