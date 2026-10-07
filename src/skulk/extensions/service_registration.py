"""Standalone standard-library helper for explicit local OS service registration.

Run this file directly with isolated base Python and site initialization disabled.
It never imports Skulk or a plugin as root and accepts no service commands, unit
contents, executable paths or storage destinations from its caller.
"""

import argparse
import contextlib
import grp
import json
import os
import platform
import plistlib
import pwd
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, final
from uuid import uuid4

# Preserve the signed/persisted Linux artifact identifier; it is not an OS-version gate.
ServicePlatform = str
"""The same derived family the runtime installer uses, spelled the same way.

This module is executed by file path in an unprivileged subprocess, so it may
import only the standard library and carries the derivation inline; a test
asserts it agrees with runtime_artifacts.current_platform(). It was a closed
pair of names, so the manager service could not register on any host the
runtime could not, and for the same reason."""

_LEGACY_PLATFORMS: dict[str, str] = {"ubuntu-24.04-x86_64": "linux-glibc-x86_64"}

# A reverse-DNS bundle identifier as Info.plist carries it; nothing else may
# reach the launchd definition from the app bundle.
_BUNDLE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,254}")


def associated_bundle(python: Path) -> str | None:
    """The identifier of the Mac app that ships ``python``, if it lives in one.

    macOS lists a login item by its program, so the plugin service would show as
    a bare interpreter name. launchd's ``AssociatedBundleIdentifiers`` lets it
    name the app instead when both are signed by the same team.

    Args:
        python: The absolute interpreter path the service will run.

    Returns:
        The enclosing ``.app`` bundle's ``CFBundleIdentifier``, or None for an
        interpreter outside an app bundle or a bundle without a valid one.
    """
    for parent in python.parents:
        if parent.suffix != ".app":
            continue
        try:
            info = cast(
                object,
                plistlib.loads((parent / "Contents" / "Info.plist").read_bytes()),
            )
        except (OSError, plistlib.InvalidFileException, ValueError):
            return None
        if not isinstance(info, dict):
            return None
        identifier = cast(dict[str, object], info).get("CFBundleIdentifier")
        if isinstance(identifier, str) and _BUNDLE_IDENTIFIER.fullmatch(identifier):
            return identifier
        return None
    return None


ServiceScope = Literal["system", "user"]
"""Where the manager service is registered.

``user`` is a launchd agent or systemd user unit with the same lifetime as the
Skulk node itself; it needs no elevation, so the node can set it up on its
own. ``system`` is a LaunchDaemon or systemd system service that also runs
without a login session; registering it needs one explicit local elevation.
"""


def _is_macos(family: str) -> bool:
    """launchd or systemd is decided by the operating system, not the CPU."""
    return _LEGACY_PLATFORMS.get(family, family).startswith("macos")


def service_platform() -> ServicePlatform:
    """Name this host's family; whether it is qualified is decided at install."""
    # The same normalization as runtime_artifacts.current_platform(): a hyphen
    # in the machine name would otherwise split the family into extra parts.
    machine = platform.machine().lower().replace("-", "_")
    architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if sys.platform == "darwin":
        return f"macos-{'arm64' if machine == 'arm64' else architecture}"
    if sys.platform == "linux":
        # Registration below drives /usr/bin/systemctl unconditionally, so a
        # Linux host without it must be refused here, before it is advertised
        # as supported, rather than at the first service command. The binary
        # alone is not enough: a container or a WSL install can carry it with
        # no manager running, and /run/systemd/system is what sd_booted()
        # checks for a live one.
        if not (
            os.path.exists("/usr/bin/systemctl")
            and os.path.exists("/run/systemd/system")
        ):
            raise ValueError("plugin services on Linux require a running systemd")
        library, _ = platform.libc_ver()
        return f"linux-{library or 'unknown'}-{architecture}"
    raise ValueError("plugin services are qualified on macOS and Linux only")


@final
@dataclass(frozen=True)
class ServiceLayout:
    """Fixed per-owner service identity and durable system-owned parent directory."""

    platform: ServicePlatform
    user_id: int
    group_id: int
    username: str
    groupname: str
    scope: ServiceScope = "system"
    user_state: Path | None = None
    """The owner's state directory that holds a user-scope root; unused for system."""
    user_units: Path | None = None
    """The owner's launchd agent or systemd user-unit directory; unused for system."""

    def __post_init__(self) -> None:
        if not 0 < self.user_id < 2147483648 or not 0 <= self.group_id < 2147483648:
            raise ValueError("service owner must be nonroot")
        if any(
            not name or len(name) > 128 or any(ord(c) < 32 for c in name)
            for name in (self.username, self.groupname)
        ):
            raise ValueError("invalid OS account identity")
        if self.scope == "user" and (
            self.user_state is None
            or self.user_units is None
            or not self.user_state.is_absolute()
            or not self.user_units.is_absolute()
        ):
            raise ValueError("a user-scope service needs absolute owner directories")

    @property
    def root(self) -> Path:
        """Return a stable service-owned leaf outside Git and login-session storage."""
        if self.scope == "user":
            assert self.user_state is not None
            return self.user_state / "plugin-service"
        parent = (
            Path("/Library/Application Support/SkulkPluginServices")
            if _is_macos(self.platform)
            else Path("/var/lib/skulk-plugin-services")
        )
        return parent / str(self.user_id)

    @property
    def label(self) -> str:
        """Return the reserved service name: per owner for system, fixed per user."""
        if self.scope == "user":
            return "foundation.foxlight.skulk.plugins"
        return f"foundation.foxlight.skulk.plugins.u{self.user_id}"

    @property
    def unit(self) -> Path:
        """Return the fixed registration file for this scope."""
        suffix = ".plist" if _is_macos(self.platform) else ".service"
        if self.scope == "user":
            assert self.user_units is not None
            return self.user_units / (self.label + suffix)
        if _is_macos(self.platform):
            return Path("/Library/LaunchDaemons") / (self.label + suffix)
        return Path("/etc/systemd/system") / (self.label + suffix)

    def definition(self, python: Path) -> bytes:
        """Render an exact nonroot OS definition with a fixed verified bootstrap.

        On macOS an interpreter inside an app bundle also names that bundle, so
        Login Items shows the app rather than the interpreter.
        """
        return self._render(
            python, associated_bundle(python) if _is_macos(self.platform) else None
        )

    def _render(self, python: Path, bundle: str | None) -> bytes:
        if not python.is_absolute() or any(ord(c) < 32 for c in str(python)):
            raise ValueError("invalid base interpreter path")
        arguments = [
            str(python),
            "-I",
            "-S",
            "-B",
            str(self.root / "service-bootstrap.py"),
            "--root",
            str(self.root),
        ]
        if _is_macos(self.platform):
            # A launchd agent already runs as its owner and may not name one.
            account = (
                {}
                if self.scope == "user"
                else {"UserName": self.username, "GroupName": self.groupname}
            )
            association: dict[str, object] = (
                {"AssociatedBundleIdentifiers": [bundle]} if bundle is not None else {}
            )
            return plistlib.dumps(
                {
                    "Label": self.label,
                    **account,
                    **association,
                    "ProgramArguments": arguments,
                    "WorkingDirectory": str(self.root),
                    "RunAtLoad": True,
                    "KeepAlive": True,
                    "ThrottleInterval": 10,
                    "ExitTimeOut": 180,
                    "Umask": 0o077,
                    "ProcessType": "Background",
                    "StandardOutPath": "/dev/null",
                    "StandardErrorPath": "/dev/null",
                },
                sort_keys=True,
            )
        command = " ".join(_systemd_argument(value) for value in arguments)
        if self.scope == "user":
            # A user unit runs as its owner under the user manager, which has
            # no local-fs ordering of its own and starts with the user's
            # session (or at boot when lingering is enabled, like Skulk's own
            # user unit).
            return (
                f"# skulk-plugin-service-v1 {json.dumps(str(python))}\n"
                "[Unit]\nDescription=Skulk managed plugin service\n"
                "StartLimitIntervalSec=0\n\n[Service]\nType=exec\n"
                f"WorkingDirectory={self.root}\nExecStart={command}\n"
                "Restart=always\nRestartSec=10\nTimeoutStopSec=180\nKillMode=mixed\n"
                "UMask=0077\nNoNewPrivileges=true\nLimitCORE=0\n"
                "StandardOutput=null\nStandardError=journal\n\n[Install]\nWantedBy=default.target\n"
            ).encode()
        return (
            f"# skulk-plugin-service-v1 {json.dumps(str(python))}\n"
            "[Unit]\nDescription=Skulk managed plugin service\nAfter=local-fs.target\n"
            "StartLimitIntervalSec=0\n\n[Service]\nType=exec\n"
            f"User={self.user_id}\nGroup={self.group_id}\n"
            # WorkingDirectory takes a path, not ExecStart's quoted argument list.
            # This root is fixed ASCII with only a numeric owner suffix.
            f"WorkingDirectory={self.root}\nExecStart={command}\n"
            "Restart=always\nRestartSec=10\nTimeoutStopSec=180\nKillMode=mixed\n"
            "UMask=0077\nNoNewPrivileges=true\nLimitCORE=0\n"
            "StandardOutput=null\nStandardError=journal\n\n[Install]\nWantedBy=multi-user.target\n"
        ).encode()

    def verify_existing(self, content: bytes) -> None:
        """Refuse an unrelated unit occupying the reserved name before any OS action."""
        if len(content) > 65536:
            raise ValueError("service definition exceeds bound")
        if _is_macos(self.platform):
            payload = cast(object, plistlib.loads(content))
            if not isinstance(payload, dict):
                raise ValueError("invalid existing service")
            arguments: object = cast(dict[str, object], payload).get("ProgramArguments")
            if (
                not isinstance(arguments, list)
                or not arguments
                or not isinstance(arguments[0], str)
            ):
                raise ValueError("invalid existing service arguments")
            python = arguments[0]
            # The bundle that named the existing agent may have moved or gone,
            # so compare against the association it was written with.
            existing: object = cast(dict[str, object], payload).get(
                "AssociatedBundleIdentifiers"
            )
            if existing is None:
                bundle = None
            elif (
                isinstance(existing, list)
                and len(cast(list[object], existing)) == 1
                and isinstance(cast(list[object], existing)[0], str)
                and _BUNDLE_IDENTIFIER.fullmatch(cast(list[str], existing)[0])
            ):
                bundle = cast(list[str], existing)[0]
            else:
                raise ValueError("invalid existing service association")
        else:
            bundle = None
            lines = content.decode().splitlines()
            if not lines:
                raise ValueError("existing service definition is empty")
            first = lines[0]
            prefix = "# skulk-plugin-service-v1 "
            if not first.startswith(prefix):
                raise ValueError("existing service is not managed by local setup")
            value = cast(object, json.loads(first[len(prefix) :]))
            if not isinstance(value, str):
                raise ValueError("invalid existing interpreter identity")
            python = value
        expected = self._render(Path(python), bundle)
        if not _is_macos(self.platform) and self.scope == "system":
            # Permit repair of only the exact prior generated unit. systemd
            # rejects its quoted WorkingDirectory before starting any process.
            previous = expected.replace(
                f"WorkingDirectory={self.root}\n".encode(),
                f'WorkingDirectory="{self.root}"\n'.encode(),
                1,
            )
            if content == previous:
                return
        if content != expected:
            raise ValueError("existing service definition differs from fixed contract")


def _systemd_argument(value: str) -> str:
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
        + '"'
    )


def _xdg_directory(variable: str, fallback: Path) -> Path:
    """An XDG base directory: the environment's when absolute, else the default."""
    value = os.environ.get(variable, "")
    return Path(value) if value and Path(value).is_absolute() else fallback


def local_layout(user_id: int, scope: ServiceScope = "system") -> ServiceLayout:
    """Resolve a local nonroot account using the OS database, not caller-supplied names.

    A user-scope layout places the service root and its agent or unit in the
    owner's own directories (the home directory from the account database,
    with the XDG overrides systemd itself honors on Linux).
    """
    account = pwd.getpwuid(user_id)
    family = service_platform()
    user_state: Path | None = None
    user_units: Path | None = None
    if scope == "user":
        home = Path(account.pw_dir)
        if _is_macos(family):
            user_state = home / "Library" / "Application Support" / "Skulk"
            user_units = home / "Library" / "LaunchAgents"
        else:
            user_state = (
                _xdg_directory("XDG_STATE_HOME", home / ".local" / "state") / "skulk"
            )
            user_units = (
                _xdg_directory("XDG_CONFIG_HOME", home / ".config") / "systemd" / "user"
            )
    return ServiceLayout(
        family,
        user_id,
        account.pw_gid,
        account.pw_name,
        grp.getgrgid(account.pw_gid).gr_name,
        scope,
        user_state,
        user_units,
    )


def _directory(path: Path) -> int:
    # Walk from / using directory descriptors; never follow an owner-controlled
    # link while the explicit setup helper has elevated filesystem authority.
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            created = False
            try:
                os.mkdir(part, 0o755, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            if created:
                # Explicitly set traversal regardless of the invoking owner's
                # restrictive umask; runtime leaves remain owner-only below.
                os.fchmod(child, 0o755)
                os.fsync(child)
                os.fsync(descriptor)
            info = os.fstat(child)
            if info.st_uid != 0 or info.st_mode & 0o022:
                os.close(child)
                raise ValueError("system service parent is not root protected")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _existing(layout: ServiceLayout) -> bytes | None:
    parent = _directory(layout.unit.parent)
    try:
        try:
            descriptor = os.open(
                layout.unit.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent
            )
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != 0
                or info.st_mode & 0o022
            ):
                raise ValueError("system service definition is not root protected")
            content = source.read(65537)
        layout.verify_existing(content)
        return content
    finally:
        os.close(parent)


def _environment(scope: ServiceScope = "system") -> dict[str, str]:
    """A fixed command environment; the user manager needs its session bus."""
    environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    if scope == "user":
        # systemctl --user reaches the owner's manager through the runtime
        # directory; it is the standard per-UID path when the caller (a
        # service, say) was started without it.
        environment["XDG_RUNTIME_DIR"] = os.environ.get(
            "XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"
        )
        bus = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        if bus:
            environment["DBUS_SESSION_BUS_ADDRESS"] = bus
    return environment


def _execute(
    arguments: tuple[str, ...],
    allowed: tuple[int, ...] = (0,),
    scope: ServiceScope = "system",
) -> None:
    result = subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=195,
        check=False,
        env=_environment(scope),
    )
    if result.returncode not in allowed:
        raise ValueError("OS service operation failed; inspect system service status")


def _stop_systemd(unit: str, scope: ServiceScope = "system") -> None:
    # A rejected generated unit reports "not loaded" (5), even though it has no
    # process to stop. Independently confirm quiescence before any replacement.
    manager = (
        ("/usr/bin/systemctl", "--user") if scope == "user" else ("/usr/bin/systemctl",)
    )
    _execute((*manager, "stop", unit), (0, 5), scope)
    result = subprocess.run(
        (
            *manager,
            "show",
            unit,
            "--property=ActiveState",
            "--property=MainPID",
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
        env=_environment(scope),
    )
    if result.returncode != 0 or set(result.stdout.splitlines()) not in (
        {b"ActiveState=inactive", b"MainPID=0"},
        {b"ActiveState=failed", b"MainPID=0"},
    ):
        raise ValueError("system service has not stopped")


def register(
    layout: ServiceLayout, action: Literal["prepare", "stop", "install"]
) -> None:
    """Perform one fixed root setup action; runtime and provider work stay nonroot."""
    if os.geteuid() != 0:
        raise ValueError("service registration requires explicit local elevation")
    if action == "prepare":
        parent = _directory(layout.root.parent)
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(layout.root.name, 0o700, dir_fd=parent)
            child = os.open(
                layout.root.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                info = os.fstat(child)
                if info.st_uid not in (0, layout.user_id) or info.st_mode & 0o077:
                    raise ValueError("existing service state ownership differs")
                if info.st_uid == 0 and os.listdir(child):
                    raise ValueError(
                        "refusing to adopt existing root-owned service data"
                    )
                os.fchown(child, layout.user_id, layout.group_id)
                os.fchmod(child, 0o700)
                os.fsync(child)
            finally:
                os.close(child)
            os.fsync(parent)
        finally:
            os.close(parent)
        return
    existing = _existing(layout)
    if action == "stop":
        if existing is None:
            return
        if _is_macos(layout.platform):
            _execute(
                ("/bin/launchctl", "bootout", "system/" + layout.label), (0, 3, 113)
            )
        else:
            _stop_systemd(layout.unit.name)
        return
    definition = layout.definition(Path(sys.executable).resolve(strict=True))
    parent = _directory(layout.unit.parent)
    temporary = "." + layout.unit.name + "." + uuid4().hex
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent,
        )
        with os.fdopen(descriptor, "wb") as target:
            os.fchmod(target.fileno(), 0o644)
            target.write(definition)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, layout.unit.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent)
    if _is_macos(layout.platform):
        _execute(("/bin/launchctl", "enable", "system/" + layout.label))
        # Setup stops the old service before activating a new copied runtime.
        _execute(("/bin/launchctl", "bootstrap", "system", str(layout.unit)))
    else:
        _execute(("/usr/bin/systemctl", "daemon-reload"))
        _execute(("/usr/bin/systemctl", "enable", "--now", layout.unit.name))


def _owned(info: os.stat_result, user_id: int) -> bool:
    return info.st_uid == user_id and not info.st_mode & 0o022


def _existing_user(layout: ServiceLayout) -> bytes | None:
    """Read and verify an existing user-scope definition, if any."""
    try:
        descriptor = os.open(layout.unit, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not _owned(info, layout.user_id):
            raise ValueError("user service definition is not owner protected")
        content = source.read(65537)
    layout.verify_existing(content)
    return content


def register_user(
    layout: ServiceLayout, action: Literal["prepare", "stop", "install"]
) -> None:
    """Perform one fixed user-scope setup action as the owner, with no elevation.

    The same three actions as :func:`register`, against the owner's own
    directories and service manager: a launchd agent in the owner's GUI domain
    (the domain Skulk's own agent uses) or a systemd user unit.
    """
    if layout.scope != "user":
        raise ValueError("system services register through the elevated helper")
    if os.geteuid() == 0 or os.getuid() != layout.user_id:
        raise ValueError("a user service registers as its owner, not as root")
    if action == "prepare":
        layout.root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):
            layout.root.mkdir(mode=0o700)
        info = layout.root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != layout.user_id
            or info.st_mode & 0o077
        ):
            raise ValueError("existing service state ownership differs")
        return
    existing = _existing_user(layout)
    domain = f"gui/{layout.user_id}"
    if action == "stop":
        if existing is None:
            return
        if _is_macos(layout.platform):
            _execute(
                ("/bin/launchctl", "bootout", f"{domain}/{layout.label}"),
                (0, 3, 113),
                "user",
            )
        else:
            _stop_systemd(layout.unit.name, "user")
        return
    definition = layout.definition(Path(sys.executable).resolve(strict=True))
    layout.unit.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = layout.unit.with_name("." + layout.unit.name + "." + uuid4().hex)
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
        )
        with os.fdopen(descriptor, "wb") as target:
            os.fchmod(target.fileno(), 0o644)
            target.write(definition)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, layout.unit)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
    if _is_macos(layout.platform):
        _execute(("/bin/launchctl", "enable", f"{domain}/{layout.label}"), scope="user")
        # Setup stops the old agent before activating a new copied runtime.
        _execute(
            ("/bin/launchctl", "bootstrap", domain, str(layout.unit)), scope="user"
        )
    else:
        _execute(("/usr/bin/systemctl", "--user", "daemon-reload"), scope="user")
        _execute(
            ("/usr/bin/systemctl", "--user", "enable", "--now", layout.unit.name),
            scope="user",
        )


def main() -> None:
    """Register only the fixed nonroot service for an explicitly selected local UID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "stop", "install"))
    parser.add_argument("--uid", type=int, required=True)
    arguments = parser.parse_args()
    try:
        register(
            local_layout(cast(int, arguments.uid)),
            cast(Literal["prepare", "stop", "install"], arguments.action),
        )
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        print(
            "Local plugin service registration failed; inspect owner and system service status.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
