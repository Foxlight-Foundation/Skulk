"""Standalone standard-library helper for explicit local OS service registration.

Run this file directly with isolated base Python and site initialization disabled.
It never imports Skulk or a plugin as root and accepts no service commands, unit
contents, executable paths or storage destinations from its caller.
"""

import argparse
import contextlib
import json
import os
import platform
import plistlib
import re
import stat
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, final
from uuid import uuid4
from xml.etree import ElementTree
from xml.sax.saxutils import escape

if sys.platform != "win32":
    # POSIX account databases. Windows registration is separate work; guarding
    # the import keeps this module loadable there for the node's API.
    import grp
    import pwd

# Preserve the signed/persisted Linux artifact identifier; it is not an OS-version gate.
ServicePlatform = str
"""The same derived family the runtime installer uses, spelled the same way.

This module is executed by file path in an unprivileged subprocess, so it may
import only the standard library and carries the derivation inline; a test
asserts it agrees with runtime_artifacts.current_platform(). It was a closed
pair of names, so the manager service could not register on any host the
runtime could not, and for the same reason."""

_LEGACY_PLATFORMS: dict[str, str] = {"ubuntu-24.04-x86_64": "linux-glibc-x86_64"}

_UNREGISTRABLE = "plugin services are qualified on macOS and Linux only"

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
    if sys.platform == "win32":
        # The machine's name only, asked of the interpreter and not of the
        # environment; see runtime_artifacts.current_platform().
        built_for = {"win-amd64": "x86_64", "win-arm64": "aarch64"}.get(
            sysconfig.get_platform()
        )
        if built_for is None:
            raise ValueError("Skulk on Windows needs a 64-bit Python (x86-64 or ARM64)")
        return f"windows-{built_for}"
    raise ValueError(_UNREGISTRABLE)


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
                    # This service owns interactive inference and management,
                    # including short-lived schema workers. Background launchd
                    # scheduling is inherited and can starve valid admissions
                    # past their deadlines even when CPU is available.
                    "ProcessType": "Interactive",
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
        if _is_macos(self.platform):
            # Scheduling changes must not prevent replacing our own old unit.
            # Accept only the exact previous renderer; owner, bootstrap, paths,
            # permissions and every other field still match the fixed contract.
            previous = cast(dict[str, object], plistlib.loads(expected))
            previous["ProcessType"] = "Background"
            if content == plistlib.dumps(previous, sort_keys=True):
                return
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


_TASK_NAMESPACE = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
_TASK_MARKER = "skulk-plugin-service-v1 "
_TASK_INSTANCES_FIELD = "Settings/MultipleInstancesPolicy"
_TASK_INSTANCES = "Parallel"
# What local setup wrote before a start was allowed beside a running one. A
# task that differs from today's in this one value is still local setup's
# own, so that setup can replace it; it is not the registered service.
_TASK_INSTANCES_BEFORE = "IgnoreNew"
_ACCOUNT_IDENTIFIER = re.compile(r"S-1-[0-9]+(-[0-9]+){1,14}")

# What Task Scheduler leaves out of the copy it hands back because it is the
# default. They are dropped from both sides of a comparison, so a Windows
# that prints them compares the same as one that does not.
_TASK_DEFAULTS = frozenset(
    {
        ("Principals/Principal/RunLevel", "LeastPrivilege"),
        ("Settings/AllowHardTerminate", "true"),
        ("Settings/AllowStartOnDemand", "true"),
        ("Settings/Enabled", "true"),
        ("Settings/Hidden", "false"),
        ("Settings/RunOnlyIfIdle", "false"),
        ("Settings/RunOnlyIfNetworkAvailable", "false"),
        ("Settings/WakeToRun", "false"),
        ("Triggers/TimeTrigger/Enabled", "true"),
        ("Triggers/TimeTrigger/Repetition/StopAtDurationEnd", "false"),
    }
)


def task_fields(definition: str) -> tuple[tuple[str, str], ...]:
    """Everything a Task Scheduler definition says, in an order-free form.

    Task Scheduler does not keep the bytes it was given: it reorders the
    elements, drops defaults and adds the task's own name. Two definitions
    are the same task exactly when this function gives the same answer for
    both.

    Args:
        definition: A task definition as text, as written here or as Task
            Scheduler hands it back.

    Returns:
        Every attribute and every element that holds text, as sorted
        ``(path, value)`` pairs, without the task's own name and without
        values that are the default.

    Raises:
        ValueError: The text is not one task definition.
    """
    if len(definition) > 65536:
        raise ValueError("service definition exceeds bound")
    # The declaration names the encoding of bytes this text no longer is.
    body = (
        definition.partition("?>")[2] if definition.startswith("<?xml") else definition
    )
    try:
        task = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        raise ValueError("invalid service definition") from None
    if task.tag != _TASK_NAMESPACE + "Task":
        raise ValueError("invalid service definition")
    fields: list[tuple[str, str]] = []

    def walk(element: ElementTree.Element, path: str) -> None:
        for name, value in element.attrib.items():
            fields.append((f"{path}@{name}", value))
        children = list(element)
        if not children:
            fields.append((path, element.text or ""))
        for child in children:
            if not child.tag.startswith(_TASK_NAMESPACE):
                raise ValueError("invalid service definition")
            name = child.tag[len(_TASK_NAMESPACE) :]
            walk(child, f"{path}/{name}" if path else name)

    walk(task, "")
    return tuple(
        sorted(
            field
            for field in fields
            if field not in _TASK_DEFAULTS and field[0] != "RegistrationInfo/URI"
        )
    )


@final
@dataclass(frozen=True)
class WindowsServiceLayout:
    """The plugin service of one Windows account: a task it registers itself.

    Windows has no per-user service manager. Task Scheduler runs a task as the
    account that registered it, in that account's signed-in session and with
    its ordinary rights, and the account needs no elevation to register one.
    A service for the whole machine is not offered on Windows.
    """

    account: str
    """The account's security identifier as text, never its name."""
    user_state: Path
    """The folder Windows keeps for the account's machine-local program data."""

    def __post_init__(self) -> None:
        if not _ACCOUNT_IDENTIFIER.fullmatch(self.account):
            raise ValueError("invalid OS account identity")
        if not self.user_state.is_absolute():
            raise ValueError("a user-scope service needs absolute owner directories")

    @property
    def scope(self) -> ServiceScope:
        """Always the account's own service."""
        return "user"

    @property
    def root(self) -> Path:
        """Return the service's own folder.

        The name is short on purpose: a machine without long paths switched on
        limits how deep the copy below it may go.
        """
        return self.user_state / "Skulk" / "plugins"

    @property
    def label(self) -> str:
        """Return the task's reserved name, which names the account.

        Task names are one namespace for the whole machine, whoever a task
        runs as, so two accounts on one machine need two names.
        """
        return f"foundation.foxlight.skulk.plugins.{self.account}"

    @staticmethod
    def launcher(python: Path) -> Path:
        """The program the task runs: the interpreter without a console window.

        A task runs in the signed-in session, where ``python.exe`` would open
        a console window for as long as the service runs. ``pythonw.exe``
        beside it is the same interpreter with no console.
        """
        return python.with_name("pythonw.exe")

    def definition(self, python: Path) -> bytes:
        """Render the exact task for ``python``, as the file ``schtasks`` reads.

        The task starts the fixed verified bootstrap and nothing else. It is
        started every minute: Task Scheduler starts a task again only after
        a failure, and the manager also ends cleanly when it wants to be
        started again, so the minute is what restarts it. The same trigger
        starts it within a minute of the account signing in.

        A start is allowed beside a running one. The bootstrap's own lock
        decides which is the service, and a start that finds it held ends at
        once with success. Task Scheduler could be told to refuse the second
        start itself, and was at first: it then shows the refusal
        (``0x800710E0``) as the task's last result, beside a healthy service,
        for as long as the service runs. Told to keep the start waiting, it
        keeps one more every minute, without limit.
        """
        if not python.is_absolute() or any(ord(c) < 32 for c in str(python)):
            raise ValueError("invalid base interpreter path")
        arguments = subprocess.list2cmdline(
            [
                "-I",
                "-S",
                "-B",
                str(self.root / "service-bootstrap.py"),
                "--root",
                str(self.root),
            ]
        )
        description = _TASK_MARKER + json.dumps(str(python))
        return (
            '<?xml version="1.0" encoding="UTF-16"?>\n'
            '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
            f"<RegistrationInfo><Description>{escape(description)}</Description></RegistrationInfo>\n"
            "<Triggers><TimeTrigger>"
            "<StartBoundary>2026-01-01T00:00:00</StartBoundary>"
            "<Repetition><Interval>PT1M</Interval></Repetition>"
            "</TimeTrigger></Triggers>\n"
            '<Principals><Principal id="Author">'
            f"<UserId>{self.account}</UserId>"
            "<LogonType>InteractiveToken</LogonType>"
            "</Principal></Principals>\n"
            "<Settings>"
            f"<MultipleInstancesPolicy>{_TASK_INSTANCES}</MultipleInstancesPolicy>"
            "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>"
            "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>"
            "<StartWhenAvailable>true</StartWhenAvailable>"
            "<IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd>"
            "<RestartOnIdle>false</RestartOnIdle></IdleSettings>"
            # No time limit, and ordinary priority: a task's default is
            # below normal, which also lowers its disk and memory priority.
            "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>"
            "<Priority>4</Priority>"
            "</Settings>\n"
            '<Actions Context="Author"><Exec>'
            f"<Command>{escape(str(self.launcher(python)))}</Command>"
            f"<Arguments>{escape(arguments)}</Arguments>"
            f"<WorkingDirectory>{escape(str(self.root))}</WorkingDirectory>"
            "</Exec></Actions>\n"
            "</Task>\n"
        ).encode("utf-16")

    def verify_existing(self, content: str) -> None:
        """Refuse an unrelated task under the reserved name before any OS action.

        Args:
            content: The task as Task Scheduler hands it back.

        Raises:
            ValueError: The task was not written by local setup, or differs
                from the one local setup writes for the interpreter it names.
                A task that is only switched off is still ours: stopping the
                service switches it off. So is one that differs only in
                refusing a second start, as local setup first wrote it.
        """
        existing = task_fields(content)
        described = [
            value for path, value in existing if path == "RegistrationInfo/Description"
        ]
        if len(described) != 1 or not described[0].startswith(_TASK_MARKER):
            raise ValueError("existing service is not managed by local setup")
        try:
            python = cast(object, json.loads(described[0][len(_TASK_MARKER) :]))
        except ValueError:
            raise ValueError("invalid existing interpreter identity") from None
        if not isinstance(python, str):
            raise ValueError("invalid existing interpreter identity")
        expected = task_fields(self.definition(Path(python)).decode("utf-16"))
        found = tuple(
            (path, _TASK_INSTANCES)
            if (path, value) == (_TASK_INSTANCES_FIELD, _TASK_INSTANCES_BEFORE)
            else (path, value)
            for path, value in existing
            if (path, value) != ("Settings/Enabled", "false")
        )
        if found != expected:
            raise ValueError("existing service definition differs from fixed contract")


OwnerLayout = ServiceLayout | WindowsServiceLayout
"""The layout of an account's own service on whichever system this is."""


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


def registrable_scopes() -> tuple[ServiceScope, ...]:
    """The scopes a service can be registered in on this system.

    Windows offers the account's own service only.
    """
    return ("user",) if sys.platform == "win32" else ("user", "system")


def owner_layout(scope: ServiceScope) -> OwnerLayout:
    """The layout of the service that belongs to the account running this process.

    Args:
        scope: Where that service is, or would be, registered.

    Returns:
        The fixed layout for this account in that scope.

    Raises:
        ValueError: On a host where the plugin service cannot be registered,
            or in a scope this system does not offer.
    """
    if sys.platform == "win32":
        # Reached only as the account itself and from Skulk, never by path as
        # another user: Windows registers nothing with elevation.
        from skulk.extensions import _service_registration_windows as windows

        return windows.owner_layout(scope)
    return local_layout(os.getuid(), scope)


def registered_for(layout: OwnerLayout, base: Path, *, protected: bool = False) -> bool:
    """Whether the registered OS definition is exactly the one that starts ``base``.

    The definition is read as itself, never through a link.

    Args:
        layout: The service whose registration is read.
        base: The interpreter the definition must invoke.
        protected: Also require that only the definition's rightful owner can
            change it: root for a system service, the account for a user
            service. On Windows the definition is not a file: Task Scheduler
            holds it and decides who may change it, so there is nothing more
            to examine.

    Returns:
        False when nothing is registered, when the definition cannot be read
        or is not protected as asked, and when it differs in any byte.

    Raises:
        ValueError: When ``base`` is not a path a definition may name.
    """
    if isinstance(layout, WindowsServiceLayout):
        if sys.platform != "win32":
            return False
        from skulk.extensions import _service_registration_windows as windows

        return windows.registered_for(layout, base)
    if sys.platform == "win32":
        # A launchd or systemd definition is no service of this system's.
        return False
    try:
        descriptor = os.open(layout.unit, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    with os.fdopen(descriptor, "rb") as source:
        if protected:
            info = os.fstat(source.fileno())
            owner = 0 if layout.scope == "system" else os.getuid()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != owner
                or info.st_mode & 0o022
            ):
                return False
        return source.read(65537) == layout.definition(base)


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
    layout: OwnerLayout, action: Literal["prepare", "stop", "install"]
) -> None:
    """Perform one fixed user-scope setup action as the owner, with no elevation.

    The same three actions as :func:`register`, against the owner's own
    directories and service manager: a launchd agent in the owner's GUI domain
    (the domain Skulk's own agent uses), a systemd user unit, or on Windows a
    Task Scheduler task.
    """
    if isinstance(layout, WindowsServiceLayout):
        if sys.platform != "win32":
            raise ValueError(_UNREGISTRABLE)
        from skulk.extensions import _service_registration_windows as windows

        windows.register_user(layout, action)
        return
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
