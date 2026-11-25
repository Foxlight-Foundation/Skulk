"""The plugin service as a Windows Task Scheduler task: its definition and its life.

Three kinds of test. The definition tests are pure and run everywhere. The
Windows tests register against a stand-in for Task Scheduler but start the real
startup file, without a console as Task Scheduler starts it, on a real sealed
copy, and stop it the way local setup does. The live tests, which run only
where ``SKULK_LIVE_TASK_SCHEDULER`` is set, do the same against the real Task
Scheduler under a name of their own, and delete what they register.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

import pytest

from skulk.extensions import service_bootstrap, service_registration
from skulk.extensions.runtime_artifacts import QualifiedHost
from skulk.extensions.runtime_files import RuntimeLock
from skulk.extensions.service_registration import (
    WindowsServiceLayout,
    registrable_scopes,
    task_fields,
)
from skulk.extensions.service_snapshot import activate_service_runtime
from skulk.extensions.tests.test_service_snapshot import staged

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="registers with Windows Task Scheduler"
)
live = pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("SKULK_LIVE_TASK_SCHEDULER"),
    reason="registers a real scheduled task; set SKULK_LIVE_TASK_SCHEDULER to run",
)

_NAMESPACE = "http://schemas.microsoft.com/windows/2004/02/mit/task"
_ACCOUNT = "S-1-5-21-1111111111-2222222222-3333333333-1001"
_LIVE_NAME = "foundation.foxlight.skulk.plugins.test"

# A manager that says in the service root that it is ready, and when the
# console asks it to stop says so there too and ends with a code of its own.
_SERVES = (
    "import os, signal, sys, time\n"
    "root = sys.argv[-1]\n"
    "def stop(*_):\n"
    "    open(os.path.join(root, 'asked'), 'w').close()\n"
    "    raise SystemExit(5)\n"
    "signal.signal(signal.SIGBREAK, stop)\n"
    "open(os.path.join(root, 'ready'), 'w').close()\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)
# A manager that hears the request and carries on regardless.
_WILL_NOT_STOP = _SERVES.replace("    raise SystemExit(5)\n", "")
# A manager that also ends cleanly, by itself, when a file asks it to: what
# the real manager does after it has selected its successor.
_RELOADS = _SERVES.replace(
    "while True:\n    time.sleep(0.05)\n",
    "while not os.path.exists(os.path.join(root, 'reload')):\n"
    "    time.sleep(0.05)\n"
    # The real manager stops half a second after it has answered.
    "time.sleep(0.5)\n"
    "os.remove(os.path.join(root, 'ready'))\n"
    "os.remove(os.path.join(root, 'reload'))\n",
)


def _as_the_scheduler_keeps_it(definition: str, name: str, enabled: bool = True) -> str:
    """Rewrite a definition the way Task Scheduler does before handing it back.

    Seen on Windows 11: the sections come back in another order, the task's
    own name is added, and the namespace may be spelled with a prefix. A task
    that was switched off says so in its settings.
    """
    task = ElementTree.fromstring(definition.partition("?>")[2])
    sections = list(task)
    for section in sections:
        task.remove(section)
    for section in reversed(sections):
        task.append(section)
    registration = task.find(f"{{{_NAMESPACE}}}RegistrationInfo")
    assert registration is not None
    ElementTree.SubElement(registration, f"{{{_NAMESPACE}}}URI").text = "\\" + name
    if not enabled:
        settings = task.find(f"{{{_NAMESPACE}}}Settings")
        assert settings is not None
        ElementTree.SubElement(settings, f"{{{_NAMESPACE}}}Enabled").text = "false"
    return '<?xml version="1.0" encoding="UTF-16"?>\n' + ElementTree.tostring(
        task, encoding="unicode"
    )


def _layout(state: Path) -> WindowsServiceLayout:
    return WindowsServiceLayout(_ACCOUNT, state)


def test_a_task_reads_the_same_after_the_scheduler_rewrites_it(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    python = tmp_path / "base" / "python.exe"
    written = layout.definition(python).decode("utf-16")
    kept = _as_the_scheduler_keeps_it(written, layout.label)
    assert kept != written
    assert task_fields(kept) == task_fields(written)
    fields = dict(task_fields(written))
    assert fields["Actions/Exec/Command"] == str(python.with_name("pythonw.exe"))
    assert fields["Principals/Principal/UserId"] == _ACCOUNT
    assert fields["Principals/Principal/LogonType"] == "InteractiveToken"
    assert fields["Settings/MultipleInstancesPolicy"] == "IgnoreNew"
    assert fields["Triggers/TimeTrigger/Repetition/Interval"] == "PT1M"
    # Nothing asks for more than the account's ordinary rights.
    assert "Principals/Principal/RunLevel" not in fields


def test_a_task_names_one_interpreter_and_one_root(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    written = layout.definition(tmp_path / "a" / "python.exe").decode("utf-16")
    other = layout.definition(tmp_path / "b" / "python.exe").decode("utf-16")
    assert task_fields(written) != task_fields(other)
    elsewhere = _layout(tmp_path / "elsewhere")
    assert task_fields(written) != task_fields(
        elsewhere.definition(tmp_path / "a" / "python.exe").decode("utf-16")
    )
    assert layout.root == tmp_path / "Skulk" / "plugins"
    assert layout.scope == "user"
    # Task names are one namespace for the whole machine: the name carries
    # the account, so another account's service is another task.
    assert layout.label == f"foundation.foxlight.skulk.plugins.{_ACCOUNT}"
    another = WindowsServiceLayout(_ACCOUNT[:-4] + "1002", tmp_path)
    assert another.label != layout.label


def test_a_path_keeps_every_character_through_the_definition(tmp_path: Path) -> None:
    """A profile folder may hold spaces, ``&`` and letters outside ASCII."""
    layout = _layout(tmp_path / "R&D <1> Jürgen 日本")
    python = tmp_path / "Python & more" / "python.exe"
    fields = dict(task_fields(layout.definition(python).decode("utf-16")))
    assert fields["Actions/Exec/WorkingDirectory"] == str(layout.root)
    assert fields["Actions/Exec/Command"] == str(python.with_name("pythonw.exe"))
    assert fields["Actions/Exec/Arguments"] == subprocess.list2cmdline(
        [
            "-I",
            "-S",
            "-B",
            str(layout.root / "service-bootstrap.py"),
            "--root",
            str(layout.root),
        ]
    )


def test_only_local_setups_own_task_is_accepted_under_the_name(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    python = tmp_path / "python.exe"
    written = layout.definition(python).decode("utf-16")
    layout.verify_existing(_as_the_scheduler_keeps_it(written, layout.label))
    # Stopping the service switches its task off; it is still ours.
    layout.verify_existing(_as_the_scheduler_keeps_it(written, layout.label, False))
    with pytest.raises(ValueError, match="not managed by local setup"):
        layout.verify_existing(written.replace("skulk-plugin-service-v1", "another"))
    with pytest.raises(ValueError, match="differs from fixed contract"):
        layout.verify_existing(written.replace("-I -S -B", "-I -B"))
    with pytest.raises(ValueError, match="differs from fixed contract"):
        layout.verify_existing(
            written.replace(
                "<LogonType>",
                "<RunLevel>HighestAvailable</RunLevel><LogonType>",
            )
        )
    with pytest.raises(ValueError, match="invalid service definition"):
        layout.verify_existing("<Task>")


def test_a_layout_refuses_what_is_not_an_account_or_a_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="account"):
        WindowsServiceLayout("thoma", tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        WindowsServiceLayout(_ACCOUNT, Path("relative"))
    with pytest.raises(ValueError, match="interpreter"):
        _layout(tmp_path).definition(Path("python.exe"))


class _Scheduler:
    """A stand-in for Task Scheduler: what ``schtasks.exe`` does, in memory.

    It keeps what it was given the way the real one does (rewritten), starts
    a task by running exactly the command the definition names, and ends one
    by ending that program and nothing else.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, str] = {}
        self.disabled: set[str] = set()
        self.readable = True
        self.ends = True
        self.calls: list[str] = []
        self.started: list[subprocess.Popen[bytes]] = []

    def _answer(
        self, arguments: list[str], code: int, output: str = ""
    ) -> "subprocess.CompletedProcess[bytes]":
        return subprocess.CompletedProcess(arguments, code, output.encode("utf-8"), b"")

    def run(self, arguments: list[str]) -> "subprocess.CompletedProcess[bytes]":
        if Path(arguments[0]).name.lower() == "cmd.exe":
            found = re.search(r"/Query /TN (\S+) /XML$", arguments[-1])
            assert found is not None
            name = found.group(1)
            if name not in self.tasks or not self.readable:
                return self._answer(arguments, 1)
            return self._answer(
                arguments,
                0,
                _as_the_scheduler_keeps_it(
                    self.tasks[name], name, name not in self.disabled
                ),
            )
        assert Path(arguments[0]).name.lower() == "schtasks.exe"
        action, name = arguments[1], arguments[3]
        self.calls.append(action)
        if action == "/Create":
            if name in self.tasks and "/F" not in arguments:
                return self._answer(arguments, 1)
            self.tasks[name] = Path(arguments[5]).read_bytes().decode("utf-16")
            self.disabled.discard(name)
            return self._answer(arguments, 0)
        if name not in self.tasks:
            return self._answer(arguments, 1)
        if action == "/Change":
            assert arguments[4] == "/DISABLE"
            self.disabled.add(name)
        elif action == "/Run":
            fields = dict(task_fields(self.tasks[name]))
            self.started.append(
                subprocess.Popen(
                    f'"{fields["Actions/Exec/Command"]}" '
                    + fields["Actions/Exec/Arguments"],
                    cwd=fields["Actions/Exec/WorkingDirectory"],
                )
            )
        elif action == "/End":
            if self.ends:
                for program in self.started:
                    program.kill()
        else:
            raise AssertionError(action)
        return self._answer(arguments, 0)

    def minute(self, name: str) -> None:
        """The task's trigger comes round: start it unless it runs or is off."""
        if name in self.disabled or any(p.poll() is None for p in self.started):
            return
        self.run(["schtasks.exe", "/Run", "/TN", name])
        # Task Scheduler's own doing, not something that was asked of it.
        self.calls.pop()


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Scheduler]:
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows

    stand_in = _Scheduler()
    monkeypatch.setattr(windows, "_run", stand_in.run)
    try:
        yield stand_in
    finally:
        for program in stand_in.started:
            program.kill()
            program.wait()


def _own_layout(state: Path) -> WindowsServiceLayout:
    assert sys.platform == "win32"
    from skulk.utils._windows_account import this_account

    return WindowsServiceLayout(this_account(), state)


def _prepared(layout: WindowsServiceLayout, manager: str) -> None:
    """Bring a service root to where local setup has it before it installs."""
    service_registration.register_user(layout, "prepare")
    activate_service_runtime(layout.root, staged(layout.root, manager))


def _appears(path: Path, program: "subprocess.Popen[bytes] | None" = None) -> None:
    deadline = time.monotonic() + 120
    while not path.exists():
        if program is not None:
            assert program.poll() is None, "the service ended early"
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)


def _held(root: Path) -> bool:
    try:
        RuntimeLock(root, service_bootstrap.LIFETIME_LOCK).close()
    except BlockingIOError:
        return True
    return False


@windows_only
def test_setup_registers_starts_and_stops_the_service(
    tmp_path: Path, scheduler: _Scheduler
) -> None:
    """Started with no console, as Task Scheduler starts it, and asked to stop."""
    layout = _own_layout(tmp_path)
    base = service_bootstrap.current_base()
    _prepared(layout, _SERVES)
    assert not service_registration.registered_for(layout, base)

    service_registration.register_user(layout, "install")
    assert scheduler.calls == ["/Create", "/Run"]
    assert not (layout.root / "service-task.xml").exists()
    assert service_registration.registered_for(layout, base, protected=True)
    assert not service_registration.registered_for(layout, base.with_name("other.exe"))
    (service,) = scheduler.started
    _appears(layout.root / "ready", service)
    assert _held(layout.root)
    # A second start, by hand or by the minute, finds the first one there.
    assert subprocess.run(scheduler.started[0].args, cwd=layout.root).returncode == 1

    service_registration.register_user(layout, "stop")
    assert (layout.root / "asked").exists()
    # The startup file waited for the manager and ended with its code.
    assert service.wait(30) == 5
    assert scheduler.calls == ["/Create", "/Run", "/Change"]
    # The request stays until the service is installed again.
    assert (layout.root / service_bootstrap.STOP_REQUEST).exists()
    assert not _held(layout.root)
    # Switched off, it starts nothing, so it is not registered for the base.
    assert not service_registration.registered_for(layout, base)

    # Installing again replaces local setup's own task and starts it.
    (layout.root / "ready").unlink()
    service_registration.register_user(layout, "install")
    assert scheduler.calls[-2:] == ["/Create", "/Run"]
    assert not (layout.root / service_bootstrap.STOP_REQUEST).exists()
    assert service_registration.registered_for(layout, base)
    _appears(layout.root / "ready", scheduler.started[-1])
    service_registration.register_user(layout, "stop")
    assert scheduler.started[-1].wait(30) == 5


@windows_only
def test_a_stop_holds_where_the_task_could_not_be_switched_off(
    tmp_path: Path, scheduler: _Scheduler
) -> None:
    """Task Scheduler gave no task to read, so none was switched off; it stays down."""
    layout = _own_layout(tmp_path)
    _prepared(layout, _SERVES)
    service_registration.register_user(layout, "install")
    (service,) = scheduler.started
    _appears(layout.root / "ready", service)
    scheduler.readable = False

    service_registration.register_user(layout, "stop")
    assert scheduler.calls == ["/Create", "/Run"]
    assert service.wait(30) == 5
    (layout.root / "ready").unlink()
    # The minute comes round with the task still switched on.
    name = layout.label
    scheduler.run(["schtasks.exe", "/Run", "/TN", name])
    assert scheduler.started[-1].wait(60) == 0
    assert not (layout.root / "ready").exists()
    assert not _held(layout.root)
    # Installing over a task that cannot be read is refused, and the request
    # to stop is still there afterwards.
    with pytest.raises(ValueError, match="Task Scheduler refused"):
        service_registration.register_user(layout, "install")
    assert (layout.root / service_bootstrap.STOP_REQUEST).exists()


@windows_only
def test_two_accounts_on_one_machine_have_a_task_each(
    tmp_path: Path, scheduler: _Scheduler
) -> None:
    """Another account's task is under another name and is not this account's concern."""
    layout = _own_layout(tmp_path)
    other = _layout(tmp_path / "other")
    theirs = other.definition(tmp_path / "other" / "python.exe").decode("utf-16")
    scheduler.tasks[other.label] = theirs
    _prepared(layout, _SERVES)
    base = service_bootstrap.current_base()
    assert not service_registration.registered_for(layout, base)

    service_registration.register_user(layout, "install")
    assert service_registration.registered_for(layout, base)
    _appears(layout.root / "ready", scheduler.started[0])
    service_registration.register_user(layout, "stop")
    assert scheduler.tasks[other.label] == theirs
    assert other.label not in scheduler.disabled


@windows_only
def test_a_service_that_ignores_the_request_is_ended_whole(
    tmp_path: Path, scheduler: _Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ending the task ends only the startup file; its job ends the manager."""
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows

    layout = _own_layout(tmp_path)
    _prepared(layout, _WILL_NOT_STOP)
    service_registration.register_user(layout, "install")
    (service,) = scheduler.started
    _appears(layout.root / "ready", service)
    monkeypatch.setattr(windows, "_STOP_WAIT_SECONDS", 3.0)

    service_registration.register_user(layout, "stop")
    assert (layout.root / "asked").exists()
    assert scheduler.calls == ["/Create", "/Run", "/Change", "/End"]
    assert service.wait(30) != 0
    assert not _held(layout.root)
    # Nothing of the manager is left: its lock is free and it writes no more.
    RuntimeLock(layout.root, "manager.lock").close()


@windows_only
def test_a_service_that_cannot_be_ended_is_reported(
    tmp_path: Path, scheduler: _Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows

    layout = _own_layout(tmp_path)
    _prepared(layout, _WILL_NOT_STOP)
    service_registration.register_user(layout, "install")
    _appears(layout.root / "ready", scheduler.started[0])
    monkeypatch.setattr(windows, "_STOP_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(windows, "_END_WAIT_SECONDS", 1.0)
    scheduler.ends = False

    with pytest.raises(ValueError, match="has not stopped"):
        service_registration.register_user(layout, "stop")
    # The request stays, so the service stops when it can, and nothing may be
    # installed over a service that is still there.
    assert (layout.root / service_bootstrap.STOP_REQUEST).exists()
    with pytest.raises(ValueError, match="still running"):
        service_registration.register_user(layout, "install")


@windows_only
def test_a_console_start_stops_on_the_same_request(tmp_path: Path) -> None:
    """With a console of its own the startup file asks the manager the same way."""
    assert sys.platform == "win32"
    import ctypes

    if not ctypes.WinDLL("kernel32").GetConsoleCP():
        pytest.skip("no console to start the service on")
    layout = _own_layout(tmp_path)
    _prepared(layout, _SERVES)
    RuntimeLock(layout.root, service_bootstrap.LIFETIME_LOCK).close()
    service = subprocess.Popen(
        [
            str(service_bootstrap.current_base()),
            "-I",
            "-S",
            "-B",
            str(layout.root / "service-bootstrap.py"),
            "--root",
            str(layout.root),
        ],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        _appears(layout.root / "ready", service)
        (layout.root / service_bootstrap.STOP_REQUEST).write_bytes(b"")
        assert service.wait(30) == 5
        assert (layout.root / "asked").exists()
    finally:
        service.kill()
        service.wait()


@windows_only
def test_a_stop_request_left_behind_starts_nothing(tmp_path: Path) -> None:
    layout = _own_layout(tmp_path)
    _prepared(layout, _SERVES)
    RuntimeLock(layout.root, service_bootstrap.LIFETIME_LOCK).close()
    (layout.root / service_bootstrap.STOP_REQUEST).write_bytes(b"")
    finished = subprocess.run(
        [
            str(service_bootstrap.current_base()),
            "-I",
            "-S",
            "-B",
            str(layout.root / "service-bootstrap.py"),
            "--root",
            str(layout.root),
        ],
        timeout=120,
    )
    assert finished.returncode == 0
    assert not (layout.root / "ready").exists()


@windows_only
def test_a_root_without_its_lock_starts_nothing(tmp_path: Path) -> None:
    """The lock is how setup knows a service is there; none, no service."""
    layout = _own_layout(tmp_path)
    _prepared(layout, _SERVES)
    (layout.root / service_bootstrap.LIFETIME_LOCK).unlink(missing_ok=True)
    finished = subprocess.run(
        [
            str(service_bootstrap.current_base()),
            "-I",
            "-S",
            "-B",
            str(layout.root / "service-bootstrap.py"),
            "--root",
            str(layout.root),
        ],
        timeout=120,
    )
    assert finished.returncode == 1
    assert not (layout.root / "ready").exists()


@windows_only
def test_another_task_under_the_name_is_left_alone(
    tmp_path: Path, scheduler: _Scheduler
) -> None:
    layout = _own_layout(tmp_path)
    _prepared(layout, _SERVES)
    foreign = (
        layout.definition(service_bootstrap.current_base())
        .decode("utf-16")
        .replace("skulk-plugin-service-v1", "someone-elses")
    )
    scheduler.tasks[layout.label] = foreign
    for action in ("stop", "install"):
        with pytest.raises(ValueError, match="not managed by local setup"):
            service_registration.register_user(layout, action)
    assert scheduler.calls == []
    assert scheduler.tasks[layout.label] == foreign
    assert not service_registration.registered_for(
        layout, service_bootstrap.current_base()
    )


@windows_only
def test_a_task_that_cannot_be_read_is_not_written_over(
    tmp_path: Path, scheduler: _Scheduler
) -> None:
    """Task Scheduler does not say "none" apart from "unreadable"; neither is a go."""
    layout = _own_layout(tmp_path)
    _prepared(layout, _SERVES)
    scheduler.tasks[layout.label] = "unreadable"
    scheduler.readable = False
    with pytest.raises(ValueError, match="Task Scheduler refused"):
        service_registration.register_user(layout, "install")
    assert scheduler.calls == ["/Create"]
    assert scheduler.tasks[layout.label] == "unreadable"
    assert scheduler.started == []


@windows_only
def test_registration_refuses_an_administrator_and_another_account(
    tmp_path: Path, scheduler: _Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows

    with pytest.raises(ValueError, match="its own account"):
        service_registration.register_user(_layout(tmp_path), "prepare")
    monkeypatch.setattr(windows, "has_root_authority", lambda: True)
    with pytest.raises(ValueError, match="its own account"):
        service_registration.register_user(_own_layout(tmp_path), "prepare")
    assert scheduler.calls == []
    assert not (tmp_path / "Skulk").exists()


@windows_only
def test_windows_offers_the_accounts_own_service_and_no_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows
    from skulk.utils._windows_account import local_data_directory, this_account

    assert registrable_scopes() == ("user",)
    with pytest.raises(ValueError, match="whole machine"):
        service_registration.owner_layout("system")
    # No release names a Windows host yet, and a service that could never be
    # given a runtime is not registered either.
    with pytest.raises(ValueError, match="qualified on macOS and Linux only"):
        service_registration.owner_layout("user")
    monkeypatch.setattr(windows, "current_platform", lambda: "a-named-host")
    layout = service_registration.owner_layout("user")
    assert isinstance(layout, WindowsServiceLayout)
    assert layout.account == this_account()
    assert layout.root == Path(local_data_directory()) / "Skulk" / "plugins"
    if "LOCALAPPDATA" in os.environ:
        assert Path(local_data_directory()) == Path(os.environ["LOCALAPPDATA"])


@windows_only
def test_the_base_of_a_windowless_start_is_the_interpreter_beside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = service_bootstrap.current_base()
    windowless = WindowsServiceLayout.launcher(base)
    if not windowless.is_file():
        pytest.skip("this Python has no pythonw.exe")
    monkeypatch.setattr(sys, "_base_executable", str(windowless), raising=False)
    assert service_bootstrap.current_base() == base
    # Asked of the interpreter itself, started as Task Scheduler starts it.
    answer = subprocess.run(
        [
            str(windowless),
            "-I",
            "-S",
            "-B",
            "-c",
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('b', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "open(sys.argv[2], 'w', encoding='utf-8').write(str(module.current_base()))\n",
            service_bootstrap.__file__,
            str(Path(os.environ["TEMP"]) / f"skulk-base-{os.getpid()}"),
        ],
        timeout=60,
    )
    told = Path(os.environ["TEMP"]) / f"skulk-base-{os.getpid()}"
    try:
        assert answer.returncode == 0
        assert Path(told.read_text(encoding="utf-8")) == base
    finally:
        told.unlink(missing_ok=True)


@windows_only
@pytest.mark.parametrize("reply", ["answered", "lost"])
async def test_a_refresh_waits_out_the_minute_and_registers_nothing_again(
    tmp_path: Path,
    scheduler: _Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    reply: str,
) -> None:
    """The successor is selected at once; the next start is the minute's to give.

    launchd and systemd start a manager again ten seconds after it ends.
    Task Scheduler does it when the minute comes round, here twelve seconds
    after the manager ended, which is longer than setup waits to see the
    selection and far shorter than it waits for the manager. Either way setup
    leaves the registration alone: the service is the same service.

    ``answered``: the manager answers the reload and stops half a second
    later, as the real one does. ``lost``: its answer never arrives and it is
    already gone, so setup has only the selection to go by and must sit out
    the whole gap.
    """
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows
    from skulk.extensions import service_setup, service_snapshot

    build = "b" * 64
    monkeypatch.setattr(
        service_setup,
        "measure_host",
        lambda: QualifiedHost("a-named-host", "3.13.13", "1.5.2", build),
    )
    monkeypatch.setattr(service_setup, "service_source_identity", lambda: build)
    monkeypatch.setattr(windows, "current_platform", lambda: "a-named-host")
    monkeypatch.setattr(windows, "local_data_directory", lambda: str(tmp_path))
    configuration = tmp_path / "c"
    configuration.mkdir()
    monkeypatch.setattr(service_setup, "SKULK_CONFIG_HOME", configuration)
    layout = service_registration.owner_layout("user")
    root = layout.root

    async def stage(path: Path) -> service_snapshot.ServiceSnapshot:
        return staged(path, _RELOADS, build[0] * 32, build)

    async def answers(path: Path) -> bool:
        # The manager of the service that runs now says it is ready.
        return (
            bool(scheduler.started)
            and scheduler.started[-1].poll() is None
            and (path / "ready").exists()
        )

    async def reload(
        path: Path, request: service_setup.ManagerRequest
    ) -> dict[str, service_setup.JsonValue]:
        # What the real manager does with this request: select the staged
        # copy, answer, and end cleanly to be started again on it.
        assert isinstance(request, service_setup.ReloadRuntimeRequest)
        service_snapshot.activate_staged_runtime(
            path,
            service_snapshot.ServiceSnapshot.model_validate_json(
                (
                    path / "core-runtimes" / request.generation / "staged.json"
                ).read_bytes()
            ),
        )
        (path / "reload").write_bytes(b"")
        if reply == "lost":
            while scheduler.started[0].poll() is None:
                await asyncio.sleep(0.05)
            raise OSError("the manager went before its answer arrived")
        return {"result": {}}

    monkeypatch.setattr(service_setup, "stage_service_runtime", stage)
    monkeypatch.setattr(service_setup, "_observe", answers)
    monkeypatch.setattr(service_setup, "manager_request", reload)

    def quiet(_message: str) -> None:
        return None

    first = await service_setup.setup_service("user", quiet)
    assert first.phase == "ready" and first.snapshot is not None
    assert scheduler.calls == ["/Create", "/Run"]
    (service,) = scheduler.started

    async def the_minute_comes_round() -> None:
        while service.poll() is None:
            await asyncio.sleep(0.1)
        await asyncio.sleep(12)
        scheduler.minute(layout.label)

    build = "c" * 64
    began = time.monotonic()
    minute = asyncio.create_task(the_minute_comes_round())
    try:
        second = await asyncio.wait_for(service_setup.setup_service("user", quiet), 180)
        waited = time.monotonic() - began
        await minute
    finally:
        # A setup that failed never ended the service, so nothing is coming.
        minute.cancel()
    try:
        assert service.returncode == 0
        if reply == "lost":
            # Longer than setup waits to see the selection: it went on waiting
            # for the manager, and did not take the gap for a failed reload.
            assert waited > service_setup._RELOAD_SETTLE_SECONDS  # pyright: ignore[reportPrivateUsage]
        assert second.phase == "ready" and second.snapshot is not None
        assert second.operation_id != first.operation_id
        assert second.snapshot.generation == "c" * 32
        # Nothing was registered again, stopped or ended: only the minute ran.
        assert scheduler.calls == ["/Create", "/Run"]
        assert len(scheduler.started) == 2
        _appears(root / "ready", scheduler.started[-1])
        assert _held(root)
        assert service_registration.registered_for(
            layout, service_bootstrap.current_base()
        )
    finally:
        service_registration.register_user(layout, "stop")
    assert scheduler.started[-1].wait(30) == 5


async def _sets_itself_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local setup from nothing to a manager that answers, and the same again.

    Everything is real but three things: where the account's data folder is
    (the test's own), the name of the host, which no release gives a Windows
    machine yet, and the copy's own answer to that name. The copy is this
    environment's, every file of it; the startup file verifies it and starts
    the real manager from it.
    """
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows
    from skulk.extensions import service_setup, service_snapshot
    from skulk.utils.runtime_files import RuntimeLock as Lock

    host = QualifiedHost("macos-arm64", "3.13.13", "1.5.2", "a" * 64)
    monkeypatch.setattr(service_snapshot, "measure_host", lambda: host)
    monkeypatch.setattr(service_setup, "measure_host", lambda: host)

    async def measured_alike(_runtime: Path, _lock: Lock) -> bytes:
        return json.dumps(
            {
                "platform": host.platform,
                "python_version": host.python_version,
                "skulk_version": host.skulk_version,
                "skulk_build_sha256": host.skulk_build_sha256,
                "inventory": service_snapshot._distributions(),  # pyright: ignore[reportPrivateUsage]
            }
        ).encode()

    monkeypatch.setattr(service_snapshot, "_qualify", measured_alike)
    monkeypatch.setattr(windows, "current_platform", lambda: host.platform)
    monkeypatch.setattr(windows, "local_data_directory", lambda: str(tmp_path))
    configuration = tmp_path / "c"
    configuration.mkdir()
    monkeypatch.setattr(service_setup, "SKULK_CONFIG_HOME", configuration)
    layout = service_registration.owner_layout("user")
    assert layout.root == tmp_path / "Skulk" / "plugins"
    try:
        completed = await service_setup.setup_service("user", lambda _message: None)
        assert completed.phase == "ready"
        assert service_setup.connected_scope() == "user"
        status = await service_setup.service_status()
        assert status["registered_runtime_verified"] is True
        assert status["management_available"] is True
        assert _held(layout.root)
        # A second setup finds a healthy service and changes nothing.
        assert await service_setup.setup_service() == completed
        assert _held(layout.root)
    finally:
        service_registration.register_user(layout, "stop")
    assert not _held(layout.root)
    Lock(layout.root, "manager.lock").close()


@windows_only
@pytest.mark.slow
async def test_local_setup_brings_the_service_up_on_windows(
    tmp_path: Path, scheduler: _Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _sets_itself_up(tmp_path, monkeypatch)
    assert scheduler.calls == ["/Create", "/Run", "/Change"]


@live
@pytest.mark.slow
async def test_local_setup_brings_the_service_up_under_the_real_task_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WindowsServiceLayout, "label", property(_live_name))
    _delete_live_task()
    try:
        await _sets_itself_up(tmp_path, monkeypatch)
    finally:
        _delete_live_task()


def _live_name(_layout: WindowsServiceLayout) -> str:
    """The live tests' own task name, so a real registration is never touched."""
    return _LIVE_NAME


def _delete_live_task() -> None:
    subprocess.run(
        ["schtasks.exe", "/Delete", "/TN", _LIVE_NAME, "/F"],
        capture_output=True,
        check=False,
    )


@live
def test_the_real_task_scheduler_runs_and_stops_the_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registered, started in the signed-in session, read back and stopped, for real.

    Needs the account to be signed in at the machine: a task that runs with
    the account's own session has nowhere to run otherwise.
    """
    monkeypatch.setattr(WindowsServiceLayout, "label", property(_live_name))
    layout = _own_layout(tmp_path)
    base = service_bootstrap.current_base()
    _prepared(layout, _SERVES)
    _delete_live_task()
    try:
        assert not service_registration.registered_for(layout, base)
        service_registration.register_user(layout, "install")
        # Read back from Task Scheduler's own copy, which it rewrote.
        assert service_registration.registered_for(layout, base, protected=True)
        _appears(layout.root / "ready")
        assert _held(layout.root)

        started = time.monotonic()
        service_registration.register_user(layout, "stop")
        assert time.monotonic() - started < 60
        assert (layout.root / "asked").exists()
        assert not _held(layout.root)
        assert not service_registration.registered_for(layout, base)

        # Over its own switched-off task, and started again without waiting
        # for the minute.
        (layout.root / "ready").unlink()
        (layout.root / "asked").unlink()
        service_registration.register_user(layout, "install")
        assert service_registration.registered_for(layout, base)
        _appears(layout.root / "ready")
        service_registration.register_user(layout, "stop")
        assert (layout.root / "asked").exists()
    finally:
        _delete_live_task()


@live
def test_the_real_task_scheduler_ends_a_service_that_will_not_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sys.platform == "win32"
    from skulk.extensions import _service_registration_windows as windows

    monkeypatch.setattr(WindowsServiceLayout, "label", property(_live_name))
    monkeypatch.setattr(windows, "_STOP_WAIT_SECONDS", 3.0)
    layout = _own_layout(tmp_path)
    _prepared(layout, _WILL_NOT_STOP)
    _delete_live_task()
    try:
        service_registration.register_user(layout, "install")
        _appears(layout.root / "ready")
        service_registration.register_user(layout, "stop")
        assert (layout.root / "asked").exists()
        assert not _held(layout.root)
        RuntimeLock(layout.root, "manager.lock").close()
    finally:
        _delete_live_task()
