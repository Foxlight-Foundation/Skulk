"""Real native bootstrap processes must stop before cache cleanup is allowed."""

from __future__ import annotations

import fcntl
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from skulk._pyinstaller import native_installer_data_files
from skulk.provisioning import native_install_process
from skulk.provisioning.native_install_process import NativeInstallCommandRunner


def _environment() -> dict[str, str]:
    return {"PATH": os.environ["PATH"]}


def _checkpoint() -> None:
    pass


def test_native_helpers_do_not_import_unrelated_provisioners() -> None:
    """A small native controller must not inherit the resident engine dependency set."""
    source = Path(__file__).resolve().parents[3]
    code = (
        "import importlib.abc,sys;"
        f"sys.path.insert(0,{str(source)!r});\n"
        "class RefuseUnrelated(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self,fullname,path=None,target=None):\n"
        "  if fullname in {'skulk.provisioning.audio_cpp','skulk.provisioning.comfy','skulk.provisioning.llama_server'}:\n"
        "   raise AssertionError('native helper imported unrelated provisioner: '+fullname)\n"
        "sys.meta_path.insert(0,RefuseUnrelated());"
        "import skulk.provisioning.native_install_process;"
        "import skulk.provisioning.demucs_package;"
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr.decode()


def _assert_stopped(identifier: int) -> None:
    deadline = time.monotonic() + 3
    while True:
        try:
            process = psutil.Process(identifier)
            if process.status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("native installation left a live descendant")
        time.sleep(0.025)


def _tree_command(marker: Path, *, exit_parent: bool = False) -> list[str]:
    descendant = (
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(60)"
    )
    command = (
        "import subprocess,sys,time;from pathlib import Path;"
        f"subprocess.Popen([sys.executable,'-I','-B','-c',{descendant!r}]);"
        f"marker=Path({str(marker)!r});\n"
        "while not marker.exists():time.sleep(.01)\n"
        + ("print('complete')" if exit_parent else "time.sleep(60)")
    )
    return [sys.executable, "-I", "-B", "-c", command]


def test_capture_and_diagnostics_survive_serial_commands(tmp_path: Path) -> None:
    """Opaque child stdout is returned only when requested; logs stay private."""
    logs = tmp_path / "logs"
    runner = NativeInstallCommandRunner(logs, checkpoint=_checkpoint)
    for capture in (True, False):
        assert runner(
            [sys.executable, "-I", "-B", "-c", "print('native facts')"],
            environment=_environment(),
            timeout_seconds=10,
            capture=capture,
        ) == ("native facts\n" if capture else "")
    records = tuple(logs.rglob("*.log"))
    assert len(records) == 4
    assert sum(path.read_bytes() == b"native facts\n" for path in records) == 2
    assert all(path.stat().st_mode & 0o077 == 0 for path in records)


def test_plain_python_descendant_cannot_rewrite_the_verified_tree(
    tmp_path: Path,
) -> None:
    """A console helper's second interpreter inherits no-bytecode policy.

    The explicit interpreter already uses -B. A plain child must also avoid
    writing import caches, including when the caller supplied a conflicting
    environment value. Otherwise library probes invalidate their own cache.
    """
    module = tmp_path / "native_helper_fixture.py"
    module.write_text("observation = 42\n")
    child = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); "
        "import native_helper_fixture; print(native_helper_fixture.observation)"
    )
    parent = (
        "import subprocess,sys; "
        f"subprocess.run([sys.executable, '-c', {child!r}], check=True)"
    )
    runner = NativeInstallCommandRunner(tmp_path / "logs", checkpoint=_checkpoint)
    assert (
        runner(
            [sys.executable, "-I", "-B", "-c", parent],
            environment=_environment() | {"PYTHONDONTWRITEBYTECODE": "0"},
            timeout_seconds=10,
            capture=True,
        )
        == "42\n"
    )
    assert not (tmp_path / "__pycache__").exists()


def test_frozen_controller_never_reexecutes_its_app_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packaged Core supplies verified Python independently of sys.executable."""
    interpreter = Path(sys.executable)
    marker = tmp_path / "launcher-called"
    launcher = tmp_path / "frozen-launcher"
    launcher.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 91\n")
    launcher.chmod(0o700)
    monkeypatch.setattr(
        native_install_process,
        "sys",
        SimpleNamespace(executable=str(launcher), frozen=True),
    )
    with pytest.raises(ValueError, match="verified guardian interpreter"):
        NativeInstallCommandRunner(tmp_path / "refused", checkpoint=_checkpoint)
    runner = NativeInstallCommandRunner(
        tmp_path / "logs", checkpoint=_checkpoint, guardian_interpreter=interpreter
    )
    assert (
        runner(
            [str(interpreter), "-I", "-B", "-c", "print('native')"],
            environment=_environment(),
            timeout_seconds=10,
            capture=True,
        )
        == "native\n"
    )
    assert not marker.exists()


def test_freezer_retains_actual_guardian_source() -> None:
    """A native Python cannot execute a helper available only as frozen bytecode."""
    [(source, target)] = native_installer_data_files()
    assert target == "skulk/provisioning"
    expected = Path(native_install_process.__file__).with_name(
        "_native_install_guard.py"
    )
    assert Path(source) == expected
    assert Path(source).read_bytes() == expected.read_bytes()


def test_command_failure_retains_bounded_diagnostics(tmp_path: Path) -> None:
    """A failed pip or device probe cannot become a successful preparation."""
    runner = NativeInstallCommandRunner(tmp_path / "logs", checkpoint=_checkpoint)
    with pytest.raises(ValueError, match=r"failed \(37\): native refusal"):
        runner(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import sys;sys.stderr.write('native refusal');sys.exit(37)",
            ],
            environment=_environment(),
            timeout_seconds=10,
            capture=True,
        )
    assert (
        next((tmp_path / "logs").rglob("*-stderr.log")).read_text() == "native refusal"
    )


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
def test_excess_output_fails_without_unbounded_logs(
    tmp_path: Path, stream: str
) -> None:
    """A noisy installer cannot exhaust memory or disk while holding a build lock."""
    runner = NativeInstallCommandRunner(tmp_path / "logs", checkpoint=_checkpoint)
    with pytest.raises(ValueError, match="output exceeds its bound"):
        runner(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                f"import sys;sys.{stream}.buffer.write(b'x'*(128*1024+1))",
            ],
            environment=_environment(),
            timeout_seconds=10,
            capture=True,
        )
    assert all(
        path.stat().st_size <= 128 * 1024 for path in (tmp_path / "logs").rglob("*.log")
    )


def test_cancel_before_spawn_cannot_start_native_command(tmp_path: Path) -> None:
    """A cancellation already accepted by the owner has no bootstrap side effect."""

    def cancelled() -> None:
        raise InterruptedError("owner cancelled")

    marker = tmp_path / "unexpected"
    runner = NativeInstallCommandRunner(tmp_path / "logs", checkpoint=cancelled)
    with pytest.raises(InterruptedError, match="owner cancelled"):
        runner(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path;Path({str(marker)!r}).touch()",
            ],
            environment=_environment(),
            timeout_seconds=10,
            capture=False,
        )
    assert not marker.exists()


@pytest.mark.parametrize("timeout", (False, True))
def test_cancel_and_deadline_drain_term_resistant_tree(
    tmp_path: Path, timeout: bool
) -> None:
    """Timeout and cancellation include the native command's grandchildren."""
    marker = tmp_path / "child.pid"

    def checkpoint() -> None:
        if not timeout and marker.exists():
            raise InterruptedError("owner cancelled")

    runner = NativeInstallCommandRunner(tmp_path / "logs", checkpoint=checkpoint)
    started = time.monotonic()
    with pytest.raises(TimeoutError if timeout else InterruptedError):
        runner(
            _tree_command(marker),
            environment=_environment(),
            timeout_seconds=1 if timeout else 10,
            capture=False,
        )
    assert time.monotonic() - started < 5
    _assert_stopped(int(marker.read_text()))


def test_completed_command_still_drains_its_live_descendants(tmp_path: Path) -> None:
    """A success status never releases an installer group before its last writer."""
    marker = tmp_path / "child.pid"
    runner = NativeInstallCommandRunner(tmp_path / "logs", checkpoint=_checkpoint)
    assert (
        runner(
            _tree_command(marker, exit_parent=True),
            environment=_environment(),
            timeout_seconds=10,
            capture=True,
        )
        == "complete\n"
    )
    _assert_stopped(int(marker.read_text()))


def test_failed_drain_retains_original_handle_and_refuses_another_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup refusal cannot authorize deletion or lose the owned group ID."""
    original = native_install_process.terminate_process_group
    observed: list[subprocess.Popen[bytes]] = []

    def drain(
        process: subprocess.Popen[bytes], *, grace_seconds: float, confirm_members: bool
    ) -> None:
        observed.append(process)
        if len(observed) <= 2:
            raise PermissionError("controlled group cleanup refusal")
        original(process, grace_seconds=grace_seconds, confirm_members=confirm_members)

    monkeypatch.setattr(native_install_process, "terminate_process_group", drain)
    runner = NativeInstallCommandRunner(tmp_path / "logs", checkpoint=_checkpoint)
    command = [sys.executable, "-I", "-B", "-c", "print('complete')"]
    try:
        with pytest.raises(PermissionError, match="cleanup refusal"):
            runner(
                command, environment=_environment(), timeout_seconds=10, capture=True
            )
        assert not runner.quiescent
        assert len(observed) == 2 and observed[0] is observed[1]
        assert observed[0].returncode is None
        with pytest.raises(RuntimeError, match="still requires drain"):
            runner(
                command, environment=_environment(), timeout_seconds=10, capture=True
            )
    finally:
        runner.drain()
    assert runner.quiescent
    assert len(observed) == 3 and observed[0] is observed[2]
    assert observed[-1].returncode is not None


def test_closed_completion_pipe_cannot_leave_native_descendants(tmp_path: Path) -> None:
    """Controller loss at command completion must not outrun the parent watcher."""
    marker = tmp_path / "child.pid"
    completion_read, completion_write = os.pipe()
    os.close(completion_read)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            str(
                Path(native_install_process.__file__).with_name(
                    "_native_install_guard.py"
                )
            ),
            str(os.getpid()),
            str(completion_write),
            "-1",
            *_tree_command(marker, exit_parent=True),
        ],
        pass_fds=(completion_write,),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.close(completion_write)
    try:
        deadline = time.monotonic() + 5
        while native_install_process.observe_process_exit(process) is None:
            assert time.monotonic() < deadline
            time.sleep(0.025)
        _assert_stopped(int(marker.read_text()))
    finally:
        native_install_process.terminate_process_group(process, confirm_members=True)


def test_guardian_retains_build_lock_until_descendant_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Controller death cannot make an in-progress generation look abandoned."""
    original = native_install_process.terminate_process_group
    lock = os.open(tmp_path / "build.lock", os.O_RDWR | os.O_CREAT, 0o600)
    contender = os.open(tmp_path / "build.lock", os.O_RDWR)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    released = False

    def drain(
        process: subprocess.Popen[bytes], *, grace_seconds: float, confirm_members: bool
    ) -> None:
        nonlocal released
        os.close(lock)
        released = True
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original(process, grace_seconds=grace_seconds, confirm_members=confirm_members)
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    monkeypatch.setattr(native_install_process, "terminate_process_group", drain)
    try:
        runner = NativeInstallCommandRunner(
            tmp_path / "logs", checkpoint=_checkpoint, lock_descriptor=lock
        )
        assert (
            runner(
                [sys.executable, "-I", "-B", "-c", "print('complete')"],
                environment=_environment(),
                timeout_seconds=10,
                capture=True,
            )
            == "complete\n"
        )
    finally:
        if not released:
            os.close(lock)
        os.close(contender)


def test_controller_death_stops_guardian_and_native_children(tmp_path: Path) -> None:
    """Kill a real controller while its distinct native process group is active."""
    marker = tmp_path / "child.pid"
    guardian_marker = tmp_path / "guardian.pid"
    core_source = str(Path(native_install_process.__file__).parents[2])
    command = (
        "import os,time;from pathlib import Path;"
        f"Path({str(marker)!r}).write_text(str(os.getpid()));"
        f"Path({str(guardian_marker)!r}).write_text(str(os.getppid()));time.sleep(60)"
    )
    controller = (
        f"import sys;sys.path.insert(0,{core_source!r});"
        "from pathlib import Path;"
        "from skulk.provisioning.native_install_process import NativeInstallCommandRunner;"
        f"runner=NativeInstallCommandRunner(Path({str(tmp_path / 'logs')!r}),checkpoint=lambda:None);"
        f"runner([{sys.executable!r},'-I','-B','-c',{command!r}],environment={{}},timeout_seconds=30,capture=False)"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", controller], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 10
        while not guardian_marker.exists():
            assert process.poll() is None, "controller failed before its native command"
            assert time.monotonic() < deadline
            time.sleep(0.025)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
        _assert_stopped(int(marker.read_text()))
        _assert_stopped(int(guardian_marker.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


def test_controller_loss_during_term_drain_keeps_guardian_and_lock(
    tmp_path: Path,
) -> None:
    """A guardian survives TERM until the owner or its watcher kills descendants."""
    marker = tmp_path / "descendant.pid"
    draining = tmp_path / "draining.pid"
    lock_path = tmp_path / "build.lock"
    core_source = str(Path(native_install_process.__file__).parents[2])
    controller = (
        "import fcntl,os,signal,sys,time;"
        f"sys.path.insert(0,{core_source!r});"
        "from pathlib import Path;"
        "from skulk.provisioning import native_install_process as native;"
        f"lock=os.open({str(lock_path)!r},os.O_RDWR|os.O_CREAT,0o600);"
        "fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);\n"
        "def pause_before_kill(process,**kwargs):\n"
        " os.close(lock)\n"
        " os.killpg(process.pid,signal.SIGTERM)\n"
        " deadline=time.monotonic()+.25\n"
        " while native.observe_process_exit(process) is None and time.monotonic()<deadline:time.sleep(.01)\n"
        f" Path({str(draining)!r}).write_text(str(process.pid))\n"
        " while True:time.sleep(.025)\n"
        "native.terminate_process_group=pause_before_kill;"
        f"runner=native.NativeInstallCommandRunner(Path({str(tmp_path / 'logs')!r}),checkpoint=lambda:None,lock_descriptor=lock);"
        f"runner({_tree_command(marker, exit_parent=True)!r},environment={{}},timeout_seconds=30,capture=False)"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", controller], start_new_session=True
    )
    descendant: psutil.Process | None = None
    guardian: psutil.Process | None = None
    contender: int | None = None
    try:
        deadline = time.monotonic() + 10
        while not draining.exists():
            assert process.poll() is None, "controller failed before drain barrier"
            assert time.monotonic() < deadline
            time.sleep(0.025)
        descendant = psutil.Process(int(marker.read_text()))
        guardian = psutil.Process(int(draining.read_text()))
        contender = os.open(lock_path, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert guardian.status() != psutil.STATUS_ZOMBIE
        process.kill()
        process.wait(timeout=3)
        deadline = time.monotonic() + 3
        while True:
            try:
                active = descendant.is_running() and descendant.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                active = False
            if not active:
                break
            assert time.monotonic() < deadline, "parent loss left a native writer alive"
            time.sleep(0.025)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        for owned in (descendant, guardian):
            if owned is not None:
                with suppress(psutil.NoSuchProcess):
                    owned.kill()
        if contender is not None:
            os.close(contender)
