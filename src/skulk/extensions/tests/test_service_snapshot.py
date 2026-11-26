"""Pre-site bootstrap integrity, stopped activation and incomplete service copying."""

import asyncio
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest

from skulk.extensions import service_bootstrap
from skulk.extensions.runtime_artifacts import QualifiedHost, measure_host
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_files import (
    RuntimeLock,
    acquire_runtime_lock,
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_manager import InventoryRequest, manager_request
from skulk.extensions.service_snapshot import (
    ServiceSnapshot,
    _qualify,  # pyright: ignore[reportPrivateUsage]
    activate_service_runtime,
    stage_service_runtime,
)
from skulk.utils.owned_process import isolated_environment
from skulk.utils.runtime_files import has_root_authority

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="examines a Windows access list or process"
)

_MANAGER = "print('verified-manager')\n"
# A manager that never stops by itself: it adds a byte to ``beats`` in the
# service root, unbuffered, for far longer than any test runs. A file that has
# stopped growing means it has stopped running.
_BEATS = (
    "import os, sys, time\n"
    "log = os.open(os.path.join(sys.argv[-1], 'beats'),"
    " os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "while True:\n"
    "    os.write(log, b'x')\n"
    "    time.sleep(0.01)\n"
)
# A manager that says it is ready and ends in order, with a code of its own,
# when the console asks it to.
_STOPS_WHEN_ASKED = (
    "import signal, time\n"
    "def stop(*_):\n"
    "    raise SystemExit(5)\n"
    "signal.signal(signal.SIGBREAK, stop)\n"
    "print('ready', flush=True)\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)


# A manager that says which copy it runs, asking the startup file that started
# it (the copy of it in the service root), and ends.
_SAYS_WHAT_IT_RUNS = (
    "import importlib.util, json, os, sys\n"
    "from pathlib import Path\n"
    "root = sys.argv[-1]\n"
    "spec = importlib.util.spec_from_file_location(\n"
    "    'startup', os.path.join(root, 'service-bootstrap.py'))\n"
    "startup = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(startup)\n"
    "print(json.dumps(startup.started_from(Path(root))), flush=True)\n"
)


def _site(runtime: Path) -> Path:
    """Where ``venv`` puts site-packages on this platform."""
    if sys.platform == "win32":
        return runtime / "Lib" / "site-packages"
    return (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _environment(runtime: Path) -> None:
    """Build an empty environment the way staging builds one on this platform."""
    if sys.platform == "win32":
        subprocess.run(
            [
                str(service_bootstrap.current_base()),
                "-I",
                "-m",
                "venv",
                "--copies",
                "--without-pip",
                str(runtime),
            ],
            check=True,
            capture_output=True,
        )
    else:
        venv.EnvBuilder(symlinks=True, with_pip=False).create(runtime)


def _lifetime_lock(root: Path) -> None:
    """Leave the lock a Windows service holds while it runs, as local setup does.

    The startup file only ever opens it: without it nothing starts there.
    """
    if sys.platform == "win32":
        # Already there, and held, when a service is running from this root.
        with contextlib.suppress(BlockingIOError):
            RuntimeLock(root, service_bootstrap.LIFETIME_LOCK).close()


def staged(
    root: Path,
    manager: str = _MANAGER,
    name: str = "a" * 32,
    build: str = "b" * 64,
) -> ServiceSnapshot:
    """Build a minimal actual venv with a fixed manager fixture and complete seal."""
    private_directory(root)
    _lifetime_lock(root)
    private_directory(root / "core-runtimes")
    generation = root / "core-runtimes" / name
    private_directory(generation)
    runtime = generation / "runtime"
    private_directory(runtime)
    _environment(runtime)
    site = _site(runtime)
    package = site / "skulk/extensions"
    package.mkdir(parents=True)
    (site / "skulk/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "runtime_manager.py").write_text(manager)
    bootstrap = Path(service_bootstrap.__file__).read_bytes()
    write_private(generation / "bootstrap.py", bootstrap)
    base = service_bootstrap.current_base()
    metadata = {
        "protocol": 1,
        "generation": name,
        **service_bootstrap.base_identity(base),
        "bootstrap_sha256": hashlib.sha256(bootstrap).hexdigest(),
        "entries": service_bootstrap.runtime_tree(runtime, base),
    }
    payload = json.dumps(metadata, sort_keys=True).encode()
    write_private(generation / "snapshot.json", payload)
    result = ServiceSnapshot(
        generation=name,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        skulk_build_sha256=build,
        copied_files=3,
        copied_bytes=32,
    )
    write_private(generation / "staged.json", result.model_dump_json().encode())
    return result


def _bootstrap_command(root: Path) -> list[str]:
    """The command the OS service runs: the base interpreter, isolated, no site."""
    return [
        str(service_bootstrap.current_base()),
        "-I",
        "-S",
        "-B",
        str(root / "service-bootstrap.py"),
        "--root",
        str(root),
    ]


def _run_bootstrap(root: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        _bootstrap_command(root),
        capture_output=True,
        timeout=60,
        env=isolated_environment(),
    )


def _reseal(root: Path, snapshot: ServiceSnapshot, change: dict[str, object]) -> None:
    """Rewrite the selected manifest with ``change`` and select the result."""
    generation = root / "core-runtimes" / snapshot.generation
    metadata = service_bootstrap.document(
        read_private(generation / "snapshot.json", 67108864)
    )
    metadata.update(change)
    raw = json.dumps(metadata).encode()
    write_private(generation / "snapshot.json", raw)
    write_private(
        root / "core-runtime.json",
        json.dumps(
            {
                "generation": snapshot.generation,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ).encode(),
    )


def _grant_everyone(path: Path, access: str) -> None:
    """Add an entry for every account to a Windows access list."""
    subprocess.run(
        ["icacls", str(path), "/grant", f"*S-1-1-0:({access})"],
        check=True,
        capture_output=True,
    )


def test_bootstrap_uses_fixed_verified_copy_and_activation_requires_stopped_owner(
    tmp_path: Path,
) -> None:
    """A sealed runtime runs without source PYTHONPATH; a live manager fences selection."""
    snapshot = staged(tmp_path)
    owner = RuntimeLock(tmp_path, "manager.lock")
    try:
        with pytest.raises(BlockingIOError):
            activate_service_runtime(tmp_path, snapshot)
        assert not (tmp_path / "core-runtime.json").exists()
    finally:
        owner.close()
    activate_service_runtime(tmp_path, snapshot)
    result = _run_bootstrap(tmp_path)
    assert (
        result.returncode == 0
        and result.stdout.splitlines() == [b"verified-manager"]
        and not result.stderr
    )


def test_the_bootstrap_ends_with_the_managers_exit_code(tmp_path: Path) -> None:
    """The OS service sees how the manager ended, whichever process it watches."""
    activate_service_runtime(tmp_path, staged(tmp_path, "raise SystemExit(7)\n"))
    assert _run_bootstrap(tmp_path).returncode == 7


def test_the_manager_starts_with_the_environment_an_owned_child_gets(
    tmp_path: Path,
) -> None:
    """The bootstrap restates the fixed environment; the two must not drift."""
    assert service_bootstrap.manager_environment(tmp_path) == {
        **isolated_environment(),
        "SKULK_HOME": str(tmp_path),
    }


def test_the_bootstrap_and_the_file_seam_agree_on_root_authority() -> None:
    assert service_bootstrap.has_root_authority() == has_root_authority()


def test_bootstrap_metadata_must_be_protected(tmp_path: Path) -> None:
    """State made by other means beside protected state is not read or trusted."""
    root = tmp_path / "root"
    private_directory(root)
    service_bootstrap.private_directory(root)
    write_private(root / "kept.json", b"{}")
    assert service_bootstrap.private_read(root / "kept.json", 16) == b"{}"
    with pytest.raises(ValueError, match="exceeds bound"):
        service_bootstrap.private_read(root / "kept.json", 1)
    made = root / "made.json"
    made.write_bytes(b"{}")
    folder = root / "made"
    folder.mkdir()
    if sys.platform != "win32":
        # Windows needs no help: what is made by other means takes its access
        # from the folder above, which is not what protected state has.
        made.chmod(0o644)
        folder.chmod(0o755)
    with pytest.raises(ValueError, match="not protected"):
        service_bootstrap.private_read(made, 16)
    with pytest.raises(ValueError, match="not protected"):
        service_bootstrap.private_directory(folder)


@pytest.mark.parametrize(
    "fault",
    [
        "file",
        "pth",
        "link",
        "manifest",
        "base",
        pytest.param("library", marks=windows_only),
        pytest.param("grant", marks=windows_only),
    ],
)
def test_bootstrap_refuses_tampering_before_site_startup(
    tmp_path: Path, fault: str
) -> None:
    """A changed dependency or startup hook cannot execute before verification."""
    snapshot = staged(tmp_path)
    activate_service_runtime(tmp_path, snapshot)
    generation = tmp_path / "core-runtimes" / snapshot.generation
    site = _site(generation / "runtime")
    marker = tmp_path / "unverified-executed"
    if fault == "file":
        (site / "skulk/extensions/runtime_manager.py").write_text(
            f"from pathlib import Path; Path({str(marker)!r}).touch()\n"
        )
    elif fault == "pth":
        (site / "unverified.pth").write_text(
            f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
        )
    elif fault == "link":
        if sys.platform == "win32":
            import _winapi

            # The link an ordinary account can make there without privilege.
            _winapi.CreateJunction(str(tmp_path), str(site / "external"))
        else:
            (site / "external").symlink_to(tmp_path)
    elif fault == "manifest":
        (generation / "snapshot.json").write_bytes(b"{}")
    elif fault == "base":
        _reseal(tmp_path, snapshot, {"base_sha256": "0" * 64})
    elif fault == "library":
        _reseal(tmp_path, snapshot, {"base_library_sha256": "0" * 64})
    else:
        # Not one byte differs; another account may now read the manager.
        _grant_everyone(site / "skulk/extensions/runtime_manager.py", "R")
    result = _run_bootstrap(tmp_path)
    assert result.returncode == 1 and not result.stdout
    assert (
        b"unavailable" in result.stderr and str(tmp_path).encode() not in result.stderr
    )
    assert not marker.exists()


async def test_interrupted_copy_does_not_publish_over_existing_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed local copy retains its incomplete generation and the current pointer."""
    private_directory(tmp_path)
    write_private(tmp_path / "core-runtime.json", b"retained current selection")
    monkeypatch.setattr("skulk.extensions.service_snapshot._sources", lambda: ())

    def disk_failure(*_: object) -> None:
        raise OSError("synthetic copy disk failure")

    monkeypatch.setattr("skulk.extensions.service_snapshot._copy", disk_failure)
    with pytest.raises(OSError, match="synthetic"):
        await stage_service_runtime(tmp_path)
    assert read_private(tmp_path / "core-runtime.json") == b"retained current selection"
    generations = list((tmp_path / "core-runtimes").iterdir())
    assert len(generations) == 1 and not (generations[0] / "staged.json").exists()
    # The failed copy's runtime (gigabytes in a real environment) is gone, so
    # retried setups cannot fill the disk; the generation stays as evidence.
    assert not (generations[0] / "runtime").exists()
    RuntimeLock(tmp_path).close()


async def test_staging_refuses_a_copy_the_volume_has_no_room_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup says how much space it needs instead of filling the disk."""
    import shutil

    from skulk.extensions import service_snapshot

    monkeypatch.setattr(service_snapshot, "_sources", lambda: ())

    def nearly_full(_path: object) -> shutil._ntuple_diskusage:  # pyright: ignore[reportPrivateUsage]
        return shutil._ntuple_diskusage(total=10**12, used=10**12, free=10**9)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(service_snapshot.shutil, "disk_usage", nearly_full)
    with pytest.raises(ValueError, match="Not enough free disk space"):
        await stage_service_runtime(tmp_path)
    assert not any((tmp_path / "core-runtimes").iterdir())
    RuntimeLock(tmp_path).close()


async def test_staging_refuses_a_copy_whose_paths_would_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Said before anything is created, with how much too long and what to change."""
    from skulk.extensions import service_snapshot

    host = QualifiedHost("macos-arm64", "3.13.13", "1.5.2", "a" * 64)
    monkeypatch.setattr(service_snapshot, "measure_host", lambda: host)
    monkeypatch.setattr(
        "skulk.extensions.runtime_integrity._paths_are_limited", lambda: True
    )
    monkeypatch.setattr(service_snapshot, "_distributions", lambda: {"skulk": "1.5.2"})
    private_directory(tmp_path)
    # ``core-runtimes``, the generation's 32 characters and ``runtime``, each
    # behind a separator, lie between the root and the copy.
    below = len(str(tmp_path)) + len("/core-runtimes/") + 32 + len("/runtime/")
    longest = "Lib/site-packages/" + "m" * (259 - below - len("Lib/site-packages/"))
    source = tmp_path / "source.py"
    source.write_bytes(b"")

    def one_file(relative: str) -> tuple[object, ...]:
        return (
            service_snapshot._SourceFile(  # pyright: ignore[reportPrivateUsage]
                relative, source, hashlib.sha256(b"").hexdigest(), 0, False
            ),
        )

    monkeypatch.setattr(service_snapshot, "_sources", lambda: one_file(longest + "mm"))
    with pytest.raises(ValueError, match="2 characters longer than Windows allows"):
        await stage_service_runtime(tmp_path)
    assert not (tmp_path / "core-runtimes").exists()
    RuntimeLock(tmp_path).close()


async def test_source_editable_dependency_is_not_silently_borrowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown editable startup paths fail setup before copying or activating code."""
    environment = tmp_path / "source"
    environment.mkdir()
    (environment / "third_party.pth").write_text("/a/development/checkout\n")

    def environment_path(name: str) -> str:
        return str(environment)

    monkeypatch.setattr(
        "skulk.extensions.service_snapshot.sysconfig.get_path", environment_path
    )
    monkeypatch.setattr(
        "skulk.extensions.service_snapshot._distributions", lambda: {"skulk": "1.5.2"}
    )
    with pytest.raises(ValueError, match="path extension"):
        await stage_service_runtime(tmp_path / "service")
    assert not (tmp_path / "service/core-runtime.json").exists()


def test_the_runtime_check_ignores_what_finder_leaves_behind(tmp_path: Path) -> None:
    """Browsing the core runtime in Finder must not stop the manager starting."""
    runtime = tmp_path / "runtime"
    private_directory(runtime)
    write_private(runtime / "module.py", b"VALUE = 1\n")
    private_directory(runtime / "lib")
    base = service_bootstrap.current_base()
    sealed = service_bootstrap.runtime_tree(runtime, base)
    write_private(runtime / ".DS_Store", b"finder")
    write_private(runtime / "lib/.DS_Store", b"finder")
    write_private(runtime / "._module.py", b"appledouble")
    assert service_bootstrap.runtime_tree(runtime, base) == sealed
    # Anything else added is still a different runtime.
    write_private(runtime / "lib/added.py", b"")
    assert service_bootstrap.runtime_tree(runtime, base) != sealed


def test_a_browsed_core_build_measures_as_the_same_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finder's files in the Skulk checkout do not change the build fingerprint."""
    packages: dict[str, Path] = {}
    for name in ("skulk", "skulk_pyo3_bindings"):
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text(f"NAME = {name!r}\n")
        packages[name] = root
    original = importlib.util.find_spec

    def find_spec(
        name: str, package: str | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if name in packages:
            return importlib.machinery.ModuleSpec(
                name, None, origin=str(packages[name] / "__init__.py")
            )
        return original(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    before = measure_host().skulk_build_sha256
    (packages["skulk"] / ".DS_Store").write_bytes(b"finder")
    (packages["skulk"] / "._module.py").write_bytes(b"appledouble")
    assert measure_host().skulk_build_sha256 == before
    (packages["skulk"] / "added.py").write_text("")
    assert measure_host().skulk_build_sha256 != before


@windows_only
@pytest.mark.parametrize("access", ["W", "R"])
def test_the_runtime_check_refuses_a_member_another_account_can_reach(
    tmp_path: Path, access: str
) -> None:
    """Any grant counts, read included, on a file and on a folder alike."""
    runtime = tmp_path / "runtime"
    private_directory(runtime)
    (runtime / "package").mkdir()
    (runtime / "package" / "module.py").write_bytes(b"VALUE = 1\n")
    base = service_bootstrap.current_base()
    # Made by other means inside the private folder: accepted as it stands.
    sealed = service_bootstrap.runtime_tree(runtime, base)
    assert set(sealed) == {".", "package", "package/module.py"}
    _grant_everyone(runtime / "package" / "module.py", access)
    with pytest.raises(ValueError, match="another user"):
        service_bootstrap.runtime_tree(runtime, base)


@windows_only
def test_the_runtime_check_refuses_a_junction(tmp_path: Path) -> None:
    assert sys.platform == "win32"
    import _winapi

    runtime = tmp_path / "runtime"
    private_directory(runtime)
    _winapi.CreateJunction(str(tmp_path), str(runtime / "elsewhere"))
    with pytest.raises(ValueError, match="unsupported link"):
        service_bootstrap.runtime_tree(runtime, service_bootstrap.current_base())


@windows_only
def test_the_runtime_check_requires_the_environment_to_name_the_base(
    tmp_path: Path,
) -> None:
    """The launcher starts whatever ``home`` names; no link says so on Windows."""
    runtime = tmp_path / "runtime"
    private_directory(runtime)
    _environment(runtime)
    base = service_bootstrap.current_base()
    service_bootstrap.runtime_tree(runtime, base)
    configuration = runtime / "pyvenv.cfg"
    lines = configuration.read_text(encoding="utf-8").splitlines()
    configuration.write_text(
        "\n".join(
            f"home = {tmp_path}" if line.startswith("home = ") else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="interpreter differs"):
        service_bootstrap.runtime_tree(runtime, base)


@windows_only
def test_the_manager_ends_when_the_bootstrap_is_ended(tmp_path: Path) -> None:
    """Nothing is left serving that the OS service does not know about."""
    activate_service_runtime(tmp_path, staged(tmp_path, _BEATS))
    beats = tmp_path / "beats"
    bootstrap = subprocess.Popen(
        _bootstrap_command(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=isolated_environment(),
    )
    try:
        deadline = time.monotonic() + 60
        while not beats.exists() or beats.stat().st_size == 0:
            assert bootstrap.poll() is None, "the bootstrap ended early"
            assert time.monotonic() < deadline, "the manager never started"
            time.sleep(0.05)
    finally:
        # Ended outright, as a service host ends a process that will not stop.
        bootstrap.kill()
        bootstrap.wait()
    deadline = time.monotonic() + 30
    size = beats.stat().st_size
    while True:
        time.sleep(1)
        now = beats.stat().st_size
        if now == size:
            break
        assert time.monotonic() < deadline, "the manager outlived the bootstrap"
        size = now


@windows_only
def test_a_console_request_to_stop_is_the_managers_to_act_on(tmp_path: Path) -> None:
    """Ctrl-Break reaches both; the bootstrap waits for the manager's own ending."""
    assert sys.platform == "win32"
    import ctypes

    if not ctypes.WinDLL("kernel32").GetConsoleCP():
        pytest.skip("no console to send Ctrl-Break through")
    activate_service_runtime(tmp_path, staged(tmp_path, _STOPS_WHEN_ASKED))
    bootstrap = subprocess.Popen(
        _bootstrap_command(tmp_path),
        stdout=subprocess.PIPE,
        env=isolated_environment(),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        assert bootstrap.stdout is not None
        assert bootstrap.stdout.readline().strip() == b"ready"
        os.kill(bootstrap.pid, signal.CTRL_BREAK_EVENT)
        assert bootstrap.wait(60) == 5
    finally:
        bootstrap.kill()
        bootstrap.wait()


def test_a_manager_the_startup_file_started_knows_which_copy_it_runs(
    tmp_path: Path,
) -> None:
    """The copy the startup file verified is the copy the manager names.

    Started by the real startup file, on its own interpreter, the manager
    asks where its environment lies and gets the selected generation and the
    digest of the manifest the startup file checked every file against.
    """
    snapshot = staged(tmp_path, _SAYS_WHAT_IT_RUNS)
    activate_service_runtime(tmp_path, snapshot)
    result = _run_bootstrap(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [snapshot.generation, snapshot.manifest_sha256]


def test_a_process_names_the_copy_it_runs_from_and_not_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the interpreter's environment lies decides, never the pointer."""
    running = staged(tmp_path)
    successor = staged(tmp_path, name="b" * 32)
    activate_service_runtime(tmp_path, successor)
    generations = tmp_path / "core-runtimes"
    # This test process runs from no service copy at all.
    assert service_bootstrap.started_from(tmp_path) is None
    monkeypatch.setattr(
        sys, "prefix", str(generations / running.generation / "runtime")
    )
    assert service_bootstrap.started_from(tmp_path) == (
        running.generation,
        running.manifest_sha256,
    )
    monkeypatch.setattr(
        sys, "prefix", str(generations / successor.generation / "runtime")
    )
    assert service_bootstrap.started_from(tmp_path) == (
        successor.generation,
        successor.manifest_sha256,
    )
    # A copy under another root is not this service's copy.
    elsewhere = tmp_path / "elsewhere"
    private_directory(elsewhere)
    assert service_bootstrap.started_from(elsewhere) is None
    # Nor is a folder inside a copy, nor a copy whose manifest is gone.
    monkeypatch.setattr(sys, "prefix", str(generations / running.generation))
    assert service_bootstrap.started_from(tmp_path) is None
    monkeypatch.setattr(
        sys, "prefix", str(generations / running.generation / "runtime")
    )
    (generations / running.generation / "snapshot.json").unlink()
    assert service_bootstrap.started_from(tmp_path) is None


def test_what_staging_copies_is_what_the_seal_accepts(tmp_path: Path) -> None:
    """Folders the copy makes on the way down are as private as the files in them."""
    from skulk.extensions import service_snapshot

    source = tmp_path / "module.py"
    source.write_bytes(b"VALUE = 1\n")
    runtime = tmp_path / "runtime"
    private_directory(runtime)
    relative = "lib/site-packages/package/inner/module.py"
    item = service_snapshot._SourceFile(  # pyright: ignore[reportPrivateUsage]
        relative, source, service_bootstrap.digest_file(source), 10, False
    )
    service_snapshot._copy((item,), runtime)  # pyright: ignore[reportPrivateUsage]
    service_snapshot._protect(runtime)  # pyright: ignore[reportPrivateUsage]
    sealed = service_bootstrap.runtime_tree(runtime, service_bootstrap.current_base())
    assert sealed[relative][0] == "file"
    assert sealed["lib/site-packages/package/inner"][0] == "directory"


# A copy's own account of itself: the state directory it was told to use in
# place of a platform, so the test can see that the directory reached it.
_MEASURES = (
    "import os\n"
    "class _Host:\n"
    "    platform = os.environ['SKULK_HOME']\n"
    "    python_version = '3.13.13'\n"
    "    skulk_version = '1.5.2'\n"
    "    skulk_build_sha256 = 'a' * 64\n"
    "def measure_host():\n"
    "    return _Host()\n"
)


def _copy_that_measures(root: Path, measurement: str) -> Path:
    """A minimal environment whose copy of Skulk answers as ``measurement`` says."""
    import shutil

    import packaging

    private_directory(root)
    private_directory(root / "core-runtimes")
    generation = root / "core-runtimes" / ("a" * 32)
    private_directory(generation)
    runtime = generation / "runtime"
    private_directory(runtime)
    _environment(runtime)
    site = _site(runtime)
    package = site / "skulk/extensions"
    package.mkdir(parents=True)
    (site / "skulk/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "runtime_artifacts.py").write_text(measurement)
    shutil.copytree(
        Path(packaging.__file__).parent,
        site / "packaging",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return runtime


@pytest.mark.skipif(has_root_authority(), reason="an owned child needs the fence")
async def test_a_copy_is_asked_what_it_is_in_its_own_state_directory(
    tmp_path: Path,
) -> None:
    """Qualification runs the copy's own interpreter, isolated, as an owned child."""
    runtime = _copy_that_measures(tmp_path, _MEASURES)
    lock = RuntimeLock(tmp_path)
    try:
        answer = service_bootstrap.document(await _qualify(runtime, lock))
    finally:
        lock.close()
    assert answer == {
        "platform": str(runtime.parent / "qualification-state"),
        "python_version": "3.13.13",
        "skulk_version": "1.5.2",
        "skulk_build_sha256": "a" * 64,
        "inventory": {},
    }


@pytest.mark.skipif(has_root_authority(), reason="an owned child needs the fence")
async def test_a_copy_that_cannot_say_what_it_is_leaves_its_evidence(
    tmp_path: Path,
) -> None:
    runtime = _copy_that_measures(tmp_path, "raise RuntimeError('not this build')\n")
    lock = RuntimeLock(tmp_path)
    try:
        with pytest.raises(ValueError, match="did not qualify"):
            await _qualify(runtime, lock)
    finally:
        lock.close()
    evidence = read_private(runtime.parent / "qualification-evidence.log")
    assert b"not this build" in evidence
    # The lock is free again: nothing of the child's is left holding it.
    RuntimeLock(tmp_path).close()


def _request_to_stop() -> int | None:
    """The request that ends a service in order, when one can be sent from here."""
    if sys.platform != "win32":
        return signal.SIGTERM
    import ctypes

    # Ctrl-Break is the one request to stop that can be sent to another
    # program's group, and only through a console.
    if ctypes.WinDLL("kernel32").GetConsoleCP():
        return signal.CTRL_BREAK_EVENT
    return None


def _new_group() -> int:
    """Start a program so that a console request reaches it and not the tests."""
    if sys.platform == "win32":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


async def _serves(root: Path, snapshot: ServiceSnapshot) -> None:
    """Select ``snapshot``, start it as the OS service would, and see it answer and end."""
    activate_service_runtime(root, snapshot)
    assert service_bootstrap.verified_runtime(root) == (
        root / "core-runtimes" / snapshot.generation / "runtime"
    )
    request_to_stop = _request_to_stop()
    bootstrap = subprocess.Popen(
        _bootstrap_command(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=isolated_environment(),
        creationflags=_new_group(),
    )
    try:
        deadline = time.monotonic() + 300
        while True:
            assert bootstrap.poll() is None, "the manager ended early"
            try:
                reply = await manager_request(root, InventoryRequest())
            except OSError:
                assert time.monotonic() < deadline, "the manager never answered"
                await asyncio.sleep(0.2)
                continue
            assert "result" in reply
            # The real manager, started by the real startup file, names the
            # copy that was selected for it.
            result = reply["result"]
            assert isinstance(result, dict)
            assert result.get("runtime") == {
                "generation": snapshot.generation,
                "manifest_sha256": snapshot.manifest_sha256,
            }
            break
        if request_to_stop is not None:
            os.kill(bootstrap.pid, request_to_stop)
            assert await asyncio.to_thread(bootstrap.wait, 120) == 0
    finally:
        # Without a console the bootstrap is ended outright, and the manager
        # with it.
        bootstrap.kill()
        bootstrap.wait()
    # It let go of everything: another manager could start here.
    (await acquire_runtime_lock(root, wait_seconds=30, name="manager.lock")).close()


def _service_root(tmp_path: Path) -> Path:
    """A provisioned, owner-only service root, as local setup leaves it."""
    root = tmp_path / "s"
    private_directory(root)
    _lifetime_lock(root)
    write_private(
        root / "host.json",
        HostSettings(transport_node_id="node-a", profile_id="a" * 32)
        .model_dump_json()
        .encode(),
    )
    return root


@windows_only
@pytest.mark.slow
@pytest.mark.skipif(has_root_authority(), reason="staging refuses root authority")
async def test_a_copy_of_this_skulk_is_sealed_and_starts_its_manager_on_windows(
    tmp_path: Path,
) -> None:
    """Nothing stands in: the host is measured and the copy answers for itself.

    The environment the tests run in is copied file by file into a service
    root and sealed. The copy's own interpreter is then asked what it is, and
    its answer has to be this host's measurement. The bootstrap verifies the
    copy on the base interpreter and starts the manager from it, and the
    manager answers and names the copy it runs.
    """
    root = _service_root(tmp_path)
    snapshot = await stage_service_runtime(root)
    assert snapshot.copied_files > 1000
    await _serves(root, snapshot)
