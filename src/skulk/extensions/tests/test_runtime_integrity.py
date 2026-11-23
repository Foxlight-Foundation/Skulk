"""Installed-file mutations are refused without importing the private runtime."""

import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.runtime_files import private_directory, write_private
from skulk.extensions.runtime_install import (
    RuntimeInstaller,
    _without_shared_write,  # pyright: ignore[reportPrivateUsage]
)
from skulk.extensions.runtime_integrity import (
    PATH_TOO_LONG,
    RuntimePathTooLongError,
    _current_base,  # pyright: ignore[reportPrivateUsage]
    _paths_are_limited,  # pyright: ignore[reportPrivateUsage]
    _pip_depth,  # pyright: ignore[reportPrivateUsage]
    adopt_current_interpreter,
    environment_command,
    path_refusal,
    path_refusal_payload,
    path_refusal_sentence,
    runtime_interpreter,
    runtime_path_excess,
    seal_runtime,
    verify_installed_runtime,
)
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.utils.runtime_files import has_root_authority

_WINDOWS = sys.platform == "win32"
_MINOR = f"python{sys.version_info.major}.{sys.version_info.minor}"
# On Windows, the file beside python.exe that holds the interpreter.
_LIBRARY = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
# Where venv puts installed packages inside an environment.
_SITE = (
    Path("Lib") / "site-packages"
    if _WINDOWS
    else Path("lib") / _MINOR / "site-packages"
)

posix_only = pytest.mark.skipif(_WINDOWS, reason="POSIX permission bits and links")
windows_only = pytest.mark.skipif(
    not _WINDOWS, reason="a Windows base is a program and its library"
)
# The installer refuses an administrator, as it refuses root.
installs = pytest.mark.skipif(
    has_root_authority(), reason="the installer refuses root authority"
)


def _stand_in_link(link: Path, directory: Path) -> None:
    """Put a link where a member should be, by the means an ordinary account has."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(directory), str(link))
    else:
        link.symlink_to(directory)


def _let_another_account_write(path: Path) -> None:
    if _WINDOWS:
        subprocess.run(
            ["icacls", str(path), "/grant", "*S-1-1-0:(W)"],
            check=True,
            capture_output=True,
        )
    else:
        path.chmod(0o666)


@pytest.mark.parametrize(
    "fault", ["changed", "added", "missing", "mode", "link", "identity"]
)
def test_installed_runtime_seal_detects_mutation(tmp_path: Path, fault: str) -> None:
    """Seal complete directory membership, including executable startup code."""
    private_directory(tmp_path)
    runtime = tmp_path / "runtime"
    private_directory(runtime)
    write_private(runtime / "library.py", b"VALUE=7\n")
    interpreter = runtime_interpreter(runtime)
    private_directory(interpreter.parent)
    if _WINDOWS:
        write_private(interpreter, b"launcher")
    else:
        interpreter.symlink_to(Path(sys.executable).resolve())
    seal_runtime(tmp_path, "a" * 64)
    verify_installed_runtime(tmp_path, "a" * 64)
    if fault == "changed":
        write_private(runtime / "library.py", b"VALUE=8\n")
    elif fault == "added":
        write_private(runtime / "startup.pth", b"import unexpected\n")
    elif fault == "missing":
        (runtime / "library.py").unlink()
    elif fault == "mode":
        _let_another_account_write(runtime / "library.py")
    elif fault == "link" and _WINDOWS:
        # The same folder with the same launcher, reached through a junction.
        elsewhere = tmp_path / "elsewhere"
        interpreter.parent.rename(elsewhere)
        _stand_in_link(interpreter.parent, elsewhere)
    elif fault == "link":
        interpreter.unlink()
        interpreter.symlink_to("/bin/sh")
    with pytest.raises(ValueError):
        verify_installed_runtime(tmp_path, ("b" if fault == "identity" else "a") * 64)


def test_a_seal_names_the_base_only_where_no_link_does(tmp_path: Path) -> None:
    """A POSIX seal is the bytes it has always been; a Windows one adds the base."""
    private_directory(tmp_path)
    runtime = tmp_path / "runtime"
    private_directory(runtime)
    write_private(runtime / "library.py", b"VALUE=7\n")
    seal_runtime(tmp_path, "a" * 64)
    sealed = cast(
        dict[str, dict[str, object]],
        json.loads((tmp_path / "installed-files.json").read_bytes()),
    )
    if _WINDOWS:
        assert sealed["base"].keys() == {"path", "executable", "library"}
        assert sealed["base"]["path"] == str(_current_base())
    else:
        assert sealed.keys() == {"runtime_digest", "entries"}


def test_environment_commands_follow_the_platform_layout(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    command = environment_command(runtime)
    assert command[0] == str(_current_base())
    assert command[1:4] == ("-I", "-m", "venv")
    assert command[-1] == str(runtime)
    if _WINDOWS:
        assert command[4] == "--copies"
        assert runtime_interpreter(runtime) == runtime / "Scripts" / "python.exe"
    else:
        assert command[4] == "--symlinks"
        assert runtime_interpreter(runtime) == runtime / "bin" / "python"


def _wheel(*members: str) -> bytes:
    """A wheel with these members and nothing in them."""
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        for member in members:
            archive.writestr(member, b"")
    return content.getvalue()


def _limit_paths(monkeypatch: pytest.MonkeyPatch, pip: tuple[int, int]) -> None:
    """Measure as a Windows machine without long paths would, with this ``pip``."""
    monkeypatch.setattr(
        "skulk.extensions.runtime_integrity._paths_are_limited", lambda: True
    )
    monkeypatch.setattr("skulk.extensions.runtime_integrity._pip_depth", lambda: pip)


def _generation(length: int) -> Path:
    """A generation folder whose path is exactly this long."""
    generation = Path("g" * length)
    assert len(str(generation)) == length
    return generation


def test_no_path_is_too_long_where_paths_have_no_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX, and Windows with long paths on, are never refused."""
    monkeypatch.setattr(
        "skulk.extensions.runtime_integrity._paths_are_limited", lambda: False
    )
    deep = _wheel("package/" + "d" * 300 + "/module.py")
    assert runtime_path_excess(_generation(900), ("bundle.pyz",), (deep,)) == 0
    if not _WINDOWS:
        monkeypatch.undo()
        assert not _paths_are_limited()


@pytest.mark.parametrize(
    ("member", "placed"),
    [
        ("{long}/module.py", "runtime/Lib/site-packages/{long}/module.py"),
        (
            "x-1.0.data/purelib/{long}/module.py",
            "runtime/Lib/site-packages/{long}/module.py",
        ),
        (
            "x-1.0.data/platlib/{long}/native.pyd",
            "runtime/Lib/site-packages/{long}/native.pyd",
        ),
        ("x-1.0.data/scripts/{long}.exe", "runtime/Scripts/{long}.exe"),
        ("x-1.0.data/data/share/{long}.txt", "runtime/share/{long}.txt"),
        # pip adds this file when it installs a wheel named on its command
        # line, which is how the installer names every wheel.
        (
            "{long}-1.0.dist-info/WHEEL",
            "runtime/Lib/site-packages/{long}-1.0.dist-info/direct_url.json",
        ),
        # Headers really go somewhere shorter; they are counted where they
        # sit in the wheel, which can only refuse early.
        (
            "x-1.0.data/headers/{long}.h",
            "runtime/Lib/site-packages/x-1.0.data/headers/{long}.h",
        ),
    ],
)
def test_a_wheel_member_is_measured_where_pip_puts_it(
    monkeypatch: pytest.MonkeyPatch, member: str, placed: str
) -> None:
    """Each member counts at its installed place, not its place in the wheel."""
    _limit_paths(monkeypatch, (0, 0))
    # Long enough that this member is the deepest thing in the generation.
    member, placed = member.format(long="p" * 120), placed.format(long="p" * 120)
    directory = len(placed.rpartition("/")[0])
    # A directory may be 247 characters long and a file 259.
    fits = min(247 - 1 - directory, 259 - 1 - len(placed))
    wheels = (_wheel(member),)
    assert runtime_path_excess(_generation(fits), (), wheels) == 0
    assert runtime_path_excess(_generation(fits + 1), (), wheels) == 1
    assert runtime_path_excess(_generation(fits + 30), (), wheels) == 30


def test_a_directory_is_refused_twelve_characters_before_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows creates no directory at 248 characters, whatever its files are named."""
    _limit_paths(monkeypatch, (0, 0))
    directory = "runtime/Lib/site-packages/package/" + "d" * 60
    fits = 247 - 1 - len(directory)
    # A short file name: the directory is what does not fit.
    short = (_wheel("package/" + "d" * 60 + "/m.py"),)
    assert runtime_path_excess(_generation(fits), (), short) == 0
    assert runtime_path_excess(_generation(fits + 1), (), short) == 1
    # A directory the wheel names without a file in it counts the same.
    empty = (_wheel("package/" + "d" * 60 + "/"),)
    assert runtime_path_excess(_generation(fits), (), empty) == 0
    assert runtime_path_excess(_generation(fits + 1), (), empty) == 1
    # A long file name: the file is what does not fit, at 260.
    name = "/" + "m" * 40 + ".py"
    long = (_wheel("package/" + "d" * 60 + name),)
    fits = 259 - 1 - len(directory + name)
    assert runtime_path_excess(_generation(fits), (), long) == 0
    assert runtime_path_excess(_generation(fits + 1), (), long) == 1


def test_the_measure_covers_pip_the_artifacts_and_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the wheels, what is known before a download is still counted."""
    _limit_paths(monkeypatch, (85, 94))
    # The deepest directory of pip decides: 247 - 1 - 85.
    assert runtime_path_excess(_generation(161)) == 0
    assert runtime_path_excess(_generation(162)) == 1
    _limit_paths(monkeypatch, (0, 0))
    artifact = "artifacts/" + "w" * 150 + ".whl"
    fits = 259 - 1 - len(artifact)
    assert runtime_path_excess(_generation(fits), (artifact[10:],)) == 0
    assert runtime_path_excess(_generation(fits + 1), (artifact[10:],)) == 1
    # With nothing installed at all, the name adoption gives a file it is
    # replacing in the environment is the deepest path.
    fits = 259 - 1 - len("runtime/.pyvenv.cfg.rebase-" + "0" * 32)
    assert runtime_path_excess(_generation(fits)) == 0
    assert runtime_path_excess(_generation(fits + 1)) == 1
    # A wheel that cannot be read is pip's to report, not this measure's.
    assert runtime_path_excess(_generation(100), (), (b"not a wheel",)) == 0


def test_the_bundled_pip_is_found_and_measured() -> None:
    """``venv`` installs the standard library's own pip; its depth is read from it."""
    _pip_depth.cache_clear()
    directory, file = _pip_depth()
    if (directory, file) == (0, 0) and not _WINDOWS:
        pytest.skip("this base interpreter ships no pip wheel")
    assert directory > len("runtime/Lib/site-packages/pip")
    assert file > directory


def test_a_path_refusal_is_a_fixed_vocabulary_of_a_code_and_one_integer() -> None:
    """The manager names the refusal with a number, never with the path."""
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(
        path_refusal_payload(RuntimePathTooLongError(12))
    )
    assert payload == {"error": PATH_TOO_LONG, "excess": 12}
    refused = path_refusal(payload)
    assert isinstance(refused, RuntimePathTooLongError) and refused.excess == 12
    assert "12 characters" in path_refusal_sentence(12)
    assert "1 character longer" in path_refusal_sentence(1)
    assert path_refusal({"error": "manager_operation_refused"}) is None
    assert path_refusal({"error": PATH_TOO_LONG}) is None
    for unusable in (True, 0, -3, 32768, "12", 1.5, None):
        assert path_refusal({"error": PATH_TOO_LONG, "excess": unusable}) is None
    # A number under another code is not this refusal.
    assert path_refusal({"error": "manager_operation_refused", "excess": 12}) is None


def test_desktop_metadata_does_not_unseal_a_generation(tmp_path: Path) -> None:
    """Browsing a generation in Finder leaves it sealed; a link by that name does not."""
    private_directory(tmp_path)
    runtime = tmp_path / "runtime"
    private_directory(runtime)
    write_private(runtime / "library.py", b"VALUE=7\n")
    private_directory(runtime / "lib")
    seal_runtime(tmp_path, "a" * 64)
    write_private(runtime / ".DS_Store", b"finder")
    write_private(runtime / "lib/.DS_Store", b"finder")
    write_private(runtime / "._library.py", b"appledouble")
    verify_installed_runtime(tmp_path, "a" * 64)
    (runtime / "lib/.DS_Store").unlink()
    _stand_in_link(runtime / "lib/.DS_Store", tmp_path)
    with pytest.raises(ValueError):
        verify_installed_runtime(tmp_path, "a" * 64)


@installs
async def test_cached_stage_refuses_startup_injection_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newly added .pth file never executes during cached-stage verification."""
    source = tmp_path / "source"
    metadata, trust, host = artifacts(source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    installer = RuntimeInstaller(tmp_path / "installation")
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    operation = await installer.stage(metadata, source)
    generation = installer.root / "generations" / operation.runtime_digest
    verify_installed_runtime(generation, operation.runtime_digest)
    marker = tmp_path / "unexpected-startup"
    injection = generation / "runtime" / _SITE / "unexpected.pth"
    injection.write_text(f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n")
    os.chmod(injection, 0o600)
    with pytest.raises(ValueError, match="integrity differs"):
        await installer.stage(metadata, source, operation_id=operation.operation_id)
    assert not marker.exists()
    assert installer.operation(operation.operation_id).state == "recovery_required"


def _interpreter(directory: Path, content: bytes) -> Path:
    """A stand-in base interpreter; adoption hashes it but never runs it."""
    if _WINDOWS:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "python.exe"
        path.write_bytes(content)
        (directory / _LIBRARY).write_bytes(b"library of " + content)
        return path.resolve()
    (directory / "bin").mkdir(parents=True, exist_ok=True)
    path = directory / "bin" / _MINOR
    path.write_bytes(content)
    path.chmod(0o755)
    return path.resolve()


def _environment(generation: Path, base: Path) -> Path:
    """Lay out a runtime the way ``venv`` does for ``base`` on this platform."""
    private_directory(generation)
    runtime = generation / "runtime"
    if _WINDOWS:
        # As venv leaves it: access inherited from the private generation.
        runtime.mkdir()
    else:
        private_directory(runtime)
    site = runtime / _SITE
    site.mkdir(parents=True)
    (site / "library.py").write_bytes(b"VALUE=7\n")
    interpreter = runtime_interpreter(runtime)
    interpreter.parent.mkdir()
    if _WINDOWS:
        interpreter.write_bytes(b"launcher")
    else:
        (runtime / "bin" / _MINOR).symlink_to(base)
        interpreter.symlink_to(_MINOR)
        (runtime / "bin/python3").symlink_to(_MINOR)
    # venv writes the file as text, so with the platform's line ending.
    (runtime / "pyvenv.cfg").write_text(
        f"home = {base.parent}\n"
        "include-system-site-packages = false\n"
        f"version = {platform.python_version()}\n"
        f"executable = {base}\n"
        f"command = {base} -m venv --without-pip {runtime}\n"
    )
    if not _WINDOWS:
        for path in (runtime, *runtime.rglob("*")):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
    return runtime


def _base_of(runtime: Path) -> Path:
    """The base interpreter an environment would start."""
    if not _WINDOWS:
        return runtime_interpreter(runtime).resolve()
    for line in (runtime / "pyvenv.cfg").read_text().splitlines():
        if line.startswith("home = "):
            return Path(line.removeprefix("home = ")) / "python.exe"
    raise AssertionError("pyvenv.cfg names no home")


def _use_base(monkeypatch: pytest.MonkeyPatch, base: Path) -> None:
    monkeypatch.setattr(
        "skulk.extensions.runtime_integrity._current_base", lambda: base
    )


@pytest.mark.parametrize("change", ["moved", "updated"])
def test_adoption_keeps_a_runtime_across_an_interpreter_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """An application update or move re-points the runtime instead of breaking it."""
    old = _interpreter(tmp_path / "Old.app", b"old interpreter")
    generation = tmp_path / "generation"
    runtime = _environment(generation, old)
    _use_base(monkeypatch, old)
    seal_runtime(generation, "a" * 64)
    assert not adopt_current_interpreter(generation, "a" * 64)
    if change == "moved":
        new = _interpreter(tmp_path / "Moved.app", b"old interpreter")
        old.unlink()
    else:
        new = _interpreter(tmp_path / "Old.app", b"updated interpreter")
    _use_base(monkeypatch, new)
    with pytest.raises((ValueError, OSError)):
        verify_installed_runtime(generation, "a" * 64)
    assert adopt_current_interpreter(generation, "a" * 64)
    verify_installed_runtime(generation, "a" * 64)
    assert _base_of(runtime) == new
    configuration = (runtime / "pyvenv.cfg").read_text()
    assert f"home = {new.parent}\n" in configuration
    assert f"command = {new} -m venv --without-pip {runtime}\n" in configuration
    # Every line still ends the way venv ended it on this platform.
    written = (runtime / "pyvenv.cfg").read_bytes()
    assert written.count(os.linesep.encode()) == written.count(b"\n") == 5
    assert not (generation / "interpreter-rebase.json").exists()
    sealed = (generation / "installed-files.json").read_bytes()
    assert not adopt_current_interpreter(generation, "a" * 64)
    assert (generation / "installed-files.json").read_bytes() == sealed


@pytest.mark.parametrize("fault", ["library", "added", "configuration", "version"])
def test_adoption_refuses_anything_but_an_interpreter_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Every sealed member except the interpreter links must still match exactly."""
    old = _interpreter(tmp_path / "Old.app", b"old interpreter")
    generation = tmp_path / "generation"
    runtime = _environment(generation, old)
    if fault == "version":
        configuration = runtime / "pyvenv.cfg"
        configuration.write_text(
            configuration.read_text().replace(
                f"version = {platform.python_version()}",
                f"version = {sys.version_info.major}.{sys.version_info.minor - 1}.0",
            )
        )
    _use_base(monkeypatch, old)
    seal_runtime(generation, "a" * 64)
    site = runtime / _SITE
    if fault == "library":
        (site / "library.py").write_bytes(b"VALUE=8\n")
    elif fault == "added":
        (site / "startup.pth").write_bytes(b"import unexpected\n")
        (site / "startup.pth").chmod(0o644)
    elif fault == "configuration":
        with (runtime / "pyvenv.cfg").open("a") as configuration:
            configuration.write("include-system-site-packages = true\n")
    sealed = (generation / "installed-files.json").read_bytes()
    _use_base(monkeypatch, _interpreter(tmp_path / "New.app", b"new interpreter"))
    with pytest.raises(ValueError):
        adopt_current_interpreter(generation, "a" * 64)
    assert (generation / "installed-files.json").read_bytes() == sealed
    assert _base_of(runtime) == old


@windows_only
def test_a_changed_library_alone_is_an_interpreter_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows the library is the interpreter; python.exe only starts it."""
    base = _interpreter(tmp_path / "App", b"interpreter")
    generation = tmp_path / "generation"
    _environment(generation, base)
    _use_base(monkeypatch, base)
    seal_runtime(generation, "a" * 64)
    base.with_name(_LIBRARY).write_bytes(b"updated library")
    with pytest.raises(ValueError, match="integrity differs"):
        verify_installed_runtime(generation, "a" * 64)
    assert adopt_current_interpreter(generation, "a" * 64)
    verify_installed_runtime(generation, "a" * 64)


@windows_only
@pytest.mark.parametrize("moment", ["sealing", "verification", "adoption"])
def test_a_base_without_its_library_is_never_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, moment: str
) -> None:
    """Such a base cannot start, so no seal may come to name it as complete."""
    base = _interpreter(tmp_path / "App", b"interpreter")
    generation = tmp_path / "generation"
    runtime = _environment(generation, base)
    _use_base(monkeypatch, base)
    if moment == "sealing":
        base.with_name(_LIBRARY).unlink()
        with pytest.raises(FileNotFoundError):
            seal_runtime(generation, "a" * 64)
        assert not (generation / "installed-files.json").exists()
        return
    seal_runtime(generation, "a" * 64)
    sealed = (generation / "installed-files.json").read_bytes()
    configuration = (runtime / "pyvenv.cfg").read_bytes()
    if moment == "verification":
        base.with_name(_LIBRARY).unlink()
        with pytest.raises(FileNotFoundError):
            verify_installed_runtime(generation, "a" * 64)
    else:
        moved = _interpreter(tmp_path / "Moved", b"interpreter")
        moved.with_name(_LIBRARY).unlink()
        _use_base(monkeypatch, moved)
        with pytest.raises(FileNotFoundError):
            adopt_current_interpreter(generation, "a" * 64)
    assert (generation / "installed-files.json").read_bytes() == sealed
    assert (runtime / "pyvenv.cfg").read_bytes() == configuration
    assert not (generation / "interpreter-rebase.json").exists()


@pytest.mark.parametrize(
    "step", [pytest.param("link", marks=posix_only), "configuration", "seal"]
)
def test_interrupted_adoption_resumes_from_its_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    """A crash at any step after the journal was written finishes on the next start."""
    old = _interpreter(tmp_path / "Old.app", b"old interpreter")
    generation = tmp_path / "generation"
    runtime = _environment(generation, old)
    _use_base(monkeypatch, old)
    seal_runtime(generation, "a" * 64)
    new = _interpreter(tmp_path / "New.app", b"new interpreter")
    _use_base(monkeypatch, new)
    if step == "link":
        leftover = runtime / "bin" / f".{_MINOR}.rebase-{'0' * 32}"
    else:
        leftover = runtime / f".pyvenv.cfg.rebase-{'0' * 32}"

    def crash(*arguments: object) -> None:
        if step == "link":
            leftover.symlink_to(str(arguments[1]))
        else:
            leftover.write_bytes(b"half written")
        raise OSError("power lost")

    monkeypatch.setattr(
        "skulk.extensions.runtime_integrity."
        + {
            "link": "_replace_link",
            "configuration": "_replace_file",
            "seal": "_write_seal",
        }[step],
        crash,
        raising=True,
    )
    with pytest.raises(OSError, match="power lost"):
        adopt_current_interpreter(generation, "a" * 64)
    rewritten = f"home = {new.parent}\n" in (runtime / "pyvenv.cfg").read_text()
    assert rewritten is (step != "configuration")
    assert (generation / "interpreter-rebase.json").exists()
    monkeypatch.undo()
    _use_base(monkeypatch, new)
    assert adopt_current_interpreter(generation, "a" * 64)
    verify_installed_runtime(generation, "a" * 64)
    assert _base_of(runtime) == new
    assert not leftover.is_symlink() and not leftover.exists()
    assert not (generation / "interpreter-rebase.json").exists()


@installs
async def test_locked_generation_adopts_a_moved_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged plugin still verifies after the base interpreter moves."""
    source = tmp_path / "source"
    metadata, trust, host = artifacts(source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    installer = RuntimeInstaller(tmp_path / "installation")
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    operation = await installer.stage(metadata, source)
    base = _current_base()
    moved = (
        tmp_path / "Moved.app" / base.name
        if _WINDOWS
        else tmp_path / "Moved.app/bin" / _MINOR
    )
    moved.parent.mkdir(parents=True)
    shutil.copy2(base, moved)
    if _WINDOWS:
        shutil.copy2(base.with_name(_LIBRARY), moved.with_name(_LIBRARY))
    _use_base(monkeypatch, moved.resolve())
    async with installer.locked_generation(operation.runtime_digest):
        pass
    generation = installer.root / "generations" / operation.runtime_digest
    assert _base_of(generation / "runtime") == moved.resolve()


def _standalone_interpreter(prefix: Path) -> Path:
    """Where a relocatable standalone CPython keeps its interpreter."""
    return prefix / "python.exe" if _WINDOWS else prefix / "bin" / _MINOR


def _run_from(monkeypatch: pytest.MonkeyPatch, interpreter: Path) -> None:
    """Make ``interpreter`` the one this process appears to be running on."""
    monkeypatch.setattr(sys, "executable", str(interpreter))
    if _WINDOWS:
        monkeypatch.setattr(sys, "_base_executable", str(interpreter), raising=False)


@installs
@pytest.mark.slow
async def test_relinked_runtime_starts_on_a_moved_standalone_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real installed runtime imports its wheel after the interpreter moves.

    Copies the running standalone CPython (relocatable, as the packaged apps
    and uv ship it), installs through the copy, moves the copy as an app move
    does, and starts the relinked environment's own Python.
    """
    prefix = Path(sys.base_prefix).resolve()
    if not _standalone_interpreter(prefix).is_file() or "Python.framework" in str(
        prefix
    ):
        pytest.skip("needs a relocatable standalone CPython")
    old = tmp_path / "Old.app" / "python"
    shutil.copytree(prefix, old, symlinks=True)
    _run_from(monkeypatch, _standalone_interpreter(old))
    source = tmp_path / "source"
    metadata, trust, host = artifacts(source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    installer = RuntimeInstaller(tmp_path / "installation")
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    operation = await installer.stage(metadata, source)
    moved = tmp_path / "Moved.app" / "python"
    moved.parent.mkdir()
    old.rename(moved)
    _run_from(monkeypatch, _standalone_interpreter(moved))
    async with installer.locked_generation(operation.runtime_digest):
        pass
    runtime = installer.root / "generations" / operation.runtime_digest / "runtime"
    started = subprocess.run(
        (
            str(runtime_interpreter(runtime)),
            "-I",
            "-B",
            "-c",
            "import example_dep, sys; print(example_dep.VALUE, sys.base_prefix)",
        ),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert started.stdout.split() == ["7", str(moved)]


@posix_only
def test_a_new_runtime_loses_group_and_other_write(tmp_path: Path) -> None:
    """Shared write bits copied from a base Python's templates are removed."""
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    script = runtime / "bin" / "activate"
    script.write_text("# activate\n")
    script.chmod(0o664)
    (runtime / "bin").chmod(0o775)
    runtime.chmod(0o775)
    (runtime / "bin" / "python").symlink_to("/usr/bin/false")

    _without_shared_write(runtime)

    assert stat.S_IMODE(runtime.stat().st_mode) == 0o755
    assert stat.S_IMODE(script.stat().st_mode) == 0o644
    assert stat.S_IMODE((runtime / "bin").stat().st_mode) == 0o755
    assert (runtime / "bin" / "python").is_symlink()


@posix_only
@pytest.mark.slow
async def test_a_base_python_with_group_writable_templates_still_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ubuntu's 0002 umask leaves uv's venv templates group-writable.

    venv copies those modes onto every runtime's activation scripts, which the
    seal refuses; staging must clear them first.
    """
    prefix = Path(sys.base_prefix).resolve()
    if not (prefix / "bin" / _MINOR).is_file() or "Python.framework" in str(prefix):
        pytest.skip("needs a relocatable standalone CPython")
    base = tmp_path / "python"
    shutil.copytree(prefix, base, symlinks=True)
    templates = base / "lib" / _MINOR / "venv" / "scripts"
    for template in templates.rglob("*"):
        if template.is_file():
            template.chmod(0o664)
    monkeypatch.setattr(sys, "executable", str(base / "bin" / _MINOR))
    source = tmp_path / "source"
    metadata, trust, host = artifacts(source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    installer = RuntimeInstaller(tmp_path / "installation")
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )

    operation = await installer.stage(metadata, source)

    assert operation.state == "staged"
