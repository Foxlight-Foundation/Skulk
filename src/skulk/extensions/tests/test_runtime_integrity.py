"""Installed-file mutations are refused without importing the private runtime."""

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from skulk.extensions.runtime_files import private_directory, write_private
from skulk.extensions.runtime_install import (
    RuntimeInstaller,
    _without_shared_write,  # pyright: ignore[reportPrivateUsage]
)
from skulk.extensions.runtime_integrity import (
    _current_base,  # pyright: ignore[reportPrivateUsage]
    adopt_current_interpreter,
    environment_command,
    runtime_interpreter,
    seal_runtime,
    verify_installed_runtime,
)
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.utils.runtime_files import has_root_authority

_WINDOWS = sys.platform == "win32"
_MINOR = f"python{sys.version_info.major}.{sys.version_info.minor}"
# Where venv puts installed packages inside an environment.
_SITE = (
    Path("Lib") / "site-packages"
    if _WINDOWS
    else Path("lib") / _MINOR / "site-packages"
)

posix_only = pytest.mark.skipif(_WINDOWS, reason="POSIX permission bits and links")
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
        base = _current_base()
        assert sealed["base"]["path"] == str(base)
        assert base.name in cast(dict[str, str], sealed["base"]["files"])
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
        library = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        (directory / library).write_bytes(b"library of " + content)
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
