"""Cached native interpreter integrity is anchored in original sealed bytes."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from skulk.provisioning.demucs_package import (
    PythonArchivePin,
    extract_python,
    file_digest,
    read_python_catalog,
    unpack_candidate,
    verify_cached_payload,
    verify_cached_runtime,
    verify_python_tree,
)


@pytest.fixture
def interpreter_fixture(tmp_path: Path) -> tuple[Path, Path, PythonArchivePin]:
    """Retain a pinned tiny interpreter tree with executable, library and link."""
    catalog = Path(__file__).parents[4] / "packaging/skulk-demucs"
    pin = read_python_catalog(catalog).for_target("macosx_14_0_arm64")
    archive_path = tmp_path / pin.filename
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in (
            ("python/bin/python3.12", b"interpreter bytes"),
            ("python/lib/python3.12/site.py", b"standard library bytes"),
            ("python/lib/libfixture.so", b"native library bytes"),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o755 if name.endswith("python3.12") else 0o644
            archive.addfile(entry, io.BytesIO(data))
        directory = tarfile.TarInfo("python/lib/python3.12/site-packages")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        link = tarfile.TarInfo("python/bin/python3")
        link.type = tarfile.SYMTYPE
        link.linkname = "python3.12"
        archive.addfile(link)
    pin = pin.model_copy(
        update={
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "size_bytes": archive_path.stat().st_size,
        }
    )
    destination = tmp_path / "expanded"
    extract_python(archive_path, destination, pin)
    return archive_path, destination, pin


def test_cached_python_checks_complete_tree_before_execution(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin],
) -> None:
    """An unchanged expanded base is reusable without running its own code."""
    archive, root, pin = interpreter_fixture
    assert verify_python_tree(archive, root, pin) == root / pin.executable


@pytest.mark.parametrize(
    "name",
    (
        "python/bin/python3.12",
        "python/lib/python3.12/site.py",
        "python/lib/libfixture.so",
    ),
)
def test_changed_python_bytes_fail_despite_unchanged_size(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin], name: str
) -> None:
    """Neither stable file size nor a cached local receipt authorizes changed code."""
    archive, root, pin = interpreter_fixture
    path = root / name
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="bytes differ from its archive"):
        verify_python_tree(archive, root, pin)


@pytest.mark.parametrize("name", ("injected.pth", "injected.pyc", "injected.py"))
def test_injected_python_import_code_is_refused(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin], name: str
) -> None:
    """Unlisted import hooks and bytecode never become trusted cached state."""
    archive, root, pin = interpreter_fixture
    (root / "python/lib/python3.12/site-packages" / name).write_bytes(b"foreign")
    with pytest.raises(ValueError, match="injected member"):
        verify_python_tree(archive, root, pin)


def test_missing_python_library_is_refused(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin],
) -> None:
    """A cached executable alone is insufficient when a sealed library is missing."""
    archive, root, pin = interpreter_fixture
    (root / "python/lib/libfixture.so").unlink()
    with pytest.raises(ValueError, match="file is missing"):
        verify_python_tree(archive, root, pin)


def test_redirected_python_root_is_refused(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin], tmp_path: Path
) -> None:
    """A valid receipt cannot move execution to another installation generation."""
    archive, root, pin = interpreter_fixture
    preserved = tmp_path / "different-generation"
    root.rename(preserved)
    root.symlink_to(preserved, target_is_directory=True)
    with pytest.raises(ValueError, match="root is missing or redirected"):
        verify_python_tree(archive, root, pin)


def test_changed_python_link_is_refused(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin],
) -> None:
    """Relative link topology remains bound to the verified original archive."""
    archive, root, pin = interpreter_fixture
    link = root / "python/bin/python3"
    link.unlink()
    link.symlink_to("../lib/libfixture.so")
    with pytest.raises(ValueError, match="link differs"):
        verify_python_tree(archive, root, pin)


def test_changed_python_archive_cannot_reauthorize_cached_bytes(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin],
) -> None:
    """Complete object identity is checked before using a cached source archive."""
    archive, root, pin = interpreter_fixture
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="differs from the pinned object"):
        verify_python_tree(archive, root, pin)


def test_cancellation_propagates_before_hashing_even_an_empty_object(
    tmp_path: Path,
) -> None:
    """An owner can stop verification without a false successful cache receipt."""
    path = tmp_path / "empty"
    path.touch()

    class CancelledError(Exception):
        """The installer's cancellation signal must reach its owner unchanged."""

    def cancel() -> None:
        raise CancelledError

    with pytest.raises(CancelledError):
        file_digest(path, limit=0, checkpoint=cancel)


@pytest.fixture
def cached_fixture(
    interpreter_fixture: tuple[Path, Path, PythonArchivePin], tmp_path: Path
) -> tuple[Path, Path, str, str]:
    """Construct sealed synthetic bytes without executing any candidate Python."""
    python_archive, expanded, pin = interpreter_fixture
    root = tmp_path / "generation"
    root.mkdir()
    expanded.rename(root / "interpreter")
    members = {
        "skulk_demucs/__init__.py": b"known adapter bytes",
        "skulk_demucs/python/" + pin.filename: python_archive.read_bytes(),
    }
    content = {
        "protocol_version": 1,
        "python": "3.12.13",
        "python_archive": pin.model_dump(mode="json"),
        "adapter_revision": "1" * 40,
        "source_revision": "2" * 40,
        "source_repository": "https://github.com/adefossez/demucs.git",
        "platform_tag": pin.platform_tag,
        "variant": "cpu",
        "members": {
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        },
        "features": ["audio.separate"],
    }
    payload_sha256 = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    members["skulk_demucs/build.json"] = json.dumps(
        content | {"payload_sha256": payload_sha256}
    ).encode()
    for name in ("METADATA", "WHEEL", "RECORD"):
        members["skulk_demucs_cpu-0.1.0.dist-info/" + name] = b"fixture metadata"
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            entry = zipfile.ZipInfo(name)
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(entry, data)
    sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    unpack_candidate(wheel, root / "payload", sha256)
    runtime = root / "runtime"
    site = runtime / "lib/python3.12/site-packages"
    for name, data in members.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    base = root / "interpreter" / pin.executable
    binary = runtime / "bin/python"
    binary.parent.mkdir()
    binary.symlink_to(base)
    (runtime / "pyvenv.cfg").write_text(
        f"home = {base.parent}\ninclude-system-site-packages = false\n"
        f"version = 3.12.13\nexecutable = {base}\n"
    )
    return wheel, root, sha256, payload_sha256


def test_cached_runtime_verification_never_runs_candidate_python(
    cached_fixture: tuple[Path, Path, str, str],
) -> None:
    """The synthetic non-executable program proves verification is data-only."""
    wheel, root, sha256, payload_sha256 = cached_fixture
    native, manifest = verify_cached_runtime(wheel, root, sha256, payload_sha256)
    assert native == root / "runtime/bin/python"
    assert manifest.payload_sha256 == payload_sha256


def test_changed_cached_sources_do_not_trust_their_own_manifest(
    cached_fixture: tuple[Path, Path, str, str],
) -> None:
    """Changing both payload code and its extracted manifest cannot authorize reuse."""
    wheel, root, sha256, _ = cached_fixture
    (root / "payload/skulk_demucs/__init__.py").write_bytes(b"evil! adapter bytes")
    (root / "payload/skulk_demucs/build.json").write_text("claims everything is valid")
    with pytest.raises(ValueError, match="differs from its trusted bytes"):
        verify_cached_payload(wheel, root / "payload", sha256)


@pytest.mark.parametrize(
    "change", ("system_packages", "duplicate_home", "foreign_python")
)
def test_changed_virtual_environment_configuration_is_refused(
    cached_fixture: tuple[Path, Path, str, str], change: str
) -> None:
    """Exact code alone cannot admit system packages or another base interpreter."""
    wheel, root, sha256, payload_sha256 = cached_fixture
    configuration = root / "runtime/pyvenv.cfg"
    text = configuration.read_text()
    if change == "system_packages":
        text = text.replace(
            "include-system-site-packages = false",
            "include-system-site-packages = true",
        )
    elif change == "duplicate_home":
        text += "home = /foreign\n"
    else:
        text = (
            "\n".join(
                "executable = /foreign/python"
                if line.startswith("executable = ")
                else line
                for line in text.splitlines()
            )
            + "\n"
        )
    configuration.write_text(text)
    with pytest.raises(ValueError, match="virtual environment"):
        verify_cached_runtime(wheel, root, sha256, payload_sha256)


def test_cached_ready_receipt_does_not_authorize_changed_installed_code(
    cached_fixture: tuple[Path, Path, str, str],
) -> None:
    """A forged completion record never replaces independent import-tree checks."""
    wheel, root, sha256, payload_sha256 = cached_fixture
    (root / "installation.json").write_text('{"verified_expanded_tree":true}')
    (
        root / "runtime/lib/python3.12/site-packages/skulk_demucs/__init__.py"
    ).write_bytes(b"changed installed code")
    with pytest.raises(ValueError, match="differs from trusted package bytes"):
        verify_cached_runtime(wheel, root, sha256, payload_sha256)


@pytest.mark.parametrize(
    "name",
    (
        "python._pth",
        "python3._pth",
        "python3.12._pth",
        "PYTHON._PTH",
        "pyvenv.cfg",
        "PYVENV.CFG",
    ),
)
def test_adjacent_python_startup_overrides_are_refused(
    cached_fixture: tuple[Path, Path, str, str], name: str
) -> None:
    """Isolated mode does not make adjacent interpreter path overrides safe."""
    wheel, root, sha256, payload_sha256 = cached_fixture
    contents = (
        "home = /unverified/python/bin\ninclude-system-site-packages = false\n"
        if name.casefold() == "pyvenv.cfg"
        else "/unverified/import/root\nimport site\n"
    )
    (root / "runtime/bin" / name).write_text(contents)
    with pytest.raises(ValueError, match="startup override"):
        verify_cached_runtime(wheel, root, sha256, payload_sha256)


def test_redirected_executable_directory_cannot_hide_startup_overrides(
    cached_fixture: tuple[Path, Path, str, str], tmp_path: Path
) -> None:
    """A verified final executable target cannot authorize a foreign bin directory."""
    wheel, root, sha256, payload_sha256 = cached_fixture
    binary_root = root / "runtime/bin"
    foreign = tmp_path / "foreign-bin"
    binary_root.rename(foreign)
    binary_root.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="executable directory"):
        verify_cached_runtime(wheel, root, sha256, payload_sha256)


@pytest.mark.parametrize("field", ("home", "executable"))
def test_virtual_environment_cannot_select_a_foreign_alias_of_verified_python(
    cached_fixture: tuple[Path, Path, str, str], tmp_path: Path, field: str
) -> None:
    """Resolving a configured alias can hide a different standard-library root."""
    wheel, root, sha256, payload_sha256 = cached_fixture
    base = root / "interpreter/python/bin/python3.12"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "bin").symlink_to(base.parent, target_is_directory=True)
    (foreign / "lib/python3.12").mkdir(parents=True)
    (foreign / "lib/python3.12/sitecustomize.py").write_text("unverified code")
    configuration = root / "runtime/pyvenv.cfg"
    configured = foreign / "bin" if field == "home" else foreign / "bin/python3.12"
    original = base.parent if field == "home" else base
    configuration.write_text(
        configuration.read_text().replace(f"{field} = {original}", f"{field} = {configured}")
    )
    with pytest.raises(ValueError, match="foreign interpreter"):
        verify_cached_runtime(wheel, root, sha256, payload_sha256)
