"""Engine package installation: fetch, verify, unpack, publish, activate, roll back, remove."""

from __future__ import annotations

import hashlib
import io
import tarfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.extensions.runtime_artifacts import RuntimeTrust
from skulk.provisioning import engine_install
from skulk.provisioning.engine_install import (
    DirectoryPackageSource,
    HttpPackageSource,
    activate_engine_generation,
    active_engine,
    install_engine_package,
    installed_engine_generations,
    remove_engine_generation,
    rollback_engine,
    verify_installed_engine,
)
from skulk.provisioning.engine_packages import (
    EnginePackageRefusedError,
    VerifiedEnginePackage,
    verify_engine_manifest,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    NOW,
    PUBLISHER,
    engine_claims,
    public_hex,
    sign_engine_claims,
)

ENGINE_ID = "foxlight.tensorfold"
ENGINE = b"#!/bin/sh\necho engine\n"
LIBRARY = b"library bytes " * 64
ADAPTER = b"print('adapter')\n"
REQUIREMENT = b"pinned requirement bytes " * 32
PACKAGE_HOST = "https://packages.example.invalid/engines/"
REQUIREMENT_URL = "https://vendor.example.invalid/whl/vendor_runtime-1.0-py3-none-any.whl"


class _Member(NamedTuple):
    """One archive member: a file with content, or a link with a target."""

    name: str
    content: bytes = b""
    mode: int = 0o644
    link: str | None = None


def _tar(members: tuple[_Member, ...]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for member in members:
            info = tarfile.TarInfo(member.name)
            info.mode = member.mode
            if member.link is not None:
                info.type = tarfile.SYMTYPE
                info.linkname = member.link
                archive.addfile(info)
            else:
                info.size = len(member.content)
                archive.addfile(info, io.BytesIO(member.content))
    return stream.getvalue()


def _zip(members: tuple[_Member, ...]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for member in members:
            info = zipfile.ZipInfo(member.name)
            info.external_attr = (0o100000 | member.mode) << 16
            archive.writestr(info, member.content)
    return stream.getvalue()


GOOD_MEMBERS = (
    _Member("bin/engine", ENGINE, 0o755),
    _Member("lib/libengine.so", LIBRARY),
    _Member("lib/libengine.so.1", link="libengine.so"),
)


@dataclass(frozen=True)
class _Built:
    """A verified package and the directory holding its files."""

    package: VerifiedEnginePackage
    files: Path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _build(
    root: Path,
    key: Ed25519PrivateKey,
    *,
    generation: int = 1,
    archive: bytes | None = None,
    archive_name: str = "engine.tar.gz",
    executable: str = "bin/engine",
    unpacked_size: int = 1_000_000,
    lane: str = "rocm",
) -> _Built:
    """Sign a manifest for real files and lay those files out for import."""
    content = archive if archive is not None else _tar(GOOD_MEMBERS)
    files = root / f"import-{lane}-{generation}"
    files.mkdir(parents=True, exist_ok=True)
    (files / archive_name).write_bytes(content)
    (files / "adapter.py").write_bytes(ADAPTER)
    (files / "vendor_runtime-1.0-py3-none-any.whl").write_bytes(REQUIREMENT)
    claims = engine_claims()
    claims["generation"] = generation
    claims["lane"] = lane
    claims["archive"] = {
        "filename": archive_name,
        "size": len(content),
        "sha256": _sha256(content),
        "unpacked_size": unpacked_size,
        "executable": executable,
    }
    claims["adapter"] = {
        "filename": "adapter.py",
        "size": len(ADAPTER),
        "sha256": _sha256(ADAPTER),
        "kind": "python",
        "python_requires": ">=3.12",
    }
    claims["external_requirements"] = [
        {
            "name": "vendor-runtime",
            "version": "1.0",
            "filename": "vendor_runtime-1.0-py3-none-any.whl",
            "size": len(REQUIREMENT),
            "sha256": _sha256(REQUIREMENT),
            "url": REQUIREMENT_URL,
        }
    ]
    trust = RuntimeTrust(
        revision=1, expires_at=NOW + 86_400, publishers={PUBLISHER: public_hex(key)}
    )
    package = verify_engine_manifest(sign_engine_claims(key, claims), trust, now=NOW)
    return _Built(package, files)


def _refused(built: _Built, root: Path, *, stop: threading.Event | None = None) -> str:
    with pytest.raises(EnginePackageRefusedError) as caught:
        install_engine_package(
            built.package, DirectoryPackageSource(built.files), root=root, stop=stop
        )
    return caught.value.code


def _leftovers(root: Path) -> list[str]:
    lane = root / ENGINE_ID / "rocm"
    return sorted(entry.name for entry in lane.iterdir()) if lane.is_dir() else []


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def packages(tmp_path: Path) -> Path:
    return tmp_path / "packages"


def test_install_from_files_lays_out_a_complete_generation(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key)

    installed = install_engine_package(
        built.package, DirectoryPackageSource(built.files), root=packages
    )

    assert installed.root == packages / ENGINE_ID / "rocm" / built.package.digest
    assert installed.executable.read_bytes() == ENGINE
    assert installed.adapter.read_bytes() == ADAPTER
    assert [path.read_bytes() for path in installed.requirements.values()] == [
        REQUIREMENT
    ]
    assert (installed.root / "manifest.json").read_bytes() == built.package.document
    assert (installed.root / "engine/lib/libengine.so.1").is_symlink()
    # Installing does not activate.
    assert active_engine(ENGINE_ID, "rocm", root=packages) is None
    assert _leftovers(packages) == [built.package.digest, "install.lock"]


def test_an_installed_generation_is_returned_without_fetching_again(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key)
    first = install_engine_package(
        built.package, DirectoryPackageSource(built.files), root=packages
    )

    second = install_engine_package(
        built.package, DirectoryPackageSource(tmp_path / "absent"), root=packages
    )

    assert second.root == first.root


def test_a_zip_archive_installs_with_its_executable_bit(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    members = (_Member("bin/engine", ENGINE, 0o755), _Member("lib/data", LIBRARY))
    built = _build(tmp_path, key, archive=_zip(members), archive_name="engine.zip")

    installed = install_engine_package(
        built.package, DirectoryPackageSource(built.files), root=packages
    )

    assert installed.executable.read_bytes() == ENGINE
    assert installed.executable.stat().st_mode & 0o100


@pytest.mark.parametrize("change", ["truncate", "alter", "remove"])
def test_a_wrong_or_missing_file_installs_nothing(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path, change: str
) -> None:
    built = _build(tmp_path, key)
    archive = built.files / "engine.tar.gz"
    content = archive.read_bytes()
    if change == "truncate":
        archive.write_bytes(content[:-1])
    elif change == "alter":
        archive.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    else:
        archive.unlink()

    expected = "source_unavailable" if change == "remove" else "archive_corrupt"
    assert _refused(built, packages) == expected
    assert _leftovers(packages) == ["install.lock"]


@pytest.mark.parametrize(
    "members",
    [
        (_Member("bin/engine", ENGINE, 0o755), _Member("../escaped", b"x")),
        (_Member("bin/engine", ENGINE, 0o755), _Member("/tmp/absolute", b"x")),
        (_Member("bin/engine", ENGINE, 0o755), _Member("lib/out", link="/etc/hosts")),
        (_Member("bin/engine", ENGINE, 0o755), _Member("lib/up", link="../../../up")),
    ],
)
def test_an_archive_that_reaches_outside_itself_is_refused(
    tmp_path: Path,
    key: Ed25519PrivateKey,
    packages: Path,
    members: tuple[_Member, ...],
) -> None:
    built = _build(tmp_path, key, archive=_tar(members))

    assert _refused(built, packages) == "archive_corrupt"
    assert _leftovers(packages) == ["install.lock"]
    assert not (packages / ENGINE_ID / "escaped").exists()
    assert not (packages / ENGINE_ID / "rocm" / "escaped").exists()


def test_a_zip_member_outside_the_archive_is_refused(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    members = (_Member("bin/engine", ENGINE, 0o755), _Member("../escaped", b"x"))
    built = _build(tmp_path, key, archive=_zip(members), archive_name="engine.zip")

    assert _refused(built, packages) == "archive_corrupt"
    assert _leftovers(packages) == ["install.lock"]


def test_an_archive_larger_than_declared_is_refused(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key, unpacked_size=len(ENGINE) + len(LIBRARY) - 1)

    assert _refused(built, packages) == "archive_corrupt"
    assert _leftovers(packages) == ["install.lock"]


@pytest.mark.parametrize(
    "members",
    [
        (_Member("bin/other", ENGINE, 0o755),),
        (_Member("bin/engine", ENGINE, 0o644),),
        (_Member("bin/engine", link="/bin/sh"),),
    ],
)
def test_an_archive_without_a_usable_executable_is_refused(
    tmp_path: Path,
    key: Ed25519PrivateKey,
    packages: Path,
    members: tuple[_Member, ...],
) -> None:
    built = _build(tmp_path, key, archive=_tar(members))

    assert _refused(built, packages) == "archive_corrupt"


def test_too_little_free_space_is_refused_before_anything_is_fetched(
    tmp_path: Path,
    key: Ed25519PrivateKey,
    packages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Usage(NamedTuple):
        total: int
        used: int
        free: int

    built = _build(tmp_path, key)

    def nearly_full(path: Path) -> _Usage:
        del path
        return _Usage(total=10**12, used=10**12 - 10**6, free=10**6)

    monkeypatch.setattr(engine_install.shutil, "disk_usage", nearly_full)

    assert _refused(built, packages) == "insufficient_space"
    assert _leftovers(packages) == ["install.lock"]


def test_a_cancelled_installation_leaves_nothing_behind(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key)
    stop = threading.Event()
    stop.set()

    assert _refused(built, packages, stop=stop) == "install_cancelled"
    assert _leftovers(packages) == ["install.lock"]


def test_what_an_interrupted_installation_left_is_swept(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key)
    lane = packages / ENGINE_ID / "rocm"
    for name in (".staging-old", ".damaged-old", ".removing-old"):
        (lane / name / "engine").mkdir(parents=True)

    install_engine_package(
        built.package, DirectoryPackageSource(built.files), root=packages
    )

    assert _leftovers(packages) == [built.package.digest, "install.lock"]


@pytest.mark.parametrize("damage", ["alter", "add", "delete", "permission"])
def test_a_changed_generation_is_noticed_and_reinstalled(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path, damage: str
) -> None:
    built = _build(tmp_path, key)
    source = DirectoryPackageSource(built.files)
    installed = install_engine_package(built.package, source, root=packages)
    library = installed.root / "engine/lib/libengine.so"
    if damage == "alter":
        # Same size, different bytes: only hashing notices.
        library.write_bytes(LIBRARY[:-1] + b"!")
    elif damage == "add":
        (installed.root / "engine/lib/extra.so").write_bytes(b"extra")
    elif damage == "delete":
        library.unlink()
    else:
        installed.executable.chmod(0o600)

    with pytest.raises(EnginePackageRefusedError) as caught:
        verify_installed_engine(installed.root)
    assert caught.value.code == "installation_damaged"
    if damage != "alter":
        with pytest.raises(EnginePackageRefusedError):
            verify_installed_engine(installed.root, deep=False)

    repaired = install_engine_package(built.package, source, root=packages)

    assert verify_installed_engine(repaired.root).root == installed.root
    assert _leftovers(packages) == [built.package.digest, "install.lock"]


def test_a_generation_under_the_wrong_name_is_refused(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key)
    installed = install_engine_package(
        built.package, DirectoryPackageSource(built.files), root=packages
    )
    renamed = installed.root.with_name("0" * 64)
    installed.root.rename(renamed)

    with pytest.raises(EnginePackageRefusedError) as caught:
        verify_installed_engine(renamed)
    assert caught.value.code == "installation_damaged"

    with pytest.raises(EnginePackageRefusedError) as missing:
        verify_installed_engine(installed.root)
    assert missing.value.code == "generation_missing"


def test_activation_rollback_and_removal(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    first = _build(tmp_path, key, generation=1)
    second = _build(tmp_path, key, generation=2)
    for built in (first, second):
        install_engine_package(
            built.package, DirectoryPackageSource(built.files), root=packages
        )
    old, new = first.package.digest, second.package.digest
    assert installed_engine_generations(ENGINE_ID, "rocm", root=packages) == tuple(
        sorted((old, new))
    )

    activate_engine_generation(ENGINE_ID, "rocm", old, root=packages)
    activate_engine_generation(ENGINE_ID, "rocm", new, root=packages)
    active = active_engine(ENGINE_ID, "rocm", root=packages)
    assert active is not None and active.digest == new
    assert active.claims.generation == 2

    # The active generation cannot be removed, and the refusal changes nothing.
    with pytest.raises(EnginePackageRefusedError) as refused:
        remove_engine_generation(ENGINE_ID, "rocm", new, root=packages)
    assert refused.value.code == "generation_active"

    assert rollback_engine(ENGINE_ID, "rocm", root=packages).digest == old
    # A second rollback undoes the first.
    assert rollback_engine(ENGINE_ID, "rocm", root=packages).digest == new

    remove_engine_generation(ENGINE_ID, "rocm", old, root=packages)
    assert installed_engine_generations(ENGINE_ID, "rocm", root=packages) == (new,)
    with pytest.raises(EnginePackageRefusedError) as nothing_earlier:
        rollback_engine(ENGINE_ID, "rocm", root=packages)
    assert nothing_earlier.value.code == "generation_missing"
    still = active_engine(ENGINE_ID, "rocm", root=packages)
    assert still is not None and still.digest == new


def test_activating_the_same_generation_twice_keeps_the_rollback_generation(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    first = _build(tmp_path, key, generation=1)
    second = _build(tmp_path, key, generation=2)
    for built in (first, second):
        install_engine_package(
            built.package, DirectoryPackageSource(built.files), root=packages
        )
    activate_engine_generation(ENGINE_ID, "rocm", first.package.digest, root=packages)
    activate_engine_generation(ENGINE_ID, "rocm", second.package.digest, root=packages)
    activate_engine_generation(ENGINE_ID, "rocm", second.package.digest, root=packages)

    assert rollback_engine(ENGINE_ID, "rocm", root=packages).digest == (
        first.package.digest
    )


def test_a_missing_or_damaged_generation_cannot_be_activated(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    first = _build(tmp_path, key, generation=1)
    second = _build(tmp_path, key, generation=2)
    install_engine_package(
        first.package, DirectoryPackageSource(first.files), root=packages
    )
    activate_engine_generation(ENGINE_ID, "rocm", first.package.digest, root=packages)

    with pytest.raises(EnginePackageRefusedError) as missing:
        activate_engine_generation(
            ENGINE_ID, "rocm", second.package.digest, root=packages
        )
    assert missing.value.code == "generation_missing"

    damaged = install_engine_package(
        second.package, DirectoryPackageSource(second.files), root=packages
    )
    (damaged.root / "engine/lib/libengine.so").write_bytes(LIBRARY[:-1] + b"!")
    with pytest.raises(EnginePackageRefusedError) as refused:
        activate_engine_generation(
            ENGINE_ID, "rocm", second.package.digest, root=packages
        )
    assert refused.value.code == "installation_damaged"

    # Both refusals left the first generation active.
    active = active_engine(ENGINE_ID, "rocm", root=packages)
    assert active is not None and active.digest == first.package.digest


def test_lanes_of_one_engine_are_installed_and_activated_separately(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    rocm = _build(tmp_path, key, lane="rocm")
    vulkan = _build(tmp_path, key, lane="vulkan")
    for built in (rocm, vulkan):
        install_engine_package(
            built.package, DirectoryPackageSource(built.files), root=packages
        )
    activate_engine_generation(ENGINE_ID, "rocm", rocm.package.digest, root=packages)

    assert active_engine(ENGINE_ID, "vulkan", root=packages) is None
    assert installed_engine_generations(ENGINE_ID, "vulkan", root=packages) == (
        vulkan.package.digest,
    )
    assert installed_engine_generations("foxlight.other", "rocm", root=packages) == ()


def test_an_unreadable_active_record_is_reported_not_guessed(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key)
    install_engine_package(
        built.package, DirectoryPackageSource(built.files), root=packages
    )
    activate_engine_generation(ENGINE_ID, "rocm", built.package.digest, root=packages)
    (packages / ENGINE_ID / "rocm" / "active.json").write_bytes(b"{not json")

    with pytest.raises(EnginePackageRefusedError) as caught:
        active_engine(ENGINE_ID, "rocm", root=packages)
    assert caught.value.code == "installation_damaged"


def _served(built: _Built) -> dict[str, bytes]:
    return {
        PACKAGE_HOST + "engine.tar.gz": (built.files / "engine.tar.gz").read_bytes(),
        PACKAGE_HOST + "adapter.py": ADAPTER,
        REQUIREMENT_URL: REQUIREMENT,
    }


def _transport(served: dict[str, bytes], seen: list[str]) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        content = served.get(str(request.url))
        if content is None:
            return httpx.Response(404)
        return httpx.Response(200, content=content)

    return httpx.MockTransport(respond)


def test_install_over_https_reads_the_package_host_and_the_pinned_address(
    tmp_path: Path, key: Ed25519PrivateKey, packages: Path
) -> None:
    built = _build(tmp_path, key)
    seen: list[str] = []
    source = HttpPackageSource(PACKAGE_HOST, transport=_transport(_served(built), seen))

    installed = install_engine_package(built.package, source, root=packages)

    assert installed.executable.read_bytes() == ENGINE
    assert sorted(seen) == sorted(_served(built))


@pytest.mark.parametrize(
    ("address", "content", "code"),
    [
        (PACKAGE_HOST + "adapter.py", None, "source_unavailable"),
        (PACKAGE_HOST + "adapter.py", ADAPTER + b"more", "archive_corrupt"),
        (PACKAGE_HOST + "adapter.py", ADAPTER[:-1], "archive_corrupt"),
        (REQUIREMENT_URL, b"x" * len(REQUIREMENT), "archive_corrupt"),
    ],
)
def test_a_bad_answer_over_https_installs_nothing(
    tmp_path: Path,
    key: Ed25519PrivateKey,
    packages: Path,
    address: str,
    content: bytes | None,
    code: str,
) -> None:
    built = _build(tmp_path, key)
    served = _served(built)
    if content is None:
        del served[address]
    else:
        served[address] = content
    source = HttpPackageSource(PACKAGE_HOST, transport=_transport(served, []))

    with pytest.raises(EnginePackageRefusedError) as caught:
        install_engine_package(built.package, source, root=packages)

    assert caught.value.code == code
    assert _leftovers(packages) == ["install.lock"]


def test_the_package_host_must_be_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        HttpPackageSource("http://packages.example.invalid/engines/")
