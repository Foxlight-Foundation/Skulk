"""Install a candidate in a fresh private directory using only its sealed bytes.

This validation tool is not Core's engine provisioner. It cannot publish a package,
reuse an installation or advertise ready capacity. Model weights remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from build_wheel import TARGETS, locked_dependencies
from package_inputs import (
    PYTHON_VERSION,
    PythonArchivePin,
    file_digest,
    verify_python_archive,
)
from pydantic import BaseModel, ConfigDict, Field

MAX_PACKAGE_BYTES = 1024 * 1024 * 1024
MAX_EXPANDED_PYTHON_BYTES = 512 * 1024 * 1024
MAX_INSTALLED_BYTES = 4 * 1024 * 1024 * 1024
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class BuildManifest(BaseModel):
    """Strict payload identity, anchored by the operator's complete package hash."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    protocol_version: Literal[1]
    python: Literal["3.12.13"]
    python_archive: PythonArchivePin
    adapter_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_repository: Literal["https://github.com/adefossez/demucs.git"]
    platform_tag: str
    variant: Literal["cpu"]
    members: dict[str, Digest] = Field(min_length=1, max_length=256)
    features: tuple[str, ...] = Field(max_length=16)
    payload_sha256: Digest


class InterpreterFacts(BaseModel):
    """Minimal facts queried from the isolated interpreter, not the controller."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    version: str
    platform: str
    machine: str
    prefix: str
    purelib: str


def safe_member(name: str) -> PurePosixPath:
    """Refuse absolute paths, traversal and alternative separators before writing."""
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or "\x00" in name
        or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
    ):
        raise ValueError("native archive member is not a safe relative path")
    return path


def extract_python(
    archive_path: Path, destination: Path, pin: PythonArchivePin
) -> Path:
    """Extract only bounded regular files and in-root links from verified Python."""
    verify_python_archive(archive_path, pin)
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    expanded = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for entry in archive:
            path = safe_member(entry.name)
            if path.parts[0] != "python" or path.as_posix() in seen:
                raise ValueError(
                    "standalone Python contains a foreign or duplicate member"
                )
            seen.add(path.as_posix())
            expanded += entry.size
            if len(seen) > 8192 or expanded > MAX_EXPANDED_PYTHON_BYTES:
                raise ValueError("standalone Python exceeds its expanded bound")
            if not (entry.isfile() or entry.isdir() or entry.issym()):
                raise ValueError(
                    "standalone Python contains an unsupported member type"
                )
            # The standard data filter resolves link targets and existing parent
            # links against the destination. Legitimate relative links survive;
            # escaping links and writes through them are refused.
            archive.extract(entry, destination, filter="data")
    executable = destination / pin.executable
    if (
        executable.is_symlink()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError("standalone Python lacks its pinned executable")
    return executable


def unpack_candidate(
    wheel: Path, destination: Path, expected_sha256: str
) -> BuildManifest:
    """Verify the complete object and every payload member before extracting it."""
    if file_digest(wheel, limit=MAX_PACKAGE_BYTES) != expected_sha256:
        raise ValueError("candidate package differs from its expected digest")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) > 256 or len(set(names)) != len(names):
            raise ValueError("candidate has excessive or duplicate members")
        if sum(entry.file_size for entry in entries) > MAX_PACKAGE_BYTES:
            raise ValueError("candidate payload exceeds its expanded bound")
        build = archive.getinfo("skulk_demucs/build.json")
        if build.file_size > 128 * 1024:
            raise ValueError("candidate manifest exceeds its bound")
        manifest = BuildManifest.model_validate_json(archive.read(build))
        content = manifest.model_dump(mode="json", exclude={"payload_sha256"})
        identity = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        metadata_root = "skulk_demucs_cpu-0.1.0.dist-info"
        expected = set(manifest.members) | {
            "skulk_demucs/build.json",
            f"{metadata_root}/METADATA",
            f"{metadata_root}/WHEEL",
            f"{metadata_root}/RECORD",
        }
        if (
            identity != manifest.payload_sha256
            or set(names) != expected
            or manifest.platform_tag not in TARGETS
            or manifest.python_archive.platform_tag != manifest.platform_tag
        ):
            raise ValueError(
                "candidate manifest identity or member inventory disagrees"
            )
        for entry in entries:
            relative = safe_member(entry.filename)
            if not stat.S_ISREG(entry.external_attr >> 16):
                raise ValueError("candidate payload must contain regular files only")
            output = destination.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(entry) as source, output.open("xb") as target:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    target.write(block)
            if (
                entry.filename in manifest.members
                and digest.hexdigest() != manifest.members[entry.filename]
            ):
                raise ValueError("candidate member differs from its manifest digest")
        return manifest


def verify_site_packages(site: Path, wheels: tuple[Path, ...]) -> None:
    """Check expanded code/data against trusted wheels, never installed RECORDs.

    Pip may rewrite RECORD and add its small installer receipts. Everything else
    in site-packages, including .pth files and native libraries, must match an
    original wheel member. Unexpected files and symlinks are refused.
    """
    expected: dict[str, str] = {}
    generated: set[str] = set()
    expanded = 0
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            wheel_names: set[str] = set()
            for entry in archive.infolist():
                path = safe_member(entry.filename)
                name = path.as_posix()
                if name in wheel_names:
                    raise ValueError("native dependency has duplicate members")
                wheel_names.add(name)
                if entry.is_dir():
                    continue
                if path.parts[0].endswith(".data"):
                    # The pinned set has only a SymPy manual outside purelib.
                    # No executable script, header or alternative import root
                    # may escape the explicitly verified native environment.
                    if name.endswith(".data/data/share/man/man1/isympy.1"):
                        continue
                    raise ValueError("unadmitted dependency installation scheme")
                if name.endswith(".dist-info/RECORD"):
                    generated.update(
                        (path.parent / suffix).as_posix()
                        for suffix in (
                            "RECORD",
                            "INSTALLER",
                            "REQUESTED",
                            "direct_url.json",
                        )
                    )
                    continue
                expanded += entry.file_size
                if expanded > MAX_INSTALLED_BYTES or len(expected) > 65536:
                    raise ValueError("installed dependency set exceeds its bound")
                with archive.open(entry) as stream:
                    hasher = hashlib.sha256()
                    while block := stream.read(1024 * 1024):
                        hasher.update(block)
                    digest = hasher.hexdigest()
                if name in expected and expected[name] != digest:
                    raise ValueError(
                        "native distributions overwrite different import bytes"
                    )
                expected[name] = digest
    actual: set[str] = set()
    for path in site.rglob("*"):
        if path.is_symlink():
            raise ValueError("installed native tree contains an unexpected symlink")
        if path.is_dir():
            continue
        name = path.relative_to(site).as_posix()
        actual.add(name)
        if name not in generated and (
            name not in expected
            or file_digest(path, limit=MAX_INSTALLED_BYTES) != expected[name]
        ):
            raise ValueError(
                f"installed native tree differs from trusted package bytes: {name}"
            )
    if not expected.keys() <= actual:
        raise ValueError("installed native tree is incomplete")


def install_candidate(wheel: Path, output: Path, expected_sha256: str) -> Path:
    """Create and verify one fresh offline environment for actual package testing."""
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    manifest = unpack_candidate(wheel, payload, expected_sha256)
    host_target = {
        ("Darwin", "arm64"): "macosx_14_0_arm64",
        ("Linux", "x86_64"): "manylinux_2_28_x86_64",
        ("Linux", "aarch64"): "manylinux_2_28_aarch64",
    }.get((platform.system(), platform.machine()))
    if host_target != manifest.platform_tag:
        raise ValueError("candidate cannot execute on this OS/CPU architecture")
    interpreter = extract_python(
        payload / "skulk_demucs" / "python" / manifest.python_archive.filename,
        output / "interpreter",
        manifest.python_archive,
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP", "LANG"}
    } | {
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_NO_COMPILE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    runtime = output / "runtime"
    subprocess.run(
        [str(interpreter), "-I", "-B", "-m", "venv", "--without-pip", str(runtime)],
        env=environment,
        check=True,
        timeout=120,
    )
    native = runtime / "bin" / "python"
    facts_code = (
        "import json,platform,sys,sysconfig; "
        "print(json.dumps(dict(version=platform.python_version(),platform=sys.platform,"
        "machine=platform.machine(),prefix=sys.prefix,purelib=sysconfig.get_path('purelib'))))"
    )
    facts = InterpreterFacts.model_validate_json(
        subprocess.check_output(
            [str(native), "-I", "-B", "-c", facts_code],
            env=environment,
            timeout=30,
        )
    )
    site = Path(facts.purelib)
    if (
        facts.version != PYTHON_VERSION
        or Path(facts.prefix) != runtime
        or not site.is_relative_to(runtime)
    ):
        raise ValueError("candidate selected the wrong private interpreter")
    offline = payload / "skulk_demucs"
    lock = (offline / "requirements.lock").read_bytes()
    locked_dependencies(lock)
    pip_wheels = tuple(
        (
            output
            / "interpreter"
            / "python"
            / "lib"
            / "python3.12"
            / "ensurepip"
            / "_bundled"
        ).glob("*.whl")
    )
    if len(pip_wheels) != 1:
        raise ValueError("bundled interpreter has an unexpected bootstrap inventory")
    # ensurepip's installation compiles bytecode independently of -B and strips
    # PIP_* settings. Run the already verified bundled pip wheel as a zip import
    # instead: the final environment contains only sealed runtime distributions.
    pip_code = (
        "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
        "runpy.run_module('pip',run_name='__main__')"
    )
    commands = (
        [
            "--require-hashes",
            "--find-links",
            str(offline / "wheels"),
            "-r",
            str(offline / "requirements.lock"),
        ],
        [str(wheel.resolve())],
    )
    for arguments in commands:
        subprocess.run(
            [
                str(native),
                "-I",
                "-B",
                "-c",
                pip_code,
                str(pip_wheels[0]),
                "install",
                "--no-index",
                "--no-compile",
                *arguments,
            ],
            env=environment,
            check=True,
            timeout=600,
        )
    verify_site_packages(site, (wheel, *sorted((offline / "wheels").glob("*.whl"))))
    (output / "installation.json").write_text(
        json.dumps(
            {
                "package_sha256": expected_sha256,
                "payload_sha256": manifest.payload_sha256,
                "python_archive_sha256": manifest.python_archive.sha256,
                "platform_tag": manifest.platform_tag,
                "offline": True,
                "verified_expanded_tree": True,
                "python": facts.version,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return native


class Arguments(argparse.Namespace):
    """Expected package identity and a fresh explicitly owned test destination."""

    wheel: Path
    sha256: str
    output: Path


def main() -> None:
    """Install sealed candidate bytes for native tests without network resolution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    print(
        install_candidate(
            arguments.wheel.resolve(), arguments.output.resolve(), arguments.sha256
        )
    )


if __name__ == "__main__":
    main()
