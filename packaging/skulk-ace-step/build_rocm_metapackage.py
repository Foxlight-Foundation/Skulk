"""Reproduce the one pinned pure ROCm metapackage using offline verified tools.

This is a producer operation. Runtime preparation installs only the resulting
sealed wheels and never downloads or executes a source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

from build_ace_step import ArchiveInventory
from package_inputs import (
    PROFILE_TARGETS,
    extract_python,
    file_digest,
    read_python_catalog,
    safe_member,
)
from packaging.tags import platform_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version
from wheel_inputs import clean_revision


def reproduce_metapackage(
    repository: Path,
    *,
    builder_profile: str,
    python_archive: Path,
    source_archive: Path,
    setuptools_wheel: Path,
    output: Path,
) -> Path:
    """Build twice and require the exact committed output, source and tool pins.

    The fresh output stays outside the clean input repository. Only the known
    official ROCm metapackage is admitted, with an explicit compiled family and
    reproducibility clock. This proves byte reproduction, not a GPU/loader ABI.
    Return the collected wheel after both independent builds match its pin.
    """
    repository = repository.resolve()
    python_archive = python_archive.resolve()
    source_archive = source_archive.resolve()
    setuptools_wheel = setuptools_wheel.resolve()
    output = output.resolve()
    revision = clean_revision(repository)
    if output.resolve().is_relative_to(repository.resolve()):
        raise ValueError("native metapackage output must stay outside clean inputs")
    root = repository / "packaging/skulk-ace-step"
    with (root / "archives/linux-amd64-rocm-py312.json").open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ValueError("native source-build inventory exceeds its bound")
    inventory = ArchiveInventory.model_validate_json(raw)
    selected = tuple(item for item in inventory.objects if item.source_build is not None)
    if inventory.variant != "rocm" or len(selected) != 1:
        raise ValueError("native build needs exactly one selected ROCm metapackage")
    expected = selected[0]
    build = expected.source_build
    assert build is not None
    target = PROFILE_TARGETS.get(builder_profile)
    if target is None or target not in set(platform_tags()):
        raise ValueError("standalone build interpreter does not match this host")
    python_pin = read_python_catalog(root).for_target(target)
    name, version, _, _ = parse_wheel_filename(setuptools_wheel.name)
    if (
        name != "setuptools" or version != Version(build.setuptools)
        or setuptools_wheel.stat().st_size != build.setuptools_size_bytes
        or file_digest(setuptools_wheel, limit=32 * 1024**2)
        != build.setuptools_archive_sha256
        or source_archive.name != f"rocm-{expected.version}.tar.gz"
        or source_archive.stat().st_size != build.source_size_bytes
        or file_digest(source_archive, limit=build.source_size_bytes) != build.source_sha256
    ):
        raise ValueError("native source-build inputs differ from their committed pins")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    base = extract_python(python_archive, output / "interpreter", python_pin)
    pip_wheels = tuple(
        (base.parents[1] / "lib/python3.12/ensurepip/_bundled").glob("*.whl")
    )
    if len(pip_wheels) != 1:
        raise ValueError("verified Python has an unexpected bootstrap inventory")
    environment = {
        name: value for name, value in os.environ.items()
        if name in {"PATH", "HOME", "TMPDIR", "LANG"}
    } | {
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ROCM_SDK_TARGET_FAMILY": build.target_family,
        "SOURCE_DATE_EPOCH": str(build.source_date_epoch),
    }
    tools = output / "tools"
    subprocess.run(
        [str(base), "-I", "-B", "-m", "venv", "--without-pip", str(tools)],
        env=environment, check=True, timeout=120,
    )
    python = tools / "bin/python"
    lock = output / "build-requirements.lock"
    lock.write_text(
        f"{setuptools_wheel.resolve().as_uri()} --hash=sha256:{build.setuptools_archive_sha256}\n"
    )
    subprocess.run(
        [
            str(python), "-I", "-B", "-c",
            "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')",
            str(pip_wheels[0]), "install", "--no-index", "--no-compile",
            "--require-hashes", "-r", str(lock),
        ], env=environment, check=True, timeout=120,
    )
    results: list[dict[str, str | int]] = []
    wheels: list[Path] = []
    for label in ("first", "second"):
        source = output / label / "source"
        source.mkdir(parents=True)
        with tarfile.open(source_archive) as archive:
            members = archive.getmembers()
            if len(members) > 128 or sum(item.size for item in members) > 2 * 1024**2:
                raise ValueError("native metapackage source exceeds its expansion bound")
            for member in members:
                path = safe_member(member.name)
                if (
                    not (member.isfile() or member.isdir())
                    or path.parts[0] != f"rocm-{expected.version}"
                ):
                    raise ValueError("native source includes an unadmitted member")
            archive.extractall(source, members=members, filter="data")
        produced = output / label / "wheels"
        produced.mkdir()
        subprocess.run(
            [
                str(python), "-I", "-B", "-c",
                "import setuptools.build_meta,sys;print(setuptools.build_meta.build_wheel(sys.argv[1]))",
                str(produced),
            ], cwd=source / f"rocm-{expected.version}", env=environment,
            check=True, timeout=120,
        )
        candidates = tuple(produced.iterdir())
        if len(candidates) != 1 or candidates[0].name != expected.filename:
            raise ValueError("native metapackage build emitted unexpected objects")
        wheel = candidates[0]
        digest = file_digest(wheel, limit=expected.size_bytes)
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.getinfo(f"rocm-{expected.version}.dist-info/METADATA")
            if metadata.file_size > 2 * 1024**2:
                raise ValueError("native metapackage metadata exceeds its bound")
            metadata_digest = hashlib.sha256(archive.read(metadata)).hexdigest()
        if (
            wheel.stat().st_size != expected.size_bytes or digest != expected.sha256
            or metadata_digest != build.metadata_sha256
        ):
            raise ValueError("native metapackage reproduction differs from its selected bytes")
        wheels.append(wheel)
        results.append(dict(label=label, sha256=digest, size_bytes=wheel.stat().st_size))
    if clean_revision(repository) != revision:
        raise ValueError("native metapackage inputs changed during reproduction")
    collected = output / "wheels"
    collected.mkdir()
    result = collected / expected.filename
    result.hardlink_to(wheels[0])
    (output / "provenance.json").write_text(json.dumps(dict(
        adapter_revision=revision, source=build.model_dump(mode="json"),
        results=results, filename=expected.filename,
        scope="Pinned pure metapackage byte reproduction; no GPU/loader/model qualification.",
    ), indent=2) + "\n")
    return result


class Arguments(argparse.Namespace):
    """Explicit retained build inputs; no download or publication action."""

    repository: Path
    builder_profile: str
    python_archive: Path
    source_archive: Path
    setuptools_wheel: Path
    output: Path


def main() -> None:
    """Reproduce only the source-controlled ROCm metapackage in a fresh output."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "python-archive", "source-archive", "setuptools-wheel", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--builder-profile", choices=tuple(PROFILE_TARGETS), required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    print(reproduce_metapackage(
        arguments.repository, builder_profile=arguments.builder_profile,
        python_archive=arguments.python_archive, source_archive=arguments.source_archive,
        setuptools_wheel=arguments.setuptools_wheel, output=arguments.output,
    ))


if __name__ == "__main__":
    main()
