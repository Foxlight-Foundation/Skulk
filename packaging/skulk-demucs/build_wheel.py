"""Assemble the closed HTDemucs adapter and exact offline dependency wheels.

The producer has no upload or promotion action. Source and dependency
identities are validated before assembling an independently verifiable candidate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Literal

from package_inputs import (
    PYTHON_VERSION,
    TARGETS,
    locked_dependencies,
    read_python_catalog,
    verify_python_archive,
)
from packaging.metadata import Metadata, RawMetadata, parse_email
from packaging.requirements import Requirement
from packaging.tags import Tag, compatible_tags, cpython_tags, mac_platforms
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field

VERSION = "0.1.0"
MAX_WHEEL_BYTES = 512 * 1024 * 1024
UPSTREAM_MODULES = (
    "__init__.py",
    "apply.py",
    "demucs.py",
    "hdemucs.py",
    "htdemucs.py",
    "hf.py",
    "spec.py",
    "states.py",
    "transformer.py",
    "utils.py",
    "wiener.py",
)
ADAPTER_MODULES: frozenset[str] = frozenset(
    (
        "__init__.py",
        "__main__.py",
        "bindings.py",
        "contracts.py",
        "engine.py",
        "server.py",
        "media.py",
        "py.typed",
    )
)


class SourcePin(BaseModel):
    """Reviewed immutable inference-library source; never runtime download input."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    repository: Literal["https://github.com/adefossez/demucs.git"]
    revision: str = Field(
        pattern=r"^[a-f0-9]{40}$", description="Immutable source commit."
    )


class Arguments(argparse.Namespace):
    """Explicit local build inputs and an architecture-selected package target."""

    repository: Path
    source: Path
    wrapper: Path
    wheelhouse: Path
    lock: Path
    output: Path
    target: str
    python_archive: Path


def target_tags(target: str) -> list[Tag]:
    """Order compatible platform and interpreter tags as an installer would."""
    if target not in TARGETS:
        raise ValueError("unknown native separation package target")
    if target.startswith("macosx_"):
        platforms = list(mac_platforms(version=(14, 0), arch="arm64"))
    else:
        architecture = target.removeprefix("manylinux_2_28_")
        platforms = [
            f"manylinux_2_{minor}_{architecture}" for minor in range(28, 16, -1)
        ]
        platforms.append(f"manylinux2014_{architecture}")
    return list(cpython_tags((3, 12), abis=["cp312"], platforms=platforms)) + list(
        compatible_tags((3, 12), interpreter="cp312", platforms=platforms)
    )


def dependency_matches(filename: str, target: str) -> bool:
    """Require the selected CPU/OS/CPython ABI rather than only a lock hash."""
    return bool(parse_wheel_filename(filename)[3] & set(target_tags(target)))


def clean_revision(root: Path) -> str:
    """Require reviewable committed input before emitting a distributable candidate."""
    revision = (
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"])
        .decode()
        .strip()
    )
    if re.fullmatch(r"[a-f0-9]{40}", revision) is None or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"]
    ):
        raise ValueError(
            "native package inputs must belong to a clean immutable commit"
        )
    return revision


def validate_dependency_closure(
    wheels: dict[str, bytes],
    required: dict[str, tuple[Version, frozenset[str]]],
    direct: bytes,
    target: str,
) -> None:
    """Check actual wheel metadata against the source and every transitive need."""
    for line in direct.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        dependency = Requirement(line)
        specifiers = tuple(dependency.specifier)
        if (
            dependency.url is not None
            or dependency.marker is not None
            or dependency.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError("adapter source must specify closed exact dependencies")
        name = canonicalize_name(dependency.name)
        if name not in required or not dependency.specifier.contains(
            required[name][0], prereleases=True
        ):
            raise ValueError("native dependency lock does not satisfy adapter source")
    environment: dict[str, str] = {
        "implementation_name": "cpython",
        "implementation_version": PYTHON_VERSION,
        "os_name": "posix",
        "platform_machine": "arm64"
        if target.startswith("macosx_")
        else target.removeprefix("manylinux_2_28_"),
        "platform_release": "",
        "platform_system": "Darwin" if target.startswith("macosx_") else "Linux",
        "platform_version": "",
        "python_full_version": PYTHON_VERSION,
        "platform_python_implementation": "CPython",
        "python_version": "3.12",
        "sys_platform": "darwin" if target.startswith("macosx_") else "linux",
        "extra": "",
    }
    for filename, data in wheels.items():
        distribution, version, _, _ = parse_wheel_filename(filename)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            paths = [
                name
                for name in archive.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            if len(paths) != 1:
                raise ValueError("dependency wheel needs one complete metadata record")
            if archive.getinfo(paths[0]).file_size > 2 * 1024 * 1024:
                raise ValueError("native dependency metadata exceeds its bound")
            raw, unparsed = parse_email(archive.read(paths[0]))
            if unparsed.keys() & {
                "metadata-version",
                "name",
                "version",
                "requires-python",
                "requires-dist",
            }:
                raise ValueError(
                    "native dependency identity or requirements are malformed"
                )
            # Some pinned upstream wheels use a newer License-File header with
            # older metadata versions. Preserve all component notices verbatim;
            # strictly validate the identity and requirements projection used
            # for closure rather than rejecting an unrelated metadata field.
            projected: RawMetadata = {}
            if "metadata_version" in raw:
                projected["metadata_version"] = raw["metadata_version"]
            if "name" in raw:
                projected["name"] = raw["name"]
            if "version" in raw:
                projected["version"] = raw["version"]
            if "requires_python" in raw:
                projected["requires_python"] = raw["requires_python"]
            if "requires_dist" in raw:
                projected["requires_dist"] = raw["requires_dist"]
            metadata = Metadata.from_raw(projected, validate=True)
            if (
                canonicalize_name(metadata.name) != distribution
                or metadata.version != version
            ):
                raise ValueError(
                    "dependency filename and native metadata identity disagree"
                )
            if (
                metadata.requires_python is not None
                and not metadata.requires_python.contains(PYTHON_VERSION)
            ):
                raise ValueError(
                    "dependency does not support the pinned native interpreter"
                )
            for dependency in metadata.requires_dist or ():
                if dependency.marker is not None and not dependency.marker.evaluate(
                    environment
                ):
                    continue
                name = canonicalize_name(dependency.name)
                if (
                    dependency.url is not None
                    or dependency.extras
                    or name not in required
                    or not dependency.specifier.contains(
                        required[name][0], prereleases=True
                    )
                ):
                    raise ValueError(
                        "offline dependency metadata closure is incomplete"
                    )
            if distribution == "soundfile":
                suffix = ".dylib" if target.startswith("macosx_") else ".so"
                if not any(
                    name.startswith("_soundfile_data/libsndfile")
                    and name.endswith(suffix)
                    for name in archive.namelist()
                ):
                    raise ValueError("SoundFile must bundle its target audio library")


def assemble_wheel(
    repository: Path,
    source: Path,
    wrapper: Path,
    wheelhouse: Path,
    lock_path: Path,
    output: Path,
    *,
    target: str,
    python_archive: Path,
    variant: Literal["cpu"] = "cpu",
) -> Path:
    """Seal known source and complete platform wheels into deterministic bytes.

    No model weights, developer paths, environment values or network resolution
    enter the package. The model remains a separate immutable registry artifact.
    """
    if target not in TARGETS:
        raise ValueError("native package variant and target are incompatible")
    canonical_wrapper = repository / "packaging" / "skulk-demucs" / "skulk_demucs"
    if wrapper.resolve() != canonical_wrapper.resolve():
        raise ValueError("adapter must come from the declared clean source repository")
    adapter_revision = clean_revision(repository)
    source_revision = clean_revision(source)
    source_document = SourcePin.model_validate_json(
        (wrapper.parent / "source.json").read_bytes()
    )
    if source_document.revision != source_revision:
        raise ValueError("native library differs from the reviewed source pin")
    python_catalog = read_python_catalog(wrapper.parent)
    python_pin = python_catalog.for_target(target)
    verify_python_archive(python_archive, python_pin)
    lock = lock_path.read_bytes()
    required = locked_dependencies(lock)
    direct = (wrapper.parent / "requirements.in").read_bytes()
    payload: dict[str, bytes] = {}
    for name in UPSTREAM_MODULES:
        payload[f"demucs/{name}"] = (source / "demucs" / name).read_bytes()
    present = {path.name for path in wrapper.iterdir() if path.name != "__pycache__"}
    if present.symmetric_difference(ADAPTER_MODULES):
        raise ValueError("native adapter members differ from the reviewed inventory")
    for path in sorted(wrapper.iterdir()):
        if path.name == "__pycache__":
            continue
        if (
            not path.is_file()
            or path.name not in ADAPTER_MODULES
            or not path.resolve().is_relative_to(repository.resolve())
        ):
            raise ValueError("unexpected file in the closed native adapter")
        # read_bytes dereferences the repository's canonical contract link;
        # standalone packages contain regular files, never developer symlinks.
        payload[f"skulk_demucs/{path.name}"] = path.read_bytes()
    seen: set[str] = set()
    dependencies: dict[str, bytes] = {}
    for path in sorted(wheelhouse.iterdir()):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix != ".whl"
            or path.stat().st_size > MAX_WHEEL_BYTES
            or not dependency_matches(path.name, target)
        ):
            raise ValueError(
                "dependency directory contains a foreign target or non-wheel"
            )
        name, version, _, _ = parse_wheel_filename(path.name)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if (
            name in seen
            or name not in required
            or required[name][0] != version
            or digest not in required[name][1]
        ):
            raise ValueError("unlocked, changed or duplicate dependency wheel")
        seen.add(name)
        dependencies[path.name] = data
        payload[f"skulk_demucs/wheels/{path.name}"] = data
    if seen != required.keys():
        raise ValueError("offline dependency set is incomplete")
    validate_dependency_closure(dependencies, required, direct, target)
    payload["skulk_demucs/requirements.in"] = direct
    payload["skulk_demucs/requirements.lock"] = lock
    payload["skulk_demucs/python.json"] = (
        python_catalog.model_dump_json(indent=2).encode() + b"\n"
    )
    payload[f"skulk_demucs/python/{python_pin.filename}"] = python_archive.read_bytes()
    payload["skulk_demucs/licenses/DEMUCS_LICENSE.txt"] = (
        source / "LICENSE"
    ).read_bytes()
    payload["skulk_demucs/licenses/SKULK_LICENSE.txt"] = (
        repository / "LICENSE"
    ).read_bytes()
    members = {
        name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())
    }
    content = {
        "protocol_version": 1,
        "python": PYTHON_VERSION,
        "python_archive": python_pin.model_dump(mode="json"),
        "adapter_revision": adapter_revision,
        "source_revision": source_revision,
        "source_repository": "https://github.com/adefossez/demucs.git",
        "platform_tag": target,
        "variant": variant,
        "members": members,
        "features": [
            "audio.separate",
            "four_stems",
            "pcm24",
            "offline_loader",
            "bundled_python",
            "resident_model",
        ],
    }
    identity = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["skulk_demucs/build.json"] = (
        json.dumps(
            content | {"payload_sha256": identity}, sort_keys=True, indent=2
        ).encode()
        + b"\n"
    )
    distribution = f"skulk_demucs_{variant}"
    metadata = f"{distribution}-{VERSION}.dist-info"
    filename = f"{distribution}-{VERSION}-cp312-cp312-{target}.whl"
    payload[f"{metadata}/METADATA"] = (
        f"Metadata-Version: 2.4\nName: skulk-demucs-{variant}\nVersion: {VERSION}\nRequires-Python: >=3.12,<3.13\nSummary: Isolated WAV source-separation runtime for Skulk\nLicense: See bundled component notices\n\n"
    ).encode()
    payload[f"{metadata}/WHEEL"] = (
        f"Wheel-Version: 1.0\nGenerator: foxlight-demucs\nRoot-Is-Purelib: false\nTag: cp312-cp312-{target}\n".encode()
    )
    records: list[str] = []
    for name, data in sorted(payload.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        records.append(f"{name},sha256={digest},{len(data)}")
    records.append(f"{metadata}/RECORD,,")
    payload[f"{metadata}/RECORD"] = ("\n".join(records) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    # Stored members avoid compression-library differences in independent
    # rebuilds. Dependency wheels already carry their own compression.
    with (
        destination.open("xb") as target_file,
        zipfile.ZipFile(target_file, "w", compression=zipfile.ZIP_STORED) as archive,
    ):
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    descriptor = {
        "filename": filename,
        "size_bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "payload_sha256": identity,
        "adapter_revision": adapter_revision,
        "source_revision": source_revision,
        "platform_tag": target,
        "variant": variant,
        "python": PYTHON_VERSION,
        "python_archive_sha256": python_pin.sha256,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(descriptor, sort_keys=True, indent=2) + "\n"
    )
    return destination


def main() -> None:
    """Assemble explicit local inputs without publishing or resolving dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "repository",
        "source",
        "wrapper",
        "wheelhouse",
        "lock",
        "output",
        "python-archive",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    print(
        assemble_wheel(
            arguments.repository,
            arguments.source,
            arguments.wrapper,
            arguments.wheelhouse,
            arguments.lock,
            arguments.output,
            target=arguments.target,
            python_archive=arguments.python_archive,
        )
    )


if __name__ == "__main__":
    main()
