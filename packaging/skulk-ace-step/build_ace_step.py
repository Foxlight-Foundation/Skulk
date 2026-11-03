"""Seal pinned native ACE source, adapter and offline dependencies; never publish."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from package_inputs import (
    MAX_PACKAGE_BYTES,
    PYTHON_VERSION,
    TARGETS,
    PythonArchivePin,
    file_digest,
    locked_dependencies,
    read_python_catalog,
    safe_member,
    verify_python_archive,
)
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, model_validator
from skulk_ace_step.weights import ModelCodeFile
from wheel_inputs import clean_revision, dependency_matches, validate_dependency_closure

VERSION = "0.1.0"
MAX_OBJECT_BYTES = 512 * 1024**2
ADAPTER_MODULES = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "audio_separation.py",
        "bindings.py",
        "contracts.py",
        "engine.py",
        "generation.py",
        "lifetime.py",
        "media.py",
        "policy.py",
        "server.py",
        "upstream_bindings.py",
        "weights.py",
        "py.typed",
    }
)


class BuildInput(BaseModel):
    """Strict immutable source-controlled producer metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class SourcePin(BuildInput):
    """Exact native library revision, independently of model/artifact downloads."""

    repository: Literal["https://github.com/ace-step/ACE-Step-1.5.git"] = Field(
        description="Fixed upstream library source."
    )
    revision: str = Field(
        pattern=r"^[a-f0-9]{40}$", description="Reviewed immutable source commit."
    )


class SourceFile(BuildInput):
    """One selected original source file, checked even when an override replaces it."""

    path: str = Field(
        min_length=1,
        max_length=512,
        description="Canonical selected Python path under acestep/.",
    )
    size_bytes: int = Field(
        ge=0,
        le=2 * 1024**2,
        description="Exact pinned original file length, including empty package initializers.",
    )
    sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="Digest of original upstream bytes."
    )

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Do not allow traversal, foreign packages or arbitrary support binaries."""
        path = safe_member(self.path)
        if not self.path.startswith("acestep/") or path.suffix != ".py":
            raise ValueError(
                "native source inventory must contain acestep Python files only"
            )
        return self


class SourceOverride(BuildInput):
    """The one reviewed namespace change avoiding eager training/UI dependencies."""

    path: Literal["acestep/training/__init__.py"] = Field(
        description="Only the inference training namespace initializer is overridden."
    )
    source: Literal["patches/inference-training-namespace.py"] = Field(
        description="Owned source-controlled replacement."
    )
    sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="Exact owned replacement digest."
    )
    reason: str = Field(
        min_length=1,
        max_length=512,
        description="Why the explicit source modification exists.",
    )


class EffectiveCode(BuildInput):
    """Known local model implementation mapping; no model-repository Python."""

    source: Literal[
        "acestep/models/base/configuration_acestep_v15.py",
        "acestep/models/base/modeling_acestep_v15_base.py",
        "acestep/models/base/apg_guidance.py",
    ] = Field(description="Selected runtime library Python path.")
    destination: Literal[
        "configuration_acestep_v15.py",
        "modeling_acestep_v15_base.py",
        "apg_guidance.py",
    ] = Field(description="Local model loader module basename.")


class SourceInventory(SourcePin):
    """Bounded complete selection and explicit patch provenance."""

    files: tuple[SourceFile, ...] = Field(
        min_length=1, max_length=192, description="Selected original library source."
    )
    overrides: tuple[SourceOverride, ...] = Field(
        min_length=1,
        max_length=1,
        description="Explicit reviewed namespace replacement.",
    )
    effective_model_code: tuple[EffectiveCode, ...] = Field(
        min_length=3,
        max_length=3,
        description="Three runtime-owned modules copied into the loader view.",
    )

    @model_validator(mode="after")
    def unique_paths(self) -> Self:
        """Refuse shadowed sources, unknown override roots or duplicate model roles."""
        paths = [item.path for item in self.files]
        if len(set(paths)) != len(paths) or any(
            item.path not in paths for item in self.overrides
        ):
            raise ValueError(
                "native source inventory has duplicate or missing override paths"
            )
        if {item.destination for item in self.effective_model_code} != {
            "configuration_acestep_v15.py",
            "modeling_acestep_v15_base.py",
            "apg_guidance.py",
        }:
            raise ValueError(
                "native source inventory lacks its complete owned model code"
            )
        return self


class ArchiveObject(BuildInput):
    """One selected immutable public wheel object, without runtime downloading."""

    distribution: str = Field(
        min_length=1, max_length=128, description="Canonical distribution identity."
    )
    version: str = Field(
        min_length=1,
        max_length=128,
        description="Exact version including local CPU build suffix.",
    )
    filename: str = Field(
        min_length=1, max_length=256, description="Exact compatible wheel basename."
    )
    size_bytes: int = Field(
        gt=0, le=MAX_OBJECT_BYTES, description="Exact complete archive size."
    )
    sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="SHA-256 from immutable intake."
    )
    source_url: str = Field(
        min_length=1,
        max_length=2048,
        description="Original public object URL retained as provenance only.",
    )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Match name/version/basename and retain only public HTTPS provenance."""
        if Path(self.filename).name != self.filename or "\\" in self.filename:
            raise ValueError("native dependency filename must be a basename")
        name, version, _, _ = parse_wheel_filename(self.filename)
        url = urlsplit(self.source_url)
        if (
            name != self.distribution
            or canonicalize_name(self.distribution) != self.distribution
            or version != Version(self.version)
            or url.scheme != "https"
            or url.netloc
            not in {
                "files.pythonhosted.org",
                "download.pytorch.org",
                "download-r2.pytorch.org",
            }
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "native dependency intake identity or public provenance is invalid"
            )
        return self


class ArchiveInventory(BuildInput):
    """One exact architecture's wheel and interpreter selection."""

    lock_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Exact committed dependency lock identity.",
    )
    target: str = Field(description="Selected OS/CPU/CPython wheel target.")
    objects: tuple[ArchiveObject, ...] = Field(
        min_length=1, max_length=128, description="One object per locked distribution."
    )
    python_archive: PythonArchivePin = Field(
        description="Independent standalone interpreter pin."
    )

    @model_validator(mode="after")
    def unique_distributions(self) -> Self:
        """No duplicates may mask an absent distribution or selected architecture."""
        if (
            len({item.distribution for item in self.objects}) != len(self.objects)
            or len({item.filename for item in self.objects}) != len(self.objects)
            or self.target != self.python_archive.platform_tag
        ):
            raise ValueError(
                "native archive inventory is duplicated or targets another interpreter"
            )
        return self


@dataclass(frozen=True)
class Member:
    """Bounded bytes or streamed object plus identity rechecked while sealing."""

    source: bytes | Path
    size: int
    sha256: str


def _small(path: Path, root: Path, limit: int = 2 * 1024**2) -> bytes:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("native package source escaped its declared repository")
    with resolved.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("native package support metadata exceeds its bound")
    return data


def _bytes(data: bytes) -> Member:
    return Member(data, len(data), hashlib.sha256(data).hexdigest())


def write_member(archive: zipfile.ZipFile, name: str, member: Member) -> None:
    """Stream one regular member and verify its admitted identity during copying."""
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    digest = hashlib.sha256()
    written = 0
    with archive.open(entry, "w", force_zip64=True) as outgoing:
        if isinstance(member.source, bytes):
            digest.update(member.source)
            written = outgoing.write(member.source)
        else:
            with member.source.open("rb") as incoming:
                while block := incoming.read(1024 * 1024):
                    written += len(block)
                    if written > member.size:
                        raise ValueError("native package object grew during sealing")
                    digest.update(block)
                    outgoing.write(block)
    if written != member.size or digest.hexdigest() != member.sha256:
        raise ValueError("native package object changed during sealing")


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
) -> Path:
    """Build deterministic, fully offline CPU package bytes from clean committed inputs.

    Large archives stream through bounded buffers, retaining selected object
    identities throughout the final copy. Model weights, machine paths and
    environment values never enter the package. No network/publish action exists.
    """
    if (
        target not in TARGETS
        or wrapper.resolve()
        != (repository / "packaging/skulk-ace-step/skulk_ace_step").resolve()
    ):
        raise ValueError(
            "native music target or declared clean adapter root is invalid"
        )
    if output.resolve().is_relative_to(
        repository.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError(
            "native music outputs must stay outside both clean source repositories"
        )
    revision, upstream_revision = clean_revision(repository), clean_revision(source)
    inputs = wrapper.parent
    pin = SourcePin.model_validate_json(_small(inputs / "source.json", repository))
    inventory_bytes = _small(inputs / "source-files.json", repository)
    inventory = SourceInventory.model_validate_json(inventory_bytes)
    if (pin.revision, pin.repository) != (
        inventory.revision,
        inventory.repository,
    ) or pin.revision != upstream_revision:
        raise ValueError(
            "native music source differs from its reviewed source pin/inventory"
        )
    catalog = read_python_catalog(inputs)
    python_pin = catalog.for_target(target)
    profile = python_pin.profile
    if lock_path.resolve() != (inputs / f"locks/{profile}-cpu-py312.lock").resolve():
        raise ValueError(
            "native music must use the selected committed architecture lock"
        )
    lock = _small(lock_path, repository)
    objects = ArchiveInventory.model_validate_json(
        _small(inputs / f"archives/{profile}-cpu-py312.json", repository)
    )
    if (
        objects.target != target
        or objects.lock_sha256 != hashlib.sha256(lock).hexdigest()
        or objects.python_archive != python_pin
    ):
        raise ValueError(
            "native music archive inventory differs from its lock/interpreter"
        )
    if python_archive.is_symlink() or not python_archive.is_file():
        raise ValueError("native music interpreter must be its ordinary pinned archive")
    verify_python_archive(python_archive, python_pin)
    required = locked_dependencies(lock)
    direct = _small(inputs / "requirements.in", repository)
    payload: dict[str, Member] = {}
    for item in inventory.files:
        data = _small(source / item.path, source)
        if (len(data), hashlib.sha256(data).hexdigest()) != (
            item.size_bytes,
            item.sha256,
        ):
            raise ValueError("native music selected source bytes differ from their pin")
        payload[item.path] = _bytes(data)
    for item in inventory.overrides:
        data = _small(inputs / item.source, repository, 16 * 1024)
        if hashlib.sha256(data).hexdigest() != item.sha256:
            raise ValueError(
                "native music source override differs from its reviewed digest"
            )
        payload[item.path] = _bytes(data)
    if any(item.source not in payload for item in inventory.effective_model_code):
        raise ValueError(
            "native music source inventory omits its owned model implementation"
        )
    code = [
        ModelCodeFile(
            source=item.source,
            destination=item.destination,
            size_bytes=payload[item.source].size,
            sha256=payload[item.source].sha256,
        )
        for item in inventory.effective_model_code
    ]
    present = {path.name for path in wrapper.iterdir() if path.name != "__pycache__"}
    if frozenset(present) != ADAPTER_MODULES:
        raise ValueError(
            "native music adapter differs from its closed module inventory"
        )
    for name in sorted(present):
        payload[f"skulk_ace_step/{name}"] = _bytes(_small(wrapper / name, repository))
    expected = {item.filename: item for item in objects.objects}
    paths = tuple(sorted(wheelhouse.iterdir()))
    if {path.name for path in paths} != expected.keys() or len(paths) != len(expected):
        raise ValueError(
            "native music offline wheel set is incomplete or contains extra objects"
        )
    dependencies: dict[str, Path] = {}
    for path in paths:
        item = expected[path.name]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item.size_bytes
            or not dependency_matches(path.name, target)
            or file_digest(path, limit=item.size_bytes) != item.sha256
        ):
            raise ValueError(
                "native music dependency differs from its exact compatible archive"
            )
        name, version, _, _ = parse_wheel_filename(path.name)
        if (
            name not in required
            or required[name][0] != version
            or item.sha256 not in required[name][1]
        ):
            raise ValueError(
                "native music archive is not selected by its exact dependency lock"
            )
        dependencies[path.name] = path
        payload[f"skulk_ace_step/wheels/{path.name}"] = Member(
            path, item.size_bytes, item.sha256
        )
    if {item.distribution for item in objects.objects} != required.keys():
        raise ValueError("native music archive inventory omits a locked distribution")
    validate_dependency_closure(dependencies, required, direct, target)
    for name, data in {
        "requirements.in": direct,
        "requirements.lock": lock,
        "python.json": catalog.model_dump_json(indent=2).encode() + b"\n",
        "source.json": pin.model_dump_json(indent=2).encode() + b"\n",
        "source-files.json": inventory_bytes,
        "archive-inputs.json": objects.model_dump_json(indent=2).encode() + b"\n",
        "model-code.json": json.dumps(
            [item.model_dump(mode="json") for item in code], sort_keys=True, indent=2
        ).encode()
        + b"\n",
    }.items():
        payload[f"skulk_ace_step/{name}"] = _bytes(data)
    payload[f"skulk_ace_step/python/{python_pin.filename}"] = Member(
        python_archive, python_pin.size_bytes, python_pin.sha256
    )
    payload["skulk_ace_step/licenses/ACE_STEP_LICENSE.txt"] = _bytes(
        _small(source / "LICENSE", source)
    )
    payload["skulk_ace_step/licenses/SKULK_LICENSE.txt"] = _bytes(
        _small(repository / "LICENSE", repository)
    )
    content = {
        "protocol_version": 1,
        "python": PYTHON_VERSION,
        "python_archive": python_pin.model_dump(mode="json"),
        "adapter_revision": revision,
        "source_revision": upstream_revision,
        "source_repository": pin.repository,
        "platform_tag": target,
        "variant": "cpu",
        "members": {name: item.sha256 for name, item in sorted(payload.items())},
        "features": [
            "music.generate",
            "music.edit",
            "source_conditioning",
            "pcm24",
            "offline_loader",
            "bundled_python",
            "resident_model",
        ],
    }
    identity = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["skulk_ace_step/build.json"] = _bytes(
        json.dumps(
            content | {"payload_sha256": identity}, sort_keys=True, indent=2
        ).encode()
        + b"\n"
    )
    distribution = "skulk_ace_step_cpu"
    metadata = f"{distribution}-{VERSION}.dist-info"
    filename = f"{distribution}-{VERSION}-cp312-cp312-{target}.whl"
    payload[f"{metadata}/METADATA"] = _bytes(
        f"Metadata-Version: 2.4\nName: skulk-ace-step-cpu\nVersion: {VERSION}\nRequires-Python: >=3.12,<3.13\nSummary: Isolated native ACE music runtime for Skulk\nLicense: See bundled component notices\n\n".encode()
    )
    payload[f"{metadata}/WHEEL"] = _bytes(
        f"Wheel-Version: 1.0\nGenerator: foxlight-native-ace\nRoot-Is-Purelib: false\nTag: cp312-cp312-{target}\n".encode()
    )
    records = [
        f"{name},sha256={base64.urlsafe_b64encode(bytes.fromhex(item.sha256)).rstrip(b'=').decode()},{item.size}"
        for name, item in sorted(payload.items())
    ]
    payload[f"{metadata}/RECORD"] = _bytes(
        ("\n".join([*records, f"{metadata}/RECORD,,"]) + "\n").encode()
    )
    if (
        len(payload) > 256
        or sum(item.size for item in payload.values()) > MAX_PACKAGE_BYTES - 1024**2
    ):
        raise ValueError(
            "native music payload exceeds its bounded sealed package layout"
        )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    target_file = destination.open("xb")
    try:
        with (
            target_file,
            zipfile.ZipFile(
                target_file, "w", compression=zipfile.ZIP_STORED
            ) as archive,
        ):
            for name, item in sorted(payload.items()):
                write_member(archive, name, item)
        if (
            clean_revision(repository) != revision
            or clean_revision(source) != upstream_revision
        ):
            raise ValueError("native package source changed during sealing")
        descriptor = {
            "filename": filename,
            "size_bytes": destination.stat().st_size,
            "sha256": file_digest(destination, limit=MAX_PACKAGE_BYTES),
            "payload_sha256": identity,
            "adapter_revision": revision,
            "source_revision": upstream_revision,
            "platform_tag": target,
            "variant": "cpu",
            "python": PYTHON_VERSION,
            "python_archive_sha256": python_pin.sha256,
        }
        with destination.with_suffix(".json").open("x") as stream:
            stream.write(json.dumps(descriptor, sort_keys=True, indent=2) + "\n")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


class Arguments(argparse.Namespace):
    """Explicit local producer inputs; there is no publication or network option."""

    repository: Path
    source: Path
    wrapper: Path
    wheelhouse: Path
    lock: Path
    output: Path
    target: str
    python_archive: Path


def main() -> None:
    """Seal the selected immutable architecture without downloading or uploading."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "source",
        "wrapper",
        "wheelhouse",
        "lock",
        "output",
        "python-archive",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
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
