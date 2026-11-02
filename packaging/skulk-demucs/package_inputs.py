"""Pinned standalone interpreter inputs shared by package intake and production."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

PYTHON_VERSION = "3.12.13"
MAX_PYTHON_ARCHIVE_BYTES = 128 * 1024 * 1024
PROFILE_TARGETS = {
    "macos-arm64": "macosx_14_0_arm64",
    "linux-amd64": "manylinux_2_28_x86_64",
    "linux-arm64": "manylinux_2_28_aarch64",
}
PROFILE_TRIPLES = {
    "macos-arm64": "aarch64-apple-darwin",
    "linux-amd64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
}


class PythonArchivePin(BaseModel):
    """One exact interpreter object, independently checked before packaging."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    profile: Literal["macos-arm64", "linux-amd64", "linux-arm64"]
    platform_tag: str = Field(description="Engine package's OS and CPU ABI tag.")
    filename: str = Field(description="Immutable release object basename.")
    url: str = Field(description="Pinned upstream release object, intake only.")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0, le=MAX_PYTHON_ARCHIVE_BYTES)
    version: Literal["3.12.13"]
    executable: Literal["python/bin/python3.12"]

    @model_validator(mode="after")
    def consistent_identity(self) -> PythonArchivePin:
        """Reject foreign interpreters, platforms and mutable or indirect URLs."""
        filename = (
            f"cpython-{PYTHON_VERSION}+20260807-{PROFILE_TRIPLES[self.profile]}"
            "-install_only_stripped.tar.gz"
        )
        url = urlsplit(self.url)
        if (
            self.platform_tag != PROFILE_TARGETS[self.profile]
            or self.filename != filename
            or url.scheme != "https"
            or url.netloc != "github.com"
            or unquote(url.path)
            != "/astral-sh/python-build-standalone/releases/download/20260807/"
            + filename
            or url.query
            or url.fragment
        ):
            raise ValueError("standalone Python identity does not match its target")
        return self


class PythonArchiveCatalog(BaseModel):
    """Closed three-platform pins, attributed to immutable upstream metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    source_repository: Literal["https://github.com/astral-sh/uv"]
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_path: Literal["crates/uv-python/download-metadata.json"]
    archives: tuple[PythonArchivePin, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def complete_targets(self) -> PythonArchiveCatalog:
        """Require one and only one reviewed interpreter per supported platform."""
        if {entry.profile for entry in self.archives} != PROFILE_TARGETS.keys():
            raise ValueError("standalone Python catalog must cover all three targets")
        return self

    def for_target(self, target: str) -> PythonArchivePin:
        """Return the exact target's interpreter; never select by the build host."""
        for entry in self.archives:
            if entry.platform_tag == target:
                return entry
        raise ValueError("no pinned standalone Python for this engine target")


def read_python_catalog(root: Path) -> PythonArchiveCatalog:
    """Bound and validate the source-controlled interpreter object inventory."""
    with (root / "python.json").open("rb") as stream:
        data = stream.read(16385)
    if len(data) > 16384:
        raise ValueError("standalone Python catalog exceeds its bound")
    return PythonArchiveCatalog.model_validate_json(data)


def file_digest(path: Path, *, limit: int) -> str:
    """Hash a bounded local object without loading a complete archive into RAM."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            if size > limit:
                raise ValueError("native archive exceeds its object bound")
            digest.update(block)
    return digest.hexdigest()


def verify_python_archive(path: Path, pin: PythonArchivePin) -> None:
    """Check the interpreter basename, complete size and digest against its pin."""
    if (
        path.name != pin.filename
        or path.stat().st_size != pin.size_bytes
        or file_digest(path, limit=pin.size_bytes) != pin.sha256
    ):
        raise ValueError("standalone Python differs from the pinned object")
