"""Collect candidate offline wheels, matching every object to an existing hash lock.

Development intake only; this command cannot publish a runtime or alter a node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from html.parser import HTMLParser
from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from build_wheel import (
    MAX_WHEEL_BYTES,
    dependency_matches,
    locked_dependencies,
    target_tags,
)
from package_inputs import (
    MAX_PYTHON_ARCHIVE_BYTES,
    PROFILE_TARGETS,
    file_digest,
    read_python_catalog,
    verify_python_archive,
)
from packaging.utils import parse_wheel_filename
from pydantic import BaseModel, ConfigDict, Field


class ObjectDigest(BaseModel):
    """Only the SHA-256 field used from upstream PyPI metadata."""

    model_config = ConfigDict(frozen=True, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReleaseFile(BaseModel):
    """An upstream candidate object, not trusted until lock and bytes agree."""

    model_config = ConfigDict(frozen=True, strict=True)
    filename: str
    url: str
    digests: ObjectDigest
    size: int = Field(gt=0)
    yanked: bool


class ReleaseFiles(BaseModel):
    """PyPI release's object inventory; other metadata is irrelevant here."""

    model_config = ConfigDict(frozen=True, strict=True)
    urls: list[ReleaseFile]


class IndexLinks(HTMLParser):
    """Read only link targets from the explicit PyTorch CPU wheel index."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Retain anchor destinations without executing any document content."""
        if tag == "a":
            self.links.extend(
                value for name, value in attrs if name == "href" and value
            )


def request_bytes(url: str, limit: int) -> bytes:
    """Bound release/index metadata before parsing it."""
    with cast(
        HTTPResponse,
        cast(
            object,
            urlopen(
                Request(url, headers={"User-Agent": "Foxlight-Demucs-intake"}),
                timeout=120,
            ),
        ),
    ) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("upstream metadata exceeds its bound")
    return data


def fetch_object(
    url: str,
    output: Path,
    digest: str,
    expected_size: int | None,
    *,
    limit: int,
) -> None:
    """Collect one immutable local object; refuse changed caches and overwrite races."""
    if expected_size is not None and expected_size > limit:
        raise ValueError("selected native archive exceeds its object bound")
    if not output.exists():
        staging = output.with_name(output.name + ".part")
        staging_stream = staging.open("xb")
        try:
            actual_digest = hashlib.sha256()
            size = 0
            with (
                staging_stream as stream,
                cast(
                    HTTPResponse,
                    cast(
                        object,
                        urlopen(
                            Request(
                                url, headers={"User-Agent": "Foxlight-Demucs-intake"}
                            ),
                            timeout=120,
                        ),
                    ),
                ) as response,
            ):
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > limit:
                        raise ValueError("native archive exceeds object bound")
                    actual_digest.update(block)
                    stream.write(block)
            if actual_digest.hexdigest() != digest or (
                expected_size is not None and expected_size != size
            ):
                raise ValueError("native archive differs from its locked object")
            # Link publication is exclusive, unlike rename's overwrite semantics.
            # Another intake racing this one cannot replace the verified object.
            output.hardlink_to(staging)
        finally:
            staging.unlink(missing_ok=True)
    if (
        output.is_symlink()
        or not output.is_file()
        or (expected_size is not None and output.stat().st_size != expected_size)
        or file_digest(output, limit=limit) != digest
    ):
        raise ValueError("cached candidate archive differs from the lock")


def fetch_profile(root: Path, profile: str, target: str, output_root: Path) -> None:
    """Keep one compatible archive per locked distribution and retain provenance."""
    lock_path = root / "locks" / f"{profile}-cpu-py312.lock"
    lock = lock_path.read_bytes()
    requirements = locked_dependencies(lock)
    if PROFILE_TARGETS.get(profile) != target:
        raise ValueError("native intake profile does not match its target")
    python_pin = read_python_catalog(root).for_target(target)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "wheels"
    destination.mkdir(exist_ok=True)
    records: list[dict[str, str | int]] = []
    priority = {tag: index for index, tag in enumerate(target_tags(target))}
    for name, (version, hashes) in sorted(requirements.items()):
        candidates: list[tuple[str, str, str, int | None]] = []
        if name == "torch" and version.local == "cpu":
            index_url = "https://download.pytorch.org/whl/cpu/torch/"
            parser = IndexLinks()
            parser.feed(request_bytes(index_url, 4 * 1024 * 1024).decode())
            for link in parser.links:
                url = urlsplit(urljoin(index_url, link))
                filename = unquote(url.path.rsplit("/", 1)[-1])
                if not filename.endswith(".whl"):
                    continue
                distribution, candidate_version, _, _ = parse_wheel_filename(filename)
                digest = url.fragment.removeprefix("sha256=")
                if (
                    distribution == name
                    and candidate_version == version
                    and digest in hashes
                    and dependency_matches(filename, target)
                ):
                    candidates.append(
                        (filename, urlunsplit(url._replace(fragment="")), digest, None)
                    )
        else:
            release = ReleaseFiles.model_validate_json(
                request_bytes(
                    f"https://pypi.org/pypi/{name}/{version}/json", 4 * 1024 * 1024
                )
            )
            for archive in release.urls:
                if archive.yanked or not archive.filename.endswith(".whl"):
                    continue
                distribution, candidate_version, _, _ = parse_wheel_filename(
                    archive.filename
                )
                if (
                    distribution == name
                    and candidate_version == version
                    and archive.digests.sha256 in hashes
                    and dependency_matches(archive.filename, target)
                ):
                    candidates.append(
                        (
                            archive.filename,
                            archive.url,
                            archive.digests.sha256,
                            archive.size,
                        )
                    )
        if not candidates:
            raise ValueError(
                f"no hash-locked compatible object for {profile}: {name}=={version}"
            )

        def preference(candidate: tuple[str, str, str, int | None]) -> tuple[int, str]:
            tags = parse_wheel_filename(candidate[0])[3]
            return min(priority[tag] for tag in tags if tag in priority), candidate[0]

        filename, url, digest, expected_size = min(candidates, key=preference)
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in {
            "files.pythonhosted.org",
            "download.pytorch.org",
            "download-r2.pytorch.org",
        }:
            raise ValueError("wheel object uses an unexpected source")
        output = destination / filename
        fetch_object(url, output, digest, expected_size, limit=MAX_WHEEL_BYTES)
        records.append(
            {
                "distribution": name,
                "version": str(version),
                "filename": filename,
                "source_url": url,
                "sha256": digest,
                "size_bytes": output.stat().st_size,
            }
        )
        print(f"{profile}: verified {filename}", flush=True)
    if {path.name for path in destination.iterdir()} != {
        record["filename"] for record in records
    }:
        raise ValueError("native wheelhouse contains unselected objects")
    python_root = output_root / "python"
    python_root.mkdir(exist_ok=True)
    python_archive = python_root / python_pin.filename
    fetch_object(
        python_pin.url,
        python_archive,
        python_pin.sha256,
        python_pin.size_bytes,
        limit=MAX_PYTHON_ARCHIVE_BYTES,
    )
    verify_python_archive(python_archive, python_pin)
    (output_root / "wheel-receipt.json").write_text(
        json.dumps(
            {
                "target": target,
                "lock_sha256": hashlib.sha256(lock).hexdigest(),
                "objects": records,
                "python_archive": python_pin.model_dump(mode="json"),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


class Arguments(argparse.Namespace):
    """A single closed dependency profile selected by the operator."""

    profile: str
    output: Path


def main() -> None:
    """Fetch a single local candidate profile; no publication or installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "profile", choices=("macos-arm64", "linux-amd64", "linux-arm64")
    )
    parsed = Arguments()
    parser.parse_args(namespace=parsed)
    fetch_profile(
        Path(__file__).parent,
        parsed.profile,
        PROFILE_TARGETS[parsed.profile],
        parsed.output,
    )


if __name__ == "__main__":
    main()
