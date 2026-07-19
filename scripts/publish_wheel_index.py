# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportExplicitAny=false
# boto3 ships no type stubs and is a CI-only dependency of this script; the
# Any-propagation suppressions are scoped to this file.
"""Publish engine wheels to the Foxlight PEP 503 index on Cloudflare R2.

The index (``wheels.foxlight.foundation``) is the source of truth for
Foxlight-built engine wheels (#614 engine provisioning): a static "simple"
index that ``uv pip install --extra-index-url`` resolves like any registry,
the same pattern PyTorch (download.pytorch.org) and llama-cpp-python use for
wheels too large or too platform-specific for PyPI.

Layout in the bucket::

    simple/index.html                      # project list (PEP 503 root)
    simple/<project>/index.html            # per-project file list, sha256 anchors
    wheels/<wheel filename>                # the artifacts themselves

The script uploads any wheels given on the command line, then regenerates
both index levels from the bucket's full current contents, so every run is
idempotent and the index can never reference a missing file. R2 access uses
the S3-compatible API via boto3; credentials come from the environment
(``R2_ACCOUNT_ID``, ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``,
``R2_BUCKET``), which in CI are repository secrets.

Run locally or from ``.github/workflows/engine-wheel.yml``::

    uv run python scripts/publish_wheel_index.py dist/*.whl
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

#: Public base URL the index is served from (R2 custom domain). One constant
#: so a domain change is a one-line edit.
INDEX_BASE_URL = "https://wheels.foxlight.foundation"

_WHEEL_NAME = re.compile(r"^(?P<distribution>[A-Za-z0-9_.]+?)-\d")


def _normalize(name: str) -> str:
    """PEP 503 name normalization: lowercase, runs of ``-_.`` become ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _client() -> Any:
    """An S3 client bound to the R2 endpoint from environment credentials."""
    import boto3  # pyright: ignore[reportMissingImports] - CI-only dependency

    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_wheels(client: Any, bucket: str, wheels: list[Path]) -> None:
    """Upload the given wheels under ``wheels/`` with correct content type."""
    for wheel in wheels:
        print(f"uploading {wheel.name} ({wheel.stat().st_size >> 20} MB)")
        client.upload_file(
            str(wheel),
            bucket,
            f"wheels/{wheel.name}",
            ExtraArgs={"ContentType": "application/octet-stream"},
        )


def _bucket_wheels(client: Any, bucket: str) -> dict[str, list[str]]:
    """Map normalized project name -> wheel filenames currently in the bucket."""
    projects: dict[str, list[str]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="wheels/"):
        for entry in page.get("Contents", []):
            filename = entry["Key"].removeprefix("wheels/")
            if not filename.endswith(".whl"):
                continue
            match = _WHEEL_NAME.match(filename)
            if match is None:
                print(f"skipping unparseable wheel name: {filename}")
                continue
            project = _normalize(match.group("distribution"))
            projects.setdefault(project, []).append(filename)
    return projects


def _put_html(client: Any, bucket: str, key: str, body: str) -> None:
    """Write one HTML page under both its index.html and trailing-slash keys.

    PEP 503 clients request ``/simple/`` and ``/simple/<project>/``; whether a
    static host rewrites those to ``index.html`` is a serving-layer detail, so
    the page is stored under both keys and works either way.
    """
    for object_key in (key, key.removesuffix("index.html")):
        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=body.encode(),
            ContentType="text/html; charset=utf-8",
        )


def _regenerate_index(
    client: Any, bucket: str, checksums: dict[str, str]
) -> None:
    """Rebuild both index levels from the bucket's current contents.

    Args:
        client: The S3 client.
        bucket: The R2 bucket name.
        checksums: sha256 by wheel filename for wheels uploaded THIS run;
            files already indexed keep their existing anchors by re-reading
            the object metadata is deliberately avoided (hashing gigabytes
            remotely per publish would be wasteful), so anchors for
            previously-published files come from the previous index.
    """
    # Merge existing per-project anchors so older files keep their hashes.
    existing_anchor: dict[str, str] = {}
    projects = _bucket_wheels(client, bucket)
    for project in projects:
        try:
            body = client.get_object(
                Bucket=bucket, Key=f"simple/{project}/index.html"
            )["Body"].read().decode()
        except Exception:  # noqa: BLE001 - first publish for this project
            continue
        for match in re.finditer(
            r'href="[^"]*/wheels/([^"#]+)#sha256=([0-9a-f]{64})"', body
        ):
            existing_anchor[match.group(1)] = match.group(2)

    project_links: list[str] = []
    for project, files in sorted(projects.items()):
        rows: list[str] = []
        for filename in sorted(files):
            digest = checksums.get(filename) or existing_anchor.get(filename)
            if digest is None:
                # Every published link carries a sha256 anchor; a wheel with
                # no known digest (not uploaded this run, absent from the
                # prior index) must never be silently republished unhashed.
                raise RuntimeError(
                    f"no sha256 known for {filename}; re-upload it through "
                    "this script to restore its anchor"
                )
            rows.append(
                f'    <a href="{INDEX_BASE_URL}/wheels/{filename}#sha256={digest}">'
                f"{filename}</a><br/>"
            )
        _put_html(
            client,
            bucket,
            f"simple/{project}/index.html",
            "<!DOCTYPE html>\n<html><head>"
            '<meta name="pypi:repository-version" content="1.0">'
            f"<title>Links for {project}</title></head>\n<body>\n"
            f"  <h1>Links for {project}</h1>\n" + "\n".join(rows) + "\n</body></html>\n",
        )
        project_links.append(f'    <a href="/simple/{project}/">{project}</a><br/>')

    _put_html(
        client,
        bucket,
        "simple/index.html",
        "<!DOCTYPE html>\n<html><head>"
        '<meta name="pypi:repository-version" content="1.0">'
        "<title>Foxlight wheel index</title></head>\n<body>\n"
        "  <h1>Foxlight wheel index</h1>\n" + "\n".join(project_links) + "\n</body></html>\n",
    )
    print(f"index regenerated for {len(projects)} project(s)")


def main(argv: list[str]) -> int:
    """Upload the given wheels and regenerate the index.

    Args:
        argv: Wheel file paths. May be empty to regenerate the index only.

    Returns:
        Process exit code.
    """
    bucket = os.environ["R2_BUCKET"]
    wheels = [Path(arg) for arg in argv]
    for wheel in wheels:
        if not wheel.is_file():
            print(f"no such wheel: {wheel}", file=sys.stderr)
            return 2
    client = _client()
    checksums = {wheel.name: _sha256(wheel) for wheel in wheels}
    _upload_wheels(client, bucket, wheels)
    _regenerate_index(client, bucket, checksums)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
