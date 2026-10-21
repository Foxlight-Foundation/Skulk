"""Build the immutable Foxlight extension for the pinned Comfy GPU engine."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

VERSION = "0.1.5"
PACKAGE = "skulk_comfy_ltx"
FILENAME = f"{PACKAGE}-{VERSION}-py3-none-any.whl"
COMFY_REVISION = "40c4fcdf513a4523e39d54a9d391908af8df8171"
LTX_REVISION = "e98f6d927d83c92ef2be6aec76c8440b19306e39"
FEATURES = ("res2s", "retake", "extend", "dfr", "prompt_relay", "union_control", "motion_tracks")
SOURCE_FILES = ("__init__.py", "comfy_sampler.py", "dfr.py", "dfr_nodes.py", "editing.py", "editing_nodes.py", "guidance.py", "layout.py", "launcher.py", "motion_tracks.py", "motion_nodes.py", "nodes.py", "relay.py", "relay_encoding.py", "relay_geometry.py", "relay_binding.py", "relay_nodes.py", "res2s.py", "union_control.py", "union_nodes.py", "py.typed")
LICENSE_FILES = ("LICENSE-APACHE-2.0", "LICENSE-LTX-2_x", "LICENSE-LTX-2", "NOTICE")


def assemble_wheel(source: Path, output: Path) -> Path:
    """Package explicit reviewed sources reproducibly, without resolving dependencies.

    Args:
        source: The checked-out packaging/skulk-comfy-ltx directory.
        output: A candidate directory; an existing wheel is never overwritten.

    Returns:
        The wheel path, beside its immutable source and content descriptor.
    """
    package = source / PACKAGE
    observed = {path.name for path in package.iterdir() if path.name != "__pycache__"}
    if observed != set(SOURCE_FILES):
        raise ValueError("extension source differs from the explicit reviewed file set")
    payload: dict[str, bytes] = {}
    for name in SOURCE_FILES:
        path = package / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("extension source must contain regular reviewed files")
        payload[f"{PACKAGE}/{name}"] = path.read_bytes()
    for name in LICENSE_FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("extension attribution must contain regular license files")
        payload[f"{PACKAGE}/{name}"] = path.read_bytes()
    members = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())}
    identity = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "protocol": 1,
        "version": VERSION,
        "comfy_revision": COMFY_REVISION,
        "ltx_revision": LTX_REVISION,
        "features": FEATURES,
        "payload_sha256": identity,
        "members": members,
    }
    payload[f"{PACKAGE}/manifest.json"] = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    info = f"{PACKAGE}-{VERSION}.dist-info"
    payload[f"{info}/METADATA"] = (
        f"Metadata-Version: 2.4\nName: skulk-comfy-ltx\nVersion: {VERSION}\n"
        "Requires-Python: >=3.13,<3.14\n"
        "Summary: Trusted LTX nodes for Skulk's pinned Comfy GPU engine\n"
        "License-Expression: Apache-2.0 AND LicenseRef-LTX-2.x AND LicenseRef-LTX-2\n"
        + "".join(f"License-File: licenses/{name}\n" for name in LICENSE_FILES)
        + "\n"
    ).encode()
    for name in LICENSE_FILES:
        payload[f"{info}/licenses/{name}"] = payload[f"{PACKAGE}/{name}"]
    payload[f"{info}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: foxlight-comfy-ltx\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    records: list[str] = []
    for name, data in sorted(payload.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        records.append(f"{name},sha256={digest},{len(data)}\n")
    records.append(f"{info}/RECORD,,\n")
    payload[f"{info}/RECORD"] = "".join(records).encode()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / FILENAME
    # Exclusive creation matters even when two local candidate builders race.
    with destination.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, data)
    descriptor = {
        "filename": FILENAME,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "payload_sha256": identity,
        "size_bytes": destination.stat().st_size,
        "comfy_revision": COMFY_REVISION,
        "ltx_revision": LTX_REVISION,
        "features": FEATURES,
    }
    with destination.with_suffix(".json").open("x") as stream:
        _ = stream.write(json.dumps(descriptor, sort_keys=True, indent=2) + "\n")
    return destination


@dataclass
class _Arguments(argparse.Namespace):
    """Typed required build paths populated by argparse."""

    source: Path = Path(".")
    output: Path = Path(".")


def main() -> None:
    """Build a local candidate; publication is a separate attested promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=_Arguments())
    print(assemble_wheel(args.source, args.output))


if __name__ == "__main__":
    main()
