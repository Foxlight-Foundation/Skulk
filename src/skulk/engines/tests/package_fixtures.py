"""Build, sign, install and activate a small real engine package for tests."""

import hashlib
import io
import tarfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.extensions.runtime_artifacts import RuntimeTrust
from skulk.provisioning.engine_install import (
    DirectoryPackageSource,
    InstalledEngine,
    activate_engine_generation,
    install_engine_package,
)
from skulk.provisioning.engine_packages import EngineLane, verify_engine_manifest
from skulk.provisioning.tests.engine_package_fixtures import (
    NOW,
    PUBLISHER,
    engine_claims,
    public_hex,
    sign_engine_claims,
)

ENGINE = b"#!/bin/sh\necho engine\n"

ECHO_ADAPTER = rb"""
import json, os, struct, sys

digest = os.environ["SKULK_ENGINE_BUILD_DIGEST"]
sys.stderr.write("adapter started\n")
sys.stderr.flush()


def read_frame():
    head = sys.stdin.buffer.read(4)
    if not head:
        return None
    (length,) = struct.unpack(">I", head)
    return json.loads(sys.stdin.buffer.read(length))


def send(message):
    body = json.dumps(message, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
    sys.stdout.buffer.flush()


while True:
    request = read_frame()
    if request is None:
        break
    kind = request["type"]
    base = {"type": kind, "sequence": request["sequence"], "build_digest": digest}
    if kind == "hello":
        send({**base, "version": max(request["versions"])})
    elif kind == "describe":
        # Report the working directory and the environment's variable names,
        # so a test can see exactly what the host passed on.
        send({**base, "engine_version": os.getcwd(), "model_families": sorted(os.environ)})
    elif kind == "stop":
        send(base)
        break
"""
"""A standard-library adapter that answers hello, describe and stop."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("bin/engine")
        info.mode = 0o755
        info.size = len(ENGINE)
        archive.addfile(info, io.BytesIO(ENGINE))
    return stream.getvalue()


def install_test_package(
    work: Path,
    root: Path,
    *,
    engine_id: str = "foxlight.tensorfold",
    lane: EngineLane = "rocm",
    generation: int = 1,
    adapter: bytes = ECHO_ADAPTER,
    protocol: tuple[int, int] = (1, 2),
    python_requires: str = ">=3.12",
    activate: bool = True,
) -> InstalledEngine:
    """Sign a manifest for real files, install it under ``root`` and activate it.

    Args:
        work: A scratch directory for the files to import.
        root: The packages root to install into.
        engine_id: The ``publisher.engine`` identity to claim.
        lane: The accelerator class to claim.
        generation: The package generation; distinct values give distinct digests.
        adapter: The adapter program's bytes.
        protocol: Lowest and highest control-protocol version the manifest declares.
        python_requires: The interpreter range the adapter declares.
        activate: Whether to make the generation the active one.
    """
    content = _archive()
    files = work / f"import-{engine_id}-{lane}-{generation}"
    files.mkdir(parents=True, exist_ok=True)
    (files / "engine.tar.gz").write_bytes(content)
    (files / "adapter.py").write_bytes(adapter)
    claims = engine_claims()
    claims["engine_id"] = engine_id
    claims["generation"] = generation
    claims["lane"] = lane
    claims["compute_targets"] = []
    claims["control_protocol"] = {"minimum": protocol[0], "maximum": protocol[1]}
    claims["archive"] = {
        "filename": "engine.tar.gz",
        "size": len(content),
        "sha256": _sha256(content),
        "unpacked_size": 1_000_000,
        "executable": "bin/engine",
    }
    claims["adapter"] = {
        "filename": "adapter.py",
        "size": len(adapter),
        "sha256": _sha256(adapter),
        "kind": "python",
        "python_requires": python_requires,
    }
    claims["external_requirements"] = []
    key = Ed25519PrivateKey.generate()
    trust = RuntimeTrust(
        revision=1, expires_at=NOW + 86_400, publishers={PUBLISHER: public_hex(key)}
    )
    package = verify_engine_manifest(sign_engine_claims(key, claims), trust, now=NOW)
    installed = install_engine_package(
        package, DirectoryPackageSource(files), root=root
    )
    if activate:
        return activate_engine_generation(engine_id, lane, installed.digest, root=root)
    return installed
