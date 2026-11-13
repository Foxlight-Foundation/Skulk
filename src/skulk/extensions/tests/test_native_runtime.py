"""Independent native-envelope fixtures and immutable legacy signing regressions."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.native_contracts import (
    NativeDescriptor,
    native_wheel,
    public_native_kernel,
)
from skulk.extensions.runtime_artifacts import (
    QualifiedHost,
    RuntimeTrust,
    canonical_json,
    verify_runtime,
)
from skulk.extensions.tests.test_runtime_install import artifacts

OBJECT = TypeAdapter(dict[str, JsonValue])
PROFILES: dict[str, JsonValue] = {
    "version": "0.4.0rc1",
    "source_commit": "43b4442c1f22dd75ccc1b19d1252272ca3eefb4b",
    "portable": "ada52fb17016f873140aa4d6cd7ae2f8c9a0895eed997a92201ed5f45a60e3d3",
    "managed": "a70a5fdf56aa24775bdce26dc4ccecc4fc50bd2d3acbd78b5b8b47f4599fe4a1",
    "handshake": "57413947cae8d31de25b3283dc1ffa87d9c9c28f0d98b5e8eb593c9dd64e07a0",
    "transport": "18e4096b5802dc4ff9c251b1819a0c5062d42713edd4e8fc30fa94224044b435",
}


def native_fixture(
    directory: Path,
) -> tuple[dict[str, JsonValue], Ed25519PrivateKey, RuntimeTrust, QualifiedHost]:
    """Author an explicit native manifest independently of private owner code."""
    key = Ed25519PrivateKey.generate()
    metadata, trust, host = artifacts(directory, signing_key=key)
    payload = OBJECT.validate_python(
        OBJECT.validate_json(metadata)["runtime"], strict=True
    )
    release = OBJECT.validate_python(payload["release"], strict=True)
    legacy = OBJECT.validate_python(release["manifest"], strict=True)
    manifest: dict[str, JsonValue] = {
        "protocol": 5,
        "contract_profile": "skulk.managed/1",
        "profile_digest": PROFILES["managed"],
        "bundle_id": "example.native",
        "bundle_version": "1.0.0",
        "executable": "bundle.pyz",
        "executable_sha256": legacy["executable_sha256"],
        "skulk_requires": ">=2.0.3,<3",
        "contracts": {
            "protocol": 5,
            "contract_profile": "skulk.portable/1",
            "profile_digest": PROFILES["portable"],
            "configuration_schema": True,
            "descriptors": [
                {
                    "contract_profile": "skulk.portable/1",
                    "profile_digest": PROFILES["portable"],
                    "id": "UpperCase:Echo@native",
                    "version": "1.0.0",
                    "title": "Native echo",
                    "description": "Independent read-only test",
                    "behavior": "read",
                    "io_mode": "unary",
                    "input_schema": True,
                    "output_schema": True,
                }
            ],
        },
    }
    record = public_native_kernel().ManagedContract.from_json(
        "Manifest", json.dumps(manifest)
    )
    release.update(protocol=2, manifest={"canonical_json": record.canonical().decode()})
    filename, digest, size = native_wheel(host.platform)
    wheels = payload["wheels"]
    assert isinstance(wheels, list)
    wheels.append({"filename": filename, "sha256": digest, "size": size})
    release["dependency_lock_sha256"] = hashlib.sha256(
        canonical_json({"example-dep": "1.0", "skulk-capability-kernel": "0.4.0rc1"})
    ).hexdigest()
    payload.update(protocol=3, release=release, kernel=dict(PROFILES))
    return payload, key, trust, replace(host, skulk_version="2.0.3")


def sign(
    payload: dict[str, JsonValue],
    key: Ed25519PrivateKey,
    domain: bytes = b"skulk.runtime/3\0",
) -> bytes:
    """Sign exact native fixture bytes under an independently stated wire domain."""
    return canonical_json(
        {
            "runtime": payload,
            "signature": key.sign(domain + canonical_json(payload)).hex(),
        }
    )


def test_native_identity_survives_core_admission(tmp_path: Path) -> None:
    """Full native names, boolean schemas and canonical bytes survive unchanged."""
    payload, key, trust, host = native_fixture(tmp_path)
    runtime = verify_runtime(sign(payload, key), trust, host, now=int(time.time()))
    assert (
        runtime.digest
        == hashlib.sha256(b"skulk.runtime/3\0" + canonical_json(payload)).hexdigest()
    )
    manifest = runtime.native_manifest
    assert manifest is not None
    assert runtime.configuration_schema is True
    assert len(manifest.digest) == 64
    contracts = OBJECT.validate_python(manifest.value()["contracts"], strict=True)
    descriptors = contracts["descriptors"]
    assert isinstance(descriptors, list)
    descriptor = OBJECT.validate_python(descriptors[0], strict=True)
    assert descriptor["id"] == "UpperCase:Echo@native"
    projected = NativeDescriptor.model_validate_json(json.dumps(descriptor))
    assert projected.id == descriptor["id"] and projected.input_schema is True
    assert projected.revision == public_native_kernel().portable_contract_digest(
        "Descriptor", json.dumps(descriptor)
    )
    assert len(projected.revision) == 64
    assert (
        manifest.digest
        == public_native_kernel()
        .ManagedContract.from_json("Manifest", manifest.canonical_json)
        .digest
    )


@pytest.mark.parametrize(
    "fault",
    [
        "unprefixed",
        "release-domain",
        "changed-wheel",
        "changed-profile",
        "noncanonical",
        "extra-field",
        "revoked",
        "revoked-release",
        "float-runtime",
        "bool-release",
        "duplicate",
    ],
)
def test_native_refuses_changed_identity_before_install(
    tmp_path: Path, fault: str
) -> None:
    """Authentication, native admission and binary pins precede installer effects."""
    payload, key, trust, host = native_fixture(tmp_path)
    domain = b"skulk.runtime/3\0"
    if fault == "unprefixed":
        domain = b""
    elif fault == "release-domain":
        domain = b"skulk.release/2\0"
    elif fault == "changed-wheel":
        wheels = payload["wheels"]
        assert isinstance(wheels, list) and isinstance(wheels[1], dict)
        wheels[1]["sha256"] = "b" * 64
    elif fault == "changed-profile":
        payload["kernel"] = {**PROFILES, "transport": "b" * 64}
    elif fault == "extra-field":
        payload["unexpected"] = True
    elif fault == "float-runtime":
        payload["protocol"] = 3.0
        domain = b""
    elif fault == "bool-release":
        release = OBJECT.validate_python(payload["release"], strict=True)
        release["protocol"] = True
        payload["release"] = release
    elif fault == "noncanonical":
        release = OBJECT.validate_python(payload["release"], strict=True)
        manifest = OBJECT.validate_python(release["manifest"], strict=True)
        source = manifest["canonical_json"]
        assert isinstance(source, str)
        manifest["canonical_json"] = " " + source
        release["manifest"] = manifest
        payload["release"] = release
    elif fault == "revoked":
        digest = hashlib.sha256(domain + canonical_json(payload)).hexdigest()
        trust = trust.model_copy(update={"revoked_artifacts": (digest,)})
    elif fault == "revoked-release":
        digest = hashlib.sha256(
            b"skulk.release/2\0" + canonical_json(payload["release"])
        ).hexdigest()
        trust = trust.model_copy(update={"revoked_artifacts": (digest,)})
    raw = sign(payload, key, domain)
    if fault == "duplicate":
        raw = raw.replace(b'"protocol":3', b'"protocol":2,"protocol":3')
    with pytest.raises(ValueError):
        verify_runtime(raw, trust, host, now=int(time.time()))
    assert not (tmp_path / "generations").exists()


def test_legacy_runtime_signature_and_digest_are_unchanged(tmp_path: Path) -> None:
    """The native domain is never retroactively applied to issued envelopes."""
    metadata, trust, host = artifacts(tmp_path)
    payload = OBJECT.validate_python(
        OBJECT.validate_json(metadata)["runtime"], strict=True
    )
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    assert runtime.native_manifest is None
    assert runtime.digest == hashlib.sha256(canonical_json(payload)).hexdigest()
    assert runtime.signature_message == canonical_json(payload)
    assert (
        runtime.release_digest
        == hashlib.sha256(canonical_json(payload["release"])).hexdigest()
    )


def test_excessively_nested_metadata_is_a_bounded_refusal(tmp_path: Path) -> None:
    """Malformed nesting never escapes the reader as an unhandled recursion error."""
    _, trust, host = artifacts(tmp_path)
    raw = b'{"runtime":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}"
    with pytest.raises(ValueError):
        verify_runtime(raw, trust, host, now=int(time.time()))


@pytest.mark.parametrize("legacy_outer", [False, True])
def test_native_and_legacy_envelopes_cannot_be_mixed(
    tmp_path: Path, legacy_outer: bool
) -> None:
    """A trusted publisher cannot relabel a native record as a legacy generation."""
    payload, key, trust, host = native_fixture(tmp_path)
    domain = b"skulk.runtime/3\0"
    if legacy_outer:
        payload["protocol"] = 2
        domain = b""
    else:
        release = OBJECT.validate_python(payload["release"], strict=True)
        release["protocol"] = 1
        payload["release"] = release
    with pytest.raises(ValueError):
        verify_runtime(sign(payload, key, domain), trust, host, now=int(time.time()))
