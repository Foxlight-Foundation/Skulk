"""Shared builders for engine package tests: valid claims and their signed envelope."""

import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue

from skulk.extensions.runtime_artifacts import canonical_json
from skulk.provisioning.engine_packages import ENGINE_SIGNATURE_DOMAIN

NOW = 1_800_000_000
PUBLISHER = "foxlight-engines"
ARCHIVE_SHA256 = "a" * 64
ADAPTER_SHA256 = "b" * 64
REQUIREMENT_SHA256 = "c" * 64

Claims = dict[str, JsonValue]
ClaimPath = tuple[str | int, ...]


def public_hex(key: Ed25519PrivateKey) -> str:
    """The raw public key as the trust document writes it."""
    return key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


def engine_claims() -> Claims:
    """Valid claims for a TensorFold build on Strix Halo, valid around ``NOW``."""
    return {
        "schema_version": 1,
        "engine_id": "foxlight.tensorfold",
        "publisher": PUBLISHER,
        "engine_version": "1.3.2",
        "generation": 1,
        "platform": "linux-glibc-x86_64",
        "lane": "rocm",
        "compute_targets": ["gfx1151"],
        "skulk_versions": ">=2.0.3,<3",
        "control_protocol": {"minimum": 1, "maximum": 2},
        "archive": {
            "filename": "tensorfold-1.3.2-linux-x86_64-gfx1151.tar.xz",
            "size": 23_939_744,
            "sha256": ARCHIVE_SHA256,
            "unpacked_size": 27_000_000,
            "executable": "bin/tensorfold-native",
        },
        "adapter": {
            "filename": "adapter.py",
            "size": 4096,
            "sha256": ADAPTER_SHA256,
            "kind": "python",
            "python_requires": ">=3.12",
        },
        "external_requirements": [
            {
                "name": "rocm-sdk-core",
                "version": "10.0.0",
                "filename": "rocm_sdk_core-10.0.0-py3-none-linux_x86_64.whl",
                "size": 900_000_000,
                "sha256": REQUIREMENT_SHA256,
                "url": "https://example.invalid/rocm/rocm_sdk_core-10.0.0-py3-none-linux_x86_64.whl",
            }
        ],
        "conservative_overhead_bytes": 2 * 1024**3,
        "provenance": {
            "repository": "ashhart/TensorFold",
            "version": "1.3.2",
            "source_commit": "b472b9f84ca9505d1d31020bed8fb4b5c7fdc33a",
            "built_by": "publisher",
        },
        "issued_at": NOW - 60,
        "expires_at": NOW + 86_400,
    }


def engine_signature(
    key: Ed25519PrivateKey,
    claims: Claims,
    *,
    domain: bytes = ENGINE_SIGNATURE_DOMAIN,
) -> str:
    """Sign the claims as a publisher does, under the given domain prefix."""
    return key.sign(domain + canonical_json(claims)).hex()


def engine_envelope(claims: Claims, signature: str) -> bytes:
    """The signed document: claims beside a detached signature."""
    return json.dumps({"engine": claims, "signature": signature}).encode()


def sign_engine_claims(
    key: Ed25519PrivateKey,
    claims: Claims,
    *,
    domain: bytes = ENGINE_SIGNATURE_DOMAIN,
) -> bytes:
    """A complete signed manifest document."""
    return engine_envelope(claims, engine_signature(key, claims, domain=domain))
