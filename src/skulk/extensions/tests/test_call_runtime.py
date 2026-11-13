"""Independent call-cohort signing and retained-state interpretation regressions."""

import hashlib
import importlib.metadata
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue

from skulk.extensions.native_call_contracts import call_native_wheel
from skulk.extensions.runtime_artifacts import (
    QualifiedHost,
    RuntimeTrust,
    VerifiedRuntime,
    canonical_json,
    verify_runtime,
)
from skulk.extensions.runtime_files import private_directory, write_private
from skulk.extensions.runtime_selection import RuntimeSelection, RuntimeSelector
from skulk.extensions.tests.test_native_runtime import (
    OBJECT,
    PROFILES,
    native_fixture,
    sign,
)

CALL_PROFILES: dict[str, JsonValue] = {
    **PROFILES,
    "version": "0.6.0rc1",
    "source_commit": "cca3dcea5ecce5e282e0afc2d205fbd586ba0add",
    "composition": "6b3b7ce596b414b40d7f74d8fe7351cb3ad93a2ecdc5c0e7623ece86b2c252d3",
    "calls": "8a2cfb90e8c986bdeb8fdc7b4dcc223740ad5b8e3c710e8e334a0a7e31e7e002",
}


def call_fixture(
    directory: Path,
) -> tuple[dict[str, JsonValue], Ed25519PrivateKey, RuntimeTrust, QualifiedHost]:
    """Select a new envelope and exact binary without changing old signed bytes."""
    payload, key, trust, host = native_fixture(directory)
    filename, digest, size = call_native_wheel(host.platform)
    wheels = payload["wheels"]
    assert isinstance(wheels, list)
    wheels[1] = {"filename": filename, "sha256": digest, "size": size}
    release = OBJECT.validate_python(payload["release"], strict=True)
    release["dependency_lock_sha256"] = hashlib.sha256(
        canonical_json({"example-dep": "1.0", "skulk-capability-kernel": "0.6.0rc1"})
    ).hexdigest()
    payload.update(
        protocol=4,
        release=release,
        kernel=dict(CALL_PROFILES),
        control_profile="skulk.calls/1",
    )
    return payload, key, trust, host


def test_call_envelope_has_independent_domain_and_explicit_control(
    tmp_path: Path,
) -> None:
    """A compatible reader preserves manifest bytes but never aliases runtime IDs."""
    payload, key, trust, host = call_fixture(tmp_path)
    runtime = verify_runtime(
        sign(payload, key, b"skulk.runtime/4\0"), trust, host, now=int(time.time())
    )
    assert runtime.claims.protocol == 4
    assert runtime.signature_message == b"skulk.runtime/4\0" + canonical_json(payload)
    assert runtime.digest == hashlib.sha256(runtime.signature_message).hexdigest()
    assert runtime.native_manifest is not None
    assert runtime.configuration_schema is True


@pytest.mark.parametrize(
    "fault",
    [
        "old-domain",
        "unprefixed",
        "missing-control",
        "old-control",
        "old-wheel",
        "composition",
        "calls",
        "source",
        "bool",
        "float",
        "revoked",
        "duplicate",
        "old-envelope",
    ],
)
def test_call_envelope_refuses_identity_substitution(
    tmp_path: Path, fault: str
) -> None:
    """No signature, control, cohort or closure mismatch reaches filesystem effects."""
    payload, key, trust, host = call_fixture(tmp_path)
    domain = b"skulk.runtime/4\0"
    if fault == "old-domain":
        domain = b"skulk.runtime/3\0"
    elif fault == "unprefixed":
        domain = b""
    elif fault == "missing-control":
        del payload["control_profile"]
    elif fault == "old-control":
        payload["control_profile"] = "skulk.transport/1"
    elif fault == "old-wheel":
        old, _, _, _ = native_fixture(tmp_path / "old")
        payload["wheels"] = old["wheels"]
    elif fault in {"composition", "calls", "source"}:
        field = "source_commit" if fault == "source" else fault
        payload["kernel"] = {**CALL_PROFILES, field: "b" * 64}
    elif fault in {"bool", "float", "old-envelope"}:
        payload["protocol"] = {"bool": True, "float": 4.0, "old-envelope": 3}[fault]
        domain = b"skulk.runtime/3\0" if fault == "old-envelope" else b""
    elif fault == "revoked":
        digest = hashlib.sha256(domain + canonical_json(payload)).hexdigest()
        trust = trust.model_copy(update={"revoked_artifacts": (digest,)})
    raw = sign(payload, key, domain)
    if fault == "duplicate":
        raw = raw.replace(b'"protocol":4', b'"protocol":3,"protocol":4')
    with pytest.raises(ValueError):
        verify_runtime(raw, trust, host, now=int(time.time()))
    assert not (tmp_path / "generations").exists()


async def test_call_selection_requires_explicit_state_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal schema names cannot silently promote old control into composed calls."""
    payload, key, trust, host = native_fixture(tmp_path / "old")
    metadata = sign(payload, key)
    old = verify_runtime(metadata, trust, host, now=int(time.time()))
    candidate_payload, new_key, new_trust, new_host = call_fixture(tmp_path / "new")
    candidate = verify_runtime(
        sign(candidate_payload, new_key, b"skulk.runtime/4\0"),
        new_trust,
        new_host,
        now=int(time.time()),
    )
    assert old.claims.release.state_schema == candidate.claims.release.state_schema
    selector = RuntimeSelector(tmp_path / "installation")
    selected = RuntimeSelection(
        revision=1,
        operation_id="a" * 32,
        runtime_digest=old.digest,
        enabled=False,
        bundle_id=old.claims.release.manifest.bundle_id,
        sequence=1,
        highest_sequence=1,
        state_schema=old.claims.release.state_schema,
        permissions=old.claims.release.permissions,
        configuration_schema=True,
        platform=host.platform,
        python_version=host.python_version,
        skulk_version=host.skulk_version,
        skulk_build_sha256=host.skulk_build_sha256,
    )
    generation = selector.root / "generations" / old.digest
    private_directory(generation)
    write_private(generation / "staged.json", metadata)
    before = selected.model_dump_json().encode()
    write_private(selector.root / "runtime-selection.json", before)
    canary = selector.root / "protected-state-canary"
    write_private(canary, b"Existing opaque settings and grants.")

    @asynccontextmanager
    async def verified_candidate(
        _digest: str,
    ) -> AsyncIterator[tuple[VerifiedRuntime, QualifiedHost]]:
        yield candidate, new_host

    monkeypatch.setattr(selector.installer, "locked_generation", verified_candidate)
    with pytest.raises(ValueError, match="profile change requires explicit migration"):
        await selector.preview(
            candidate.digest,
            expected_revision=1,
            operation_id="b" * 32,
            rollback=False,
            accept_permissions=False,
        )
    assert (selector.root / "runtime-selection.json").read_bytes() == before
    assert canary.read_bytes() == b"Existing opaque settings and grants."


def test_call_envelope_never_downgrades_to_loaded_old_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading frozen old records does not establish support for the call ABI."""
    payload, key, trust, host = call_fixture(tmp_path)
    version = importlib.metadata.version

    def old_reader(name: str) -> str:
        return "0.4.0rc1" if name == "skulk-capability-kernel" else version(name)

    monkeypatch.setattr(importlib.metadata, "version", old_reader)
    with pytest.raises(ValueError, match="native call reader is unavailable"):
        verify_runtime(
            sign(payload, key, b"skulk.runtime/4\0"), trust, host, now=int(time.time())
        )
    assert not (tmp_path / "generations").exists()
