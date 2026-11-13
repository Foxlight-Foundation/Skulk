"""Signed inference admission is a distinct closure, never an execution grant."""

import hashlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue

from skulk.extensions.native_inference_contracts import (
    InferenceManifest,
    InferenceProfiles,
    inference_native_wheel,
)
from skulk.extensions.runtime_artifacts import (
    QualifiedHost,
    RuntimeTrust,
    VerifiedRuntime,
    canonical_json,
    verify_runtime,
)
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_selection import (
    RuntimeSelection,
    RuntimeSelector,
    runtime_contract_identity,
)
from skulk.extensions.tests.test_call_runtime import call_fixture
from skulk.extensions.tests.test_native_runtime import OBJECT, sign


def inference_fixture(
    directory: Path,
) -> tuple[dict[str, JsonValue], Ed25519PrivateKey, RuntimeTrust, QualifiedHost]:
    """Independently select the new signed domains and actual published wheel."""
    payload, key, trust, host = call_fixture(directory)
    profiles = InferenceProfiles.qualified()
    release = OBJECT.validate_python(payload["release"], strict=True)
    wrapper = OBJECT.validate_python(release["manifest"], strict=True)
    manifest = OBJECT.validate_json(str(wrapper["canonical_json"]))
    manifest.update(
        protocol=6,
        contract_profile="skulk.managed/2",
        profile_digest=profiles.managed,
        effect="inference",
    )
    bounds = OBJECT.validate_python(manifest["bounds"], strict=True)
    bounds["inference_seconds"] = None
    manifest["bounds"] = bounds
    contracts = OBJECT.validate_python(manifest["contracts"], strict=True)
    contracts.update(
        protocol=6,
        contract_profile="skulk.portable/2",
        profile_digest=profiles.portable,
        descriptors=[
            {
                "contract_profile": "skulk.portable/2",
                "profile_digest": profiles.portable,
                "id": "example.generate",
                "version": "1.0.0",
                "title": "Generate",
                "description": "Independent synthetic inference",
                "behavior": "inference",
                "processing_domain": "local",
                "input_schema": True,
                "output_schema": True,
            }
        ],
    )
    manifest["contracts"] = contracts
    record = InferenceManifest.from_json(json.dumps(manifest))
    release.update(protocol=3, manifest={"canonical_json": record.canonical_json})
    filename, digest, size = inference_native_wheel(host.platform)
    wheels = payload["wheels"]
    assert isinstance(wheels, list)
    wheels[1] = {"filename": filename, "sha256": digest, "size": size}
    release["dependency_lock_sha256"] = hashlib.sha256(
        canonical_json({"example-dep": "1.0", "skulk-capability-kernel": "0.7.0rc3"})
    ).hexdigest()
    payload.update(
        protocol=5,
        release=release,
        kernel=profiles.model_dump(mode="json"),
        control_profile="skulk.calls/2",
    )
    return payload, key, trust, host


def test_inference_closure_preserves_exact_domains_and_no_generation_ceiling(
    tmp_path: Path,
) -> None:
    """The real compiled reader accepts the new manifest without old identity reuse."""
    payload, key, trust, host = inference_fixture(tmp_path)
    runtime = verify_runtime(
        sign(payload, key, b"skulk.runtime/5\0"), trust, host, now=int(time.time())
    )
    assert runtime.signature_message == b"skulk.runtime/5\0" + canonical_json(payload)
    assert runtime.digest == hashlib.sha256(runtime.signature_message).hexdigest()
    release = OBJECT.validate_python(payload["release"], strict=True)
    assert (
        runtime.release_digest
        == hashlib.sha256(b"skulk.release/3\0" + canonical_json(release)).hexdigest()
    )
    assert isinstance(runtime.native_manifest, InferenceManifest)
    bounds = OBJECT.validate_python(
        runtime.native_manifest.value()["bounds"], strict=True
    )
    assert bounds["inference_seconds"] is None
    assert runtime.configuration_schema is True


@pytest.mark.parametrize(
    "fault",
    [
        "old-signature",
        "unprefixed",
        "old-control",
        "missing-control",
        "old-wheel",
        "old-author",
        "old-envelope",
        "old-release",
        "wrong-portable",
        "wrong-composition",
        "wrong-source",
        "float-protocol",
        "bool-protocol",
        "revoked",
        "duplicate",
    ],
)
def test_inference_closure_refuses_relabeling_and_substitution(
    tmp_path: Path,
    fault: str,
) -> None:
    """A signed label cannot translate old contracts or activate mismatched bytes."""
    payload, key, trust, host = inference_fixture(tmp_path)
    domain = b"skulk.runtime/5\0"
    if fault == "old-signature":
        domain = b"skulk.runtime/4\0"
    elif fault == "unprefixed":
        domain = b""
    elif fault == "old-control":
        payload["control_profile"] = "skulk.calls/1"
    elif fault == "missing-control":
        del payload["control_profile"]
    elif fault in ("old-wheel", "old-author"):
        old, _, _, _ = call_fixture(tmp_path / "old")
        if fault == "old-wheel":
            payload["wheels"] = old["wheels"]
        else:
            release = OBJECT.validate_python(payload["release"], strict=True)
            release["manifest"] = OBJECT.validate_python(old["release"], strict=True)[
                "manifest"
            ]
            payload["release"] = release
    elif fault == "old-envelope":
        payload["protocol"] = 4
        domain = b"skulk.runtime/4\0"
    elif fault == "old-release":
        release = OBJECT.validate_python(payload["release"], strict=True)
        release["protocol"] = 2
        payload["release"] = release
    elif fault.startswith("wrong-"):
        kernel = OBJECT.validate_python(payload["kernel"], strict=True)
        field = {
            "wrong-source": "source_commit",
            "wrong-composition": "composition",
            "wrong-portable": "portable",
        }[fault]
        kernel[field] = "a" * (40 if field == "source_commit" else 64)
        payload["kernel"] = kernel
    elif fault == "float-protocol":
        payload["protocol"] = 5.0
    elif fault == "bool-protocol":
        payload["protocol"] = True
    elif fault == "revoked":
        digest = hashlib.sha256(domain + canonical_json(payload)).hexdigest()
        trust = trust.model_copy(update={"revoked_artifacts": (digest,)})
    raw = sign(payload, key, domain)
    if fault == "duplicate":
        raw = raw.replace(b'"protocol":5', b'"protocol":4,"protocol":5')
    with pytest.raises(ValueError):
        verify_runtime(raw, trust, host, now=int(time.time()))


def test_equal_state_labels_do_not_authorize_an_inference_interpretation_upgrade(
    tmp_path: Path,
) -> None:
    """Native inference introduces a distinct migration boundary despite a shared label."""
    old, _, _, _ = call_fixture(tmp_path / "old")
    new, _, _, _ = inference_fixture(tmp_path / "new")
    assert (
        OBJECT.validate_python(old["release"], strict=True)["state_schema"]
        == OBJECT.validate_python(new["release"], strict=True)["state_schema"]
    )
    assert runtime_contract_identity(old) != runtime_contract_identity(new)
    assert runtime_contract_identity(new)[0] == "native-inference"


async def test_inference_update_preview_preserves_retained_identity_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained inference generation can update without weakening its digest check.

    Only the installer's already-verified generation observation is injected;
    signatures, compiled admission and stopped selection compatibility are real.
    """
    payload, key, trust, host = inference_fixture(tmp_path / "source")
    metadata = sign(payload, key, b"skulk.runtime/5\0")
    previous = verify_runtime(metadata, trust, host, now=int(time.time()))
    release = OBJECT.validate_python(payload["release"], strict=True)
    release["sequence"] = 2
    payload["release"] = release
    candidate = verify_runtime(
        sign(payload, key, b"skulk.runtime/5\0"), trust, host, now=int(time.time())
    )
    selector = RuntimeSelector(tmp_path / "installation")
    selected = RuntimeSelection(
        revision=1,
        operation_id="a" * 32,
        runtime_digest=previous.digest,
        enabled=False,
        bundle_id=previous.claims.release.manifest.bundle_id,
        sequence=1,
        highest_sequence=1,
        state_schema=previous.claims.release.state_schema,
        permissions=previous.claims.release.permissions,
        configuration_schema=previous.configuration_schema,
        platform=host.platform,
        python_version=host.python_version,
        skulk_version=host.skulk_version,
        skulk_build_sha256=host.skulk_build_sha256,
    )
    generation = selector.root / "generations" / previous.digest
    private_directory(generation)
    write_private(generation / "staged.json", metadata)
    before = selected.model_dump_json().encode()
    write_private(selector.root / "runtime-selection.json", before)
    canary = selector.root / "protected-state-canary"
    write_private(canary, b"Opaque retained application state.")

    @asynccontextmanager
    async def verified_candidate(
        _digest: str,
    ) -> AsyncIterator[tuple[VerifiedRuntime, QualifiedHost]]:
        yield candidate, host

    monkeypatch.setattr(selector.installer, "locked_generation", verified_candidate)
    preview = await selector.preview(
        candidate.digest, expected_revision=1, operation_id="b" * 32, enabled=False
    )
    assert preview.runtime_digest == candidate.digest and preview.sequence == 2
    assert read_private(selector.root / "runtime-selection.json") == before
    assert read_private(canary) == b"Opaque retained application state."
    altered = OBJECT.validate_json(metadata)
    altered["runtime"] = payload
    write_private(generation / "staged.json", canonical_json(altered))
    with pytest.raises(ValueError, match="selected contract metadata differs"):
        await selector.preview(
            candidate.digest, expected_revision=1, operation_id="c" * 32, enabled=False
        )
