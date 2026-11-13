"""Current policy enrollment against real signed metadata, without live children."""

import hashlib
import secrets
import time
from pathlib import Path

import pytest

from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.inference_callers import InferenceCallers
from skulk.extensions.inference_enrollment import (
    InferenceEnrollment,
    InferencePolicyUpdate,
)
from skulk.extensions.native_inference_authority import InferenceGrant, PrincipalAccess
from skulk.extensions.native_inference_contracts import InferenceManifest
from skulk.extensions.runtime_artifacts import (
    QualifiedHost,
    RuntimeTrust,
    verify_runtime,
)
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_files import read_private, write_private
from skulk.extensions.runtime_selection import RuntimeSelection
from skulk.extensions.tests.test_inference_runtime import inference_fixture
from skulk.extensions.tests.test_native_runtime import sign
from skulk.shared.types.common import NodeId


def installation(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[InferenceCallers, InferenceGrant, QualifiedHost, RuntimeTrust]:
    """Create exact signed/selected metadata and a persisted identity; start no server."""
    write_private(
        root / "host.json",
        HostSettings(profile_id="a" * 32, transport_node_id="enrollment-core")
        .model_dump_json()
        .encode(),
    )
    callers = InferenceCallers(root, "a" * 32, "enrollment-core")
    payload, key, trust, host = inference_fixture(root / "fixture")
    raw = sign(payload, key, b"skulk.runtime/5\0")
    runtime = verify_runtime(raw, trust, host, now=int(time.time()))
    manifest = runtime.native_manifest
    assert isinstance(manifest, InferenceManifest)
    selected = RuntimeSelection(
        revision=1,
        operation_id=secrets.token_hex(16),
        runtime_digest=runtime.digest,
        enabled=False,
        bundle_id=manifest.bundle_id,
        sequence=1,
        highest_sequence=1,
        state_schema=runtime.claims.release.state_schema,
        permissions=runtime.claims.release.permissions,
        configuration_schema=True,
        platform=host.platform,
        python_version=host.python_version,
        skulk_version=host.skulk_version,
        skulk_build_sha256=host.skulk_build_sha256,
    )
    path = root / "installations" / "managed.fixture"
    write_private(path / "runtime-selection.json", selected.model_dump_json().encode())
    write_private(path / "publisher-trust.json", trust.model_dump_json().encode())
    write_private(path / "generations" / runtime.digest / "staged.json", raw)
    write_private(
        path
        / ("bundle-" + hashlib.sha256(manifest.bundle_id.encode()).hexdigest()[:24])
        / "identity",
        b"enrolled-node",
    )
    descriptor = manifest.descriptors[0]
    grant = InferenceGrant(
        plugin_id="managed.fixture",
        runtime_digest=runtime.digest,
        node_id="enrolled-node",
        capability_id=descriptor.id,
        version=descriptor.version,
        descriptor_revision=descriptor.revision,
        behavior="inference",
        processing_domain=descriptor.processing_domain,
        io_mode=descriptor.io_mode,
    )
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    return callers, grant, host, trust


async def test_permission_enrollment_is_independent_of_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected signed app may be enrolled offline; no discovery/child socket is used."""
    callers, grant, _, _ = installation(tmp_path, monkeypatch)
    owner = callers.owner(AuthenticatedOwner(NodeId("enrollment-core")))
    update = InferencePolicyUpdate(
        expected_revision=callers.policy().revision,
        enabled=True,
        callers=(PrincipalAccess(principal=owner.principal, grants=(grant,)),),
    )
    policy = await InferenceEnrollment(callers).update(update)
    authorized = callers.authorize(owner, grant)
    assert authorized.access.maximum_seconds is None
    assert authorized.access.grants == (grant,)
    assert policy.enabled
    # No credential file, provider connection, live host socket or application
    # process was needed to establish permission. None is advertised as capacity.
    assert not (tmp_path / "inference-clients.json").exists()
    assert not (tmp_path / "native-read-policy.json").exists()
    with pytest.raises(ValueError, match="revision"):
        await InferenceEnrollment(callers).update(update)


@pytest.mark.parametrize(
    "fault",
    [
        "runtime",
        "descriptor",
        "domain",
        "effect",
        "mode",
        "node",
        "revoked",
        "missing-identity",
        "changed-selection",
    ],
)
async def test_signed_policy_enrollment_refuses_substituted_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Owner approval cannot relabel a signed effect, domain, mode or active generation."""
    callers, grant, host, trust = installation(tmp_path, monkeypatch)
    path = tmp_path / "installations" / grant.plugin_id
    changes = {
        "runtime": {"runtime_digest": "e" * 64},
        "descriptor": {"descriptor_revision": "e" * 64},
        "domain": {"processing_domain": "external"},
        "effect": {"behavior": "read"},
        "mode": {"io_mode": "server_streaming"},
        "node": {"node_id": "different-node"},
    }
    if fault in changes:
        grant = InferenceGrant.model_validate({**grant.model_dump(), **changes[fault]})
    elif fault == "revoked":
        trust = RuntimeTrust.model_validate(
            {**trust.model_dump(), "revoked_artifacts": (grant.runtime_digest,)}
        )
        write_private(path / "publisher-trust.json", trust.model_dump_json().encode())
    elif fault == "missing-identity":
        next(path.glob("bundle-*/identity")).unlink()
    else:

        def measure() -> QualifiedHost:
            selected = RuntimeSelection.model_validate_json(
                read_private(path / "runtime-selection.json")
            )
            write_private(
                path / "runtime-selection.json",
                selected.model_copy(update={"revision": 2}).model_dump_json().encode(),
            )
            return host

        monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", measure)
    owner = callers.owner(AuthenticatedOwner(NodeId("enrollment-core")))
    with pytest.raises((OSError, ValueError)):
        await InferenceEnrollment(callers).update(
            InferencePolicyUpdate(
                expected_revision=callers.policy().revision,
                enabled=True,
                callers=(PrincipalAccess(principal=owner.principal, grants=(grant,)),),
            )
        )
    assert not callers.policy().enabled


async def test_revoked_client_cannot_be_reenrolled_by_application_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring capability state never recreates a Core-owned client credential."""
    callers, grant, _, _ = installation(tmp_path, monkeypatch)
    client, token = callers.create(callers.inventory().revision, "API", None)
    caller = callers.authenticate(token)
    access = PrincipalAccess(principal=caller.principal, grants=(grant,))
    policy = await InferenceEnrollment(callers).update(
        InferencePolicyUpdate(
            expected_revision=callers.policy().revision, enabled=True, callers=(access,)
        )
    )
    callers.revoke(callers.inventory().revision, client.identifier)
    with pytest.raises(ValueError):
        callers.authorize(caller, grant)
    with pytest.raises(ValueError, match="not enrolled"):
        await InferenceEnrollment(callers).update(
            InferencePolicyUpdate(
                expected_revision=policy.revision, enabled=True, callers=(access,)
            )
        )
    assert callers.inventory().clients == ()
