"""Actual protected credential/policy custody, never mocked bearer validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.inference_callers import InferenceCallers
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    InferenceGrant,
    PrincipalAccess,
)
from skulk.extensions.runtime_attachment import (
    AttachmentJournal,
    HostSettings,
    finish_attachment,
)
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private
from skulk.shared.types.common import NodeId


@pytest.fixture
def callers(tmp_path: Path) -> InferenceCallers:
    """Create private manager custody with an independently controlled clock."""
    write_private(
        tmp_path / "host.json",
        HostSettings(profile_id="a" * 32, transport_node_id="inference-core")
        .model_dump_json()
        .encode(),
    )
    return InferenceCallers(tmp_path, "a" * 32, "inference-core", unix_ms=lambda: 1000)


def grant(**changes: str) -> InferenceGrant:
    """Make a host policy fixture without claiming a signed installed runtime."""
    return InferenceGrant.model_validate(
        {
            "plugin_id": "managed.fixture",
            "runtime_digest": "b" * 64,
            "node_id": "fixture-node",
            "capability_id": "fixture.infer",
            "version": "1.0.0",
            "descriptor_revision": "c" * 64,
            "behavior": "inference",
            "processing_domain": "external",
            "io_mode": "server_streaming",
            **changes,
        }
    )


def test_new_credential_identifies_but_does_not_authorize(
    callers: InferenceCallers,
) -> None:
    """A connected account, installed package and real bearer supply no grant."""
    client, token = callers.create(
        callers.inventory().revision, "Stock API client", None
    )
    caller = callers.authenticate(token)
    assert caller.principal.identifier == f"client:{client.identifier}"
    assert caller.principal.kind == "client"
    assert caller.principal.issuer_node_id == "core-profile:" + callers.profile_id
    assert callers.current(caller) is None
    assert not callers.policy().enabled
    with pytest.raises(ValueError, match="not enrolled"):
        callers.authorize(caller, grant())
    stored = read_private(callers.root / "inference-clients.json")
    assert token.encode() not in stored
    assert token not in repr(caller)
    assert b"token_digest" not in callers.inventory().model_dump_json().encode()
    assert b"digest_key" not in callers.inventory().model_dump_json().encode()


@pytest.mark.parametrize(
    "fault",
    ["secret", "identifier", "trailing", "prefix", "missing", "owner", "gateway"],
)
def test_forged_or_malformed_bearers_never_authenticate(
    callers: InferenceCallers, fault: str
) -> None:
    """The identifier, original caller text and provider/gateway credentials are inert."""
    _, token = callers.create(callers.inventory().revision, "Client", None)
    invalid = {
        "secret": token[:-1] + ("0" if token[-1] != "0" else "1"),
        "identifier": "ski_" + "f" * 32 + token[36:],
        "trailing": token + " ",
        "prefix": token.upper(),
        "missing": "",
        "owner": "direct-owner",
        "gateway": "gateway-signed-body",
    }[fault]
    with pytest.raises(ValueError):
        callers.authenticate(invalid)


def test_revocation_invalidates_an_already_authenticated_caller_without_touching_peers(
    callers: InferenceCallers,
) -> None:
    """No cached caller object can outlive removal of its exact credential lifetime."""
    first, token = callers.create(callers.inventory().revision, "First", None)
    held = callers.authenticate(token)
    _, second_token = callers.create(callers.inventory().revision, "Second", None)
    callers.revoke(callers.inventory().revision, first.identifier)
    with pytest.raises(ValueError):
        callers.authenticate(token)
    with pytest.raises(ValueError, match="revoked"):
        callers.current(held)
    assert (
        callers.authenticate(second_token).principal.identifier
        != held.principal.identifier
    )


def test_client_expiry_is_optional_and_checked_after_queueing(
    callers: InferenceCallers,
) -> None:
    """No default generation ceiling is invented; a real finite expiry still binds."""
    _, token = callers.create(callers.inventory().revision, "Finite", 1100)
    caller = callers.authenticate(token)
    assert callers.current(caller) == 1100
    future = InferenceCallers(
        callers.root,
        callers.profile_id,
        callers.transport_node_id,
        unix_ms=lambda: 1100,
    )
    with pytest.raises(ValueError, match="expired"):
        future.current(caller)
    with pytest.raises(ValueError):
        future.authenticate(token)
    with pytest.raises(ValueError):
        callers.create(callers.inventory().revision, "Already expired", 1000)


def test_mutations_use_actual_cross_process_dispatch_lock_and_revision(
    callers: InferenceCallers,
) -> None:
    """A management race cannot bypass the same lock used by the final owner write."""
    old = callers.inventory().revision
    client, _ = callers.create(old, "First", None)
    current = callers.inventory().revision
    with pytest.raises(ValueError, match="revision"):
        callers.create(old, "Stale", None)
    with pytest.raises(ValueError, match="revision"):
        callers.revoke(old, client.identifier)
    lock = RuntimeLock(callers.root, "native-policy.lock")
    try:
        with pytest.raises(BlockingIOError):
            callers.revoke(current, client.identifier)
        with pytest.raises(BlockingIOError):
            callers.create(current, "Blocked", None)
    finally:
        lock.close()
    assert callers.inventory().revision == current
    assert len(callers.inventory().clients) == 1


@pytest.mark.parametrize("fault", ["profile", "issuer", "symlink", "public", "corrupt"])
def test_custody_substitution_is_refused(
    callers: InferenceCallers, fault: str, tmp_path: Path
) -> None:
    """A copied, symlinked, nonprivate or corrupt store cannot establish identity."""
    _, token = callers.create(callers.inventory().revision, "First", None)
    path = callers.root / "inference-clients.json"
    if fault in {"profile", "issuer"}:
        host = HostSettings(
            profile_id="d" * 32 if fault == "profile" else callers.profile_id,
            transport_node_id="another-core"
            if fault == "issuer"
            else callers.transport_node_id,
        )
        write_private(callers.root / "host.json", host.model_dump_json().encode())
    elif fault == "symlink":
        target = tmp_path / "copied.json"
        path.rename(target)
        path.symlink_to(target)
    elif fault == "public":
        path.chmod(0o644)
    else:
        write_private(path, b"{invalid")
    with pytest.raises((OSError, ValueError)):
        callers.authenticate(token)


def test_permissions_bind_caller_effect_domain_runtime_mode_and_policy_revision(
    callers: InferenceCallers,
) -> None:
    """A policy grant is unrelated to provider defaults, readiness or the caller token."""
    _, token = callers.create(callers.inventory().revision, "API", None)
    caller = callers.authenticate(token)
    destination = grant()
    policy = InferenceAccessPolicy(
        revision="d" * 64,
        enabled=True,
        callers=(PrincipalAccess(principal=caller.principal, grants=(destination,)),),
    )
    write_private(
        callers.root / "native-inference-policy.json", policy.model_dump_json().encode()
    )
    admitted = callers.authorize(caller, destination)
    assert admitted.access.maximum_seconds is None
    for changed in (
        grant(behavior="read"),
        grant(processing_domain="local"),
        grant(runtime_digest="e" * 64),
        grant(io_mode="unary"),
        grant(descriptor_revision="f" * 64),
    ):
        with pytest.raises(ValueError):
            callers.authorize(caller, changed)
    owner = callers.owner(AuthenticatedOwner(NodeId(callers.transport_node_id)))
    with pytest.raises(ValueError):
        callers.authorize(owner, destination)
    changed_policy = InferenceAccessPolicy(
        revision="e" * 64, enabled=True, callers=policy.callers
    )
    write_private(
        callers.root / "native-inference-policy.json",
        changed_policy.model_dump_json().encode(),
    )
    assert callers.authorize(caller, destination).revision != admitted.revision


def test_credential_capacity_refuses_without_mutating_custody(
    callers: InferenceCallers,
) -> None:
    """A full registry has a deterministic refusal and no evicted live identity."""
    client, token = callers.create(callers.inventory().revision, "Template", None)
    state = callers.state()
    assert state is not None
    from skulk.extensions.native_inference_authority import InferenceCredentialState

    full = InferenceCredentialState(
        profile_id=state.profile_id,
        issuer_node_id=state.issuer_node_id,
        revision=state.revision,
        digest_key=state.digest_key,
        clients=tuple(
            state.clients[0].model_copy(update={"identifier": f"{i:032x}"})
            for i in range(255)
        )
        + state.clients,
    )
    write_private(
        callers.root / "inference-clients.json", full.model_dump_json().encode()
    )
    before = read_private(callers.root / "inference-clients.json")
    with pytest.raises(ValidationError):
        callers.create(full.revision, "Too many", None)
    assert read_private(callers.root / "inference-clients.json") == before
    assert (
        callers.authenticate(token).principal.identifier
        == f"client:{client.identifier}"
    )


def test_normal_transport_renewal_preserves_credentials_grants_and_revocation(
    callers: InferenceCallers,
) -> None:
    """A journaled restart changes the live role binding, not the durable caller realm."""
    kept, token = callers.create(callers.inventory().revision, "Kept", None)
    revoked, old_token = callers.create(callers.inventory().revision, "Revoked", None)
    callers.revoke(callers.inventory().revision, revoked.identifier)
    original = callers.authenticate(token)
    destination = grant()
    policy = InferenceAccessPolicy(
        revision="c" * 64,
        enabled=True,
        callers=(PrincipalAccess(principal=original.principal, grants=(destination,)),),
    )
    write_private(
        callers.root / "native-inference-policy.json", policy.model_dump_json().encode()
    )
    authorization = callers.authorize(original, destination)
    before = read_private(callers.root / "inference-clients.json")
    previous = HostSettings.model_validate_json(
        read_private(callers.root / "host.json")
    )
    current = HostSettings(
        profile_id=callers.profile_id, transport_node_id="restarted-core"
    )
    lock = RuntimeLock(callers.root, "manager.lock")
    try:
        finish_attachment(
            callers.root,
            AttachmentJournal(
                previous=previous, current=current, installations=(), state="pending"
            ),
        )
    finally:
        lock.close()
    with pytest.raises(ValueError, match="attachment differs"):
        callers.inventory()
    restarted = InferenceCallers(
        callers.root,
        callers.profile_id,
        current.transport_node_id,
        unix_ms=lambda: 1000,
    )
    authenticated = restarted.authenticate(token)
    assert authenticated == original
    assert restarted.authorize(authenticated, destination) == authorization
    assert read_private(callers.root / "inference-clients.json") == before
    assert restarted.inventory().clients[0].identifier == kept.identifier
    with pytest.raises(ValueError):
        restarted.authenticate(old_token)
    with pytest.raises(ValueError, match="owner issuer differs"):
        restarted.owner(AuthenticatedOwner(NodeId(previous.transport_node_id)))
    assert (
        restarted.owner(
            AuthenticatedOwner(NodeId(current.transport_node_id))
        ).principal.issuer_node_id
        == original.principal.issuer_node_id
    )
    added, _ = restarted.create(restarted.inventory().revision, "After restart", None)
    restarted.revoke(restarted.inventory().revision, added.identifier)
    assert restarted.authenticate(token) == original
