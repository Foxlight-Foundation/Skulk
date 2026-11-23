"""Original-caller discovery admission; installed transport is qualified separately."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, TypeAdapter

from skulk.api.tests.test_inference_access import application
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_calls import NativeCalls, NativeHostEndpoint
from skulk.extensions.native_host_protocol import RequestFrame
from skulk.extensions.native_host_rpc import Peer, Reply
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    PrincipalAccess,
)
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_contracts import InferenceDescriptor
from skulk.extensions.native_inference_hosts import InferenceHosts, InferenceListing
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import write_private
from skulk.extensions.tests.test_native_host_protocol import Channels, binding
from skulk.extensions.tests.test_native_inference_selection import listing

_OBJECT = TypeAdapter(dict[str, JsonValue])


def destination(name: str, *, version: str = "1.0.0") -> InferenceListing:
    """Create actual compiled declarations; never forge a descriptor revision."""
    base = listing(False)
    definition = base.descriptor.model_dump(mode="json")
    definition.update(id=name, version=version)
    descriptor = InferenceDescriptor.model_validate_json(json.dumps(definition))
    value = base.model_dump(mode="json")
    value.update(
        descriptor=descriptor.model_dump(mode="json"),
        target={
            "node_id": "node-" + name,
            "capability_id": descriptor.id,
            "version": descriptor.version,
            "descriptor_revision": descriptor.revision,
            "behavior": descriptor.behavior,
            "processing_domain": descriptor.processing_domain,
        },
    )
    return InferenceListing.model_validate_json(json.dumps(value))


@dataclass
class Parent:
    """Real caller credentials and compiled scope, with injected host observations."""

    service: NativeInferenceCalls
    peer: Peer
    foreign_peer: Peer
    call_id: str
    caller: AuthenticatedInferenceCaller
    root: Path
    visible: tuple[InferenceListing, ...]
    replies: list[Reply] = field(default_factory=list)
    scans: int = 0

    @property
    def active(self):
        """Observe custody created by the real public root dispatch."""
        return next(iter(self.service.active.values()))

    async def query(self, **changes: JsonValue) -> Reply:
        """Use the same protected handler and retain every reply for cleanup."""
        fields = _OBJECT.validate_json(self.active.context.canonical())
        parameters: dict[str, JsonValue] = {
            "query_kind": "directory",
            "parent_call_id": self.call_id,
            "parent_scope_id": fields["scope_id"],
            "parent_request_sha256": fields["request_sha256"],
            "request_id": f"{self.active.next_discovery:064x}",
        }
        parameters.update(changes)
        handler = self.service.hosts.inference_handler
        assert handler is not None
        result = await handler(
            self.peer,
            RequestFrame(
                request_id=1,
                operation="describe_inference",
                parameters=parameters,
                payload_size=0,
            ),
            b"",
        )
        self.replies.append(result)
        return result


@pytest.fixture
async def parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: Channels
) -> AsyncIterator[Parent]:
    """Inject observation/selection only; keep real caller and native scope admission."""
    application(tmp_path)
    service = NativeInferenceCalls(
        ServiceConnection(manager_root=str(tmp_path), profile_id="a" * 32),
        "access-core",
    )
    _, token = service.callers.create(
        service.callers.inventory().revision, "Scope", None
    )
    caller = service.callers.authenticate(token)
    source, allowed, broader = (
        destination("source"),
        destination("allowed"),
        destination("broader"),
    )
    policy = InferenceAccessPolicy(
        revision="f" * 64,
        enabled=True,
        callers=(
            PrincipalAccess(
                principal=caller.principal,
                grants=tuple(v.grant() for v in (source, allowed, broader)),
            ),
        ),
    )
    write_private(
        tmp_path / "native-inference-policy.json", policy.model_dump_json().encode()
    )
    entered = asyncio.Event()

    async def owner_handler(_peer: Peer, _request: RequestFrame, _body: bytes) -> Reply:
        """Hold the actual protected root open while its nested query is observed."""
        entered.set()
        await asyncio.Future[None]()
        return Reply("complete", {})

    handler = service.hosts.inference_handler
    assert handler is not None
    peer = Peer(channels.core, handler, frozenset({"describe_inference"}), lambda: None)
    foreign = Peer(channels.owner, owner_handler, frozenset({"infer"}), lambda: None)
    service.hosts.endpoints[source.plugin_id] = NativeHostEndpoint(
        binding(Ed25519PrivateKey.generate()), peer
    )
    state = Parent(
        service, peer, foreign, "", caller, tmp_path, (source, allowed, broader)
    )

    async def observe(
        _hosts: InferenceHosts, *, streaming: bool = False
    ) -> tuple[InferenceListing, ...]:
        """Inject ready observations while preserving all production authorization."""
        if streaming:
            state.scans += 1
        return state.visible

    def selected(_hosts: NativeCalls, _plugin: str, _runtime: str, _peer: Peer) -> None:
        """Installed host custody is independently exercised by normal installed tests."""

    monkeypatch.setattr(InferenceHosts, "directory", observe)
    monkeypatch.setattr(NativeCalls, "check_owner", selected)
    task = asyncio.create_task(
        service.invoke(
            caller,
            source.plugin_id,
            source.target.capability_id,
            source.target.version,
            b"{}",
        )
    )
    try:
        async with asyncio.timeout(5):
            await entered.wait()
        state.call_id = next(iter(service.active))[1]
        context = _OBJECT.validate_json(state.active.context.canonical())
        context["allowed_targets"] = [
            v.target.model_dump(mode="json") for v in (source, allowed)
        ]
        state.active.context = service.kernel.CompositionV3Contract.from_json(
            "Context", json.dumps(context)
        )
        yield state
    finally:
        for reply in state.replies:
            reply.release()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.close()
        await asyncio.gather(peer.close(), foreign.close())


async def test_discovery_intersects_parent_ceiling_without_an_ambient_principal(
    parent: Parent,
) -> None:
    """A broader real caller grant cannot widen an already attenuated invocation."""
    reply = await parent.query()
    try:
        reply.check()
        assert reply.outcome == "complete"
        observed = TypeAdapter(list[dict[str, JsonValue]]).validate_json(reply.payload)
        assert [v["target"] for v in observed] == [
            parent.visible[1].target.model_dump(mode="json")
        ]
        assert observed[0]["inference_seconds"] is None
        assert parent.active.next_delegation == 1
        assert parent.active.discovering and parent.active.children
        with pytest.raises(ValueError):
            parent.service.callers.authorize(
                parent.service.callers.authenticate("not-a-client"),
                parent.visible[1].grant(),
            )
    finally:
        reply.release()
    assert not parent.active.discovering and not parent.active.children


@pytest.mark.parametrize(
    "field", ["caller_id", "issuer_node_id", "policy_revision", "destination"]
)
async def test_discovery_refuses_replacement_identity_fields(
    parent: Parent, field: str
) -> None:
    """An app cannot turn a scope query into a service or another caller's query."""
    reply = await parent.query(**{field: "ambient"})
    assert reply.outcome == "pre_dispatch" and not reply.payload
    assert parent.scans == 0


@pytest.mark.parametrize(
    "field", ["parent_call_id", "parent_scope_id", "parent_request_sha256"]
)
async def test_discovery_refuses_foreign_parent_bindings(
    parent: Parent, field: str
) -> None:
    """Each parent correlation is independently checked before observation."""
    reply = await parent.query(**{field: "b" * 64})
    assert reply.outcome == "pre_dispatch" and not reply.payload
    assert parent.scans == 0


async def test_discovery_cannot_cross_the_actual_role_peer(parent: Parent) -> None:
    """Copied original context is not a grant on a different authenticated connection."""
    parent.peer = parent.foreign_peer
    reply = await parent.query()
    assert reply.outcome == "pre_dispatch" and parent.scans == 0


@pytest.mark.parametrize(
    "state",
    ["uncertain", "execution_complete", "retiring", "delegating", "discovering"],
)
async def test_unavailable_parent_never_observes_the_directory(
    parent: Parent, state: str
) -> None:
    """Retired or occupied source custody supplies no catalog metadata."""
    setattr(parent.active, state, True)
    reply = await parent.query()
    assert reply.outcome == "pre_dispatch" and parent.scans == 0


@pytest.mark.parametrize(
    "fault", ["policy", "credential", "completion", "retiring", "removed"]
)
async def test_queued_discovery_rechecks_authority_at_physical_delivery(
    parent: Parent, fault: str
) -> None:
    """A prepared response cannot outlive original caller or parent authorization."""
    reply = await parent.query()
    try:
        assert reply.outcome == "complete"
        if fault == "policy":
            write_private(
                parent.root / "native-inference-policy.json",
                InferenceAccessPolicy(revision="e" * 64, enabled=False, callers=())
                .model_dump_json()
                .encode(),
            )
        elif fault == "credential":
            parent.service.callers.revoke(
                parent.service.callers.inventory().revision,
                parent.caller.principal.identifier.removeprefix("client:"),
            )
        elif fault == "completion":
            parent.active.execution_complete = True
        elif fault == "retiring":
            parent.active.retiring = True
        else:
            parent.service.active.clear()
        with pytest.raises(ValueError):
            reply.check()
    finally:
        reply.release()


async def test_discovery_sequence_is_separate_and_refuses_replay(
    parent: Parent,
) -> None:
    """A repeated query never consumes an effect sequence or reuses cached metadata."""
    first = await parent.query(request_id="0" * 63 + "1")
    first.release()
    replay = await parent.query(request_id="0" * 63 + "1")
    assert replay.outcome == "pre_dispatch"
    assert parent.scans == 1 and parent.active.next_delegation == 1
    second = await parent.query()
    second.release()
    assert second.outcome == "complete" and parent.scans == 2


async def test_concurrent_query_has_no_implicit_queue(parent: Parent) -> None:
    """A pending guarded response owns its single source slot until release."""
    first = await parent.query()
    try:
        second = await parent.query()
        assert second.outcome == "pre_dispatch" and parent.scans == 1
    finally:
        first.release()


@pytest.mark.parametrize("cancel", [False, True])
async def test_discovery_observation_retains_cancellation_custody(
    parent: Parent, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """Withdrawal during an observation discards bytes and retires the query slot."""
    entered, release, drained = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def observe(
        _hosts: InferenceHosts, *, streaming: bool = False
    ) -> tuple[InferenceListing, ...]:
        assert streaming
        entered.set()
        try:
            await release.wait()
            return parent.visible
        finally:
            drained.set()

    monkeypatch.setattr(InferenceHosts, "directory", observe)
    task = asyncio.create_task(parent.query())
    try:
        await entered.wait()
        assert parent.active.discovering and task in parent.active.children
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            parent.active.execution_complete = True
            release.set()
            result = await task
            assert result.outcome == "pre_dispatch" and not result.payload
        assert drained.is_set()
        assert not parent.active.discovering and task not in parent.active.children
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_discovery_excludes_executor_alias_even_when_explicitly_allowed(
    parent: Parent,
) -> None:
    """Changing a version or descriptor cannot turn self-recursion into a new target."""
    alias = destination("source", version="2.0.0")
    context = _OBJECT.validate_json(parent.active.context.canonical())
    context["allowed_targets"] = [alias.target.model_dump(mode="json")]
    parent.active.context = parent.service.kernel.CompositionV3Contract.from_json(
        "Context", json.dumps(context)
    )
    parent.visible = (alias,)
    reply = await parent.query()
    try:
        assert reply.outcome == "complete" and reply.payload == b"[]"
    finally:
        reply.release()


@pytest.mark.parametrize("keep_other_destination", [False, True])
async def test_discovery_omits_an_unenrolled_runtime_generation(
    parent: Parent, keep_other_destination: bool
) -> None:
    """An upgraded generation cannot suppress other exact-granted observations."""
    source, allowed, broader = parent.visible
    if keep_other_destination:
        # This fixture normally narrows to one child. Retain two already
        # enrolled destinations for the mixed-generation observation case.
        context = _OBJECT.validate_json(parent.active.context.canonical())
        context["allowed_targets"] = [
            v.target.model_dump(mode="json") for v in (source, allowed, broader)
        ]
        parent.active.context = parent.service.kernel.CompositionV3Contract.from_json(
            "Context", json.dumps(context)
        )
    fields = allowed.model_dump(mode="json")
    fields["runtime_digest"] = "d" * 64
    upgraded = InferenceListing.model_validate_json(json.dumps(fields))
    assert upgraded.target == allowed.target and upgraded.grant() != allowed.grant()
    with pytest.raises(ValueError):
        parent.service.callers.authorize(parent.caller, upgraded.grant())
    parent.visible = (source, upgraded, broader)
    response = await parent.query()
    try:
        assert response.outcome == "complete"
        response.check()
        observed = TypeAdapter(list[dict[str, JsonValue]]).validate_json(
            response.payload
        )
        assert [item["target"] for item in observed] == (
            [broader.target.model_dump(mode="json")] if keep_other_destination else []
        )
        assert parent.active.next_delegation == 1
    finally:
        response.release()
