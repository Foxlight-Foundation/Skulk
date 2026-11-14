"""Original-caller admission for exact installed native inference destinations.

This is deterministic host authorization and dispatch, not model routing. It
contains no provider client, account session, task classifier or retry policy.
"""

import asyncio
import hashlib
import json
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.extensions.inference_callers import (
    AuthenticatedInferenceCaller,
    InferenceCallers,
)
from skulk.extensions.native_calls import NativeCalls
from skulk.extensions.native_host_protocol import (
    Digest,
    Received,
    RequestFrame,
    ResponseFrame,
)
from skulk.extensions.native_host_rpc import Peer, Reply
from skulk.extensions.native_inference_contracts import (
    CompositionRecord,
    public_inference_kernel,
)
from skulk.extensions.native_inference_hosts import (
    InferenceHosts,
    InferenceListing,
    InferenceTarget,
)
from skulk.extensions.native_inference_streams import NativeInferenceStream
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_install import finish_runtime_work

_OBJECT = TypeAdapter(dict[str, JsonValue])
_TEXT = TypeAdapter(str)
_DEADLINE: TypeAdapter[int | None] = TypeAdapter(int | None)


@final
class InferenceDelegation(BaseModel):
    """Core-role proxy claims, matched against the actual active source parent."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )
    parent_call_id: Digest = Field(
        description="Actual host-assigned active source call."
    )
    request: dict[str, JsonValue] = Field(
        description="Compiled delegation request and exact body binding."
    )
    context: dict[str, JsonValue] = Field(
        description="Candidate metadata independently recomputed by Core."
    )
    remaining_seconds: float | None = Field(
        gt=0, description="Actual attenuated budget, including explicit null."
    )


@final
@dataclass(slots=True, repr=False)
class _Active:
    peer: Peer
    listing: InferenceListing
    context: CompositionRecord
    caller: AuthenticatedInferenceCaller
    revision: str
    expires_at: float | None
    deadline: int | None
    task: asyncio.Task[object]
    caller_guard: Callable[[], None] | None = None
    children: set[asyncio.Task[object]] = field(default_factory=set)
    next_delegation: int = 1
    delegating: bool = False
    uncertain: bool = False


@final
class NativeInferenceCalls:
    """Own exact inference dispatch and a bounded active original-caller tree."""

    def __init__(self, connection: ServiceConnection, issuer_node_id: str) -> None:
        """Bind actual host custody and require the explicitly qualified new binary."""
        self.connection, self.issuer_node_id = connection, issuer_node_id
        self.kernel = public_inference_kernel()
        self.hosts = NativeCalls(
            connection, issuer_node_id, inference_handler=self._delegate
        )
        self.callers = InferenceCallers(
            self.hosts.root, connection.profile_id, issuer_node_id
        )
        self.callers.check()
        self.directory = InferenceHosts(self.hosts)
        self.active: dict[tuple[Peer, str], _Active] = {}
        self.closed = False
        self.watch: asyncio.Task[None] | None = None

    def _remaining(self, active: _Active) -> float | None:
        if active.expires_at is None and active.deadline is None:
            return None
        if active.expires_at is None or active.deadline is None:
            raise ValueError("inference budget binding differs")
        seconds = min(
            active.expires_at - asyncio.get_running_loop().time(),
            (active.deadline - time.time_ns() // 1_000_000) / 1000,
        )
        if seconds <= 0:
            raise ValueError("inference caller deadline elapsed")
        return seconds

    def _permitted(self, active: _Active) -> None:
        if active.caller_guard is not None:
            active.caller_guard()
        if self.closed:
            raise ValueError("inference Core host unavailable")
        authorization = self.callers.authorize(active.caller, active.listing.grant())
        if authorization.revision != active.revision:
            raise ValueError("inference caller policy changed")
        self.hosts.check_owner(
            active.listing.plugin_id, active.listing.runtime_digest, active.peer
        )
        self._remaining(active)

    async def _watch_authority(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            for active in list(self.active.values()):
                try:
                    self._permitted(active)
                except (OSError, ValueError):
                    active.task.cancel()

    async def invoke(
        self,
        caller: AuthenticatedInferenceCaller,
        plugin_id: str,
        capability_id: str,
        version: str,
        payload: bytes,
        seconds: float | None = None,
    ) -> Received:
        """Invoke one exact unary destination under the authenticated original caller."""
        return await self._invoke(
            caller, plugin_id, capability_id, version, payload, seconds
        )

    async def invoke_stream(
        self,
        caller: AuthenticatedInferenceCaller,
        plugin_id: str,
        capability_id: str,
        version: str,
        payload: bytes,
        consume: Callable[[NativeInferenceStream], Awaitable[None]],
        seconds: float | None = None,
        *,
        caller_guard: Callable[[], None] | None = None,
    ) -> None:
        """Own one exact stream and drain it through a trusted application adapter.

        The adapter must release borrowed output and consume through its terminal.
        Failure or cancellation drains the same invocation; it never replays work.
        A trusted adapter may provide a synchronous liveness guard, rechecked at
        discovery, dispatch, descendant calls and each protected physical write.
        """
        result = await self._invoke(
            caller,
            plugin_id,
            capability_id,
            version,
            payload,
            seconds,
            consume,
            caller_guard,
        )
        result.release()

    async def _invoke(
        self,
        caller: AuthenticatedInferenceCaller,
        plugin_id: str,
        capability_id: str,
        version: str,
        payload: bytes,
        seconds: float | None = None,
        consume: Callable[[NativeInferenceStream], Awaitable[None]] | None = None,
        caller_guard: Callable[[], None] | None = None,
    ) -> Received:
        """Dispatch one exact caller-authorized implementation without automatic replay.

        Nullable request, credential, signed owner and per-caller ceilings are
        intersected. Control/health/partial-I/O bounds remain separate resources.
        """
        loop = asyncio.get_running_loop()
        started, started_wall = loop.time(), time.time_ns() // 1_000_000
        if seconds is not None and (
            isinstance(seconds, bool) or not 0 < seconds < float("inf")
        ):
            raise ValueError("inference request duration refused")
        async with asyncio.timeout_at(None if seconds is None else started + seconds):
            if caller_guard is not None:
                caller_guard()
            directory = await self.directory.directory(streaming=consume is not None)
            if caller_guard is not None:
                caller_guard()
            matches = [
                value
                for value in directory
                if (value.plugin_id, value.target.capability_id, value.target.version)
                == (plugin_id, capability_id, version)
            ]
            if len(matches) != 1:
                raise ValueError("inference exact destination unavailable")
            listing = matches[0]
            authorization = self.callers.authorize(caller, listing.grant())
            deadlines = [
                started_wall + int(value * 1000)
                for value in (
                    seconds,
                    authorization.access.maximum_seconds,
                    listing.inference_seconds,
                )
                if value is not None
            ]
            if authorization.credential_deadline_unix_ms is not None:
                deadlines.append(authorization.credential_deadline_unix_ms)
            deadline = min(deadlines) if deadlines else None
            expiry = (
                None if deadline is None else started + (deadline - started_wall) / 1000
            )
            targets = [
                InferenceTarget.from_grant(grant).model_dump(mode="json")
                for grant in authorization.access.grants
                if listing.target.behavior == "inference" or grant.behavior == "read"
            ]
            identifier = secrets.token_hex(32)
            context = self.kernel.CompositionV3Contract.from_json(
                "Context",
                json.dumps(
                    {
                        "protocol": 3,
                        "contract_profile": "skulk.composition/3",
                        "profile_digest": self.kernel.composition_v3_profile_digest(),
                        "kind": "context",
                        "scope_id": identifier,
                        "root_scope_id": identifier,
                        "parent_scope_id": None,
                        "caller_id": caller.principal.identifier,
                        "caller_kind": caller.principal.kind,
                        "issuer_node_id": caller.principal.issuer_node_id,
                        "executor": listing.target.model_dump(mode="json"),
                        "request_sha256": hashlib.sha256(payload).hexdigest(),
                        "policy_revision": authorization.revision,
                        "allowed_targets": targets,
                        "allowed_processing_domains": list(
                            dict.fromkeys(
                                grant.processing_domain
                                for grant in authorization.access.grants
                            )
                        ),
                        "ancestry": [],
                        "deadline_unix_ms": deadline,
                    },
                    allow_nan=False,
                ),
            )
            return await self._dispatch(
                listing, context, caller, payload, expiry, consume, caller_guard
            )

    async def _dispatch(
        self,
        listing: InferenceListing,
        context: CompositionRecord,
        caller: AuthenticatedInferenceCaller,
        payload: bytes,
        expiry: float | None,
        consume: Callable[[NativeInferenceStream], Awaitable[None]] | None = None,
        caller_guard: Callable[[], None] | None = None,
    ) -> Received:
        fields = _OBJECT.validate_json(context.canonical())
        if (
            len(self.active) >= 64
            or (listing.descriptor.io_mode == "unary") != (consume is None)
            or not self.kernel.PortableSchema(
                json.dumps(listing.descriptor.input_schema)
            ).matches(payload.decode())
        ):
            raise ValueError("inference capacity, mode or input unavailable")
        endpoint = self.hosts.endpoints.get(listing.plugin_id)
        if endpoint is None:
            raise ValueError("inference owner connection unavailable")
        peer = endpoint.peer
        current = asyncio.current_task()
        assert current is not None
        active = _Active(
            peer,
            listing,
            context,
            caller,
            _TEXT.validate_python(fields["policy_revision"], strict=True),
            expiry,
            _DEADLINE.validate_python(fields["deadline_unix_ms"], strict=True),
            current,
            caller_guard=caller_guard,
        )
        self._permitted(active)
        call_id = secrets.token_hex(32)
        self.active[peer, call_id] = active
        if self.watch is None:
            self.watch = asyncio.create_task(self._watch_authority())
        response: Received | None = None
        transferred = False
        control: asyncio.Task[Received] | None = None
        stream: NativeInferenceStream | None = None
        try:
            async with asyncio.timeout_at(expiry):
                request = peer.request(
                    "infer" if consume is None else "infer_stream",
                    {
                        "call_id": call_id,
                        "node_id": listing.target.node_id,
                        "incarnation": listing.incarnation,
                        "context": fields,
                        "remaining_seconds": self._remaining(active),
                    },
                    payload,
                    check=lambda: self._permitted(active),
                    cleanup_seconds=listing.cleanup_seconds,
                )
                if consume is None:
                    response = await request
                else:
                    control = asyncio.create_task(request)

                    def completed() -> None:
                        self._permitted(active)
                        if active.children or active.uncertain:
                            raise ValueError(
                                "native stream descendant completion unavailable"
                            )

                    stream = NativeInferenceStream(
                        peer,
                        call_id,
                        listing.descriptor,
                        control,
                        lambda: self._permitted(active),
                        completed,
                        listing.stream_seconds,
                        listing.cleanup_seconds,
                    )
                    await stream.ready(listing.startup_seconds)
                    await consume(stream)
                    if not stream.finished:
                        raise ValueError("native stream consumer omitted terminal")
                    response = await asyncio.shield(control)
                assert response is not None
                if (
                    not isinstance(response.frame, ResponseFrame)
                    or response.frame.outcome != "complete"
                    or active.children
                    or active.uncertain
                ):
                    raise ValueError("inference execution outcome unavailable")
                self._permitted(active)
                schema = listing.descriptor.output_schema
                if (
                    consume is None
                    and schema is not None
                    and not self.kernel.PortableSchema(json.dumps(schema)).matches(
                        response.payload.decode()
                    )
                ):
                    raise ValueError("inference output schema refused")
                transferred = True
                return response
        finally:
            if stream is not None:
                stream.close()

            async def cleanup() -> None:
                try:
                    if control is not None:
                        if not control.done() and control.cancelling() == 0:
                            control.cancel()
                        await asyncio.gather(control, return_exceptions=True)
                        if (
                            response is None
                            and not control.cancelled()
                            and control.exception() is None
                        ):
                            control.result().release()
                finally:
                    self.active.pop((peer, call_id), None)
                    for child in list(active.children):
                        child.cancel()
                    await self._drain(active.children)

            try:
                await finish_runtime_work(asyncio.create_task(cleanup()))
            except BaseException:
                if response is not None:
                    response.release()
                raise
            if response is not None and not transferred:
                response.release()

    @staticmethod
    async def _drain(tasks: set[asyncio.Task[object]]) -> None:
        await asyncio.gather(*list(tasks), return_exceptions=True)

    def _policy_record(self, active: _Active) -> CompositionRecord:
        authorization = self.callers.authorize(active.caller, active.listing.grant())
        fields = _OBJECT.validate_json(active.context.canonical())
        return self.kernel.CompositionV3Contract.from_json(
            "Policy",
            json.dumps(
                {
                    "protocol": 3,
                    "contract_profile": "skulk.composition/3",
                    "profile_digest": self.kernel.composition_v3_profile_digest(),
                    "kind": "policy",
                    "caller_id": active.caller.principal.identifier,
                    "caller_kind": active.caller.principal.kind,
                    "issuer_node_id": active.caller.principal.issuer_node_id,
                    "policy_revision": authorization.revision,
                    "allowed_targets": [
                        InferenceTarget.from_grant(grant).model_dump(mode="json")
                        for grant in authorization.access.grants
                    ],
                    "allowed_processing_domains": list(
                        dict.fromkeys(
                            grant.processing_domain
                            for grant in authorization.access.grants
                        )
                    ),
                    "deadline_unix_ms": fields["deadline_unix_ms"],
                }
            ),
        )

    async def _delegate(
        self, peer: Peer, request: RequestFrame, payload: bytes
    ) -> Reply:
        parent: _Active | None = None
        claimed = False
        response: Received | None = None
        transferred = False
        current = asyncio.current_task()
        assert current is not None
        try:
            delegation = InferenceDelegation.model_validate(request.parameters)
            parent = self.active.get((peer, delegation.parent_call_id))
            if parent is None or parent.delegating or parent.uncertain:
                raise ValueError("inference parent custody unavailable")
            self._permitted(parent)
            public = self.kernel.CompositionV3Contract.from_json(
                "DelegateRequest", json.dumps(delegation.request)
            )
            fields = _OBJECT.validate_json(public.canonical())
            sequence = int(_TEXT.validate_python(fields["request_id"], strict=True), 16)
            if not parent.next_delegation <= sequence < 2**64:
                raise ValueError("inference delegation sequence refused")
            parent.next_delegation = sequence + 1
            parent.delegating = claimed = True
            parent.children.add(current)
            destination = InferenceTarget.model_validate(fields["destination"])
            directory = await self.directory.directory()
            self._permitted(parent)
            if self.active.get((peer, delegation.parent_call_id)) is not parent:
                raise ValueError("inference parent was withdrawn")
            matches = [value for value in directory if value.target == destination]
            if len(matches) != 1:
                raise ValueError("inference exact destination unavailable")
            listing = matches[0]
            candidate = self.kernel.CompositionV3Contract.from_json(
                "Context", json.dumps(delegation.context)
            )
            candidate_fields = _OBJECT.validate_json(candidate.canonical())
            admitted = parent.context.admit(
                public,
                self._policy_record(parent),
                parent.listing.target.model_dump_json(),
                payload,
                time.time_ns() // 1_000_000,
                _TEXT.validate_python(candidate_fields["scope_id"], strict=True),
            )
            if admitted.canonical() != candidate.canonical():
                raise ValueError("inference delegated context differs")
            deadline = _DEADLINE.validate_python(
                candidate_fields["deadline_unix_ms"], strict=True
            )
            if (deadline is None) != (delegation.remaining_seconds is None):
                raise ValueError("inference nullable delegation budget differs")
            expiries = [
                value
                for value in (
                    parent.expires_at,
                    None
                    if delegation.remaining_seconds is None
                    else asyncio.get_running_loop().time()
                    + delegation.remaining_seconds,
                    None
                    if deadline is None
                    else asyncio.get_running_loop().time()
                    + (deadline - time.time_ns() // 1_000_000) / 1000,
                )
                if value is not None
            ]
            response = await self._dispatch(
                listing,
                admitted,
                parent.caller,
                payload,
                min(expiries) if expiries else None,
                caller_guard=parent.caller_guard,
            )
            transferred = True
            return Reply("complete", {}, response.payload, response.release)
        except (OSError, ValueError, TimeoutError):
            if parent is not None:
                parent.uncertain = True
            return Reply("unknown", {})
        finally:
            if parent is not None and claimed:
                parent.children.discard(current)
                parent.delegating = False
            if response is not None and not transferred:
                response.release()

    async def close(self) -> None:
        """Withdraw caller work, drain host connections and release every owned lease."""
        self.closed = True
        if self.watch is not None:
            self.watch.cancel()
            await asyncio.gather(self.watch, return_exceptions=True)
        tasks = {active.task for active in self.active.values()}
        for task in tasks:
            task.cancel()
        await finish_runtime_work(asyncio.create_task(self._drain(tasks)))
        await self.hosts.close()
