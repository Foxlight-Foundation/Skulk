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
from typing import Literal, final

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

type NativeCompletionDelivery = Callable[[Callable[[], None]], Awaitable[None]]
"""Trusted final delivery receiving a current original-caller authorization guard."""


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
class InferenceDestinationQuery(BaseModel):
    """Read-only exact-target resolution under the actual active source parent."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    parent_call_id: Digest = Field(description="Actual source call on this role peer.")
    destination: InferenceTarget = Field(
        description="Exact compiled target to resolve."
    )


@final
class InferenceDiscoveryQuery(BaseModel):
    """Enumerate observations for one actual live parent, never a claimed caller."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    query_kind: Literal["directory"] = Field(
        description="Explicit scoped directory form."
    )
    parent_call_id: Digest = Field(description="Actual call on the source role peer.")
    parent_scope_id: Digest = Field(description="Original compiled parent scope.")
    parent_request_sha256: Digest = Field(description="Original admitted input bytes.")
    request_id: Digest = Field(description="Separate monotonic discovery sequence.")


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
    next_discovery: int = 1
    discovering: bool = False
    delegating: bool = False
    uncertain: bool = False
    execution_complete: bool = False
    retiring: bool = False


@final
@dataclass(slots=True)
class _DelegatedMedia:
    stream: NativeInferenceStream
    authorize: Callable[[], None]
    withdraw: Callable[[], None]
    expires_at: float | None
    deadline: int | None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    pull: asyncio.Lock = field(default_factory=asyncio.Lock)
    failed: bool = False


@final
class NativeInferenceCalls:
    """Own exact inference dispatch and a bounded active original-caller tree."""

    def __init__(self, connection: ServiceConnection, issuer_node_id: str) -> None:
        """Bind actual host custody and require the explicitly qualified new binary."""
        self.connection, self.issuer_node_id = connection, issuer_node_id
        self.kernel = public_inference_kernel()
        self.hosts = NativeCalls(
            connection, issuer_node_id, inference_handler=self._handle_delegation
        )
        self.callers = InferenceCallers(
            self.hosts.root, connection.profile_id, issuer_node_id
        )
        self.callers.check()
        self.directory = InferenceHosts(self.hosts)
        self.active: dict[tuple[Peer, str], _Active] = {}
        self.delegated: dict[tuple[Peer, str], _DelegatedMedia] = {}
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
        if active.execution_complete and (active.children or active.uncertain):
            raise ValueError("completed inference custody is uncertain")
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
        *,
        expected_listing: InferenceListing | None = None,
    ) -> Received:
        """Invoke one exact unary destination under the authenticated original caller.

        A trusted adapter may pin an earlier full ``expected_listing``. It must
        match fresh discovery before current caller permission is checked.
        The returned response retains its byte lease until delivery or discard.
        """
        return await self._invoke(
            caller,
            plugin_id,
            capability_id,
            version,
            payload,
            seconds,
            expected_listing=expected_listing,
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
        expected_listing: InferenceListing | None = None,
        complete_delivery: NativeCompletionDelivery | None = None,
    ) -> None:
        """Own one exact stream and drain it through a trusted application adapter.

        The adapter must release borrowed output and consume through its terminal.
        Failure or cancellation drains the same invocation; it never replays work.
        A trusted adapter may provide a synchronous liveness guard, rechecked at
        discovery, dispatch, descendant calls and each protected physical write.
        A selected listing additionally pins the full signed target, runtime and
        child incarnation; it never grants permission or replaces fresh discovery.
        Optional trusted final delivery runs after native retirement drains but
        before caller custody is removed. Its guard and the active watcher
        protect the final physical write; completed execution cannot delegate.
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
            expected_listing=expected_listing,
            complete_delivery=complete_delivery,
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
        *,
        expected_listing: InferenceListing | None = None,
        complete_delivery: NativeCompletionDelivery | None = None,
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
            if expected_listing is not None and listing != expected_listing:
                # A catalog is meaningful only for the generation that produced
                # it. Operator authority must not silently retarget stale work.
                raise ValueError("inference selected destination changed")
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
                listing,
                context,
                caller,
                payload,
                expiry,
                consume,
                caller_guard,
                complete_delivery,
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
        complete_delivery: NativeCompletionDelivery | None = None,
        *,
        delegated_call_id: str | None = None,
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
        call_id = delegated_call_id or secrets.token_hex(32)
        if (peer, call_id) in self.active:
            raise ValueError("inference active call identity already owned")
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
                        active.execution_complete = True

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
                active.execution_complete = True
                transferred = True
                return response
        finally:
            # Keep final response custody, but admit no new descendants or
            # observations once execution has entered its cleanup phase.
            active.retiring = True
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
                    for child in list(active.children):
                        child.cancel()
                    await self._drain(active.children)

            try:
                await finish_runtime_work(asyncio.create_task(cleanup()))
                if transferred and complete_delivery is not None:
                    # Keep this in the captured invocation task, outside the
                    # shielded cleanup task: withdrawal must interrupt a blocked
                    # final delivery even though provider execution has ended.
                    self._permitted(active)
                    await complete_delivery(lambda: self._permitted(active))
            except BaseException:
                if response is not None:
                    response.release()
                raise
            finally:
                self.active.pop((peer, call_id), None)
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

    async def _handle_delegation(
        self, peer: Peer, request: RequestFrame, payload: bytes
    ) -> Reply:
        if request.operation == "describe_inference":
            if request.parameters.get("query_kind") == "directory":
                return await self._discover_destinations(peer, request, payload)
            return await self._describe_destination(peer, request, payload)
        if request.operation in ("stream_ready", "stream_input", "stream_next"):
            return await self._delegated_media(peer, request, payload)
        return await self._delegate(peer, request, payload)

    async def _discover_destinations(
        self, peer: Peer, request: RequestFrame, payload: bytes
    ) -> Reply:
        parent: _Active | None = None
        claimed = False
        transferred = False
        current = asyncio.current_task()
        assert current is not None
        try:
            if payload:
                raise ValueError("native discovery query has a body")
            query = InferenceDiscoveryQuery.model_validate(request.parameters)
            parent = self.active.get((peer, query.parent_call_id))
            if (
                parent is None
                or parent.uncertain
                or parent.execution_complete
                or parent.retiring
                or parent.task.cancelling()
                or parent.task.done()
                or parent.delegating
                or parent.discovering
            ):
                raise ValueError("native discovery parent unavailable")
            self._permitted(parent)
            context = _OBJECT.validate_json(parent.context.canonical())
            allowed = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
                context["allowed_targets"], strict=True
            )
            domains = TypeAdapter(list[str]).validate_python(
                context["allowed_processing_domains"], strict=True
            )
            ancestry = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
                context["ancestry"], strict=True
            )
            executor = _OBJECT.validate_python(context["executor"], strict=True)
            if (
                query.parent_scope_id != context["scope_id"]
                or query.parent_request_sha256 != context["request_sha256"]
            ):
                raise ValueError("native discovery parent binding differs")
            sequence = int(query.request_id, 16)
            if not parent.next_discovery <= sequence < 2**64:
                raise ValueError("native discovery sequence refused")
            parent.next_discovery = sequence + 1
            parent.discovering = claimed = True
            parent.children.add(current)
            directory = await self.directory.directory(streaming=True)
            selected = tuple(
                listing
                for listing in directory
                if len(ancestry) < 8
                and listing.target.model_dump(mode="json") in allowed
                and listing.target.processing_domain in domains
                and all(
                    listing.target.node_id != ancestor["node_id"]
                    or listing.target.capability_id != ancestor["capability_id"]
                    for ancestor in (*ancestry, executor)
                )
            )

            def authorize() -> None:
                assert parent is not None
                self._permitted(parent)
                if (
                    self.active.get((peer, query.parent_call_id)) is not parent
                    or parent.execution_complete
                    or parent.retiring
                    or parent.uncertain
                    or parent.task.cancelling()
                ):
                    raise ValueError("native discovery parent changed")
                for listing in selected:
                    if (
                        self.callers.authorize(parent.caller, listing.grant()).revision
                        != parent.revision
                    ):
                        raise ValueError("native discovery caller policy changed")

            authorize()
            if len(selected) > 4096:
                raise ValueError("native discovery offer bound exceeded")
            parts: list[bytes] = []
            size = 2
            for listing in selected:
                part = listing.model_dump_json().encode()
                size += len(part) + bool(parts)
                if size > 8 * 1024 * 1024:
                    raise ValueError("native discovery byte bound exceeded")
                parts.append(part)

            def release() -> None:
                assert parent is not None
                parent.discovering = False
                parent.children.discard(current)

            # Query custody includes the guarded physical response write. A
            # completed source parent cannot leave a queued observation alive.
            transferred = True
            return Reply(
                "complete", {}, b"[" + b",".join(parts) + b"]", release, authorize
            )
        except (OSError, ValueError, TimeoutError):
            return Reply("pre_dispatch", {})
        finally:
            if parent is not None and claimed and not transferred:
                parent.discovering = False
                parent.children.discard(current)

    async def _describe_destination(
        self, peer: Peer, request: RequestFrame, payload: bytes
    ) -> Reply:
        try:
            if payload:
                raise ValueError("native descriptor query has a body")
            query = InferenceDestinationQuery.model_validate(request.parameters)
            parent = self.active.get((peer, query.parent_call_id))
            if (
                parent is None
                or parent.uncertain
                or parent.execution_complete
                or parent.retiring
                or parent.task.cancelling()
                or parent.task.done()
                or parent.delegating
                or parent.discovering
            ):
                raise ValueError("native descriptor parent unavailable")
            self._permitted(parent)
            allowed = _OBJECT.validate_json(parent.context.canonical())[
                "allowed_targets"
            ]
            if (
                not isinstance(allowed, list)
                or query.destination.model_dump(mode="json") not in allowed
            ):
                raise ValueError("native descriptor target is outside parent authority")
            directory = await self.directory.directory(streaming=True)
            self._permitted(parent)
            if self.active.get((peer, query.parent_call_id)) is not parent:
                raise ValueError("native descriptor parent changed")
            matches = [
                entry for entry in directory if entry.target == query.destination
            ]
            if len(matches) != 1:
                raise ValueError("native descriptor target unavailable")
            listing = matches[0]
            self.callers.authorize(parent.caller, listing.grant())
            return Reply(
                "complete",
                {"descriptor": listing.descriptor.model_dump(mode="json")},
                check=lambda: self._permitted(parent),
            )
        except (OSError, ValueError, TimeoutError):
            # Resolution is read-only: refusal does not dispatch or consume a
            # delegation sequence, and never returns ambient catalog metadata.
            return Reply("pre_dispatch", {})

    async def _delegated_media(
        self, peer: Peer, request: RequestFrame, payload: bytes
    ) -> Reply:
        if set(request.parameters) != {"call_id"}:
            return Reply("pre_dispatch", {})
        identifier = request.parameters["call_id"]
        media = (
            self.delegated.get((peer, identifier))
            if isinstance(identifier, str)
            else None
        )
        if media is None:
            return Reply("pre_dispatch", {})
        try:
            media.authorize()
            if request.operation == "stream_ready":
                if payload:
                    return Reply("pre_dispatch", {})
                remaining = (
                    None
                    if media.expires_at is None
                    else min(
                        media.expires_at - asyncio.get_running_loop().time(),
                        (media.deadline - time.time_ns() // 1_000_000) / 1000
                        if media.deadline is not None
                        else float("inf"),
                    )
                )
                if remaining is not None and remaining <= 0:
                    raise ValueError("delegated stream expired")
                return Reply(
                    "complete",
                    {
                        "ready": True,
                        "remaining_seconds": remaining,
                        "deadline_unix_ms": media.deadline,
                    },
                    check=media.authorize,
                    cancelled=media.withdraw,
                )
            if request.operation == "stream_input":
                await media.stream.send_input(payload)
                return Reply(
                    "complete", {}, check=media.authorize, cancelled=media.withdraw
                )
            if payload or media.pull.locked():
                return Reply("pre_dispatch", {})
            async with media.pull:
                value = await media.stream.next_output()
                if value is None:
                    raise ValueError("delegated output already completed")
                transferred = False
                try:
                    terminal = value.kind != "chunk"

                    def released() -> None:
                        value.release()
                        if terminal:
                            media.done.set()

                    reply = Reply(
                        "complete",
                        {"kind": value.kind},
                        value.packet,
                        released,
                        media.authorize,
                        media.withdraw,
                    )
                    transferred = True
                    return reply
                finally:
                    if not transferred:
                        value.release()
        except (OSError, ValueError, TimeoutError, asyncio.CancelledError):
            media.withdraw()
            raise

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
            if (
                parent is None
                or parent.delegating
                or parent.discovering
                or parent.uncertain
                or parent.execution_complete
                or parent.retiring
                or parent.task.cancelling()
                or parent.task.done()
            ):
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
            streaming = request.operation == "delegate_inference_stream"
            directory = await self.directory.directory(streaming=streaming)
            self._permitted(parent)
            if self.active.get((peer, delegation.parent_call_id)) is not parent:
                raise ValueError("inference parent was withdrawn")
            matches = [value for value in directory if value.target == destination]
            if len(matches) != 1:
                raise ValueError("inference exact destination unavailable")
            listing = matches[0]
            if (listing.descriptor.io_mode == "unary") == streaming:
                raise ValueError("inference delegated mode differs")
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
            expiry = min(expiries) if expiries else None
            consume: Callable[[NativeInferenceStream], Awaitable[None]] | None = None
            delegated_id: str | None = None
            if streaming:
                authorization = self.callers.authorize(parent.caller, listing.grant())
                now = asyncio.get_running_loop().time()
                for ceiling in (
                    authorization.access.maximum_seconds,
                    listing.inference_seconds,
                ):
                    if ceiling is not None:
                        expiries.append(now + ceiling)
                expiry = min(expiries) if expiries else None
                actual_deadline = None
                if expiry is not None:
                    actual_deadline = time.time_ns() // 1_000_000 + int(
                        (expiry - now) * 1000
                    )
                    if actual_deadline <= time.time_ns() // 1_000_000:
                        raise TimeoutError
                    policy_value = _OBJECT.validate_json(
                        self._policy_record(parent).canonical()
                    )
                    policy_value["deadline_unix_ms"] = actual_deadline
                    admitted = parent.context.admit(
                        public,
                        self.kernel.CompositionV3Contract.from_json(
                            "Policy", json.dumps(policy_value)
                        ),
                        parent.listing.target.model_dump_json(),
                        payload,
                        time.time_ns() // 1_000_000,
                        _TEXT.validate_python(
                            candidate_fields["scope_id"], strict=True
                        ),
                    )
                    actual_deadline = _DEADLINE.validate_python(
                        _OBJECT.validate_json(admitted.canonical())["deadline_unix_ms"],
                        strict=True,
                    )
                    assert actual_deadline is not None
                    expiry = min(
                        expiry,
                        now + (actual_deadline - time.time_ns() // 1_000_000) / 1000,
                    )
                delegated_id = _TEXT.validate_python(
                    candidate_fields["scope_id"], strict=True
                )
                if delegated_id == delegation.parent_call_id:
                    raise ValueError("delegated media identity is the parent")
                key = peer, delegated_id

                async def consume_stream(stream: NativeInferenceStream) -> None:
                    assert parent is not None
                    if key in self.delegated:
                        raise ValueError("delegated media identity is already owned")

                    def authorize() -> None:
                        assert parent is not None
                        if (
                            media.failed
                            or current.cancelling()
                            or self.active.get((peer, delegation.parent_call_id))
                            is not parent
                            or self.delegated.get(key) is not media
                        ):
                            raise ValueError("delegated stream was withdrawn")
                        self._permitted(parent)
                        stream.authorize()

                    def withdraw() -> None:
                        assert parent is not None
                        media.failed = True
                        parent.uncertain = True
                        if (
                            current is not asyncio.current_task()
                            and not current.done()
                            and not current.cancelling()
                        ):
                            current.cancel()
                        media.done.set()

                    media = _DelegatedMedia(
                        stream, authorize, withdraw, expiry, actual_deadline
                    )
                    self.delegated[key] = media
                    try:
                        await media.done.wait()
                        authorize()
                        if not stream.finished:
                            raise ValueError(
                                "delegated stream has no verified terminal"
                            )
                    finally:
                        if self.delegated.get(key) is media:
                            del self.delegated[key]

                consume = consume_stream
            response = await self._dispatch(
                listing,
                admitted,
                parent.caller,
                payload,
                expiry,
                consume,
                caller_guard=parent.caller_guard,
                delegated_call_id=delegated_id,
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
