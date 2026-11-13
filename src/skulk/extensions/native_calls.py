"""Core-owned admission and delegation for installed native unary reads.

This adapter uses the actual local manager profile and signed installed contracts.
It supplies generic host primitives, with no provider, account or routing policy.
"""

import asyncio
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
)

from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.native_call_contracts import CompositionRecord, public_call_kernel
from skulk.extensions.native_contracts import NativeDescriptor, NativeManifest
from skulk.extensions.native_host_binding import load_core_host_key
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    Digest,
    HostBinding,
    Identifier,
    Profile,
    ReadGrant,
    ReadPolicy,
    Received,
    RequestFrame,
    ResponseFrame,
    admit_owner,
)
from skulk.extensions.native_host_rpc import Peer, Reply
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private
from skulk.extensions.runtime_selection import RuntimeSelector

_OBJECT = TypeAdapter(dict[str, JsonValue])
_TEXT = TypeAdapter(str)


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class Target(_Contract):
    """Exact native destination, never a bearer grant or caller identity."""

    node_id: Identifier = Field(description="Actual live native child identity.")
    capability_id: Identifier = Field(description="Exact signed capability id.")
    version: str = Field(description="Exact signed capability version.")
    descriptor_revision: Digest = Field(
        description="Full compiled descriptor identity."
    )
    processing_domain: Literal["local"] = Field(
        default="local", description="Explicitly enrolled local processing."
    )
    behavior: Literal["read"] = Field(
        default="read", description="Read-only native adapter."
    )


@final
class NativeListing(_Contract):
    """Signed installed declaration plus actual live child custody."""

    plugin_id: str = Field(description="Registered installation.")
    runtime_digest: Digest = Field(description="Exact signed installed runtime.")
    incarnation: Profile = Field(description="Actual protected child channel lifetime.")
    target: Target = Field(description="Complete destination identity.")
    descriptor: NativeDescriptor = Field(
        description="Compiled signed author declaration."
    )

    def grant(self) -> ReadGrant:
        """Project exact enrollment identity; discovery itself grants nothing."""
        return ReadGrant(
            plugin_id=self.plugin_id,
            runtime_digest=self.runtime_digest,
            capability_id=self.target.capability_id,
            version=self.target.version,
            descriptor_revision=self.target.descriptor_revision,
            processing_domain="local",
        )


@final
class PolicyUpdate(_Contract):
    """Revision-fenced owner enrollment, disabled until explicitly configured."""

    expected_revision: Digest = Field(
        description="Current policy revision; stale updates fail."
    )
    enabled: bool = Field(description="Whether enrolled reads may execute.")
    grants: tuple[ReadGrant, ...] = Field(
        max_length=64, description="Exact trusted local reads."
    )

    @field_validator("grants", mode="before")
    @classmethod
    def json_grants(cls, value: object) -> object:
        """Convert the JSON array container; individual grants stay strictly typed."""
        if isinstance(value, list):
            return tuple(
                TypeAdapter(list[ReadGrant]).validate_python(value, strict=True)
            )
        return value


@final
class Delegation(_Contract):
    """Private source-owner proxy metadata, checked against Core's active parent."""

    parent_call_id: Digest = Field(description="Core-assigned active parent call.")
    request: dict[str, JsonValue] = Field(
        description="Public compiled delegation request."
    )
    context: dict[str, JsonValue] = Field(
        description="Candidate context to independently recompute."
    )
    remaining_seconds: float = Field(
        gt=0, le=30, description="Attenuated unary time budget."
    )


@final
@dataclass(slots=True)
class _Endpoint:
    binding: HostBinding
    peer: Peer


@final
@dataclass(slots=True)
class _Active:
    peer: Peer
    listing: NativeListing
    context: CompositionRecord
    expires_at: float
    task: asyncio.Task[object]
    children: set[asyncio.Task[object]] = field(default_factory=set)
    next_delegation: int = 1
    delegating: bool = False
    uncertain: bool = False


@final
class NativeCalls:
    """Own authenticated host connections and a bounded active call tree."""

    def __init__(self, connection: ServiceConnection, issuer_node_id: str) -> None:
        """Bind only host-configured profile state, never a request-supplied path."""
        self.connection, self.issuer_node_id = connection, issuer_node_id
        self.root = Path(connection.manager_root)
        self.kernel = public_call_kernel()
        self.budget = NativePayloadBudget()
        self.endpoints: dict[str, _Endpoint] = {}
        self.active: dict[tuple[Peer, str], _Active] = {}
        self._connect_lock = asyncio.Lock()
        self.closed = False

    def _profile(self) -> None:
        if self.closed:
            raise ValueError("native Core host unavailable")
        actual = HostSettings.model_validate_json(read_private(self.root / "host.json"))
        if (
            actual.profile_id != self.connection.profile_id
            or actual.transport_node_id != self.issuer_node_id
        ):
            raise ValueError("native manager attachment differs")

    def policy(self) -> ReadPolicy:
        """Read atomic owner enrollment without inferring provider allowances."""
        self._profile()
        try:
            return ReadPolicy.model_validate_json(
                read_private(self.root / "native-read-policy.json")
            )
        except FileNotFoundError:
            return ReadPolicy(revision="0" * 64, enabled=False, grants=())

    async def update_policy(self, update: PolicyUpdate) -> ReadPolicy:
        """Enroll only exact signed live reads with a nonblocking revision fence."""
        if update.enabled:
            available = tuple(value.grant() for value in await self.directory())
            if any(grant not in available for grant in update.grants):
                raise ValueError("native enrollment requires an exact live unary read")
        lock = RuntimeLock(self.root, "native-policy.lock")
        try:
            prior = self.policy()
            if prior.revision != update.expected_revision:
                raise ValueError("native policy revision changed")
            policy = ReadPolicy(
                revision=secrets.token_hex(32),
                enabled=update.enabled,
                grants=update.grants,
            )
            write_private(
                self.root / "native-read-policy.json", policy.model_dump_json().encode()
            )
            # Already dispatched work is cancelled too. The synchronous final
            # owner read separately closes the queued-before-physical-write gap.
            for active in list(self.active.values()):
                active.task.cancel()
            return policy
        finally:
            lock.close()

    def _binding(self, plugin_id: str) -> HostBinding:
        self._profile()
        binding = HostBinding.model_validate_json(
            read_private(self.root / "native-host-bindings" / (plugin_id + ".json"))
        )
        selection = RuntimeSelector(
            self.root / "installations" / binding.plugin_id
        ).current()
        if (
            binding.plugin_id != plugin_id
            or binding.profile_id != self.connection.profile_id
            or binding.transport_node_id != self.issuer_node_id
            or selection is None
            or not selection.enabled
            or selection.runtime_digest != binding.runtime_digest
        ):
            raise ValueError("native runtime attachment changed")
        return binding

    def _current(self, binding: HostBinding) -> None:
        if self._binding(binding.plugin_id) != binding:
            raise ValueError("native host lifetime changed")

    async def _endpoint(self, binding: HostBinding) -> _Endpoint:
        async with self._connect_lock:
            existing = self.endpoints.get(binding.plugin_id)
            if existing is not None:
                if existing.binding == binding and not existing.peer.stream.closed:
                    existing.peer.check()
                    return existing
                await existing.peer.close()
                del self.endpoints[binding.plugin_id]
            self._current(binding)
            key = load_core_host_key(self.root, self.connection.profile_id).key()
            installation = self.root / "installations" / binding.plugin_id
            name = hashlib.sha256(str(installation.absolute()).encode()).hexdigest()[
                :24
            ]
            socket = (
                Path("/tmp")
                / f"skulk-control-{os.getuid()}-{name}"
                / "native-host.sock"
            )
            reader, writer = await asyncio.open_unix_connection(socket, limit=262144)
            try:
                keys = await admit_owner(
                    reader, writer, binding, key, lambda: self._current(binding)
                )
                peer = Peer(
                    AuthenticatedStream(reader, writer, keys, self.budget),
                    self._delegate,
                    frozenset({"delegate"}),
                    lambda: self._current(binding),
                )
                endpoint = _Endpoint(binding, peer)
                self.endpoints[binding.plugin_id] = endpoint
                return endpoint
            except BaseException:
                writer.close()
                raise

    async def directory(self) -> tuple[NativeListing, ...]:
        """Match actual owner discovery against exact signed installed declarations."""
        self._profile()
        paths = sorted((self.root / "native-host-bindings").glob("*.json"))
        if len(paths) > 64:
            raise ValueError("native installation capacity exceeded")
        listings: list[NativeListing] = []
        for path in paths:
            try:
                binding = self._binding(path.stem)
                selector = RuntimeSelector(
                    self.root / "installations" / binding.plugin_id
                )
                metadata = read_private(
                    selector.root
                    / "generations"
                    / binding.runtime_digest
                    / "staged.json",
                    262144,
                )
                runtime = await selector.installer.inspect_metadata(metadata)
                manifest = runtime.claims.release.manifest
                if (
                    runtime.claims.protocol != 4
                    or not isinstance(manifest, NativeManifest)
                    or runtime.digest != binding.runtime_digest
                ):
                    continue
                contracts = _OBJECT.validate_python(
                    manifest.value()["contracts"], strict=True
                )
                signed = TypeAdapter(tuple[NativeDescriptor, ...]).validate_json(
                    json.dumps(contracts["descriptors"])
                )
                endpoint = await self._endpoint(binding)
                async with asyncio.timeout(3):
                    response = await endpoint.peer.request("describe", {})
                try:
                    if (
                        not isinstance(response.frame, ResponseFrame)
                        or response.frame.outcome != "complete"
                        or response.payload
                    ):
                        raise ValueError("native directory reply refused")
                    children = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
                        response.frame.parameters["children"], strict=True
                    )
                    if len(children) > 64:
                        raise ValueError("native directory exceeds bound")
                    for child in children:
                        if child["manifest_digest"] != manifest.digest:
                            raise ValueError("native child manifest differs")
                        descriptors = TypeAdapter(
                            tuple[NativeDescriptor, ...]
                        ).validate_json(json.dumps(child["descriptors"]))
                        if descriptors != signed:
                            raise ValueError(
                                "native owner declaration differs from signed contract"
                            )
                        for descriptor in descriptors:
                            if (
                                descriptor.behavior != "read"
                                or descriptor.io_mode != "unary"
                            ):
                                continue
                            listings.append(
                                NativeListing(
                                    plugin_id=binding.plugin_id,
                                    runtime_digest=binding.runtime_digest,
                                    incarnation=TypeAdapter[str](
                                        Profile
                                    ).validate_python(
                                        child["incarnation"], strict=True
                                    ),
                                    target=Target(
                                        node_id=_TEXT.validate_python(
                                            child["node_id"], strict=True
                                        ),
                                        capability_id=descriptor.id,
                                        version=descriptor.version,
                                        descriptor_revision=descriptor.revision,
                                    ),
                                    descriptor=descriptor,
                                )
                            )
                finally:
                    response.release()
            except (OSError, ValueError, TimeoutError):
                # Offline/old/changed owners advertise no native executable
                # capacity; their installed metadata remains on the normal API.
                continue
        if len(listings) > 64 or len(
            {listing.target.model_dump_json() for listing in listings}
        ) != len(listings):
            raise ValueError("native directory is ambiguous or exceeds bound")
        return tuple(listings)

    def _permitted(self, listing: NativeListing, revision: str) -> None:
        policy = self.policy()
        if (
            not policy.enabled
            or policy.revision != revision
            or listing.grant() not in policy.grants
        ):
            raise ValueError("native read enrollment changed")
        endpoint = self.endpoints.get(listing.plugin_id)
        if endpoint is None:
            raise ValueError("native owner unavailable")
        endpoint.peer.check()

    def _policy_record(
        self, context: dict[str, JsonValue], targets: list[JsonValue], revision: str
    ) -> CompositionRecord:
        return self.kernel.CompositionContract.from_json(
            "Policy",
            json.dumps(
                {
                    "protocol": 1,
                    "contract_profile": "skulk.composition/1",
                    "profile_digest": self.kernel.composition_profile_digest(),
                    "kind": "policy",
                    "caller_id": context["caller_id"],
                    "caller_kind": context["caller_kind"],
                    "issuer_node_id": context["issuer_node_id"],
                    "policy_revision": revision,
                    "allowed_targets": targets,
                    "allowed_processing_domains": ["local"],
                    "deadline_unix_ms": context["deadline_unix_ms"],
                }
            ),
        )

    async def invoke(
        self,
        owner: AuthenticatedOwner,
        plugin_id: str,
        capability_id: str,
        version: str,
        payload: bytes,
        seconds: float,
    ) -> Received:
        """Apply the unary budget to admission, discovery and actual execution."""
        expires_at = asyncio.get_running_loop().time() + seconds
        async with asyncio.timeout_at(expires_at):
            return await self._invoke(
                owner, plugin_id, capability_id, version, payload, seconds, expires_at
            )

    async def _invoke(
        self,
        owner: AuthenticatedOwner,
        plugin_id: str,
        capability_id: str,
        version: str,
        payload: bytes,
        seconds: float,
        expires_at: float,
    ) -> Received:
        """Mint a root only from the actual authenticated owner and current policy."""
        if owner.issuer_node_id != self.issuer_node_id or not 0 < seconds <= 30:
            raise ValueError("native original caller unavailable")
        directory = await self.directory()
        matches = [
            value
            for value in directory
            if (value.plugin_id, value.target.capability_id, value.target.version)
            == (plugin_id, capability_id, version)
        ]
        if len(matches) != 1:
            raise ValueError("native destination unavailable")
        listing = matches[0]
        policy = self.policy()
        self._permitted(listing, policy.revision)
        identifier = secrets.token_hex(32)
        remaining = expires_at - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ValueError("native admission deadline elapsed")
        deadline = time.time_ns() // 1_000_000 + int(remaining * 1000)
        targets: list[JsonValue] = [
            value.target.model_dump(mode="json")
            for value in directory
            if value.grant() in policy.grants
        ]
        context = self.kernel.CompositionContract.from_json(
            "Context",
            json.dumps(
                {
                    "protocol": 1,
                    "contract_profile": "skulk.composition/1",
                    "profile_digest": self.kernel.composition_profile_digest(),
                    "kind": "context",
                    "scope_id": identifier,
                    "root_scope_id": identifier,
                    "parent_scope_id": None,
                    "caller_id": owner.identifier,
                    "caller_kind": owner.kind,
                    "issuer_node_id": str(owner.issuer_node_id),
                    "executor": listing.target.model_dump(mode="json"),
                    "request_sha256": hashlib.sha256(payload).hexdigest(),
                    "policy_revision": policy.revision,
                    "allowed_targets": targets,
                    "allowed_processing_domains": ["local"],
                    "ancestry": [],
                    "deadline_unix_ms": deadline,
                }
            ),
        )
        return await self._dispatch(listing, context, payload, expires_at)

    async def _dispatch(
        self,
        listing: NativeListing,
        context: CompositionRecord,
        payload: bytes,
        expires_at: float,
    ) -> Received:
        fields = _OBJECT.validate_json(context.canonical())
        revision = _TEXT.validate_python(fields["policy_revision"], strict=True)
        self._permitted(listing, revision)
        if len(self.active) >= 64 or not self.kernel.PortableSchema(
            json.dumps(listing.descriptor.input_schema)
        ).matches(payload.decode()):
            raise ValueError("native call capacity or input unavailable")
        peer = self.endpoints[listing.plugin_id].peer
        call_id = secrets.token_hex(32)
        task = asyncio.current_task()
        assert task is not None
        active = _Active(peer, listing, context, expires_at, task)
        self.active[peer, call_id] = active
        response: Received | None = None
        transferred = False
        try:
            async with asyncio.timeout_at(expires_at):
                remaining = expires_at - asyncio.get_running_loop().time()
                response = await peer.request(
                    "invoke",
                    {
                        "call_id": call_id,
                        "node_id": listing.target.node_id,
                        "incarnation": listing.incarnation,
                        "context": fields,
                        "remaining_seconds": remaining,
                    },
                    payload,
                    check=lambda: self._permitted(listing, revision),
                )
                if (
                    not isinstance(response.frame, ResponseFrame)
                    or response.frame.outcome != "complete"
                    or active.uncertain
                    or active.children
                ):
                    raise ValueError(
                        "native execution outcome unavailable; do not replay"
                    )
                self._permitted(listing, revision)
                schema = listing.descriptor.output_schema
                if schema is not None and not self.kernel.PortableSchema(
                    json.dumps(schema)
                ).matches(response.payload.decode()):
                    raise ValueError("native result schema refused")
                transferred = True
                return response
        finally:
            self.active.pop((peer, call_id), None)
            for task in list(active.children):
                task.cancel()
            await asyncio.gather(*list(active.children), return_exceptions=True)
            if response is not None and not transferred:
                response.release()

    async def _delegate(
        self, peer: Peer, request: RequestFrame, payload: bytes
    ) -> Reply:
        parent: _Active | None = None
        current = asyncio.current_task()
        assert current is not None
        response: Received | None = None
        transferred = False
        claimed = False
        try:
            delegation = Delegation.model_validate(request.parameters)
            parent = self.active.get((peer, delegation.parent_call_id))
            if parent is None or parent.delegating or parent.next_delegation >= 2**64:
                return Reply("pre_dispatch", {})
            fields = _OBJECT.validate_json(parent.context.canonical())
            public = self.kernel.CompositionContract.from_json(
                "DelegateRequest", json.dumps(delegation.request)
            )
            public_fields = _OBJECT.validate_json(public.canonical())
            number = int(
                _TEXT.validate_python(public_fields["request_id"], strict=True), 16
            )
            # The author-facing owner consumes admitted local refusals before
            # proxying. Core permits their gaps, but independently rejects each
            # reused/backward number on this actual parent peer. A sequence is
            # not authority: policy, ancestry and exact bytes are recomputed.
            if not parent.next_delegation <= number < 2**64:
                return Reply("pre_dispatch", {})
            parent.next_delegation = number + 1
            parent.delegating = True
            claimed = True
            parent.children.add(current)
            directory = await self.directory()
            self._permitted(
                parent.listing,
                _TEXT.validate_python(fields["policy_revision"], strict=True),
            )
            if self.active.get((peer, delegation.parent_call_id)) is not parent:
                raise ValueError("native parent withdrawn")
            destination = Target.model_validate(public_fields["destination"])
            matches = [value for value in directory if value.target == destination]
            if len(matches) != 1:
                raise ValueError("native destination changed")
            listing = matches[0]
            policy = self.policy()
            targets: list[JsonValue] = [
                value.target.model_dump(mode="json")
                for value in directory
                if value.grant() in policy.grants
            ]
            candidate = self.kernel.CompositionContract.from_json(
                "Context", json.dumps(delegation.context)
            )
            candidate_fields = _OBJECT.validate_json(candidate.canonical())
            admitted = parent.context.admit(
                public,
                self._policy_record(fields, targets, policy.revision),
                parent.listing.target.model_dump_json(),
                payload,
                time.time_ns() // 1_000_000,
                _TEXT.validate_python(candidate_fields["scope_id"], strict=True),
            )
            if admitted.canonical() != candidate.canonical():
                raise ValueError("native delegated context differs")
            response = await self._dispatch(
                listing,
                admitted,
                payload,
                min(
                    parent.expires_at,
                    asyncio.get_running_loop().time() + delegation.remaining_seconds,
                ),
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
        """Fence connections and drain active handlers without replaying any call."""
        self.closed = True
        await asyncio.gather(
            *(value.peer.close() for value in self.endpoints.values()),
            return_exceptions=True,
        )
        self.endpoints.clear()
