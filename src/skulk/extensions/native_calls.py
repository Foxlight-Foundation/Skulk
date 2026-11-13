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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
from skulk.extensions.managed_limits import MAX_RUNTIME_METADATA_BYTES
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
from skulk.extensions.runtime_artifacts import VerifiedRuntime
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.extensions.runtime_selection import RuntimeSelector

_OBJECT = TypeAdapter(dict[str, JsonValue])
_TEXT = TypeAdapter(str)
_DIRECTORY_SECONDS = 3.0
_MAX_INSTALLATIONS = 64


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
class _ChildDescription(_Contract):
    """Untrusted owner metadata, validated before signed contract comparison."""

    node_id: Identifier = Field(description="Owner-observed child identity.")
    incarnation: Profile = Field(description="Actual child channel lifetime.")
    manifest_digest: Digest = Field(description="Claimed signed manifest identity.")
    descriptors: tuple[NativeDescriptor, ...] = Field(
        max_length=64, description="Declarations to compare with the signed manifest."
    )


@final
class _OwnerDescription(_Contract):
    """Closed discovery wrapper; parsing this record never admits execution."""

    children: tuple[_ChildDescription, ...] = Field(
        max_length=64, description="Bounded owner-observed native children."
    )
    diagnostics: dict[str, JsonValue] | None = Field(
        default=None, description="Inert owner diagnostics, unused for admission."
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
@dataclass(frozen=True, slots=True, repr=False)
class _Inspection:
    """Own in-flight verification until its thread and installer fence finish."""

    binding: HostBinding
    metadata_sha256: str
    task: asyncio.Task[VerifiedRuntime]


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
        self._connect_locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self._probe_locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self._directory_slots = asyncio.Semaphore(_MAX_INSTALLATIONS)
        self._inspections: dict[str, _Inspection] = {}
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

    @asynccontextmanager
    async def _owner_lock(
        self, locks: dict[str, tuple[asyncio.Lock, int]], plugin_id: str
    ) -> AsyncIterator[None]:
        """Serialize one owner's work without reserving another owner's capacity."""
        lock, users = locks.get(plugin_id, (asyncio.Lock(), 0))
        if not users and len(locks) >= _MAX_INSTALLATIONS:
            raise ValueError("native owner lock capacity unavailable")
        locks[plugin_id] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            _, users = locks[plugin_id]
            if users == 1:
                del locks[plugin_id]
            else:
                locks[plugin_id] = (lock, users - 1)

    async def _endpoint(self, binding: HostBinding) -> _Endpoint:
        async with self._owner_lock(self._connect_locks, binding.plugin_id):
            existing = self.endpoints.get(binding.plugin_id)
            if existing is not None:
                if existing.binding == binding and not existing.peer.stream.closed:
                    existing.peer.check()
                    return existing
                await existing.peer.close()
                self.endpoints.pop(binding.plugin_id, None)
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
                self._current(binding)
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
        """Return signed live reads found within one bounded concurrent discovery.

        Unavailable, invalid and unfinished owners contribute no capacity. The
        three-second discovery budget is shared across owners; interrupted RPC
        cleanup has its own bounded drain before this method returns. No stale
        directory result is cached or substituted for dispatch-time validation.
        """
        self._profile()
        paths = sorted((self.root / "native-host-bindings").glob("*.json"))
        if len(paths) > _MAX_INSTALLATIONS:
            raise ValueError("native installation capacity exceeded")
        results: list[tuple[NativeListing, ...]] = [()] * len(paths)

        async def inspect(index: int, path: Path) -> None:
            try:
                # Duplicate scans wait before shared admission. Otherwise many
                # waiters for stalled owners can occupy every host-wide slot.
                async with (
                    self._owner_lock(self._probe_locks, path.stem),
                    self._directory_slots,
                ):
                    results[index] = await self._discover(path)
            except (OSError, ValueError, TimeoutError):
                # A failing owner contributes no executable capacity. Other
                # owners retain their independently verified observations.
                pass

        try:
            async with asyncio.timeout(_DIRECTORY_SECONDS):
                async with asyncio.TaskGroup() as group:
                    for index, path in enumerate(paths):
                        group.create_task(inspect(index, path))
        except TimeoutError:
            # TaskGroup drains cancellation before exposing partial discovery;
            # no response leases escape this request. Threaded metadata work
            # retains separate bounded ownership until its OS fence releases.
            pass
        # Preserve installed-path order regardless of response timing: compiled
        # policy attenuation compares the complete ordered target collection.
        listings = tuple(value for result in results for value in result)
        if len(listings) > 64 or len(
            {listing.target.model_dump_json() for listing in listings}
        ) != len(listings):
            raise ValueError("native directory is ambiguous or exceeds bound")
        return listings

    def _inspection_finished(
        self, plugin_id: str, task: asyncio.Task[VerifiedRuntime]
    ) -> None:
        work = self._inspections.get(plugin_id)
        if work is not None and work.task is task:
            del self._inspections[plugin_id]
        # A timed-out observer may leave no waiter. Consume only the terminal
        # exception here; active observers still receive it through shield.
        if not task.cancelled():
            task.exception()

    async def _inspect_metadata(
        self, binding: HostBinding, selector: RuntimeSelector, metadata: bytes
    ) -> VerifiedRuntime:
        self._profile()
        work = self._inspections.get(binding.plugin_id)
        if work is not None and work.task.done():
            self._inspection_finished(binding.plugin_id, work.task)
            work = None
        fingerprint = hashlib.sha256(metadata).hexdigest()
        if work is None:
            if len(self._inspections) >= _MAX_INSTALLATIONS:
                raise ValueError("native metadata inspection capacity unavailable")
            task = asyncio.create_task(selector.installer.inspect_metadata(metadata))
            work = _Inspection(binding, fingerprint, task)
            self._inspections[binding.plugin_id] = work
            task.add_done_callback(
                lambda done: self._inspection_finished(binding.plugin_id, done)
            )
        elif work.binding != binding or work.metadata_sha256 != fingerprint:
            raise ValueError("native metadata inspection lifetime differs")
        # Cancelling an observer cannot stop to_thread hashing. Retain the
        # actual installer task/fence and one slot per owner until completion;
        # repeated scans may join it but never cache a completed observation.
        runtime = await asyncio.shield(work.task)
        self._current(binding)
        return runtime

    async def _discover(self, path: Path) -> tuple[NativeListing, ...]:
        """Verify one signed installation and its actual protected owner reply."""
        listings: list[NativeListing] = []
        try:
            binding = self._binding(path.stem)
            selector = RuntimeSelector(self.root / "installations" / binding.plugin_id)
            metadata = read_private(
                selector.root / "generations" / binding.runtime_digest / "staged.json",
                MAX_RUNTIME_METADATA_BYTES,
            )
            runtime = await self._inspect_metadata(binding, selector, metadata)
            manifest = runtime.claims.release.manifest
            if (
                runtime.claims.protocol != 4
                or not isinstance(manifest, NativeManifest)
                or runtime.digest != binding.runtime_digest
            ):
                return ()
            contracts = _OBJECT.validate_python(
                manifest.value()["contracts"], strict=True
            )
            signed = TypeAdapter(tuple[NativeDescriptor, ...]).validate_json(
                json.dumps(contracts["descriptors"])
            )
            endpoint = await self._endpoint(binding)
            response = await endpoint.peer.request("describe", {})
            try:
                if (
                    not isinstance(response.frame, ResponseFrame)
                    or response.frame.outcome != "complete"
                    or response.payload
                ):
                    raise ValueError("native directory reply refused")
                description = _OwnerDescription.model_validate_json(
                    json.dumps(response.frame.parameters)
                )
                for child in description.children:
                    if child.manifest_digest != manifest.digest:
                        raise ValueError("native child manifest differs")
                    descriptors = child.descriptors
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
                                incarnation=child.incarnation,
                                target=Target(
                                    node_id=child.node_id,
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
            return ()
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
        """Fence connections and drain handlers and owned metadata workers."""
        self.closed = True

        async def drain() -> None:
            await asyncio.gather(
                *(value.peer.close() for value in self.endpoints.values()),
                return_exceptions=True,
            )
            self.endpoints.clear()
            await asyncio.gather(
                *(work.task for work in self._inspections.values()),
                return_exceptions=True,
            )

        await finish_runtime_work(asyncio.create_task(drain()))
