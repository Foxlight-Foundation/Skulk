"""Exercise two normally signed installed apps through real owner HTTP admission.

The full frozen Core runtime serves its production native router over a real
loopback TCP socket. No caller authentication, signed installer, manager/service,
owner process, compiled native reader or author process is substituted.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import secrets
import socket
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, cast, final
from uuid import uuid4

import httpx
from fastapi import FastAPI
from hypercorn.config import Config
from pydantic import JsonValue, TypeAdapter
from starlette.types import ASGIApp

from skulk.api.inference_access import ClientCreated, create_inference_access_router
from skulk.api.native_inference import create_native_inference_router
from skulk.extensions.inference_callers import ClientInventory
from skulk.extensions.managed_limits import MAX_RUNTIME_METADATA_BYTES
from skulk.extensions.native_host_binding import load_core_host_key
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    HostBinding,
    RequestFrame,
    ResponseFrame,
    admit_owner,
)
from skulk.extensions.native_host_rpc import Peer, Reply
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    inference_issuer,
)
from skulk.extensions.native_inference_hosts import InferenceListing
from skulk.extensions.runtime_artifacts import (
    RuntimeTrust,
    measure_host,
    verify_runtime,
)
from skulk.extensions.runtime_attachment import (
    AttachmentRequest,
    HostSettings,
    ServiceConnection,
)
from skulk.extensions.runtime_controller import LifecycleRequest
from skulk.extensions.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    RuntimeManager,
    SubmitRequest,
)
from skulk.extensions.runtime_service import RuntimeServiceStatus
from skulk.shared.types.common import NodeId

_OBJECT = TypeAdapter(dict[str, JsonValue])
_ARRAY = TypeAdapter(list[JsonValue])


class _HTTPServer(Protocol):
    def serve(
        self,
        app: ASGIApp,
        config: Config,
        *,
        shutdown_trigger: Callable[[], Awaitable[object]],
    ) -> Awaitable[None]:
        """Start the actual Hypercorn ASGI runtime with an explicit shutdown owner."""
        ...


@final
class _Arguments(argparse.Namespace):
    caller: Path
    receiver: Path
    work: Path
    receipt: Path


async def _serve_http(app: ASGIApp, config: Config, stopped: asyncio.Event) -> None:
    server = cast(
        _HTTPServer, cast(object, importlib.import_module("hypercorn.asyncio"))
    )
    await server.serve(app, config, shutdown_trigger=stopped.wait)


def _binding(work: Path, role: str) -> HostBinding:
    return HostBinding.model_validate_json(
        read_private(
            work / "native-host-bindings" / f"managed.native-inference-{role}.json"
        )
    )


def _socket(work: Path, role: str) -> Path:
    root = work / "installations" / f"managed.native-inference-{role}"
    digest = hashlib.sha256(str(root.absolute()).encode()).hexdigest()[:24]
    return Path("/tmp") / f"skulk-control-{os.getuid()}-{digest}" / "native-host.sock"


async def _observe(work: Path, role: str) -> dict[str, JsonValue]:
    # Read-only host diagnostics use the actual normal manager role attachment.
    # Root invocation still goes exclusively through production HTTP admission.
    binding = _binding(work, role)
    budget = NativePayloadBudget()
    reader, writer = await asyncio.open_unix_connection(
        _socket(work, role), limit=262144
    )
    peer: Peer | None = None

    def current() -> None:
        if _binding(work, role) != binding:
            raise ValueError("diagnostic attachment changed")

    async def refuse(_peer: Peer, _request: RequestFrame, _body: bytes) -> Reply:
        return Reply("pre_dispatch", {})

    try:
        async with asyncio.timeout(3):
            keys = await admit_owner(
                reader,
                writer,
                binding,
                load_core_host_key(work, binding.profile_id).key(),
                current,
            )
            peer = Peer(
                AuthenticatedStream(reader, writer, keys, budget),
                refuse,
                frozenset(),
                current,
            )
            response = await peer.request("describe_inference", {})
            try:
                assert (
                    isinstance(response.frame, ResponseFrame)
                    and response.frame.outcome == "complete"
                )
                return _OBJECT.validate_python(
                    response.frame.parameters["diagnostics"], strict=True
                )
            finally:
                response.release()
    finally:
        if peer is not None:
            await peer.close()
        writer.close()
        assert budget.used == 0


async def _quiescent(
    client: httpx.AsyncClient, work: Path
) -> tuple[InferenceListing, ...]:
    try:
        return await _wait_quiescent(client, work)
    except TimeoutError:
        # Keep a failed installed gate actionable without retaining author
        # output, request bodies, role keys or protected application state.
        statuses: dict[str, JsonValue] = {}
        for role in ("caller", "receiver"):
            try:
                status = RuntimeServiceStatus.model_validate_json(
                    read_private(
                        work
                        / "installations"
                        / f"managed.native-inference-{role}"
                        / "service-status.json"
                    )
                )
                statuses[role] = {
                    "state": status.state,
                    "error_code": status.error_code,
                }
            except (OSError, ValueError):
                statuses[role] = "unavailable"
        print(
            json.dumps({"phase": "quiescence-timeout", "owners": statuses}),
            flush=True,
        )
        raise


async def _wait_quiescent(
    client: httpx.AsyncClient, work: Path
) -> tuple[InferenceListing, ...]:
    async with asyncio.timeout(60):
        while True:
            try:
                response = await client.get("/v1/native-inference")
                assert response.status_code == 200
                listings = TypeAdapter(tuple[InferenceListing, ...]).validate_json(
                    response.content
                )
                if len(listings) == 2:
                    observations = [
                        await _observe(work, role) for role in ("caller", "receiver")
                    ]
                    if all(
                        value["retained_payload_bytes"] == 0
                        and value["dispatched_calls"] == 0
                        and _OBJECT.validate_python(value["scopes"], strict=True)[
                            "active_scopes"
                        ]
                        == 0
                        and _OBJECT.validate_python(value["scopes"], strict=True)[
                            "owned_tasks"
                        ]
                        == 0
                        for value in observations
                    ):
                        return listings
            except (OSError, ValueError, TimeoutError):
                pass  # Only read-only discovery/diagnostics are polled.
            await asyncio.sleep(0.05)


async def _dispatched(work: Path, role: str) -> None:
    async with asyncio.timeout(10):
        while True:
            observation = await _observe(work, role)
            if observation["dispatched_calls"] == 1:
                return
            await asyncio.sleep(0.01)


async def _refuse_role(work: Path, role: str, stale: HostBinding | None = None) -> None:
    binding = stale or _binding(work, role)
    reader, writer = await asyncio.open_unix_connection(_socket(work, role))
    try:
        async with asyncio.timeout(3):
            writer.write(
                (
                    json.dumps(
                        {
                            "protocol": 1,
                            "binding_digest": binding.digest,
                            "nonce": secrets.token_hex(32),
                        }
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()
            challenge = await reader.readline()
            if stale is None:
                assert challenge
                writer.write(
                    (
                        json.dumps({"protocol": 1, "signature": "0" * 128}) + "\n"
                    ).encode()
                )
                await writer.drain()
                assert await reader.read() == b""
            else:
                assert challenge == b""
    finally:
        writer.close()
        await writer.wait_closed()


async def qualify(caller: Path, receiver: Path, work: Path, receipt: Path) -> None:
    """Use two disposable publishers and retain payload-free qualification evidence."""
    work = work.absolute()
    if work.exists() or receipt.exists():
        raise ValueError("qualification requires fresh owned paths")
    private_directory(work)
    host = measure_host()
    binding = HostSettings(
        transport_node_id="native-inference-qualification", profile_id=uuid4().hex
    )
    write_private(work / "host.json", binding.model_dump_json().encode())
    connection = ServiceConnection(
        manager_root=str(work), profile_id=binding.profile_id or ""
    )
    write_private(work / "connection.json", connection.model_dump_json().encode())
    manager = RuntimeManager(work)
    app = FastAPI()
    app.include_router(
        create_native_inference_router(
            NodeId(binding.transport_node_id), connection_path=work / "connection.json"
        )
    )
    app.include_router(
        create_inference_access_router(
            NodeId(binding.transport_node_id), connection_path=work / "connection.json"
        )
    )
    child_roots: dict[str, Path] = {}
    stopped = asyncio.Event()
    network = socket.socket()
    network.bind(("127.0.0.1", 0))
    port = TypeAdapter(tuple[str, int]).validate_python(
        cast(object, network.getsockname()), strict=True
    )[1]
    config = Config()
    config.bind = [f"fd://{network.fileno()}"]
    config.accesslog = None
    config.errorlog = None
    server: asyncio.Task[None] | None = None
    checks: list[str] = []
    runtimes: list[dict[str, str]] = []
    large_metadata_bytes = 0
    try:
        await manager.start()
        for role, source in (("caller", caller), ("receiver", receiver)):
            identifier = f"managed.native-inference-{role}"
            metadata = (source / "release/release.json").read_bytes()
            if role == "receiver":
                # Whitespace changes transport bytes, not the signed canonical
                # claims. The normal installer accepts this valid envelope;
                # discovery must use the same bound rather than hide the app.
                assert len(metadata) < 300_000
                metadata += b" " * (300_000 - len(metadata))
                large_metadata_bytes = len(metadata)
            trust = RuntimeTrust.model_validate_json(
                (source / "public-trust.json").read_bytes()
            )
            runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
            native = runtime.native_manifest
            assert native is not None
            child_roots[role] = (
                work
                / "installations"
                / identifier
                / (
                    "bundle-"
                    + hashlib.sha256(native.bundle_id.encode()).hexdigest()[:24]
                )
            )
            await manager.dispatch(
                InstallationRequest(action="register", plugin_id=identifier)
            )
            controller = manager.controllers[identifier]
            write_private(
                controller.root / "protected-state-canary",
                b"Synthetic retained state; no account material.",
            )
            write_private(
                controller.root / "publisher-trust.json",
                trust.model_dump_json().encode(),
            )
            staged = await controller.selector.installer.stage(
                metadata, source / "release"
            )
            assert staged.state == "staged"
            assert (
                read_private(
                    controller.root / "generations" / runtime.digest / "staged.json",
                    MAX_RUNTIME_METADATA_BYTES,
                )
                == metadata
            )
            activation = LifecycleRequest(
                operation_id=uuid4().hex,
                action="activate",
                expected_revision=0,
                runtime_digest=runtime.digest,
            )
            await manager.dispatch(
                SubmitRequest(plugin_id=identifier, request=activation)
            )
            assert controller.work is not None
            await controller.work
            assert controller.operation(activation.operation_id).state == "complete"
            runtimes.append(
                {
                    "runtime_digest": runtime.digest,
                    "manifest_digest": runtime.native_manifest.digest
                    if runtime.native_manifest is not None
                    else "",
                }
            )
        checks.extend(
            [
                "normal-signed-two-application-install",
                "normal-manager-service-owner-activation",
            ]
        )
        network.detach()
        server = asyncio.create_task(_serve_http(app, config, stopped))
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=45,
            headers={
                "x-skulk-dashboard": "pairing-v1",
                "origin": f"http://127.0.0.1:{port}",
            },
        ) as client:
            async with asyncio.timeout(90):
                while True:
                    try:
                        response = await client.get("/v1/native-inference")
                        if (
                            response.status_code == 200
                            and len(_ARRAY.validate_json(response.content)) == 2
                        ):
                            break
                    except httpx.TransportError:
                        pass
                    if server.done():
                        server.result()
                    await asyncio.sleep(0.1)
            listings = TypeAdapter(tuple[InferenceListing, ...]).validate_json(
                response.content
            )
            assert len(listings) == 2
            assert large_metadata_bytes == 300_000
            checks.append(
                "production-router-actual-http-owner-authentication-and-signed-discovery"
            )
            checks.append(
                "normal-installer-and-discovery-accept-signed-envelope-above-256kib"
            )
            await _refuse_role(work, "receiver")
            checks.append("new-inference-owner-refuses-unproven-core-role")
            paths = {
                value.target.capability_id: f"/v1/native-inference/{value.plugin_id}/{value.target.capability_id}/{value.target.version}/call"
                for value in listings
            }
            policy_reply = await client.get("/v1/inference-access/policy")
            policy = InferenceAccessPolicy.model_validate_json(policy_reply.content)
            assert policy_reply.status_code == 200 and not policy.enabled
            denied = await client.post(paths["InferenceReceiver"], json={})
            assert denied.status_code == 503 and _counter(child_roots["receiver"]) == 0
            inventory_reply = await client.get("/v1/inference-access/clients")
            inventory = ClientInventory.model_validate_json(inventory_reply.content)
            created = await client.post(
                "/v1/inference-access/clients",
                json={
                    "expected_revision": inventory.revision,
                    "label": "Independent stock client",
                },
            )
            assert (
                created.status_code == 201
                and created.headers["cache-control"] == "no-store"
            )
            credential = ClientCreated.model_validate_json(created.content)
            client_id = "client:" + credential.client.identifier

            async def enroll(
                identifier: str, grants: tuple[InferenceListing, ...]
            ) -> None:
                current = await client.get("/v1/inference-access/policy")
                previous = InferenceAccessPolicy.model_validate_json(current.content)
                response = await client.put(
                    "/v1/inference-access/policy",
                    json={
                        "expected_revision": previous.revision,
                        "enabled": True,
                        "callers": [
                            {
                                "principal": {
                                    "identifier": identifier,
                                    "kind": "client",
                                    "issuer_node_id": inference_issuer(
                                        binding.profile_id
                                    ),
                                },
                                "grants": [
                                    value.grant().model_dump(mode="json")
                                    for value in grants
                                ],
                                "maximum_seconds": None,
                            }
                        ],
                    },
                )
                assert response.status_code == 200, (
                    "normal signed caller enrollment refused"
                )

            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=45,
                headers={"authorization": "Bearer " + credential.token},
            ) as user:
                assert (
                    await user.get("/v1/inference-access/clients")
                ).status_code == 403
                assert (
                    await user.post(paths["InferenceReceiver"], json={})
                ).status_code == 503
                assert _counter(child_roots["receiver"]) == 0
                assert (
                    await user.post(
                        paths["InferenceReceiver"],
                        json={},
                        headers={"authorization": "Bearer invalid"},
                    )
                ).status_code == 401
                assert (
                    await user.get(
                        "/v1/native-inference", headers={"x-forwarded-for": "127.0.0.1"}
                    )
                ).status_code == 403
                checks.extend(
                    [
                        "disabled-policy-writes-no-native-effect",
                        "real-client-bearer-authentication-is-not-an-execution-grant",
                        "invalid-bearer-never-falls-back-to-owner",
                        "client-cannot-administer-credentials",
                        "forwarded-client-transport-refused",
                    ]
                )
                await enroll(client_id, listings)
                no_owner = await client.post(paths["InferenceReceiver"], json={})
                assert (
                    no_owner.status_code == 503
                    and _counter(child_roots["receiver"]) == 0
                )
                raw = b'{ "caller_id": "forged data only", "exact": 1 }'
                echoed = await user.post(
                    paths["InferenceReceiver"],
                    content=raw,
                    headers={"content-type": "application/json"},
                )
                assert echoed.status_code == 200, (
                    "authenticated installed inference call failed"
                )
                value = _OBJECT.validate_json(echoed.content)
                context = _OBJECT.validate_python(value["context"], strict=True)
                assert (
                    value["raw"] == raw.decode()
                    and context["caller_id"] == client_id
                    and context["caller_kind"] == "client"
                )
                assert (
                    context["issuer_node_id"] == inference_issuer(binding.profile_id)
                    and context["deadline_unix_ms"] is None
                )
                assert _counter(child_roots["receiver"]) == 1
                checks.extend(
                    [
                        "owner-realm-does-not-inherit-client-permission",
                        "real-unmodified-stock-bearer-client-through-production-tcp-api",
                        "full-original-client-and-issuer-survive-installed-native-dispatch",
                        "raw-input-hash-binding-and-forged-caller-data-is-inert",
                        "null-ceiling-preserved-through-real-owner-and-author",
                    ]
                )
                target = next(
                    value.target
                    for value in listings
                    if value.target.capability_id == "InferenceReceiver"
                )
                for action in ("delegate", "recover-denied"):
                    response = await user.post(
                        paths["InferenceCaller"],
                        json={
                            "action": action,
                            "destination": target.model_dump(mode="json"),
                        },
                    )
                    assert response.status_code == 200, (
                        "installed composed inference call failed"
                    )
                    value = _OBJECT.validate_json(response.content)
                    source = _OBJECT.validate_python(value["caller"], strict=True)
                    receiver_value = _OBJECT.validate_python(
                        value["receiver"], strict=True
                    )
                    destination = _OBJECT.validate_python(
                        receiver_value["context"], strict=True
                    )
                    assert source["caller_id"] == destination["caller_id"] == client_id
                    assert (
                        source["root_scope_id"] == destination["root_scope_id"]
                        and destination["parent_scope_id"] == source["scope_id"]
                    )
                    assert destination["deadline_unix_ms"] is None
                    assert receiver_value["pid"] != value["caller_pid"]
                checks.extend(
                    [
                        "two-independent-installed-apps-preserve-client-ancestry-and-null-budget",
                        "local-refusal-consumes-sequence-without-breaking-core-proxy-order",
                    ]
                )
                before = _counter(child_roots["receiver"])
                await enroll(
                    client_id,
                    tuple(
                        value
                        for value in listings
                        if value.target.capability_id == "InferenceCaller"
                    ),
                )
                forbidden = await user.post(
                    paths["InferenceCaller"],
                    json={
                        "action": "delegate",
                        "destination": target.model_dump(mode="json"),
                    },
                )
                assert (
                    forbidden.status_code == 503
                    and _counter(child_roots["receiver"]) == before
                )
                checks.append(
                    "destination-and-processing-domain-permission-denial-before-receiver-effect"
                )
                await enroll(client_id, listings)
                before = _counter(child_roots["receiver"])
                prior = next(
                    value.incarnation
                    for value in listings
                    if value.target.capability_id == "InferenceReceiver"
                )
                active = asyncio.create_task(
                    user.post(paths["InferenceReceiver"], json={"delay": 30})
                )
                async with asyncio.timeout(5):
                    while _counter(child_roots["receiver"]) == before:
                        await asyncio.sleep(0.01)
                await _dispatched(work, "receiver")
                current = ClientInventory.model_validate_json(
                    (await client.get("/v1/inference-access/clients")).content
                )
                revoked = await client.request(
                    "DELETE",
                    "/v1/inference-access/clients/" + credential.client.identifier,
                    json={"expected_revision": current.revision},
                )
                assert revoked.status_code == 200
                revoked_call = await active
                assert revoked_call.status_code == 503
                assert (
                    await user.post(paths["InferenceReceiver"], json={})
                ).status_code == 401
                assert _counter(child_roots["receiver"]) == before + 1
                async with asyncio.timeout(60):
                    while True:
                        refreshed = TypeAdapter(
                            tuple[InferenceListing, ...]
                        ).validate_json(
                            (await client.get("/v1/native-inference")).content
                        )
                        changed = [
                            value
                            for value in refreshed
                            if value.target.capability_id == "InferenceReceiver"
                            and value.incarnation != prior
                        ]
                        if len(refreshed) == 2 and changed:
                            listings = refreshed
                            break
                        await asyncio.sleep(0.05)
                checks.extend(
                    [
                        "revocation-cancels-actually-dispatched-uncapped-work",
                        "cancelled-child-is-reaped-and-replaced-with-a-new-incarnation",
                        "revoked-credential-cannot-replay-and-effects-are-not-retried",
                    ]
                )
            current = ClientInventory.model_validate_json(
                (await client.get("/v1/inference-access/clients")).content
            )
            replacement_response = await client.post(
                "/v1/inference-access/clients",
                json={
                    "expected_revision": current.revision,
                    "label": "Replacement independent client",
                },
            )
            assert replacement_response.status_code == 201
            replacement = ClientCreated.model_validate_json(
                replacement_response.content
            )
            assert replacement.client.identifier != credential.client.identifier
            await enroll("client:" + replacement.client.identifier, listings)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=45,
                headers={"authorization": "Bearer " + replacement.token},
            ) as user:
                started = time.monotonic()
                long_call = await user.post(
                    paths["InferenceReceiver"], json={"delay": 35}
                )
                assert long_call.status_code == 200 and time.monotonic() - started >= 34
                long_context = _OBJECT.validate_python(
                    _OBJECT.validate_json(long_call.content)["context"], strict=True
                )
                assert long_context["deadline_unix_ms"] is None
                checks.append(
                    "real-generation-silence-beyond-thirty-seconds-with-independent-health"
                )
                await _quiescent(client, work)
                checks.append(
                    "all-native-scope-tasks-payload-leases-and-dispatches-quiescent"
                )
            for role in ("caller", "receiver"):
                controller = manager.controllers[f"managed.native-inference-{role}"]
                selection = controller.selector.current()
                assert selection is not None
                withdrawal = LifecycleRequest(
                    operation_id=uuid4().hex,
                    action="disable",
                    expected_revision=selection.revision,
                )
                await manager.dispatch(
                    SubmitRequest(
                        plugin_id=f"managed.native-inference-{role}", request=withdrawal
                    )
                )
                assert controller.work is not None
                await controller.work
                assert controller.operation(withdrawal.operation_id).state == "complete"
                RuntimeLock(controller.root, "supervisor.lock").close()
                retained = controller.selector.current()
                assert retained is not None and not retained.enabled
                preview = await controller.selector.preview(
                    retained.runtime_digest,
                    expected_revision=retained.revision,
                    operation_id=uuid4().hex,
                    rollback=True,
                    enabled=False,
                )
                assert preview.runtime_digest == retained.runtime_digest
                assert controller.selector.current() == retained
                assert (
                    read_private(controller.root / "protected-state-canary")
                    == b"Synthetic retained state; no account material."
                )
            assert (
                _ARRAY.validate_json((await client.get("/v1/native-inference")).content)
                == []
            )
            checks.extend(
                [
                    "normal-disable-reaps-both-owners-and-children",
                    "synthetic-retained-state-survives-lifecycle",
                    "retained-inference-generation-previews-without-changing-selection",
                    "disabled-native-inference-withdraws-capacity",
                ]
            )
            stopped.set()
            await server
            server = None
            await _reattach_http(manager, work, replacement, credential.token)
            checks.extend(
                [
                    "normal-transport-renewal-retains-enrolled-http-client-and-policy",
                    "http-client-management-and-revocation-survive-core-api-restart",
                ]
            )
    finally:
        stopped.set()
        try:
            if server is not None:
                await server
        finally:
            try:
                network.close()
            finally:
                await manager.close()
    assert measure_host() == host
    checks.append("manager-http-owner-child-cleanup-and-host-build-unchanged")
    write_private(
        receipt,
        (
            json.dumps(
                {
                    "format": 1,
                    "passed": True,
                    "platform": host.platform,
                    "checks": checks,
                    "runtimes": runtimes,
                    "skulk_build_sha256": host.skulk_build_sha256,
                    "large_signed_metadata_bytes": large_metadata_bytes,
                    "scope": "Full frozen Core; real production TCP owner and enrolled-client authentication; two normally signed installed inference applications; exact unary effects and delegation. No provider account, live migration, streaming or distributed qualification claim.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    print(json.dumps({"passed": True, "checks": len(checks)}))


def _counter(root: Path) -> int:
    """Observe actual effects only inside the disposable qualification application."""
    try:
        return int(read_private(root / "qualification-effects", 32))
    except FileNotFoundError:
        return 0


async def _reattach_http(
    manager: RuntimeManager, work: Path, kept: ClientCreated, revoked_token: str
) -> None:
    """Renew through the normal manager and authenticate over a fresh real HTTP server."""
    previous = manager.settings
    assert previous.profile_id is not None
    credentials = read_private(work / "inference-clients.json")
    policy = read_private(work / "native-inference-policy.json")
    transport = "native-inference-restarted"
    fence = RuntimeLock(work, "attachment.lock")
    try:
        await manager.dispatch(
            AttachmentRequest(
                profile_id=previous.profile_id,
                transport_node_id=transport,
                skulk_build_sha256=measure_host().skulk_build_sha256,
            )
        )
    finally:
        fence.close()
    assert manager.settings.transport_node_id == transport
    assert read_private(work / "inference-clients.json") == credentials
    assert read_private(work / "native-inference-policy.json") == policy
    app = FastAPI()
    app.include_router(
        create_inference_access_router(
            NodeId(transport), connection_path=work / "connection.json"
        )
    )
    app.include_router(
        create_native_inference_router(
            NodeId(transport), connection_path=work / "connection.json"
        )
    )
    network = socket.socket()
    network.bind(("127.0.0.1", 0))
    port = TypeAdapter(tuple[str, int]).validate_python(
        cast(object, network.getsockname()), strict=True
    )[1]
    config = Config()
    config.bind = [f"fd://{network.fileno()}"]
    config.accesslog = None
    config.errorlog = None
    network.detach()
    stopped = asyncio.Event()
    server = asyncio.create_task(_serve_http(app, config, stopped))
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=10,
            headers={
                "x-skulk-dashboard": "pairing-v1",
                "origin": f"http://127.0.0.1:{port}",
            },
        ) as client:
            async with asyncio.timeout(30):
                while True:
                    try:
                        inventory = await client.get("/v1/inference-access/clients")
                        if inventory.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    await asyncio.sleep(0.01)
            current = ClientInventory.model_validate_json(inventory.content)
            assert kept.client.identifier in {
                item.identifier for item in current.clients
            }
            headers = {"authorization": "Bearer " + kept.token}
            assert (
                await client.get("/v1/native-inference", headers=headers)
            ).status_code == 200
            assert (
                await client.get(
                    "/v1/native-inference",
                    headers={"authorization": "Bearer " + revoked_token},
                )
            ).status_code == 401
            response = await client.post(
                "/v1/inference-access/clients",
                json={
                    "expected_revision": current.revision,
                    "label": "After transport renewal",
                },
            )
            assert response.status_code == 201
            added = ClientCreated.model_validate_json(response.content)
            response = await client.request(
                "DELETE",
                "/v1/inference-access/clients/" + added.client.identifier,
                json={"expected_revision": added.revision},
            )
            assert response.status_code == 200
            assert (
                await client.get(
                    "/v1/native-inference",
                    headers={"authorization": "Bearer " + added.token},
                )
            ).status_code == 401
            assert (
                await client.get("/v1/native-inference", headers=headers)
            ).status_code == 200
    finally:
        stopped.set()
        await server


def main() -> None:
    """Qualify independently published inference caller/receiver fixture directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = _Arguments()
    _ = parser.parse_args(namespace=arguments)
    asyncio.run(
        qualify(arguments.caller, arguments.receiver, arguments.work, arguments.receipt)
    )


if __name__ == "__main__":
    main()
