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

from skulk.api.native_calls import create_native_calls_router
from skulk.extensions.native_calls import NativeListing
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
from skulk.extensions.runtime_artifacts import (
    RuntimeTrust,
    measure_host,
    verify_runtime,
)
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
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
        read_private(work / "native-host-bindings" / f"managed.native-{role}.json")
    )


def _socket(work: Path, role: str) -> Path:
    root = work / "installations" / f"managed.native-{role}"
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
            response = await peer.request("describe", {})
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
) -> tuple[NativeListing, ...]:
    async with asyncio.timeout(60):
        while True:
            try:
                response = await client.get("/v1/native-capabilities")
                assert response.status_code == 200
                listings = TypeAdapter(tuple[NativeListing, ...]).validate_json(
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
        transport_node_id="native-call-qualification", profile_id=uuid4().hex
    )
    write_private(work / "host.json", binding.model_dump_json().encode())
    connection = ServiceConnection(
        manager_root=str(work), profile_id=binding.profile_id or ""
    )
    write_private(work / "connection.json", connection.model_dump_json().encode())
    manager = RuntimeManager(work)
    app = FastAPI()
    app.include_router(
        create_native_calls_router(
            NodeId(binding.transport_node_id), connection_path=work / "connection.json"
        )
    )
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
    try:
        await manager.start()
        for role, source in (("caller", caller), ("receiver", receiver)):
            identifier = f"managed.native-{role}"
            metadata = (source / "release/release.json").read_bytes()
            trust = RuntimeTrust.model_validate_json(
                (source / "public-trust.json").read_bytes()
            )
            runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
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
                        response = await client.get("/v1/native-capabilities")
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
            listings = TypeAdapter(tuple[NativeListing, ...]).validate_json(
                response.content
            )
            assert len(listings) == 2
            checks.append(
                "production-router-actual-http-owner-authentication-and-signed-discovery"
            )
            policy = await client.get("/v1/native-capabilities/policy")
            assert (
                policy.status_code == 200
                and _OBJECT.validate_json(policy.content)["enabled"] is False
            )
            paths = {
                value.target.capability_id: f"/v1/native-capabilities/{value.plugin_id}/{value.target.capability_id}/{value.target.version}/call"
                for value in listings
            }
            denied = await client.post(paths["Receiver"], json={})
            assert denied.status_code == 503
            checks.append("disabled-by-default")
            changed = await client.put(
                "/v1/native-capabilities/policy",
                json={
                    "expected_revision": _OBJECT.validate_json(policy.content)[
                        "revision"
                    ],
                    "enabled": True,
                    "grants": [
                        value.grant().model_dump(mode="json") for value in listings
                    ],
                },
            )
            assert changed.status_code == 200, changed.text
            raw = b'{ "caller_id": "forged data only", "exact": 1 }'
            echoed = await client.post(
                paths["Receiver"],
                content=raw,
                headers={
                    "content-type": "application/json",
                    "x-skulk-caller-id": "untrusted-header",
                },
            )
            assert echoed.status_code == 200, echoed.text
            echo = _OBJECT.validate_json(echoed.content)
            assert echo["raw"] == raw.decode()
            assert (
                _OBJECT.validate_python(echo["context"], strict=True)["caller_id"]
                == "direct-owner"
            )
            assert (
                _OBJECT.validate_python(echo["context"], strict=True)["issuer_node_id"]
                == binding.transport_node_id
            )
            checks.append("exact-raw-input-and-host-minted-original-caller")
            target = next(
                value.target
                for value in listings
                if value.target.capability_id == "Receiver"
            )
            delegated = await client.post(
                paths["Caller"],
                json={
                    "action": "delegate",
                    "destination": target.model_dump(mode="json"),
                },
            )
            assert delegated.status_code == 200, delegated.text
            value = _OBJECT.validate_json(delegated.content)
            original = _OBJECT.validate_python(value["caller"], strict=True)
            receiver_value = _OBJECT.validate_python(value["receiver"], strict=True)
            child = _OBJECT.validate_python(receiver_value["context"], strict=True)
            assert child["caller_id"] == original["caller_id"] == "direct-owner"
            assert (
                child["issuer_node_id"]
                == original["issuer_node_id"]
                == binding.transport_node_id
            )
            assert child["root_scope_id"] == original["root_scope_id"]
            assert child["parent_scope_id"] == original["scope_id"]
            assert child["ancestry"] == [original["executor"]]
            assert value["caller_pid"] != receiver_value["pid"]
            assert receiver_value["raw"] == '{ "data": "exact delegated bytes" }'
            checks.extend(
                [
                    "native-call-between-independent-installed-applications",
                    "root-parent-original-caller-and-ancestry-preserved",
                    "exact-delegated-bytes",
                ]
            )
            for headers in (
                {"authorization": "Bearer synthetic-paired-token"},
                {"x-forwarded-for": "127.0.0.1"},
            ):
                refused = await client.post(paths["Receiver"], json={}, headers=headers)
                assert refused.status_code == 403
            checks.append("paired-and-forwarded-caller-claims-refused")
            checks_before_faults = len(checks)
            print(
                json.dumps(
                    {"phase": "two-app-call-passed", "checks": checks_before_faults}
                ),
                flush=True,
            )
            listings = await _quiescent(client, work)
            await _refuse_role(work, "caller")
            checks.append("installed-owner-refuses-forged-Core-role-signature")
            current_policy = _OBJECT.validate_json(changed.content)
            stale_update = await client.put(
                "/v1/native-capabilities/policy",
                json={"expected_revision": "0" * 64, "enabled": True, "grants": []},
            )
            assert stale_update.status_code == 409
            checks.append("stale-policy-revision-refuses-without-changing-enrollment")
            malformed = await client.post(
                paths["Receiver"],
                content=b'{"incomplete":',
                headers={"content-type": "application/json"},
            )
            assert malformed.status_code == 503
            await _quiescent(client, work)
            checks.append("malformed-body-refuses-before-native-dispatch")
            caller_target = next(
                item.target
                for item in listings
                if item.target.capability_id == "Caller"
            )
            caller_incarnation = next(
                item.incarnation
                for item in listings
                if item.target.capability_id == "Caller"
            )
            for action, destination in (
                ("cycle", caller_target),
                ("wrong-parent", target),
            ):
                refused = await client.post(
                    paths["Caller"],
                    json={
                        "action": action,
                        "destination": destination.model_dump(mode="json"),
                    },
                )
                assert refused.status_code == 503, refused.text
                ready = await _quiescent(client, work)
                assert (
                    next(
                        item.incarnation
                        for item in ready
                        if item.target.capability_id == "Caller"
                    )
                    == caller_incarnation
                )
            checks.append(
                "cycles-and-cross-root-parent-claims-refuse-without-crashing-app"
            )
            recovered = await client.post(
                paths["Caller"],
                json={
                    "action": "recover-denied",
                    "destination": target.model_dump(mode="json"),
                },
            )
            assert recovered.status_code == 200, recovered.text
            recovered_value = _OBJECT.validate_json(recovered.content)
            recovered_child = _OBJECT.validate_python(
                _OBJECT.validate_python(recovered_value["receiver"], strict=True)[
                    "context"
                ],
                strict=True,
            )
            assert (
                recovered_child["parent_scope_id"]
                == _OBJECT.validate_python(recovered_value["caller"], strict=True)[
                    "scope_id"
                ]
            )
            await _quiescent(client, work)
            checks.append(
                "local-refusal-consumes-author-sequence-without-blocking-next-valid-delegation"
            )
            orphan = await client.post(
                paths["Caller"],
                json={
                    "action": "orphan",
                    "destination": target.model_dump(mode="json"),
                },
            )
            assert orphan.status_code == 503
            ready = await _quiescent(client, work)
            assert (
                next(
                    item.incarnation
                    for item in ready
                    if item.target.capability_id == "Caller"
                )
                != caller_incarnation
            )
            checks.append(
                "premature-parent-completion-is-unknown-and-drains-descendants"
            )

            original = asyncio.create_task(
                client.post(paths["Receiver"], json={"delay": 2})
            )
            try:
                await _dispatched(work, "receiver")
                competing = await client.post(
                    paths["Caller"],
                    json={
                        "action": "delegate",
                        "destination": target.model_dump(mode="json"),
                    },
                )
                assert competing.status_code == 503
                first_result = await original
                assert first_result.status_code == 200
                assert (
                    _OBJECT.validate_json(first_result.content)["raw"] == '{"delay":2}'
                )
            finally:
                if not original.done():
                    original.cancel()
                await asyncio.gather(original, return_exceptions=True)
            await _quiescent(client, work)
            checks.append("competing-root-cannot-replace-or-cancel-another-active-call")

            timed = asyncio.create_task(
                client.post(
                    paths["Receiver"],
                    json={"delay": 10},
                    params={"timeout_seconds": 1.5},
                )
            )
            try:
                await _dispatched(work, "receiver")
                expired = await timed
                assert expired.status_code == 503
            finally:
                if not timed.done():
                    timed.cancel()
                await asyncio.gather(timed, return_exceptions=True)
            await _quiescent(client, work)
            checks.append("actual-dispatched-unary-deadline-retires-child-and-drains")

            revoking = asyncio.create_task(
                client.post(paths["Receiver"], json={"delay": 20})
            )
            try:
                await _dispatched(work, "receiver")
                revoked = await client.put(
                    "/v1/native-capabilities/policy",
                    json={
                        "expected_revision": current_policy["revision"],
                        "enabled": False,
                        "grants": [],
                    },
                )
                assert revoked.status_code == 200
                assert (await revoking).status_code == 503
            finally:
                if not revoking.done():
                    revoking.cancel()
                await asyncio.gather(revoking, return_exceptions=True)
            await _quiescent(client, work)
            ready = await _quiescent(client, work)
            restored = await client.put(
                "/v1/native-capabilities/policy",
                json={
                    "expected_revision": _OBJECT.validate_json(revoked.content)[
                        "revision"
                    ],
                    "enabled": True,
                    "grants": [item.grant().model_dump(mode="json") for item in ready],
                },
            )
            assert restored.status_code == 200
            checks.append(
                "live-policy-revocation-cancels-dispatched-work-without-replay"
            )

            for _ in range(3):
                disconnected = asyncio.create_task(
                    client.post(paths["Receiver"], json={"delay": 20})
                )
                try:
                    await _dispatched(work, "receiver")
                finally:
                    disconnected.cancel()
                    await asyncio.gather(disconnected, return_exceptions=True)
                await _quiescent(client, work)
            checks.append(
                "repeated-http-disconnects-clean-up-without-spending-crash-budget"
            )

            old_binding = _binding(work, "receiver")
            interrupted = asyncio.create_task(
                client.post(paths["Receiver"], json={"delay": 20})
            )
            try:
                await _dispatched(work, "receiver")
                service = manager.controllers["managed.native-receiver"].service
                assert service is not None and service.process is not None
                service.process.kill()
                assert (await interrupted).status_code == 503
            finally:
                if not interrupted.done():
                    interrupted.cancel()
                await asyncio.gather(interrupted, return_exceptions=True)
            await _quiescent(client, work)
            assert (
                _binding(work, "receiver").service_instance
                != old_binding.service_instance
            )
            await _refuse_role(work, "receiver", old_binding)
            recovery = await client.post(
                paths["Receiver"], json={"explicit_new_request": True}
            )
            assert recovery.status_code == 200
            await _quiescent(client, work)
            checks.extend(
                [
                    "owner-loss-produces-unknown-without-replay",
                    "normal-owner-restart-fences-old-host-lifetime",
                    "explicit-new-request-after-recovery",
                ]
            )

            for role in ("caller", "receiver"):
                identifier = f"managed.native-{role}"
                controller = manager.controllers[identifier]
                selection = controller.selector.current()
                assert selection is not None
                withdrawal = LifecycleRequest(
                    operation_id=uuid4().hex,
                    action="disable",
                    expected_revision=selection.revision,
                )
                await manager.dispatch(
                    SubmitRequest(plugin_id=identifier, request=withdrawal)
                )
                assert controller.work is not None
                await controller.work
                assert controller.operation(withdrawal.operation_id).state == "complete"
                RuntimeLock(controller.root, "supervisor.lock").close()
                assert (
                    read_private(controller.root / "protected-state-canary")
                    == b"Synthetic retained state; no account material."
                )
            empty = await client.get("/v1/native-capabilities")
            assert (
                empty.status_code == 200 and _ARRAY.validate_json(empty.content) == []
            )
            checks.extend(
                [
                    "normal-disable-reaps-both-owner-child-fences",
                    "synthetic-protected-state-remains-intact",
                    "disabled-installations-withdraw-native-directory",
                ]
            )
            print(
                json.dumps(
                    {"phase": "installed-fault-gates-passed", "checks": len(checks)}
                ),
                flush=True,
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
    checks.append("manager-owner-child-and-http-cleanup-host-unchanged")
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
                    "scope": "Full frozen Core; real production native router, TCP and direct-owner authentication; two normal signed installed applications. Same-host unary local reads only; no remote Fabric or individual-client admission claim.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    print(json.dumps({"passed": True, "checks": len(checks)}))


def main() -> None:
    """Qualify independently published caller/receiver fixture directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = _Arguments()
    _ = parser.parse_args(namespace=args)
    asyncio.run(qualify(args.caller, args.receiver, args.work, args.receipt))


if __name__ == "__main__":
    main()
