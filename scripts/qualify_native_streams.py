"""Qualify normal signed streaming through real owner/Core and bearer WebSockets.

This account-free installed gate uses the public SDK application fixture, not
a replacement installer, authorization function, host channel or native codec.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
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
from qualify_native_inference import observe_native_inference
from starlette.types import ASGIApp
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from skulk.api.inference_access import ClientCreated, create_inference_access_router
from skulk.api.native_inference import create_native_inference_router
from skulk.api.server_config import create_local_api_server_config
from skulk.extensions.inference_callers import ClientInventory
from skulk.extensions.native_host_protocol import ResponseFrame
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    inference_issuer,
)
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_contracts import public_inference_kernel
from skulk.extensions.native_inference_hosts import InferenceListing
from skulk.extensions.native_inference_streams import NativeInferenceStream
from skulk.extensions.runtime_artifacts import (
    RuntimeTrust,
    measure_host,
    verify_runtime,
)
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
from skulk.extensions.runtime_controller import LifecycleRequest
from skulk.extensions.runtime_files import (
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

_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_LISTINGS: TypeAdapter[tuple[InferenceListing, ...]] = TypeAdapter(
    tuple[InferenceListing, ...]
)
_TEXT = TypeAdapter(str)


class _HTTPServer(Protocol):
    def serve(
        self,
        app: ASGIApp,
        config: Config,
        *,
        shutdown_trigger: Callable[[], Awaitable[object]],
    ) -> Awaitable[None]:
        """Run actual Hypercorn with explicit owned shutdown."""
        ...


@final
class _Arguments(argparse.Namespace):
    streamer: Path
    work: Path
    receipt: Path


async def qualify(source: Path, work: Path, receipt: Path) -> None:
    """Install a fresh public-SDK app and exercise actual TCP caller admission."""
    work = work.absolute()
    if work.exists() or receipt.exists():
        raise ValueError("qualification requires fresh owned paths")
    private_directory(work)
    measured = measure_host()
    settings = HostSettings(
        transport_node_id="native-stream-qualification", profile_id=uuid4().hex
    )
    assert settings.profile_id is not None
    issuer = inference_issuer(settings.profile_id)
    write_private(work / "host.json", settings.model_dump_json().encode())
    service_connection = ServiceConnection(
        manager_root=str(work), profile_id=settings.profile_id or ""
    )
    write_private(
        work / "connection.json", service_connection.model_dump_json().encode()
    )
    manager = RuntimeManager(work)
    node = NodeId(settings.transport_node_id)
    app = FastAPI()
    app.include_router(
        create_native_inference_router(node, connection_path=work / "connection.json")
    )
    app.include_router(
        create_inference_access_router(node, connection_path=work / "connection.json")
    )
    native = public_inference_kernel()
    stopped = asyncio.Event()
    network = socket.socket()
    network.bind(("127.0.0.1", 0))
    port = TypeAdapter(tuple[str, int]).validate_python(
        cast(object, network.getsockname()), strict=True
    )[1]
    config = create_local_api_server_config(port)
    config.bind = [f"fd://{network.fileno()}"]
    server: asyncio.Task[None] | None = None
    checks: list[str] = []
    runtime_digest = ""
    try:
        await manager.start()
        identifier = "managed.native-inference-streamer"
        metadata = (source / "release/release.json").read_bytes()
        trust = RuntimeTrust.model_validate_json(
            (source / "public-trust.json").read_bytes()
        )
        runtime = verify_runtime(metadata, trust, measured, now=int(time.time()))
        manifest = runtime.native_manifest
        assert manifest is not None
        runtime_digest = runtime.digest
        await manager.dispatch(
            InstallationRequest(action="register", plugin_id=identifier)
        )
        controller = manager.controllers[identifier]
        child_root = controller.root / (
            "bundle-" + hashlib.sha256(manifest.bundle_id.encode()).hexdigest()[:24]
        )
        write_private(
            controller.root / "publisher-trust.json", trust.model_dump_json().encode()
        )
        assert (
            await controller.selector.installer.stage(metadata, source / "release")
        ).state == "staged"
        activation = LifecycleRequest(
            operation_id=uuid4().hex,
            action="activate",
            expected_revision=0,
            runtime_digest=runtime.digest,
        )
        await manager.dispatch(SubmitRequest(plugin_id=identifier, request=activation))
        assert controller.work is not None
        await controller.work
        assert controller.operation(activation.operation_id).state == "complete"
        checks.append("normal-signed-public-sdk-application-install-and-startup")

        server_api = cast(
            _HTTPServer, cast(object, importlib.import_module("hypercorn.asyncio"))
        )
        network.detach()

        async def serve() -> None:
            await server_api.serve(app, config, shutdown_trigger=stopped.wait)

        server = asyncio.create_task(serve())
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=45,
            headers={
                "x-skulk-dashboard": "pairing-v1",
                "origin": f"http://127.0.0.1:{port}",
            },
        ) as owner:
            async with asyncio.timeout(90):
                while True:
                    try:
                        response = await owner.get("/v1/native-inference")
                        if (
                            response.status_code == 200
                            and len(_LISTINGS.validate_json(response.content)) == 4
                        ):
                            break
                    except httpx.TransportError:
                        pass
                    if server.done():
                        server.result()
                    await asyncio.sleep(0.05)
            listings = _LISTINGS.validate_json(response.content)
            assert all(
                value.startup_seconds == 30 and value.stream_seconds == 1
                for value in listings
            )
            inventory = ClientInventory.model_validate_json(
                (await owner.get("/v1/inference-access/clients")).content
            )
            created = await owner.post(
                "/v1/inference-access/clients",
                json={
                    "expected_revision": inventory.revision,
                    "label": "Streaming qualification client",
                },
            )
            assert created.status_code == 201
            credential = ClientCreated.model_validate_json(created.content)
            client_id = "client:" + credential.client.identifier

            async def policy(enabled: bool) -> None:
                current = InferenceAccessPolicy.model_validate_json(
                    (await owner.get("/v1/inference-access/policy")).content
                )
                result = await owner.put(
                    "/v1/inference-access/policy",
                    json={
                        "expected_revision": current.revision,
                        "enabled": enabled,
                        "callers": [
                            {
                                "principal": {
                                    "identifier": client_id,
                                    "kind": "client",
                                    "issuer_node_id": issuer,
                                },
                                "grants": [
                                    value.grant().model_dump(mode="json")
                                    for value in listings
                                ],
                                "maximum_seconds": None,
                            }
                        ],
                    },
                )
                assert result.status_code == 200

            def endpoint(suffix: str) -> str:
                value = next(
                    value
                    for value in listings
                    if value.target.capability_id == "InferenceStreamer" + suffix
                )
                return f"ws://127.0.0.1:{port}/v1/native-inference/{value.plugin_id}/{value.target.capability_id}/{value.target.version}/stream"

            headers = {"authorization": "Bearer " + credential.token}
            for refused in (
                {},
                {"authorization": "Bearer invalid"},
                {**headers, "x-forwarded-for": "127.0.0.1"},
            ):
                try:
                    async with connect(
                        endpoint("Server"), additional_headers=refused, proxy=None
                    ):
                        raise AssertionError("unproven WebSocket caller accepted")
                except InvalidStatus as error:
                    assert error.response.status_code == 403
            assert not (child_root / "qualification-effects").exists()
            checks.append("missing-forged-and-forwarded-bearers-refused-before-effects")
            await policy(True)

            async def start(connection: ClientConnection, payload: str) -> str:
                await connection.send(payload)
                ready = _OBJECT.validate_json(await connection.recv())
                assert ready["type"] == "ready"
                return _TEXT.validate_python(ready["call_id"], strict=True)

            async def input_frames(connection: ClientConnection, call_id: str) -> None:
                for sequence, kind in enumerate(("started", "chunk", "completed")):
                    frame: dict[str, JsonValue] = {
                        "call_id": call_id,
                        "direction": "caller_to_provider",
                        "sequence": sequence,
                        "kind": kind,
                    }
                    media = b""
                    if kind == "chunk":
                        frame.update(
                            payload={"input": "exact media"},
                            media={
                                "kind": "inline",
                                "media_type": "application/octet-stream",
                            },
                        )
                        media = b"\x00\xfe\x02"
                    await connection.send(
                        native.encode_transport_v2_packet(json.dumps(frame), media)
                    )

            async def output(
                connection: ClientConnection,
            ) -> tuple[dict[str, JsonValue], bytes]:
                packet = await connection.recv()
                assert isinstance(packet, bytes)
                raw, media = native.decode_transport_v2_packet(packet)
                return _OBJECT.validate_json(raw), media

            expected_process: JsonValue = None
            for suffix in ("Server", "Client", "Bidi"):
                for terminal in ("completed", "failed", "cancelled"):
                    raw = json.dumps(
                        {
                            "exact": "unchanged",
                            "terminal": terminal,
                            "cleanup_delay": 0.1,
                            "delay": 1.2,
                        }
                    )
                    async with connect(
                        endpoint(suffix),
                        additional_headers=headers,
                        proxy=None,
                        max_size=2**21,
                    ) as connection:
                        call_id = await start(connection, raw)
                        if suffix != "Server":
                            await input_frames(connection, call_id)
                        if suffix != "Client":
                            chunk, media = await output(connection)
                            assert chunk["kind"] == "chunk" and media == b"\x00\xff\x01"
                        result, media = await output(connection)
                        assert result["kind"] == terminal and media == b""
                        assert (
                            read_private(child_root / ("cleanup-" + call_id))
                            == b"closed"
                        )
                        if terminal == "completed":
                            body = _OBJECT.validate_python(
                                result["payload"], strict=True
                            )
                            if expected_process is None:
                                expected_process = body["pid"]
                            assert body["pid"] == expected_process
                            context = _OBJECT.validate_python(
                                body["context"], strict=True
                            )
                            assert (
                                body["raw"] == raw and context["caller_id"] == client_id
                            )
                            assert context["deadline_unix_ms"] is None
                            if suffix != "Server":
                                assert (
                                    body["input_media_sha256"]
                                    == hashlib.sha256(b"\x00\xfe\x02").hexdigest()
                                )
                                assert body["input_kinds"] == [
                                    "started",
                                    "chunk",
                                    "completed",
                                ]
                        else:
                            assert result["error"] == {
                                "code": "provider_error"
                                if terminal == "failed"
                                else "cancelled",
                                "message": "",
                            }
                        end, media = await output(connection)
                        assert end["kind"] == "end" and not media
                    checks.append(
                        f"installed-{suffix.lower()}-{terminal}-exact-media-cleanup-and-uncapped-silence"
                    )
                    print(
                        json.dumps(
                            {"mode": suffix, "terminal": terminal, "passed": True}
                        ),
                        flush=True,
                    )

            for fault in ("call", "sequence", "profile", "schema", "media-bound"):
                async with connect(
                    endpoint("Client"),
                    additional_headers=headers,
                    proxy=None,
                    max_size=2**21,
                ) as connection:
                    call_id = await start(connection, "{}")
                    await connection.send(
                        native.encode_transport_v2_packet(
                            json.dumps(
                                {
                                    "call_id": call_id,
                                    "direction": "caller_to_provider",
                                    "sequence": 0,
                                    "kind": "started",
                                }
                            ),
                            b"",
                        )
                    )
                    wire = native.encode_transport_v2_packet(
                        json.dumps(
                            {
                                "call_id": call_id,
                                "direction": "caller_to_provider",
                                "sequence": 1,
                                "kind": "chunk",
                                "payload": {"input": "bounded"},
                                "media": {
                                    "kind": "inline",
                                    "media_type": "application/octet-stream",
                                },
                            }
                        ),
                        b"x",
                    )
                    length = int.from_bytes(wire[:4], "big")
                    envelope = _OBJECT.validate_json(wire[4 : 4 + length])
                    frame = _OBJECT.validate_python(envelope["frame"], strict=True)
                    body = b"x"
                    if fault == "call":
                        frame["call_id"] = "0" * 64
                    elif fault == "sequence":
                        frame["sequence"] = 7
                    elif fault == "schema":
                        frame["payload"] = "not an object"
                    elif fault == "profile":
                        envelope["profile_digest"] = "0" * 64
                    else:
                        envelope["media_size"] = 1048577
                        body = b"x" * 1048577
                    envelope["frame"] = frame
                    encoded = json.dumps(envelope).encode()
                    # Corrupt actual codec output as untrusted network input;
                    # production admission still uses the real compiled reader.
                    await connection.send(
                        len(encoded).to_bytes(4, "big") + encoded + body
                    )
                    try:
                        await connection.recv()
                        raise AssertionError("invalid caller packet produced output")
                    except ConnectionClosed as error:
                        assert error.code == 1011
                    async with asyncio.timeout(10):
                        while not (child_root / ("cleanup-" + call_id)).exists():
                            await asyncio.sleep(0.01)
                checks.append("invalid-" + fault + "-input-cancels-without-replay")

            # Portable/2 fixes JSON control values at 64 KiB. Large contexts
            # use media rather than silently changing that issued language.
            before = read_private(child_root / "qualification-effects", 32)
            for size in (128 * 1024, 3 * 1024 * 1024):
                async with connect(
                    endpoint("Server"), additional_headers=headers, proxy=None
                ) as connection:
                    try:
                        await connection.send(json.dumps({"large": "x" * size}))
                        await connection.recv()
                        raise AssertionError("oversized JSON control was admitted")
                    except ConnectionClosed as error:
                        assert error.code == 1009
                assert read_private(child_root / "qualification-effects", 32) == before
            checks.append(
                "production-json-control-bound-refuses-large-initial-messages-before-effects"
            )
            large_media = b"x" * (3 * 1024 * 1024)
            initial = json.dumps(
                {"input_media_sha256": hashlib.sha256(large_media).hexdigest()}
            )
            async with connect(
                endpoint("Client"), additional_headers=headers, proxy=None
            ) as connection:
                call_id = await start(connection, initial)
                for sequence in range(5):
                    kind = (
                        "started"
                        if sequence == 0
                        else "completed"
                        if sequence == 4
                        else "chunk"
                    )
                    frame: dict[str, JsonValue] = {
                        "call_id": call_id,
                        "direction": "caller_to_provider",
                        "sequence": sequence,
                        "kind": kind,
                    }
                    media = b""
                    if kind == "chunk":
                        frame.update(
                            payload={"input": "large"},
                            media={
                                "kind": "inline",
                                "media_type": "application/octet-stream",
                            },
                        )
                        media = large_media[
                            (sequence - 1) * 1048576 : sequence * 1048576
                        ]
                    await connection.send(
                        native.encode_transport_v2_packet(json.dumps(frame), media)
                    )
                terminal, media = await output(connection)
                value = _OBJECT.validate_python(terminal["payload"], strict=True)
                assert terminal["kind"] == "completed" and not media
                assert value["input_media_bytes"] == len(large_media)
                assert (
                    value["input_media_sha256"]
                    == hashlib.sha256(large_media).hexdigest()
                )
                assert value["pid"] == expected_process
                end, _ = await output(connection)
                assert end["kind"] == "end"
            checks.append(
                "production-server-preserves-three-mib-streamed-input-under-per-frame-media-bounds"
            )

            async with (
                connect(
                    endpoint("Server"), additional_headers=headers, proxy=None
                ) as first,
                connect(
                    endpoint("Server"), additional_headers=headers, proxy=None
                ) as second,
            ):
                first_id = await start(first, '{"block":true,"cleanup_delay":0.1}')
                await output(first)
                second_id = await start(second, '{"delay":0.3}')
                await first.close()
                await output(second)
                result, _ = await output(second)
                assert result["kind"] == "completed"
                assert (
                    _OBJECT.validate_python(result["payload"], strict=True)["pid"]
                    == expected_process
                )
                await output(second)
                assert first_id != second_id
                async with asyncio.timeout(10):
                    while not (child_root / ("cleanup-" + first_id)).exists():
                        await asyncio.sleep(0.01)
            checks.append("one-disconnected-root-does-not-cancel-same-app-sibling")

            # Hold a legitimate Core consumer open after an invalid input. The
            # owner must cancel independently of consumer/connection teardown,
            # even when the app never reads its already-poisoned media socket.
            broker = NativeInferenceCalls(
                service_connection, settings.transport_node_id
            )
            independently_withdrawn = False

            async def hold_failed_input(port: NativeInferenceStream) -> None:
                nonlocal independently_withdrawn
                for sequence, kind in enumerate(("started", "chunk")):
                    frame: dict[str, JsonValue] = {
                        "call_id": port.call_id,
                        "direction": "caller_to_provider",
                        "sequence": sequence,
                        "kind": kind,
                    }
                    if kind == "chunk":
                        frame["payload"] = {"input": "invalid sequence"}
                        frame["sequence"] = 7
                    response = await port.peer.request(
                        "stream_input",
                        {"call_id": port.call_id},
                        native.encode_transport_v2_packet(json.dumps(frame), b""),
                    )
                    try:
                        assert isinstance(
                            response.frame, ResponseFrame
                        ) and response.frame.outcome == (
                            "complete" if kind == "started" else "unknown"
                        )
                    finally:
                        response.release()
                async with asyncio.timeout(5):
                    await asyncio.shield(port.control)
                assert (
                    read_private(child_root / ("cleanup-" + port.call_id)) == b"closed"
                )
                observation = await observe_native_inference(work, "streamer")
                scopes = _OBJECT.validate_python(observation["scopes"], strict=True)
                assert (
                    observation["dispatched_calls"] == 0
                    and scopes["active_scopes"] == 0
                )
                independently_withdrawn = True
                raise ValueError("expected failed input outcome")

            try:
                caller = broker.callers.authenticate(credential.token)
                try:
                    await broker.invoke_stream(
                        caller,
                        identifier,
                        "InferenceStreamerClient",
                        "1.0.0",
                        b'{"block_before_input":true}',
                        hold_failed_input,
                    )
                    raise AssertionError("failed input unexpectedly succeeded")
                except ValueError:
                    assert independently_withdrawn
            finally:
                await broker.close()
            checks.append(
                "owner-withdraws-unread-input-failure-before-core-consumer-closes"
            )

            async with connect(
                endpoint("Server"), additional_headers=headers, proxy=None
            ) as connection:
                call_id = await start(connection, '{"block":true}')
                await output(connection)
                await policy(False)
                try:
                    await connection.recv()
                    raise AssertionError("withdrawn authority delivered output")
                except ConnectionClosed as error:
                    assert error.code in (1008, 1011)
                async with asyncio.timeout(10):
                    while not (child_root / ("cleanup-" + call_id)).exists():
                        await asyncio.sleep(0.01)
            checks.append(
                "active-policy-withdrawal-cancels-and-drains-public-sdk-resources"
            )
            await policy(True)
            value = next(
                value
                for value in listings
                if value.target.capability_id == "InferenceStreamerUnary"
            )
            result = await owner.post(
                f"/v1/native-inference/{value.plugin_id}/{value.target.capability_id}/{value.target.version}/call",
                json={},
                headers=headers,
            )
            assert result.status_code == 200
            assert _OBJECT.validate_json(result.content)["pid"] == expected_process
            checks.append(
                "same-process-unary-reuse-after-streams-cancellation-and-revocation"
            )
            async with asyncio.timeout(10):
                while True:
                    observation = await observe_native_inference(work, "streamer")
                    scopes = _OBJECT.validate_python(observation["scopes"], strict=True)
                    if (
                        observation["retained_payload_bytes"] == 0
                        and observation["dispatched_calls"] == 0
                        and scopes["active_scopes"] == 0
                        and scopes["owned_tasks"] == 0
                    ):
                        break
                    await asyncio.sleep(0.05)
            checks.append("actual-owner-quiescent-payload-scope-and-task-custody")

    finally:
        stopped.set()
        try:
            if server is not None:
                await server
            else:
                network.close()
        finally:
            await manager.close()
    write_private(
        receipt,
        (
            json.dumps(
                {
                    "format": 1,
                    "passed": True,
                    "platform": measured.platform,
                    "runtime_digest": runtime_digest,
                    "skulk_build_sha256": measured.skulk_build_sha256,
                    "checks": checks,
                    "scope": "Real TCP bearer WebSocket; normal signed public-SDK application; same-host streaming. No provider-account, migration, routing or distributed claim.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    print(json.dumps({"passed": True, "checks": len(checks)}))


def main() -> None:
    """Run account-free installed stream acceptance in new task-owned paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("streamer", "work", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = _Arguments()
    parser.parse_args(namespace=arguments)
    asyncio.run(qualify(arguments.streamer, arguments.work, arguments.receipt))


if __name__ == "__main__":
    main()
