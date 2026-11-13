"""Actual Unix host role proofs and authenticated byte/resource boundaries."""

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.native_host_binding import CoreHostKey, provision_native_host
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import (
    AuthenticatedStream,
    HostBinding,
    RequestFrame,
    admit_core,
)
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_files import read_private, write_private


def binding(key: Ed25519PrivateKey) -> HostBinding:
    """Independent synthetic profile, unrelated to any account or installed app."""
    return HostBinding(
        protocol=1,
        profile_id="a" * 32,
        plugin_id="managed.example",
        runtime_digest="b" * 64,
        transport_node_id="serving-node",
        service_instance="c" * 32,
        core_public_key=key.public_key().public_bytes_raw().hex(),
        server_secret="d" * 64,
    )


@dataclass(frozen=True)
class Channels:
    """Two actual peers after the real mutual challenge."""

    core: AuthenticatedStream
    owner: AuthenticatedStream
    core_budget: NativePayloadBudget
    owner_budget: NativePayloadBudget


async def test_roles_preserve_exact_body_and_owned_capacity(channels: Channels) -> None:
    """Host role proof alone does not modify exact body bytes or release their lease."""
    payload = b'{ "caller_id": "untrusted data only" }'
    await channels.core.send(
        RequestFrame(
            request_id=1, operation="invoke", parameters={}, payload_size=len(payload)
        ),
        payload,
        check=lambda: None,
    )
    frame = await channels.owner.receive()
    try:
        assert frame.payload == payload
        assert channels.owner_budget.used == len(payload)
        assert channels.core.keys.outbound != channels.owner.keys.outbound
    finally:
        frame.release()
        frame.release()


@pytest.mark.parametrize("fault", ["replay", "payload", "reflection"])
async def test_corrupt_or_replayed_frame_cannot_reach_a_handler(
    channels: Channels, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Capture real writes, then inject stale/damaged/wrong-direction bytes."""
    source = channels.owner if fault == "reflection" else channels.core
    receiver = channels.core if fault == "reflection" else channels.owner
    captured: list[bytes] = []
    write = source.writer.write
    payload = b'{"value":1}'

    def observe(data: bytes) -> None:
        captured.append(data)
        # Fault injection changes only the body on its first physical write;
        # the real sender's sequence, metadata and authentication stay intact.
        write(data[:-1] + b" " if fault == "payload" and data == payload else data)

    monkeypatch.setattr(source.writer, "write", observe)
    await source.send(
        RequestFrame(
            request_id=1, operation="invoke", parameters={}, payload_size=len(payload)
        ),
        payload,
        check=lambda: None,
    )
    if fault == "payload":
        with pytest.raises(ValueError, match="authentication"):
            await receiver.receive()
        assert receiver.closed
        return
    frame = await receiver.receive()
    frame.release()
    packet = bytearray(b"".join(captured))
    if fault == "reflection":
        target = channels.core
        destination = channels.owner
    else:
        target, destination = source, receiver
    target.writer.write(bytes(packet))
    await target.writer.drain()
    with pytest.raises(ValueError, match="sequence|authentication"):
        await destination.receive()
    assert destination.closed


async def test_queued_policy_refusal_sends_no_bytes_and_keeps_channel(
    channels: Channels,
) -> None:
    """A policy decision before writer queueing cannot be reused after revocation."""
    await channels.core.write_lock.acquire()
    permitted = True
    checked = False

    def check() -> None:
        nonlocal checked
        checked = True
        if not permitted:
            raise ValueError("revoked")

    task = asyncio.create_task(
        channels.core.send(
            RequestFrame(
                request_id=1, operation="describe", parameters={}, payload_size=0
            ),
            b"",
            check=check,
        )
    )
    try:
        await asyncio.sleep(0)
        assert not checked
        permitted = False
        channels.core.write_lock.release()
        with pytest.raises(ValueError, match="revoked"):
            await task
        assert not channels.core.closed
        await channels.core.send(
            RequestFrame(
                request_id=2, operation="describe", parameters={}, payload_size=0
            ),
            b"",
            check=lambda: None,
        )
        actual = await channels.owner.receive()
        try:
            assert actual.frame.request_id == 2
        finally:
            actual.release()
    finally:
        if channels.core.write_lock.locked():
            channels.core.write_lock.release()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_normal_launcher_provisions_private_profile_and_generation(
    tmp_path: Path,
) -> None:
    """The actual launcher sidecar binds runtime/lifetime and excludes the Core private key."""
    settings = HostSettings(profile_id="a" * 32, transport_node_id="serving-node")
    write_private(tmp_path / "host.json", settings.model_dump_json().encode())
    installation = tmp_path / "installations/managed.example"
    provision_native_host(installation, "b" * 64, "c" * 32)
    first = HostBinding.model_validate_json(
        read_private(installation / "core-host.json")
    )
    key = CoreHostKey.model_validate_json(
        read_private(tmp_path / "native-host-key.json")
    )
    assert key.private_key not in read_private(installation / "core-host.json").decode()
    assert first.runtime_digest == "b" * 64
    provision_native_host(installation, "e" * 64, "f" * 32)
    second = HostBinding.model_validate_json(
        read_private(installation / "core-host.json")
    )
    assert second.server_secret != first.server_secret
    assert second.core_public_key == first.core_public_key
    assert second.digest != first.digest


@pytest.mark.parametrize(
    "fault", ["replayed_proof", "wrong_signature", "stale_lifetime"]
)
async def test_role_proof_cannot_be_replayed_or_outlive_its_attachment(
    fault: str,
) -> None:
    """Actual fresh challenges refuse stolen signatures, other keys and stale custody."""
    key = Ed25519PrivateKey.generate()
    expected = binding(key)
    stale = False
    outcomes: asyncio.Queue[bool] = asyncio.Queue()
    tasks: set[asyncio.Task[None]] = set()

    def current() -> None:
        if stale:
            raise ValueError("synthetic lifetime changed")

    async def accept(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        try:
            await admit_core(reader, writer, expected, current)
        except (OSError, ValueError):
            outcomes.put_nowait(False)
        else:
            outcomes.put_nowait(True)
        finally:
            writer.close()
            tasks.discard(task)

    with tempfile.TemporaryDirectory(prefix="role-proof-", dir="/tmp") as directory:
        path = Path(directory) / "host.sock"
        server = await asyncio.start_unix_server(accept, path)
        saved: str | None = None
        try:
            for first in (True, False):
                async with asyncio.timeout(3):
                    reader, writer = await asyncio.open_unix_connection(path)
                    try:
                        nonce = "f" * 64
                        writer.write(
                            (
                                json.dumps(
                                    {
                                        "protocol": 1,
                                        "binding_digest": expected.digest,
                                        "nonce": nonce,
                                    }
                                )
                                + "\n"
                            ).encode()
                        )
                        await writer.drain()
                        challenge = TypeAdapter(dict[str, JsonValue]).validate_json(
                            await reader.readline()
                        )
                        server_nonce = TypeAdapter(str).validate_python(
                            challenge["nonce"], strict=True
                        )
                        transcript = (
                            expected.transcript()
                            + b"\0"
                            + nonce.encode()
                            + b"\0"
                            + server_nonce.encode()
                        )
                        signature = key.sign(
                            b"skulk.native-host.core/1\0" + transcript
                        ).hex()
                        if not first:
                            if fault == "replayed_proof":
                                assert saved is not None
                                signature = saved
                            elif fault == "wrong_signature":
                                signature = (
                                    Ed25519PrivateKey.generate()
                                    .sign(b"skulk.native-host.core/1\0" + transcript)
                                    .hex()
                                )
                            else:
                                stale = True
                        writer.write(
                            (
                                json.dumps({"protocol": 1, "signature": signature})
                                + "\n"
                            ).encode()
                        )
                        await writer.drain()
                        ready = await reader.readline()
                        assert bool(ready) is first
                        assert await outcomes.get() is first
                        saved = signature
                    finally:
                        writer.close()
                        await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
