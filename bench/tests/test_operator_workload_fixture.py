"""Local fixture validation, including opt-in real signed connector traffic."""

import asyncio
import base64
import hashlib
import os
import socket
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast
from uuid import UUID

import hypercorn.asyncio as hypercorn_asyncio
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from pydantic import ValidationError

import bench.operator_workload_fixture as fixture_module
from bench.operator_fixture_app import generated_responses
from bench.operator_fixture_client import request_fixture
from bench.operator_fixture_observer import FixtureObserver, ObservationEvent
from bench.operator_workload_fixture import (
    FixtureLeaseExpiredError,
    FixtureSettings,
    copy_verified_fixture_binary,
    fixture_lease,
    isolated_fixture,
    verify_binary,
)
from skulk.operator.pairing import pairing_signature_message


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


async def test_only_the_session_deadline_is_normal_expiry() -> None:
    """Startup timeouts and teardown timeouts cannot become a successful expiry."""
    with pytest.raises(FixtureLeaseExpiredError):
        async with fixture_lease(0.01):
            await asyncio.sleep(1)
    with pytest.raises(TimeoutError, match="startup failed"):
        async with fixture_lease(1):
            raise TimeoutError("startup failed")
    with pytest.raises(TimeoutError, match="cleanup failed"):
        async with fixture_lease(0.01):
            try:
                await asyncio.sleep(1)
            finally:
                raise TimeoutError("cleanup failed")


async def test_external_cancellation_is_not_reported_as_expiry() -> None:
    """Caller cancellation retains its identity instead of forging a deadline."""

    async def wait() -> None:
        async with fixture_lease(10):
            await asyncio.sleep(10)

    task = asyncio.create_task(wait())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_fixture_refuses_unpinned_binary_and_unbounded_lifetime(tmp_path: Path) -> None:
    """Digest validation precedes provisioning; no hosted target parameter exists."""
    binary = tmp_path / "relay"
    binary.write_bytes(b"not a relay")
    binary.chmod(0o700)
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_binary(FixtureSettings(relay_binary=binary, relay_sha256="0" * 64))
    for duration in (0, 59, 7201):
        with pytest.raises(ValidationError):
            FixtureSettings(
                relay_binary=binary, relay_sha256="0" * 64, lifetime_seconds=duration
            )
    with pytest.raises(ValidationError):
        FixtureSettings.model_validate(
            {
                "relay_binary": binary,
                "relay_sha256": "0" * 64,
                "relay_url": "wss://example.invalid",
            }
        )


def test_fixture_executes_copied_verified_bytes_after_source_replacement(
    tmp_path: Path,
) -> None:
    """A build at the selected pathname cannot change this fixture's executable."""
    source = tmp_path / "source"
    source.write_bytes(b"selected-binary")
    source.chmod(0o700)
    settings = FixtureSettings(
        relay_binary=source,
        relay_sha256=hashlib.sha256(b"selected-binary").hexdigest(),
    )
    destination = copy_verified_fixture_binary(settings, tmp_path / "owned-copy")
    source.write_bytes(b"new-build")
    assert destination.read_bytes() == b"selected-binary"
    assert destination.stat().st_mode & 0o777 == 0o500
    with pytest.raises(ValueError, match="digest mismatch"):
        copy_verified_fixture_binary(settings, tmp_path / "rejected-copy")
    assert (tmp_path / "rejected-copy").stat().st_mode & 0o111 == 0


@pytest.mark.parametrize("observed", [False, True])
async def test_real_fixture_pairs_reads_and_cleans_up(
    observed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real Rust relay and Python auth, not released-app capacity.

    Uses a no-retry, loopback-only TLS-over-WebSocket protocol test adapter.
    """
    configured = os.environ.get("SKULK_PAIRED_RELAY_BINARY")
    if configured is None:
        pytest.skip("requires an explicitly selected local relay binary")
    binary = Path(configured).resolve(strict=True)
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    settings = FixtureSettings(
        relay_binary=binary, relay_sha256=digest, lifetime_seconds=60
    )
    events: list[ObservationEvent] = []
    observer = FixtureObserver(events.append) if observed else None
    listener_started = False
    original_serve = cast(Callable[..., Awaitable[None]], hypercorn_asyncio.serve)

    async def delayed_serve(
        app: ASGIFramework,
        config: Config,
        *,
        shutdown_trigger: Callable[[], Awaitable[object]],
    ) -> None:
        nonlocal listener_started
        await asyncio.sleep(0.5)
        listener_started = True
        await original_serve(app, config, shutdown_trigger=shutdown_trigger)

    monkeypatch.setattr(fixture_module, "_serve", delayed_serve)
    async with isolated_fixture(settings, observer=observer) as fixture:
        assert listener_started
        if observer is not None:
            observer.begin("cold-launch")
        directory = fixture.directory
        relay_port = fixture.relay_port
        gateway_port = fixture.configuration.operator_api_port
        assert (directory / "pairing.txt").stat().st_mode & 0o777 == 0o600
        package = fixture.pairing_service.create_session()
        remote = package.remote_access
        assert remote is not None
        assert (await request_fixture(remote, "GET", "/state")).status == 401
        key = Ed25519PrivateKey.generate()
        challenge = await request_fixture(
            remote,
            "POST",
            "/v1/auth/pairing-sessions/challenge",
            body={
                "nonce": package.nonce,
                "deviceName": "Synthetic protocol test",
                "devicePublicKey": _base64url(
                    key.public_key().public_bytes(
                        serialization.Encoding.Raw, serialization.PublicFormat.Raw
                    )
                ),
            },
        )
        assert challenge.status == 200
        proof = key.sign(
            pairing_signature_message(
                cluster_id=UUID(str(package.cluster_id)),
                nonce=package.nonce,
                challenge=str(challenge.body["challenge"]),
            )
        )
        exchange = await request_fixture(
            remote,
            "POST",
            "/v1/auth/pairing-sessions/exchange",
            body={
                "nonce": package.nonce,
                "signature": _base64url(proof),
            },
        )
        assert exchange.status == 200
        token = str(exchange.body["accessToken"])
        for path, expected in generated_responses().items():
            response = await request_fixture(remote, "GET", path, bearer=token)
            assert response.status == 200
            assert response.body == expected
        # The fixture cannot forward an arbitrary cluster mutation.
        assert (
            await request_fixture(
                remote, "POST", "/place_instance", body={}, bearer=token
            )
        ).status == 404
        if observer is not None:
            async with asyncio.timeout(3):
                while not observer.idle():
                    await asyncio.sleep(0.01)
            observer.end()
            assert (
                sum(event["type"] == "connection-open" for event in events)
                == len(generated_responses()) + 4
            )
            assert sum(event.get("outcome") == "failed" for event in events) == 2
            assert (
                sum(event.get("outcome") == "completed" for event in events)
                == len(generated_responses()) + 2
            )
    assert not directory.exists()
    # Two independent connection attempts after the context exits verify both
    # listeners are gone; this is local lifecycle evidence, not cloud inventory.
    for _ in range(2):
        for port in (relay_port, gateway_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.2)
                assert connection.connect_ex(("127.0.0.1", port)) != 0
        await asyncio.sleep(0.05)
