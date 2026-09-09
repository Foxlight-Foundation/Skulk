"""Opt-in loopback fixture for observing unchanged released operator clients.

Creates only fresh local authority and relay state, with a bounded lifetime.
The physical-device workload recorder is a separate integration; this command
does not claim capacity qualification or export captured user data.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import ssl
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol, cast, final

import aiohttp
import hypercorn.asyncio as hypercorn_asyncio
import qrcode
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from pydantic import BaseModel, ConfigDict, Field
from qrcode.constants import ERROR_CORRECT_L

from bench.operator_fixture_app import create_fixture_app
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import OperatorPairingService
from skulk.operator.relay import (
    OperatorGatewayConnector,
    OperatorRelayConfiguration,
    OperatorRelayConfigurationRepository,
    OperatorRelayProvisioning,
)


class _HypercornServe(Protocol):
    def __call__(
        self,
        app: ASGIFramework,
        config: Config,
        *,
        shutdown_trigger: Callable[[], Awaitable[object]] | None = None,
        mode: Literal["asgi", "wsgi"] | None = None,
    ) -> Coroutine[object, object, None]: ...


_serve = cast(_HypercornServe, hypercorn_asyncio.serve)


class FixtureSettings(BaseModel):
    """Local-only fixture inputs; no remote URL or existing authority is accepted."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    relay_binary: Path = Field(description="Explicit local on-demand relay executable.")
    relay_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Expected executable digest."
    )
    lifetime_seconds: int = Field(
        default=600, ge=60, le=7200, description="Whole fixture lease, including setup."
    )


@dataclass(frozen=True)
class RunningFixture:
    """Ephemeral local handles; never serialize this object as campaign evidence."""

    directory: Path
    relay_port: int
    configuration: OperatorRelayConfiguration
    pairing_service: OperatorPairingService


class FixtureLeaseExpiredError(Exception):
    """The whole-session deadline cancelled an otherwise clean fixture."""


@asynccontextmanager
async def fixture_lease(lifetime_seconds: float) -> AsyncIterator[None]:
    """Translate only this lease's cancellation, never inner or cleanup timeouts.

    An inner TimeoutError must retain its failure status even if cleanup runs
    past the session deadline. Catch the lease cancellation before asyncio
    converts it to the indistinguishable builtin TimeoutError.
    """
    async with asyncio.timeout(lifetime_seconds) as lease:
        try:
            yield
        except asyncio.CancelledError:
            if lease.expired():
                raise FixtureLeaseExpiredError from None
            raise


@final
class _Tls13Configuration(Config):
    def create_ssl_context(self) -> ssl.SSLContext | None:
        """Require pinned inner TLS 1.3, including in this synthetic listener."""
        context = super().create_ssl_context()
        if context is not None:
            context.minimum_version = ssl.TLSVersion.TLSv1_3
        return context


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(tuple[str, int], listener.getsockname())[1]


def verify_binary(settings: FixtureSettings) -> Path:
    """Verify the exact executable before creating authority, files, or sockets."""
    binary = settings.relay_binary.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("fixture relay must be an executable file")
    with binary.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != settings.relay_sha256:
        raise ValueError("fixture relay digest mismatch")
    return binary


def _private_file(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(value)


def _private_qr(path: Path, value: str) -> None:
    code = qrcode.QRCode(border=4, error_correction=ERROR_CORRECT_L)
    code.add_data(value)
    code.make(fit=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        code.make_image().save(output)


async def _wait_ready(port: int, guardian: asyncio.subprocess.Process) -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as client:
        for _ in range(100):
            if guardian.returncode is not None:
                raise RuntimeError("fixture relay exited before readiness")
            try:
                async with client.get(
                    f"http://127.0.0.1:{port}/readyz", allow_redirects=False
                ) as response:
                    if response.status == 204:
                        return
            except (aiohttp.ClientError, TimeoutError):
                pass
            await asyncio.sleep(0.05)
    raise RuntimeError("fixture relay did not become ready")


async def _stop_guardian(guardian: asyncio.subprocess.Process) -> None:
    if guardian.stdin is not None:
        guardian.stdin.close()
    # The independent guardian owns termination and escalation of its relay.
    # Do not kill it before it has had time to reap that child.
    await asyncio.wait_for(guardian.wait(), timeout=8)


async def _watch_guardian(guardian: asyncio.subprocess.Process) -> None:
    await guardian.wait()
    raise RuntimeError("fixture relay lease ended unexpectedly")


@asynccontextmanager
async def isolated_fixture(settings: FixtureSettings) -> AsyncIterator[RunningFixture]:
    """Start real local relay/gateway/auth with synthetic API bodies, then reap all.

    The context cancels at its whole-session deadline. Temporary keys are never
    printed and are removed after the gateway and relay stop. A separate relay
    watchdog also observes parent death and expiry. No production URL is accepted.
    """
    binary = verify_binary(settings)
    deadline = asyncio.get_running_loop().time() + settings.lifetime_seconds
    async with fixture_lease(settings.lifetime_seconds):
        with TemporaryDirectory(prefix="skulk-operator-fixture-") as temporary:
            directory = Path(temporary)
            relay_port, gateway_port = _available_port(), _available_port()
            while gateway_port == relay_port:
                gateway_port = _available_port()
            relay_path = directory / "relay.json"
            provisioning_path = directory / "provisioning.json"
            provisioning_process = await asyncio.create_subprocess_exec(
                str(binary),
                "provision-on-demand",
                f"ws://127.0.0.1:{relay_port}",
                str(relay_path),
                str(provisioning_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                if await asyncio.wait_for(provisioning_process.wait(), 10) != 0:
                    raise RuntimeError("fixture provisioning failed")
            finally:
                if provisioning_process.returncode is None:
                    provisioning_process.kill()
                    await provisioning_process.wait()
            document = cast(dict[str, object], json.loads(relay_path.read_text()))
            document["bind"] = f"127.0.0.1:{relay_port}"
            relay_path.write_text(json.dumps(document))
            provider = LocalFileAuthorityKeyProvider(directory / "authority-key.bin")
            authority = EncryptedAuthorityStore(
                provider, directory / "authority.sqlite3"
            )
            repository = OperatorRelayConfigurationRepository(
                authority,
                certificate_path=directory / "tls.pem",
                private_key_path=directory / "tls-key.pem",
            )
            service = OperatorPairingService(
                authority,
                provider,
                relay_repository=repository,
            )
            configuration = service.configure_relay(
                OperatorRelayProvisioning.model_validate_json(
                    provisioning_path.read_text()
                ),
                operator_api_port=gateway_port,
                cluster_name="Fixture",
            )
            configuration.server_ssl_context()
            server = _Tls13Configuration()
            server.bind = [f"127.0.0.1:{gateway_port}"]
            server.certfile = str(configuration.certificate_path)
            server.keyfile = str(configuration.private_key_path)
            server.accesslog = None
            server.errorlog = None
            server.alpn_protocols = ["http/1.1"]
            server.graceful_timeout = 2
            server.shutdown_timeout = 2
            server.read_timeout = 20
            server.ssl_handshake_timeout = 10
            shutdown = asyncio.Event()
            guardian = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "bench.operator_fixture_lease",
                str(binary),
                str(relay_path),
                str(max(0.001, deadline - asyncio.get_running_loop().time())),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                async with asyncio.TaskGroup() as group:
                    monitor = group.create_task(_watch_guardian(guardian))
                    listener = group.create_task(
                        _serve(
                            cast(ASGIFramework, create_fixture_app(service)),
                            server,
                            shutdown_trigger=shutdown.wait,
                        )
                    )
                    connector = group.create_task(
                        OperatorGatewayConnector(
                            configuration,
                            next_connector_generation=service.reserve_relay_connector_generation,
                        ).run()
                    )
                    try:
                        await _wait_ready(relay_port, guardian)
                        invitation = service.create_invitation(
                            lifetime=timedelta(seconds=settings.lifetime_seconds),
                            max_pairings=4,
                        )
                        _private_file(directory / "pairing.txt", invitation.as_url())
                        _private_qr(directory / "pairing.png", invitation.as_url())
                        yield RunningFixture(
                            directory, relay_port, configuration, service
                        )
                    finally:
                        shutdown.set()
                        connector.cancel()
                        monitor.cancel()
                        await asyncio.wait_for(listener, timeout=5)
            finally:
                await asyncio.shield(_stop_guardian(guardian))


async def _run(settings: FixtureSettings) -> None:
    try:
        async with isolated_fixture(settings) as fixture:
            print(
                json.dumps(
                    {
                        "schema": "operator-fixture-ready.v1",
                        "synthetic": True,
                        "pairingFile": str(fixture.directory / "pairing.txt"),
                        "pairingQr": str(fixture.directory / "pairing.png"),
                        "relayPort": fixture.relay_port,
                        "lifetimeSeconds": settings.lifetime_seconds,
                        "capacityQualified": False,
                    }
                ),
                flush=True,
            )
            await asyncio.Event().wait()
    except FixtureLeaseExpiredError:
        print(
            '{"schema":"operator-fixture-stopped.v1","reason":"lease-expired"}',
            flush=True,
        )


def main(arguments: Sequence[str] | None = None) -> None:
    """Run a local, expiring fixture pinned to an explicit relay binary digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-binary", type=Path, required=True)
    parser.add_argument("--relay-sha256", required=True)
    parser.add_argument("--lifetime-seconds", type=int, default=600)
    options = vars(parser.parse_args(arguments))
    settings = FixtureSettings.model_validate(options)
    asyncio.run(_run(settings))


if __name__ == "__main__":
    main()
