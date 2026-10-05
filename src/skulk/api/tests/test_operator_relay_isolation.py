# pyright: reportPrivateUsage=false
"""Tests that optional relay ingress cannot take down the local API."""

from collections.abc import Callable
from typing import cast, final

import anyio
import pytest
from fastapi.testclient import TestClient

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.api.operator_remote_access import OperatorRemoteAccessSupervisor
from skulk.operator.key_provider import AuthorityKeyUnavailableError
from skulk.operator.pairing import OperatorPairingService
from skulk.operator.relay import OperatorRelayConfiguration
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel


@final
class _UnavailablePairingService:
    """Pairing-service stand-in with unreadable protected relay state."""

    def relay_configuration(self) -> OperatorRelayConfiguration | None:
        """Raise the expected protected-key availability failure."""

        raise AuthorityKeyUnavailableError("credential-shaped-secret")


def _build_api(pairing_service: OperatorPairingService | None = None) -> API:
    """Create a minimal API component for relay-isolation tests."""

    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("local-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        operator_pairing_service=pairing_service,
    )


def test_unreadable_relay_state_does_not_prevent_local_api_startup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Damaged optional authority material leaves canonical local reads usable."""

    service = cast(
        OperatorPairingService,
        cast(object, _UnavailablePairingService()),
    )
    api = _build_api(service)

    assert api.operator_remote_access_state == "not_configured"
    assert TestClient(api.app).get("/state").status_code == 200
    assert "credential-shaped-secret" not in caplog.text


async def test_operator_listener_failure_is_isolated_from_local_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relay listener failure cancels its lanes and leaves the local API up."""

    api = _build_api()
    configuration = OperatorRelayConfiguration.model_construct(
        routing_locator="a" * 43,
        operator_api_port=52417,
        gateway_server_name="skulk-isolation.remote",
        connector_authority_key_id=None,
    )
    connector_started = anyio.Event()
    connector_stopped = anyio.Event()

    @final
    class _GenerationService:
        """Pairing-service stand-in exposing the connector generation callback."""

        @staticmethod
        def reserve_relay_connector_generation() -> int:
            """Return one synthetic durable generation."""

            return 1

    api._operator_pairing_service = cast(
        OperatorPairingService,
        cast(object, _GenerationService()),
    )

    @final
    class _BlockingConnector:
        """Connector stand-in that records sibling cancellation."""

        async def run(self) -> None:
            """Block until the supervising task group cancels this lane."""

            connector_started.set()
            try:
                await anyio.sleep_forever()
            finally:
                connector_stopped.set()

    async def _failing_operator_api(
        _configuration: OperatorRelayConfiguration,
        _event: anyio.Event,
    ) -> None:
        await connector_started.wait()
        raise OSError("private-listener-detail")

    def _connector_factory(
        received: OperatorRelayConfiguration,
        *,
        next_connector_generation: Callable[[], int],
    ) -> _BlockingConnector:
        assert received is configuration
        assert callable(next_connector_generation)
        return _BlockingConnector()

    monkeypatch.setattr(api, "run_operator_api", _failing_operator_api)
    monkeypatch.setattr(
        api_main,
        "OperatorGatewayConnector",
        _connector_factory,
    )

    supervisor = OperatorRemoteAccessSupervisor(
        load_configuration=lambda: configuration,
        run_session=api._run_operator_remote_access_session,
        poll_seconds=0.01,
        minimum_failure_retry_seconds=60.0,
    )
    api._operator_remote_access = supervisor
    shutdown = anyio.Event()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(api.run_operator_remote_access, shutdown)
        with anyio.fail_after(5):
            await connector_stopped.wait()
            while api.operator_remote_access_state != "failed":
                await anyio.sleep(0.005)
        shutdown.set()

    assert TestClient(api.app).get("/state").status_code == 200
