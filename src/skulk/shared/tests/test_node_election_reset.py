"""Regression coverage for election-driven event-router replacement ordering."""

# pyright: reportAny=false, reportPrivateUsage=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

import skulk.main as main_module
from skulk.api.main import API
from skulk.download.coordinator import DownloadCoordinator
from skulk.main import Node
from skulk.routing.event_router import EventRouter
from skulk.routing.router import Router
from skulk.shared.election import Election, ElectionResult
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSnapshot, StateSyncMessage
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.store.config import ModelStoreConfig, SkulkConfig
from skulk.store.model_store_client import ModelStoreClient
from skulk.store.model_store_server import ModelStoreServer
from skulk.utils.channels import channel
from skulk.worker.main import Worker


@dataclass
class _FakeTaskGroup:
    events: list[str]

    def start_soon(self, func: Any, *args: Any, name: object = None) -> None:
        owner = getattr(func, "__self__", None)
        owner_name = type(owner).__name__ if owner is not None else ""
        label = f"{owner_name}.{func.__name__}".strip(".")
        self.events.append(f"start:{label}")


class _FakeRouter:
    def sender(self, _topic: object) -> object:
        return object()

    def telemetry_sender(self) -> object:
        return object()

    def receiver(self, _topic: object) -> object:
        return object()

    def receiver_with_origin(self, _topic: object) -> object:
        return object()

    def current_session_connections(self) -> dict[str, tuple[str, int, int]]:
        return {}


class _OldEventRouter:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def shutdown(self) -> None:
        self._events.append("old_event_router.shutdown")


class _OldWorker:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        # The promotion path reads the outgoing worker's replicated state to
        # seed the new master's session (#273) — the stub carries the same
        # contract as the real Worker.
        self.state = State()

    async def shutdown(self) -> None:
        self._events.append("old_worker.shutdown")


class _OldStoreServer:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def stop(self) -> None:
        self._events.append("old_store_server.stop")


class _FakeApi:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def reset(
        self,
        _result_clock: int,
        _event_receiver: object,
        _master_node_id: NodeId,
    ) -> None:
        self._events.append("api.reset")

    def unpause(
        self,
        _result_clock: int,
        master_node_id: NodeId | None = None,
    ) -> None:
        self._events.append("api.unpause")

    def set_runner_diagnostics_provider(self, _provider: object) -> None:
        self._events.append("api.runner_diagnostics_provider")

    def set_runner_cancel_provider(self, _provider: object) -> None:
        self._events.append("api.runner_cancel_provider")

    def set_vision_media_ingress_provider(self, _provider: object) -> None:
        self._events.append("api.vision_media_ingress_provider")

    def refresh_config_dependent_capabilities(self) -> None:
        self._events.append("api.refresh_config_dependent_capabilities")

    def set_model_store_runtime(
        self,
        _skulk_config: SkulkConfig | None,
        _store_client: ModelStoreClient | None,
    ) -> None:
        self._events.append("api.set_model_store_runtime")


@pytest.mark.asyncio
async def test_election_restarts_event_router_after_receivers_are_rewired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_events: list[str] = []

    class NewEventRouter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_event_router.created")

        def sender(self) -> object:
            order_events.append("new_event_router.sender")
            return object()

        def receiver(self) -> object:
            order_events.append("new_event_router.receiver")
            return object()

        async def run(self) -> None:
            return

    class NewWorker:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_worker.created")

        async def run(self) -> None:
            return

        def collect_runner_diagnostics(self) -> list[object]:
            return []

        async def cancel_runner_task(self, *_args: Any, **_kwargs: Any) -> object:
            return object()

        def collect_vision_media_ingress_diagnostics(self) -> object:
            return object()

    monkeypatch.setattr("skulk.main.EventRouter", NewEventRouter)
    monkeypatch.setattr("skulk.main.Worker", NewWorker)
    async def _fake_request_cluster_config(
        *_args: Any, **_kwargs: Any
    ) -> str | None:
        return None

    monkeypatch.setattr(Node, "_request_cluster_config", _fake_request_cluster_config)

    election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, cast(object, _FakeRouter())),
        event_router=cast(EventRouter, cast(object, _OldEventRouter(order_events))),
        download_coordinator=None,
        worker=cast(Worker, cast(object, _OldWorker(order_events))),
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=cast(API, cast(object, _FakeApi(order_events))),
        node_id=NodeId("self"),
        offline=False,
        skulk_config=None,
        store_client=None,
        store_server=None,
        telemetry_view=TelemetryView(),
        telemetry_receiver=channel[NodeTelemetry]()[1],
    )
    node._tg = _FakeTaskGroup(order_events)  # pyright: ignore[reportAttributeAccessIssue]

    await election_sender.send(
        ElectionResult(
            session_id=SessionId(master_node_id=NodeId("other"), election_clock=7),
            won_clock=7,
            is_new_master=True,
        )
    )
    election_sender.close()

    await node._elect_loop()

    assert order_events.index("new_event_router.receiver") < order_events.index(
        "api.reset"
    )
    assert order_events.index("api.reset") < order_events.index(
        "start:NewEventRouter.run"
    )
    assert order_events.index("new_worker.created") < order_events.index(
        "start:NewEventRouter.run"
    )


@pytest.mark.asyncio
async def test_election_attaches_capacity_guard_before_coordinator_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire staging admission before the replacement coordinator can receive."""
    order_events: list[str] = []

    class NewEventRouter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def sender(self) -> object:
            return object()

        def receiver(self) -> object:
            return object()

        async def run(self) -> None:
            return

    class NewWorker:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_worker.created")

        async def run(self) -> None:
            return

        async def prepare_staging_transfer(self, *_args: Any) -> None:
            return

        def collect_runner_diagnostics(self) -> list[object]:
            return []

        async def cancel_runner_task(self, *_args: Any, **_kwargs: Any) -> object:
            return object()

        def collect_vision_media_ingress_diagnostics(self) -> object:
            return object()

    class NewModelStoreDownloader:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def set_staging_capacity_callback(self, _callback: object) -> None:
            order_events.append("staging_capacity_callback.attached")

    class NewDownloadCoordinator:
        def __init__(
            self,
            _node_id: NodeId,
            shard_downloader: object,
            **_kwargs: Any,
        ) -> None:
            self.shard_downloader = shard_downloader

        async def run(self) -> None:
            return

    class OldDownloadCoordinator:
        async def shutdown(self) -> None:
            order_events.append("old_download_coordinator.shutdown")

    monkeypatch.setattr("skulk.main.EventRouter", NewEventRouter)
    monkeypatch.setattr("skulk.main.Worker", NewWorker)
    monkeypatch.setattr("skulk.main.ModelStoreDownloader", NewModelStoreDownloader)
    monkeypatch.setattr("skulk.main.DownloadCoordinator", NewDownloadCoordinator)

    def _fake_shard_downloader(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(
        "skulk.main.skulk_shard_downloader",
        _fake_shard_downloader,
    )

    async def _fake_request_cluster_config(
        *_args: Any, **_kwargs: Any
    ) -> str | None:
        return None

    async def _skip_config_broadcast(*_args: Any, **_kwargs: Any) -> None:
        return

    monkeypatch.setattr(Node, "_request_cluster_config", _fake_request_cluster_config)
    monkeypatch.setattr(
        Node, "_broadcast_config_if_store_host", _skip_config_broadcast
    )

    election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, cast(object, _FakeRouter())),
        event_router=cast(EventRouter, cast(object, _OldEventRouter(order_events))),
        download_coordinator=cast(
            DownloadCoordinator, cast(object, OldDownloadCoordinator())
        ),
        worker=cast(Worker, cast(object, _OldWorker(order_events))),
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=cast(API, cast(object, _FakeApi(order_events))),
        node_id=NodeId("self"),
        offline=False,
        skulk_config=SkulkConfig(
            model_store=ModelStoreConfig(
                store_host="store",
                store_path="/unused/store",
            )
        ),
        store_client=cast(ModelStoreClient, object()),
        store_server=None,
        telemetry_view=TelemetryView(),
        telemetry_receiver=channel[NodeTelemetry]()[1],
    )
    node._tg = _FakeTaskGroup(order_events)  # pyright: ignore[reportAttributeAccessIssue]

    await election_sender.send(
        ElectionResult(
            session_id=SessionId(master_node_id=NodeId("other"), election_clock=8),
            won_clock=8,
            is_new_master=True,
        )
    )
    election_sender.close()

    await node._elect_loop()

    assert order_events.index(
        "staging_capacity_callback.attached"
    ) < order_events.index("start:NewDownloadCoordinator.run")


@pytest.mark.asyncio
async def test_new_master_does_not_wait_on_unavailable_state_sync_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_events: list[str] = []

    class NewEventRouter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_event_router.created")

        def sender(self) -> object:
            return object()

        def receiver(self) -> object:
            return object()

        async def run(self) -> None:
            return

    class NewMaster:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_master.created")

        async def run(self) -> None:
            return

    class NewWorker:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_worker.created")

        async def run(self) -> None:
            return

        def collect_runner_diagnostics(self) -> list[object]:
            return []

        async def cancel_runner_task(self, *_args: Any, **_kwargs: Any) -> object:
            return object()

        def collect_vision_media_ingress_diagnostics(self) -> object:
            return object()

    monkeypatch.setattr("skulk.main.EventRouter", NewEventRouter)
    monkeypatch.setattr("skulk.main.Master", NewMaster)
    monkeypatch.setattr("skulk.main.Worker", NewWorker)

    async def _unexpected_request_cluster_config(
        *_args: Any, **_kwargs: Any
    ) -> str | None:
        raise AssertionError("new master should not request config before startup")

    monkeypatch.setattr(
        Node, "_request_cluster_config", _unexpected_request_cluster_config
    )

    election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, cast(object, _FakeRouter())),
        event_router=cast(EventRouter, cast(object, _OldEventRouter(order_events))),
        download_coordinator=None,
        worker=cast(Worker, cast(object, _OldWorker(order_events))),
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=cast(API, cast(object, _FakeApi(order_events))),
        node_id=NodeId("self"),
        offline=False,
        skulk_config=None,
        store_client=None,
        store_server=None,
        telemetry_view=TelemetryView(),
        telemetry_receiver=channel[NodeTelemetry]()[1],
    )
    node._tg = _FakeTaskGroup(order_events)  # pyright: ignore[reportAttributeAccessIssue]

    await election_sender.send(
        ElectionResult(
            session_id=SessionId(master_node_id=NodeId("self"), election_clock=8),
            won_clock=8,
            is_new_master=True,
        )
    )
    election_sender.close()

    await node._elect_loop()

    assert "new_master.created" in order_events


class _StateSyncOnlyRouter:
    def __init__(self) -> None:
        self.request_sender, self.request_receiver = channel[StateSyncMessage]()
        self.response_sender, self.response_receiver = channel[
            tuple[str | None, StateSyncMessage]
        ]()

    def sender(self, _topic: object) -> object:
        return self.request_sender.clone()

    def receiver_with_origin(self, _topic: object) -> object:
        return self.response_receiver


@pytest.mark.asyncio
async def test_request_cluster_config_ignores_non_master_origin() -> None:
    router = _StateSyncOnlyRouter()
    _election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, cast(object, router)),
        event_router=cast(EventRouter, object()),
        download_coordinator=None,
        worker=None,
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=None,
        node_id=NodeId("self"),
        offline=False,
        skulk_config=None,
        store_client=None,
        store_server=None,
        telemetry_view=TelemetryView(),
        telemetry_receiver=channel[NodeTelemetry]()[1],
    )

    session_id = SessionId(master_node_id=NodeId("master"), election_clock=9)
    empty_snapshot = StateSnapshot(
        session_id=session_id,
        last_event_applied_idx=-1,
        state=State(last_event_applied_idx=-1),
    )
    result: dict[str, str | None] = {}
    request_finished = anyio.Event()

    async def _request() -> None:
        result["config"] = await node._request_cluster_config(session_id)
        request_finished.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_request)

        request = await router.request_receiver.receive()
        assert request.kind == "request"
        assert request.session_id == session_id

        await router.response_sender.send(
            (
                "not-the-master",
                StateSyncMessage(
                    kind="response",
                    requester=request.requester,
                    session_id=session_id,
                    snapshot=empty_snapshot,
                    config_yaml="bad: true\n",
                ),
            )
        )
        await router.response_sender.send(
            (
                str(session_id.master_node_id),
                StateSyncMessage(
                    kind="response",
                    requester=request.requester,
                    session_id=session_id,
                    snapshot=empty_snapshot,
                    config_yaml="good: true\n",
                ),
            )
        )

        await request_finished.wait()
        assert result["config"] == "good: true\n"


@pytest.mark.asyncio
async def test_request_cluster_config_survives_delayed_master_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _StateSyncOnlyRouter()
    _election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, cast(object, router)),
        event_router=cast(EventRouter, object()),
        download_coordinator=None,
        worker=None,
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=None,
        node_id=NodeId("self"),
        offline=False,
        skulk_config=None,
        store_client=None,
        store_server=None,
        telemetry_view=TelemetryView(),
        telemetry_receiver=channel[NodeTelemetry]()[1],
    )
    monkeypatch.setattr(main_module, "_CLUSTER_CONFIG_SYNC_ATTEMPTS", 6)
    monkeypatch.setattr(
        main_module,
        "_CLUSTER_CONFIG_SYNC_RESPONSE_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        main_module,
        "_CLUSTER_CONFIG_SYNC_RETRY_INTERVAL_SECONDS",
        0.001,
    )

    session_id = SessionId(master_node_id=NodeId("master"), election_clock=10)
    snapshot = StateSnapshot(
        session_id=session_id,
        last_event_applied_idx=-1,
        state=State(last_event_applied_idx=-1),
    )
    result: dict[str, str | None] = {}

    async def _request() -> None:
        result["config"] = await node._request_cluster_config(session_id)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_request)
        for attempt in range(4):
            request = await router.request_receiver.receive()
            if attempt == 3:
                await router.response_sender.send(
                    (
                        str(session_id.master_node_id),
                        StateSyncMessage(
                            kind="response",
                            requester=request.requester,
                            session_id=session_id,
                            snapshot=snapshot,
                            config_yaml="model_store: {}\n",
                        ),
                    )
                )

        while "config" not in result:
            await anyio.sleep(0)

        assert result["config"] == "model_store: {}\n"
        tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_authoritative_config_rewires_api_and_stops_superseded_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    config_path = tmp_path / "skulk.yaml"
    monkeypatch.setattr(main_module, "resolve_config_path", lambda: config_path)
    replacement_client = cast(ModelStoreClient, object())

    def _replacement_runtime(
        _node_id: NodeId,
        _config: SkulkConfig | None,
    ) -> tuple[ModelStoreClient, None]:
        return replacement_client, None

    monkeypatch.setattr(
        main_module,
        "_configure_model_store_runtime",
        _replacement_runtime,
    )

    _election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, object()),
        event_router=cast(EventRouter, object()),
        download_coordinator=None,
        worker=None,
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=cast(API, cast(object, _FakeApi(events))),
        node_id=NodeId("worker"),
        offline=False,
        skulk_config=SkulkConfig(
            model_store=ModelStoreConfig(
                store_host="worker",
                store_path="/old-store",
            )
        ),
        store_client=cast(ModelStoreClient, object()),
        store_server=cast(ModelStoreServer, cast(object, _OldStoreServer(events))),
        telemetry_view=TelemetryView(),
        telemetry_receiver=channel[NodeTelemetry]()[1],
    )

    await node._apply_authoritative_cluster_config(
        "model_store:\n"
        "  store_host: elected-master\n"
        "  store_http_host: 192.0.2.44\n"
        "  store_path: /master-store\n"
    )

    assert node.store_client is replacement_client
    assert node.store_server is None
    assert node.skulk_config is not None
    assert node.skulk_config.model_store is not None
    assert node.skulk_config.model_store.store_host == "elected-master"
    assert events == [
        "old_store_server.stop",
        "api.set_model_store_runtime",
    ]
