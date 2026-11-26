"""The engines report: what every node said about engine packages, grouped by engine."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import skulk.api.main as api_main
from skulk.api.engines import EngineChangeResponse, EnginesResponse, engines_view
from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import (
    ChangeEnginePackage,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.engine_packages import NodeEnginePackage
from skulk.shared.types.events import EnginePackageChangeCompleted, IndexedEvent
from skulk.shared.types.profiling import NodeResources
from skulk.utils.channels import channel

BUILD = "c" * 64
OLDER = "b" * 64
OFFERED = "d" * 64

HELD = NodeEnginePackage(
    engine_id="foxlight.tensorfold",
    lane="rocm",
    backend_tag="tensorfold-rocm",
    status="active",
    build=BUILD,
    engine_version="1.3.3",
    generation=4,
    publisher="foxlight",
    installed_builds=(OLDER, BUILD),
    rollback_build=OLDER,
    overhead_bytes=2 * 1024**3,
)
DAMAGED = NodeEnginePackage(
    engine_id="foxlight.tensorfold",
    lane="cpu",
    backend_tag="tensorfold-cpu",
    status="unusable",
    installed_builds=(BUILD,),
    problem="installation_damaged",
)
OFFER = NodeEnginePackage(
    engine_id="acme.other_engine",
    lane="cuda",
    backend_tag="other_engine-cuda",
    status="offered",
    build=OFFERED,
    engine_version="0.9",
    generation=1,
    channel="candidate",
    download_bytes=70 * 1024**2,
    overhead_bytes=1024**3,
)


def _build_api() -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("test-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


def test_packages_are_grouped_by_engine_and_ordered_by_node_then_lane() -> None:
    view = engines_view(
        {
            NodeId("node-b"): NodeResources(engine_packages=(HELD,)),
            NodeId("node-a"): NodeResources(engine_packages=(HELD, DAMAGED, OFFER)),
            NodeId("node-c"): NodeResources(),
        },
        packages_enabled=True,
    )

    assert [engine.engine_id for engine in view.engines] == [
        "acme.other_engine",
        "foxlight.tensorfold",
    ]
    assert [engine.name for engine in view.engines] == ["other_engine", "tensorfold"]
    assert [
        (entry.node_id, entry.package.lane) for entry in view.engines[1].nodes
    ] == [("node-a", "cpu"), ("node-a", "rocm"), ("node-b", "rocm")]
    assert view.engines[0].nodes[0].package == OFFER


def test_a_cluster_with_no_engine_package_reports_none() -> None:
    """No packages, no difference: built-in engines are not listed here."""
    view = engines_view(
        {NodeId("node-a"): NodeResources(backends=frozenset({"llama_cpp-vulkan"}))},
        packages_enabled=True,
    )

    assert view.engines == []


def test_a_node_with_no_package_sends_the_telemetry_it_sent_before() -> None:
    """The new field is absent from the wire unless a node has something to say."""
    plain = NodeResources().model_dump(mode="json", by_alias=True)
    reporting = NodeResources(engine_packages=(HELD,))

    assert "enginePackages" not in plain
    assert NodeResources.model_validate_json(
        reporting.model_dump_json(by_alias=True)
    ).engine_packages == (HELD,)


def test_the_route_serves_what_the_nodes_advertised() -> None:
    api = _build_api()
    api._telemetry_view.node_resources[NodeId("node-a")] = NodeResources(  # pyright: ignore[reportPrivateUsage]
        engine_packages=(HELD, DAMAGED)
    )

    response = TestClient(api.app).get("/v1/engines")

    assert response.status_code == 200
    # The wire uses camel-case names.
    assert '"packagesEnabled":true' in response.text.replace(" ", "")
    assert '"rollbackBuild"' in response.text
    body = EnginesResponse.model_validate_json(response.text)
    assert body.packages_enabled is True
    assert [engine.engine_id for engine in body.engines] == ["foxlight.tensorfold"]
    lanes = body.engines[0].nodes
    assert [entry.package for entry in lanes] == [DAMAGED, HELD]
    assert {entry.node_id for entry in lanes} == {"node-a"}


def test_the_route_says_when_packages_are_turned_off_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "engine_packages_disabled", lambda: True)

    response = TestClient(_build_api().app).get("/v1/engines")

    assert EnginesResponse.model_validate_json(response.text) == EnginesResponse(
        packages_enabled=False, engines=[]
    )


def test_the_route_is_in_the_openapi_document() -> None:
    document = cast("dict[str, dict[str, dict[str, dict[str, object]]]]", _build_api().app.openapi())
    operation = document["paths"]["/v1/engines"]["get"]

    assert operation["tags"] == ["Engines"]
    assert operation["summary"]
    assert operation["description"]


NODE = NodeId("node-a")


def _api_holding(*packages: NodeEnginePackage) -> API:
    api = _build_api()
    api._telemetry_view.node_resources[NODE] = NodeResources(  # pyright: ignore[reportPrivateUsage]
        engine_packages=packages
    )
    return api


def _answering(
    api: API,
    *,
    code: str | None = None,
    after: NodeResources | None = None,
    sent: list[ChangeEnginePackage] | None = None,
) -> AsyncMock:
    """Stand in for the master and the node: answer each change at once."""

    async def send(command: object) -> None:
        assert isinstance(command, ChangeEnginePackage)
        if sent is not None:
            sent.append(command)
        api._engine_change_results[command.command_id] = EnginePackageChangeCompleted(  # pyright: ignore[reportPrivateUsage]
            request_id=command.command_id,
            target_node=command.target_node,
            owner_node=command.owner_node,
            success=code is None,
            code=code,
            resources=(after or NodeResources()) if code is None else None,
        )
        api._engine_change_events[command.command_id].set()  # pyright: ignore[reportPrivateUsage]

    return AsyncMock(side_effect=send)


def test_activating_a_build_asks_the_node_and_returns_what_it_then_holds() -> None:
    api = _api_holding(HELD)
    moved = HELD.model_copy(update={"build": OLDER, "rollback_build": BUILD})
    sent: list[ChangeEnginePackage] = []

    with patch.object(
        api,
        "_send",
        new=_answering(api, after=NodeResources(engine_packages=(moved,)), sent=sent),
    ):
        response = TestClient(api.app).post(
            "/v1/engines/foxlight.tensorfold/activate",
            json={"nodeId": "node-a", "lane": "rocm", "build": OLDER},
        )

    assert response.status_code == 200
    body = EngineChangeResponse.model_validate_json(response.text)
    assert (body.node_id, body.engine_id, body.lane, body.action) == (
        "node-a",
        "foxlight.tensorfold",
        "rocm",
        "activate",
    )
    assert body.package == moved
    assert '"rollbackBuild"' in response.text
    assert [
        (each.target_node, each.owner_node, each.engine_id, each.lane, each.action, each.build)
        for each in sent
    ] == [(NODE, api.node_id, "foxlight.tensorfold", "rocm", "activate", OLDER)]
    # Nothing is kept once the answer has been read.
    assert api._engine_change_events == {}  # pyright: ignore[reportPrivateUsage]
    assert api._engine_change_results == {}  # pyright: ignore[reportPrivateUsage]


def test_a_rollback_names_the_earlier_build_and_removing_the_package_may_leave_none() -> None:
    api = _api_holding(HELD)
    sent: list[ChangeEnginePackage] = []

    with patch.object(api, "_send", new=_answering(api, sent=sent)):
        client = TestClient(api.app)
        rolled_back = client.post(
            "/v1/engines/foxlight.tensorfold/rollback",
            json={"nodeId": "node-a", "lane": "rocm"},
        )
        removed_one = client.post(
            "/v1/engines/foxlight.tensorfold/remove",
            json={"nodeId": "node-a", "lane": "rocm", "build": OLDER},
        )
        removed_all = client.post(
            "/v1/engines/foxlight.tensorfold/remove",
            json={"nodeId": "node-a", "lane": "rocm"},
        )

    assert [rolled_back.status_code, removed_one.status_code, removed_all.status_code] == [
        200,
        200,
        200,
    ]
    assert [(each.action, each.build) for each in sent] == [
        # The earlier build the node reported, so the request is safe to read twice.
        ("rollback", OLDER),
        ("remove", OLDER),
        ("remove", None),
    ]
    assert removed_all.json()["package"] is None


def test_a_node_that_goes_on_offering_a_removed_package_is_reported_so() -> None:
    api = _api_holding(HELD)
    offered = HELD.model_copy(
        update={"status": "offered", "installed_builds": (), "rollback_build": None}
    )

    with patch.object(
        api, "_send", new=_answering(api, after=NodeResources(engine_packages=(offered,)))
    ):
        response = TestClient(api.app).post(
            "/v1/engines/foxlight.tensorfold/remove",
            json={"nodeId": "node-a", "lane": "rocm"},
        )

    assert response.json()["package"]["status"] == "offered"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        # A node nobody has heard from.
        ("foxlight.tensorfold/rollback", {"nodeId": "node-z", "lane": "rocm"}),
        # A lane the node does not report.
        ("foxlight.tensorfold/rollback", {"nodeId": "node-a", "lane": "metal"}),
        # An engine the node does not report.
        ("acme.absent/rollback", {"nodeId": "node-a", "lane": "rocm"}),
        # A package the node is only offered has nothing on disk to change.
        ("acme.other_engine/remove", {"nodeId": "node-a", "lane": "cuda"}),
    ],
)
def test_a_package_no_node_holds_is_not_found_and_nothing_is_sent(
    path: str, body: dict[str, str]
) -> None:
    api = _api_holding(HELD, OFFER)

    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(f"/v1/engines/{path}", json=body)

    assert response.status_code == 404
    assert "/v1/engines" in response.json()["error"]["message"]
    send.assert_not_awaited()


def test_a_rollback_with_no_earlier_build_is_not_found_and_nothing_is_sent() -> None:
    api = _api_holding(HELD.model_copy(update={"rollback_build": None}))

    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(
            "/v1/engines/foxlight.tensorfold/rollback",
            json={"nodeId": "node-a", "lane": "rocm"},
        )

    assert response.status_code == 404
    assert "keeps no earlier" in response.json()["error"]["message"]
    send.assert_not_awaited()


@pytest.mark.parametrize(
    ("code", "status", "words"),
    [
        ("generation_missing", 404, "does not hold that build"),
        ("node_unavailable", 404, "not a live member"),
        ("generation_active", 409, "Activate another build first"),
        ("generation_in_use", 409, "Stop it first"),
        ("installation_damaged", 409, "not intact"),
        ("skulk_version_unsupported", 409, "cannot run that build"),
        ("protocol_unsupported", 409, "cannot run that build"),
        ("adapter_python_unsupported", 409, "cannot run that build"),
        ("packages_disabled", 409, "turned off"),
        ("change_failed", 500, "did not confirm"),
        ("some_code_from_a_newer_node", 500, "did not confirm"),
    ],
)
def test_a_refusal_is_explained_to_a_person(code: str, status: int, words: str) -> None:
    api = _api_holding(HELD)

    with patch.object(api, "_send", new=_answering(api, code=code)):
        response = TestClient(api.app).post(
            "/v1/engines/foxlight.tensorfold/remove",
            json={"nodeId": "node-a", "lane": "rocm", "build": BUILD},
        )

    assert response.status_code == status
    assert words in response.json()["error"]["message"]


def test_a_node_that_does_not_answer_is_reported_as_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "_ENGINE_CHANGE_CONFIRM_SECONDS", 0.05)
    api = _api_holding(HELD)

    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(
            "/v1/engines/foxlight.tensorfold/rollback",
            json={"nodeId": "node-a", "lane": "rocm"},
        )

    assert response.status_code == 503
    assert "has not answered" in response.json()["error"]["message"]
    send.assert_awaited_once()
    assert api._engine_change_events == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("activate", {"nodeId": "node-a", "lane": "rocm"}),
        ("activate", {"nodeId": "node-a", "lane": "rocm", "build": "abc"}),
        ("activate", {"nodeId": "node-a", "lane": "gpu", "build": BUILD}),
        ("remove", {"nodeId": "node-a", "lane": "rocm", "build": "../" + "c" * 61}),
        ("rollback", {"lane": "rocm"}),
    ],
)
def test_a_malformed_change_is_refused_before_anything_is_sent(
    path: str, body: dict[str, str]
) -> None:
    api = _api_holding(HELD)

    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(
            f"/v1/engines/foxlight.tensorfold/{path}", json=body
        )

    assert response.status_code == 422
    send.assert_not_awaited()
