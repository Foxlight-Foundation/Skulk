"""The engines report: what every node said about engine packages, grouped by engine."""

from typing import cast

import pytest
from fastapi.testclient import TestClient

import skulk.api.main as api_main
from skulk.api.engines import EnginesResponse, engines_view
from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.engine_packages import NodeEnginePackage
from skulk.shared.types.events import IndexedEvent
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
