"""API half of the topology layer for capability nodes.

`ExtensionContext.publish_capability_node` records a bounded summary on the
shared TelemetryView outbound map, `withdraw_capability_node` removes it, and
`GET /state` projects the received summaries of live hosts as
`capabilityNodes` with the host's last receipt time.
"""

from datetime import datetime, timezone
from typing import cast

import pytest

from skulk.api.main import (
    API,
    TEST_CAPABILITY_NODE_ENV_VAR,
    TEST_CAPABILITY_NODE_PLUGIN_ID,
)
from skulk.shared.election import ElectionMessage
from skulk.shared.types.capability_nodes import (
    MAX_CAPABILITY_NODES_PER_HOST,
    CapabilityNodeSummary,
    CapabilityNodeSurface,
)
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import TelemetryView
from skulk.utils.channels import channel


def _build_api(view: TelemetryView) -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        telemetry_view=view,
    )


def _summary(node_id: str = "studio", status: str = "ready") -> CapabilityNodeSummary:
    return CapabilityNodeSummary.model_validate(
        {
            "plugin_id": "foxlight.video-studio",
            "node_id": node_id,
            "bundle_id": "foxlight.video-studio",
            "version": "1.0.0",
            "title": "Video Studio",
            "status": status,
            "owner_available": True,
            "surfaces": (
                CapabilityNodeSurface(
                    surface_id="studio", title="Studio", url="http://127.0.0.1:8188/"
                ),
            ),
        }
    )


def test_publish_replaces_in_place_and_withdraw_removes() -> None:
    view = TelemetryView()
    api = _build_api(view)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    context.publish_capability_node(_summary("a"))
    context.publish_capability_node(_summary("b"))
    context.publish_capability_node(_summary("a", status="degraded"))
    assert [s.node_id for s in view.local_capability_nodes.values()] == ["a", "b"]
    assert view.local_capability_nodes["foxlight.video-studio/a"].status == "degraded"
    context.withdraw_capability_node("foxlight.video-studio", "a")
    context.withdraw_capability_node("foxlight.video-studio", "never")
    assert list(view.local_capability_nodes) == ["foxlight.video-studio/b"]


def test_publish_refuses_a_seventeenth_node_per_host() -> None:
    view = TelemetryView()
    api = _build_api(view)
    publish = api._extension_context.publish_capability_node  # pyright: ignore[reportPrivateUsage]
    for index in range(MAX_CAPABILITY_NODES_PER_HOST + 1):
        publish(_summary(f"n{index}"))
    assert len(view.local_capability_nodes) == MAX_CAPABILITY_NODES_PER_HOST
    # Replacing an existing key is still allowed at the bound.
    publish(_summary("n0", status="failed"))
    assert view.local_capability_nodes["foxlight.video-studio/n0"].status == "failed"


async def test_state_projects_capability_nodes_of_live_hosts() -> None:
    view = TelemetryView()
    api = _build_api(view)
    host = NodeId("n-host")
    dead = NodeId("n-dead")
    observed = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    view.node_capability_nodes[host] = (_summary(),)
    view.node_capability_nodes_received_at[host] = observed
    view.node_capability_nodes[dead] = (_summary("ghost"),)
    view.node_capability_nodes_received_at[dead] = observed
    api.state = api.state.model_copy(
        update={"last_seen": {host: datetime.now(tz=timezone.utc)}}
    )
    payload = await api.get_cluster_state()
    nodes = cast("dict[str, list[dict[str, object]]]", payload["capabilityNodes"])
    assert list(nodes) == ["n-host"]
    entry = nodes["n-host"][0]
    assert entry["pluginId"] == "foxlight.video-studio"
    surfaces = cast("list[dict[str, object]]", entry["surfaces"])
    assert surfaces[0]["url"] == "http://127.0.0.1:8188/"
    assert entry["observedAt"] == observed.isoformat()


def test_env_fake_node_publishes_one_link_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    view = TelemetryView()
    api = _build_api(view)
    monkeypatch.setenv(TEST_CAPABILITY_NODE_ENV_VAR, "http://127.0.0.1:8188/")
    api._publish_test_capability_node_from_env()  # pyright: ignore[reportPrivateUsage]
    summary = view.local_capability_nodes[f"{TEST_CAPABILITY_NODE_PLUGIN_ID}/studio"]
    assert summary.status == "ready"
    assert summary.surfaces[0].kind == "link"
    assert summary.surfaces[0].url == "http://127.0.0.1:8188/"
    assert summary.actions[0].surface_id == "studio"


def test_env_fake_node_ignores_invalid_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    view = TelemetryView()
    api = _build_api(view)
    for value in ("", "   ", "ftp://x/", "http://user:pw@host/"):
        monkeypatch.setenv(TEST_CAPABILITY_NODE_ENV_VAR, value)
        api._publish_test_capability_node_from_env()  # pyright: ignore[reportPrivateUsage]
    assert view.local_capability_nodes == {}
