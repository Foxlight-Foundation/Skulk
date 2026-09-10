"""Capability-node summaries: bounds, wire shape, and the telemetry view.

The summary is gossiped to every node and rendered by every dashboard, so the
model refuses anything unbounded or credential-bearing at construction, and
the view keeps exactly one snapshot per host that clears when the host
withdraws its last node or leaves.
"""

from datetime import datetime, timezone

import pytest
from pydantic import JsonValue, ValidationError

from skulk.shared.types.capability_nodes import (
    MAX_ACTION_PAYLOAD_BYTES,
    MAX_CAPABILITY_NODES_PER_HOST,
    MAX_SURFACE_URL_LENGTH,
    CapabilityNodeAction,
    CapabilityNodeSummary,
    CapabilityNodeSurface,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.utils.info_gatherer.info_gatherer import NodeCapabilityNodes


def _summary(node_id: str = "studio", **overrides: object) -> CapabilityNodeSummary:
    fields: dict[str, object] = {
        "plugin_id": "foxlight.video-studio",
        "node_id": node_id,
        "bundle_id": "foxlight.video-studio",
        "version": "1.0.0",
        "title": "Video Studio",
        "status": "ready",
        "owner_available": True,
        "surfaces": (
            CapabilityNodeSurface(
                surface_id="studio", title="Studio", url="http://127.0.0.1:8188/"
            ),
        ),
        "actions": (
            CapabilityNodeAction(
                action_id="open", title="Open", kind="surface", surface_id="studio"
            ),
        ),
    }
    fields.update(overrides)
    return CapabilityNodeSummary.model_validate(fields)


def test_surface_url_must_be_absolute_http_without_credentials() -> None:
    for bad in ("ftp://host/", "/relative", "http://user:pw@host/", "javascript:x"):
        with pytest.raises(ValidationError):
            CapabilityNodeSurface(surface_id="s", title="S", url=bad)
    CapabilityNodeSurface(surface_id="s", title="S", url="https://host:8443/ui")
    CapabilityNodeSurface(surface_id="s", title="S", url="https://host/ui?workflow=h3&tab=2")


def test_surface_url_is_bounded_and_refuses_credential_query_names() -> None:
    with pytest.raises(ValidationError):
        CapabilityNodeSurface(
            surface_id="s", title="S", url="https://host/ui?" + "x" * MAX_SURFACE_URL_LENGTH
        )
    for query in ("token=abc", "Access_Token=abc", "api_key=abc", "sig=abc", "auth="):
        with pytest.raises(ValidationError):
            CapabilityNodeSurface(surface_id="s", title="S", url=f"https://host/ui?{query}")


def test_key_segments_refuse_slashes_so_the_host_key_is_injective() -> None:
    with pytest.raises(ValidationError):
        _summary(plugin_id="a/b", node_id="c")
    with pytest.raises(ValidationError):
        _summary(node_id="b/c")
    assert _summary(plugin_id="a.b", node_id="c").key == "a.b/c"


def test_action_shapes_require_their_target() -> None:
    with pytest.raises(ValidationError):
        CapabilityNodeAction(action_id="a", title="A", kind="surface")
    with pytest.raises(ValidationError):
        CapabilityNodeAction(action_id="a", title="A", kind="link")
    with pytest.raises(ValidationError):
        CapabilityNodeAction(action_id="a", title="A", kind="descriptor")
    oversized: dict[str, JsonValue] = {"blob": "x" * MAX_ACTION_PAYLOAD_BYTES}
    with pytest.raises(ValidationError):
        CapabilityNodeAction(
            action_id="a",
            title="A",
            kind="descriptor",
            capability_id="video.plan@1",
            payload=oversized,
        )


def test_summary_rejects_duplicate_or_dangling_ids() -> None:
    surface = CapabilityNodeSurface(surface_id="s", title="S", url="http://h/")
    with pytest.raises(ValidationError):
        _summary(surfaces=(surface, surface))
    with pytest.raises(ValidationError):
        _summary(
            actions=(
                CapabilityNodeAction(
                    action_id="a", title="A", kind="surface", surface_id="missing"
                ),
            )
        )
    with pytest.raises(ValidationError):
        _summary(title=" padded")


def test_summary_bounds_surfaces_actions_and_host_count() -> None:
    surfaces = tuple(
        CapabilityNodeSurface(surface_id=f"s{i}", title="S", url="http://h/")
        for i in range(5)
    )
    with pytest.raises(ValidationError):
        _summary(surfaces=surfaces, actions=())
    nodes = tuple(_summary(node_id=f"n{i}") for i in range(MAX_CAPABILITY_NODES_PER_HOST + 1))
    with pytest.raises(ValidationError):
        NodeCapabilityNodes(nodes=nodes)
    NodeCapabilityNodes(nodes=nodes[:MAX_CAPABILITY_NODES_PER_HOST])


def test_reading_round_trips_through_the_telemetry_wire_shape() -> None:
    # The wire path dumps by alias to JSON and validates back; the tuple fields
    # arrive as lists and must coerce under strict mode.
    telemetry = NodeTelemetry(
        node_id=NodeId("host"), info=NodeCapabilityNodes(nodes=(_summary(),))
    )
    decoded = NodeTelemetry.model_validate_json(telemetry.model_dump_json(by_alias=True))
    assert isinstance(decoded.info, NodeCapabilityNodes)
    assert decoded.info.nodes == (_summary(),)
    assert decoded.info.nodes[0].surfaces[0].url == "http://127.0.0.1:8188/"


def test_view_keeps_latest_snapshot_and_clears_on_empty_or_removal() -> None:
    view = TelemetryView()
    host = NodeId("host")

    def reading(*nodes: CapabilityNodeSummary) -> NodeTelemetry:
        return NodeTelemetry(node_id=host, info=NodeCapabilityNodes(nodes=nodes))

    first = datetime(2026, 9, 10, tzinfo=timezone.utc)
    view.apply(reading(_summary()), received_at=first)
    assert view.node_capability_nodes[host] == (_summary(),)
    assert view.node_capability_nodes_received_at[host] == first

    later = datetime(2026, 9, 10, 0, 1, tzinfo=timezone.utc)
    degraded = _summary(status="degraded")
    view.apply(reading(degraded), received_at=later)
    assert view.node_capability_nodes[host] == (degraded,)
    assert view.node_capability_nodes_received_at[host] == later

    view.apply(reading(), received_at=later)
    assert host not in view.node_capability_nodes
    assert host not in view.node_capability_nodes_received_at

    view.apply(reading(_summary()), received_at=later)
    view.prune(host)
    assert host not in view.node_capability_nodes
    assert host not in view.node_capability_nodes_received_at
