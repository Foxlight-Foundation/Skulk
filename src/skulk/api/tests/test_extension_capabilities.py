"""API-side telemetry-plane advertise surface (fabric-citizenship Phase 1).

`ExtensionContext.advertise_capability` records a tag on the shared TelemetryView
outbound set (which the worker's gatherer later gossips), and the round trip is
observable locally: a node advertising a capability sees it in its own
`read_cluster` snapshot once the reading coalesces back into the view.
"""

from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import NodeCapabilities


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


def test_advertise_capability_records_on_outbound_set() -> None:
    view = TelemetryView()
    api = _build_api(view)
    api._extension_context.advertise_capability("memory")  # pyright: ignore[reportPrivateUsage]
    assert view.local_advertised_capabilities == {"memory"}


def test_advertise_capability_is_additive_and_idempotent() -> None:
    view = TelemetryView()
    api = _build_api(view)
    advertise = api._extension_context.advertise_capability  # pyright: ignore[reportPrivateUsage]
    advertise("memory")
    advertise("memory")  # idempotent
    advertise("search")  # additive
    assert view.local_advertised_capabilities == {"memory", "search"}


def test_advertise_capability_ignores_blank_tags() -> None:
    view = TelemetryView()
    api = _build_api(view)
    advertise = api._extension_context.advertise_capability  # pyright: ignore[reportPrivateUsage]
    advertise("   ")
    advertise("")
    assert view.local_advertised_capabilities == set()
    # a surrounding-whitespace tag is trimmed, not dropped
    advertise("  memory  ")
    assert view.local_advertised_capabilities == {"memory"}


def test_advertised_capability_round_trips_into_read_cluster() -> None:
    # End to end within one node: advertise -> the gatherer would emit a
    # NodeCapabilities reading -> the view coalesces it -> read_cluster surfaces
    # it. We stand in for the gossip hop by applying the reading the gatherer
    # would send for this node.
    view = TelemetryView()
    api = _build_api(view)
    api._extension_context.advertise_capability("memory")  # pyright: ignore[reportPrivateUsage]
    view.apply(
        NodeTelemetry(
            node_id=NodeId("api-node"),
            info=NodeCapabilities(
                capabilities=frozenset(view.local_advertised_capabilities)
            ),
        )
    )
    snapshot = api._extension_context.read_cluster()  # pyright: ignore[reportPrivateUsage]
    node = next(n for n in snapshot if n.node_id == NodeId("api-node"))
    assert node.capabilities == ("memory",)
