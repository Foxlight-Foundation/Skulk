"""API-side telemetry-plane advertise surface (fabric-citizenship Phase 1).

`ExtensionContext.advertise_capability` records a tag on the shared TelemetryView
outbound set (which the worker's gatherer later gossips), and the round trip is
observable locally: a node advertising a capability sees it in its own
`read_cluster` snapshot once the reading coalesces back into the view.
"""

from typing import cast

from skulk.api.main import API
from skulk.extensions import (
    CapabilityDescriptor,
    ExtensionContext,
    LoadedExtensions,
    descriptor_revision,
)
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import NodeCapabilities


def _build_api(
    view: TelemetryView, extensions: LoadedExtensions | None = None
) -> API:
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
        extensions=extensions,
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


# --- Provider surface (fabric-citizenship Phase 2a) --------------------------

_ECHO = CapabilityDescriptor(
    id="echo",
    version="1.0.0",
    title="Echo",
    description="Returns the input text unchanged.",
    input_schema={"type": "object"},
)


class _ProviderExtension:
    """Minimal provider extension for API wiring tests."""

    name = "test-provider"
    skulk_requires = ">=0"

    def __init__(self) -> None:
        self.started_with: list[ExtensionContext] = []

    def chat_middleware(self) -> None:
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_ECHO]

    def on_start(self, context: ExtensionContext) -> None:
        self.started_with.append(context)


def test_provider_descriptor_id_is_auto_advertised_and_on_start_runs() -> None:
    provider = _ProviderExtension()
    view = TelemetryView()
    _build_api(view, extensions=LoadedExtensions([provider]))
    # The descriptor's id became the telemetry discovery tag without the
    # extension calling advertise itself, and on_start ran with the live
    # context (a pure provider has no chat hook through which to reach it).
    assert view.local_advertised_capabilities == {"echo"}
    assert len(provider.started_with) == 1


async def test_list_node_capabilities_serves_descriptors_and_revisions() -> None:
    api = _build_api(TelemetryView(), extensions=LoadedExtensions([_ProviderExtension()]))
    payload = await api.list_node_capabilities()
    assert payload["node_id"] == "api-node"
    capabilities = cast("list[object]", payload["capabilities"])
    assert isinstance(capabilities, list) and len(capabilities) == 1
    restored = CapabilityDescriptor.model_validate(capabilities[0])
    assert restored == _ECHO
    revisions = payload["revisions"]
    assert isinstance(revisions, dict)
    assert revisions["echo@1.0.0"] == descriptor_revision(_ECHO)


async def test_list_node_capabilities_empty_without_extensions() -> None:
    payload = await _build_api(TelemetryView()).list_node_capabilities()
    assert payload["capabilities"] == []
    assert payload["revisions"] == {}


async def test_describe_node_local_returns_descriptors() -> None:
    api = _build_api(TelemetryView(), extensions=LoadedExtensions([_ProviderExtension()]))
    descriptors = await api._extension_context.describe_node(NodeId("api-node"))  # pyright: ignore[reportPrivateUsage]
    assert descriptors == (_ECHO,)


async def test_describe_node_unknown_peer_returns_empty() -> None:
    api = _build_api(TelemetryView())
    # No topology, so the peer is unreachable: degrade to (), never raise.
    descriptors = await api._extension_context.describe_node(NodeId("n-ghost"))  # pyright: ignore[reportPrivateUsage]
    assert descriptors == ()


def test_withdraw_capability_removes_tag() -> None:
    view = TelemetryView()
    api = _build_api(view)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    context.advertise_capability("memory")
    context.advertise_capability("search")
    context.withdraw_capability("memory")
    assert view.local_advertised_capabilities == {"search"}
    # Withdrawing an unknown or already-withdrawn tag is a no-op.
    context.withdraw_capability("memory")
    context.withdraw_capability("never-advertised")
    assert view.local_advertised_capabilities == {"search"}
