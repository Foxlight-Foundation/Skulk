"""A managed installation is a satellite of its host, the way the bridge's nodes were."""

import asyncio
from dataclasses import replace
from pathlib import Path

from skulk.extensions.managed import (
    ManagedConnection,
    ManagedNode,
    ManagedOwner,
    ManagedSurface,
    summaries_for,
)
from skulk.extensions.managed_attachment import ManagedAttachment
from skulk.extensions.tests.test_steward_tools import context
from skulk.shared.types.capability_nodes import CapabilityNodeSummary

PROFILE = "299f2b629ac04a99ab20b6c8538789b6"


def _node(
    status: str = "ready", ready: bool = True, url: str | None = "http://127.0.0.1:1/s/"
) -> ManagedNode:
    return ManagedNode(
        node_id="node-1",
        bundle_id="foxlight.video-studio",
        version="0.1.0",
        status=status,
        configurable=True,
        descriptors=(),
        title="Skulk Video Studio",
        surfaces=(
            ManagedSurface(
                surface_id="studio", title="Studio", kind="link", ready=ready, url=url
            ),
            ManagedSurface(
                surface_id="comfy", title="ComfyUI", kind="link", ready=False, url=None
            ),
        ),
    )


def test_summaries_carry_identity_status_and_ready_link_surfaces_only() -> None:
    (summary,) = summaries_for("managed.fixture", (_node(),), True)
    assert (summary.plugin_id, summary.node_id, summary.bundle_id) == (
        "managed.fixture",
        "node-1",
        "foxlight.video-studio",
    )
    assert summary.title == "Skulk Video Studio" and summary.status == "ready"
    assert [s.surface_id for s in summary.surfaces] == ["studio"]
    assert [a.action_id for a in summary.actions] == ["open-studio"]
    assert summary.owner_available
    # The owner's terminal states fold into the summary's failed; unready or
    # URL-less surfaces are not carried; an unreachable owner is said so.
    (summary,) = summaries_for(
        "managed.fixture", (_node("restart_exhausted", ready=False),), False
    )
    assert (
        summary.status == "failed"
        and summary.surfaces == ()
        and not summary.owner_available
    )


async def test_the_owner_publishes_after_refresh_and_withdraws_on_stop(
    tmp_path: Path,
) -> None:
    published: list[CapabilityNodeSummary] = []
    withdrawn: list[tuple[str, str]] = []
    attachment = ManagedAttachment(tmp_path, PROFILE)
    owner = ManagedOwner(
        ManagedConnection(plugin_id="managed.fixture", state_root=str(tmp_path)),
        attachment=attachment,
    )

    def withdraw(plugin_id: str, node_id: str) -> None:
        withdrawn.append((plugin_id, node_id))

    owner.context = replace(
        context(),
        publish_capability_node=published.append,
        withdraw_capability_node=withdraw,
    )
    owner.nodes = (_node(),)
    owner.available = True
    owner._publish_summaries()  # pyright: ignore[reportPrivateUsage]
    assert [s.key for s in published] == ["managed.fixture/node-1"]
    # A node that leaves the description is withdrawn; the rest republished.
    owner.nodes = ()
    owner._publish_summaries()  # pyright: ignore[reportPrivateUsage]
    assert withdrawn == [("managed.fixture", "node-1")]
    owner.nodes = (_node(),)
    owner._publish_summaries()  # pyright: ignore[reportPrivateUsage]
    attachment.retain("test-node")

    async def observer() -> None:
        await asyncio.Event().wait()

    owner.poll_task = asyncio.create_task(observer())
    await owner.on_stop()
    assert withdrawn[-1] == ("managed.fixture", "node-1")


def test_malformed_display_metadata_never_decides_availability() -> None:
    """A surface the topology cannot carry is left off; the node is still summarized."""
    node = ManagedNode(
        node_id="node-1",
        bundle_id="foxlight.video-studio",
        version="0.1.0",
        status="degraded",
        configurable=True,
        descriptors=(),
        surfaces=(
            ManagedSurface(
                surface_id="bad", title="Bad", kind="link", ready=True, url="ftp://x"
            ),
            ManagedSurface(
                surface_id="studio",
                title="Studio",
                kind="link",
                ready=True,
                url="http://127.0.0.1:1/s/",
            ),
        ),
    )
    (summary,) = summaries_for("managed.fixture", (node,), True)
    assert [s.surface_id for s in summary.surfaces] == ["studio"]
    # Readiness is the surface's own report, not the node's lifecycle.
    assert summary.status == "degraded" and summary.surfaces[0].ready


async def test_an_owner_answering_only_describe_stays_admitted(tmp_path: Path) -> None:
    owner = ManagedOwner(
        ManagedConnection(plugin_id="managed.fixture", state_root=str(tmp_path)),
        attachment=ManagedAttachment(tmp_path, PROFILE),
    )
    owner.attachment = None
    ctx = context()
    owner.context = ctx
    asked: list[str] = []

    async def request(
        message: dict[str, object], *, timeout: float
    ) -> dict[str, object]:
        asked.append(str(message["operation"]))
        if message["operation"] == "describe-extended":
            raise ValueError("managed operation refused")
        return {
            "transport_node_id": str(ctx.node_id),
            "nodes": [_node().model_dump(mode="json")],
        }

    owner._request = request  # type: ignore[method-assign]
    await owner.refresh()
    assert asked == ["describe-extended", "describe"] and owner.available
