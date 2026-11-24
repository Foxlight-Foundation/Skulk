"""The cluster's engine packages report, read from what each node last advertised.

Nothing here reads a disk, starts a process or changes anything. Every fact
comes from node telemetry the API already holds, so the report says what each
node reported about itself and when a node stops reporting it drops out.
"""

from collections.abc import Mapping

from pydantic import Field

from skulk.shared.types.common import NodeId
from skulk.shared.types.engine_packages import (
    ENGINE_BUILD_PATTERN,
    EnginePackageAction,
    EnginePackageLane,
    NodeEnginePackage,
)
from skulk.shared.types.profiling import NodeResources
from skulk.utils.pydantic_ext import CamelCaseModel


class EngineNodePackage(CamelCaseModel):
    """One engine package lane on one node."""

    node_id: str = Field(description="The node that reported this lane.")
    package: NodeEnginePackage = Field(
        description="What the node holds or may install for the lane."
    )


class EnginePackages(CamelCaseModel):
    """One engine ID and every node lane that reports it."""

    engine_id: str = Field(description="The publisher.engine identity.")
    name: str = Field(
        description=(
            "The engine's name in backend tags and engine orders, such as "
            "tensorfold."
        )
    )
    nodes: list[EngineNodePackage] = Field(
        description="One entry per node and lane, sorted by node then lane."
    )


class EnginesResponse(CamelCaseModel):
    """Every engine package the cluster's nodes hold or may install."""

    packages_enabled: bool = Field(
        description=(
            "Whether engine packages are turned on for the node that "
            "answered. A node on which they are turned off reports none."
        )
    )
    engines: list[EnginePackages] = Field(
        description="One entry per engine ID, sorted by engine ID."
    )


class EngineRollbackRequest(CamelCaseModel):
    """Which node and accelerator class an engine package change is for."""

    node_id: str = Field(
        min_length=1,
        max_length=128,
        description="The node whose package is changed, as GET /v1/engines names it.",
    )
    lane: EnginePackageLane = Field(
        description="The accelerator class of the package: metal, cuda, rocm, vulkan or cpu."
    )


class EngineActivateRequest(EngineRollbackRequest):
    """The installed build to make the one new launches use."""

    build: str = Field(
        pattern=ENGINE_BUILD_PATTERN,
        description="Manifest digest of a build on the node's disk.",
    )


class EngineRemoveRequest(EngineRollbackRequest):
    """One build to delete, or the whole package when no build is named."""

    build: str | None = Field(
        default=None,
        pattern=ENGINE_BUILD_PATTERN,
        description=(
            "Manifest digest of the build to delete. Leave it out to delete "
            "every build of the package on the node."
        ),
    )


class EngineChangeResponse(CamelCaseModel):
    """What a node holds for one package lane after a change was made."""

    node_id: str = Field(description="The node that made the change.")
    engine_id: str = Field(description="The publisher.engine identity.")
    lane: EnginePackageLane = Field(description="The accelerator class.")
    action: EnginePackageAction = Field(description="The change that was made.")
    package: NodeEnginePackage | None = Field(
        description=(
            "The lane as the node reported it straight after the change. "
            "Null when the node no longer holds or is offered the package."
        )
    )


_REFUSAL_SENTENCES: dict[str, str] = {
    "generation_missing": (
        "The node does not hold that build, or has no earlier build to return to."
    ),
    "generation_active": (
        "That build is the one new launches use. Activate another build first, "
        "or remove the whole package."
    ),
    "generation_in_use": (
        "A model on the node is running on a build this would delete. Stop it first."
    ),
    "installation_damaged": (
        "The build's files on the node are not intact, so nothing was changed."
    ),
    "skulk_version_unsupported": "The node's Skulk version cannot run that build.",
    "protocol_unsupported": "The node's Skulk version cannot run that build.",
    "adapter_python_unsupported": "The node's Skulk version cannot run that build.",
    "packages_disabled": "Engine packages are turned off on the node.",
    "node_unavailable": "The node is not a live member of the cluster.",
}


def engine_change_refusal(code: str | None) -> tuple[int, str]:
    """Turn a node's refusal code into an HTTP status and a sentence for a person.

    Args:
        code: The code the node answered with.

    Returns:
        404 for a build, package or node that is not there, 409 for a change
        the node's present state does not allow, and 500 when the node could
        not say whether the change was made.
    """
    sentence = _REFUSAL_SENTENCES.get(code or "")
    if sentence is None:
        return 500, (
            "The node did not confirm the change. Read /v1/engines to see what "
            "it holds now; the node's log has the reason."
        )
    missing = code in ("generation_missing", "node_unavailable")
    return (404 if missing else 409), sentence


def held_engine_package(
    resources: NodeResources | None, engine_id: str, lane: str
) -> NodeEnginePackage | None:
    """Find the package lane a node holds, ignoring one it is only offered."""
    if resources is None:
        return None
    return next(
        (
            package
            for package in resources.engine_packages
            if package.engine_id == engine_id
            and package.lane == lane
            and package.status != "offered"
        ),
        None,
    )


def reported_engine_package(
    resources: NodeResources | None, engine_id: str, lane: str
) -> NodeEnginePackage | None:
    """Find what a node reports for one package lane, held or offered."""
    if resources is None:
        return None
    return next(
        (
            package
            for package in resources.engine_packages
            if package.engine_id == engine_id and package.lane == lane
        ),
        None,
    )


def engines_view(
    node_resources: Mapping[NodeId, NodeResources], *, packages_enabled: bool
) -> EnginesResponse:
    """Group what every node reported about engine packages by engine ID.

    Args:
        node_resources: The latest resources each node advertised.
        packages_enabled: Whether engine packages are on for this node.

    Returns:
        The report, with engines, nodes and lanes in a stable order.
    """
    grouped: dict[str, list[EngineNodePackage]] = {}
    for node_id, resources in node_resources.items():
        for package in resources.engine_packages:
            grouped.setdefault(package.engine_id, []).append(
                EngineNodePackage(node_id=str(node_id), package=package)
            )
    return EnginesResponse(
        packages_enabled=packages_enabled,
        engines=[
            EnginePackages(
                engine_id=engine_id,
                name=engine_id.split(".", 1)[-1],
                nodes=sorted(
                    grouped[engine_id],
                    key=lambda entry: (entry.node_id, entry.package.lane),
                ),
            )
            for engine_id in sorted(grouped)
        ],
    )
