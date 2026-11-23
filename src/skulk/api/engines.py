"""The cluster's engine packages report, read from what each node last advertised.

Nothing here reads a disk, starts a process or changes anything. Every fact
comes from node telemetry the API already holds, so the report says what each
node reported about itself and when a node stops reporting it drops out.
"""

from collections.abc import Mapping

from pydantic import Field

from skulk.shared.types.common import NodeId
from skulk.shared.types.engine_packages import NodeEnginePackage
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
