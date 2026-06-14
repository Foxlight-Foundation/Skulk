from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)
from pydantic.alias_generators import to_camel

from skulk.shared.topology import Topology, TopologySnapshot
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import (
    NodeNetworkInfo,
    NodeThunderboltInfo,
    ThunderboltBridgeStatus,
)
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.downloads import DownloadProgress
from skulk.shared.types.worker.instances import Instance, InstanceId
from skulk.shared.types.worker.runners import RunnerId, RunnerStatus
from skulk.utils.pydantic_ext import CamelCaseModel


class State(CamelCaseModel):
    """Global system state.

    The :class:`Topology` instance is encoded/decoded via an immutable
    :class:`~shared.topology.TopologySnapshot` to ensure compatibility with
    standard JSON serialisation.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        # I want to reenable this ASAP, but it's causing an issue with TaskStatus
        strict=True,
        arbitrary_types_allowed=True,
    )
    instances: Mapping[InstanceId, Instance] = {}
    runners: Mapping[RunnerId, RunnerStatus] = {}
    downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {}
    tasks: Mapping[TaskId, Task] = {}
    last_seen: Mapping[NodeId, datetime] = {}
    topology: Topology = Field(default_factory=Topology)
    tracing_enabled: bool = False
    last_event_applied_idx: int = Field(default=-1, ge=-1)

    # Connectivity mappings stay on the control plane: apply() builds the
    # topology graph (RDMA edges, thunderbolt-bridge cycles) from them and the
    # placement planner reads node_network, so they must be ordered, not
    # last-write-wins telemetry (#279 slice 3 scoping).
    node_network: Mapping[NodeId, NodeNetworkInfo] = {}
    node_thunderbolt: Mapping[NodeId, NodeThunderboltInfo] = {}
    node_thunderbolt_bridge: Mapping[NodeId, ThunderboltBridgeStatus] = {}
    # node_resources (#279 slice 1), node_memory + node_system (#279 slice 2),
    # and node_identities + node_disk + node_rdma_ctl (#279 slice 3, the
    # observational readings) moved to the telemetry plane — gossiped
    # last-write-wins off the event log, held in TelemetryView, not here.
    # NB: do NOT add a model_validator(mode=
    # "before") to strip legacy keys for cross-version snapshots — that forces
    # the whole model into strict PYTHON-mode validation, where ISO datetime
    # strings (last_seen, serialized over the wire) are rejected, which silently
    # broke state-sync entirely (followers livelock requesting the event log
    # from 0). Mixed-version clusters are unsupported anyway (see #293 / the
    # versioning policy in CLAUDE.md), so extra="forbid" simply rejecting an
    # old snapshot's removed keys is the correct, intended behavior.

    # Detected cycles where all nodes have Thunderbolt bridge enabled (>2 nodes)
    thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]] = []

    @field_serializer("topology", mode="plain")
    def _encode_topology(self, value: Topology) -> TopologySnapshot:
        return value.to_snapshot()

    @field_validator("topology", mode="before")
    @classmethod
    def _deserialize_topology(cls, value: object) -> Topology:  # noqa: D401 – Pydantic validator signature
        """Convert an incoming *value* into a :class:`Topology` instance.

        Accepts either an already constructed :class:`Topology` or a mapping
        representing :class:`~shared.topology.TopologySnapshot`.
        """

        if isinstance(value, Topology):
            return value

        if isinstance(value, Mapping):  # likely a snapshot-dict coming from JSON
            snapshot = TopologySnapshot(**cast(dict[str, Any], value))  # type: ignore[arg-type]
            return Topology.from_snapshot(snapshot)

        raise TypeError("Invalid representation for Topology field in State")
