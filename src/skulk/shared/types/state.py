from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

from skulk.shared.topology import Topology, TopologySnapshot
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeIdentity,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
    NodeThunderboltInfo,
    SystemPerformanceProfile,
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

    # Granular node state mappings (update independently at different frequencies)
    node_identities: Mapping[NodeId, NodeIdentity] = {}
    node_memory: Mapping[NodeId, MemoryUsage] = {}
    node_disk: Mapping[NodeId, DiskUsage] = {}
    node_system: Mapping[NodeId, SystemPerformanceProfile] = {}
    node_network: Mapping[NodeId, NodeNetworkInfo] = {}
    node_thunderbolt: Mapping[NodeId, NodeThunderboltInfo] = {}
    node_thunderbolt_bridge: Mapping[NodeId, ThunderboltBridgeStatus] = {}
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] = {}
    # node_resources moved to the telemetry plane (#279) — it is gossiped
    # last-write-wins off the event log and lives in TelemetryView, not here.

    # Detected cycles where all nodes have Thunderbolt bridge enabled (>2 nodes)
    thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]] = []

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_node_resources(cls, data: object) -> object:
        """Strip the removed ``node_resources`` field from inbound payloads.

        ``node_resources`` moved to the telemetry plane (#279), but ``State``
        keeps ``extra="forbid"`` so a newer binary's unknown fields are caught
        rather than silently dropped. Without this, a state-sync snapshot from
        a pre-#279 master (which still carries ``nodeResources``) fails
        validation on an upgraded follower; the follower then discards the
        snapshot and falls back to replay from index 0, losing any
        instances/topology that lived only in an already-compacted log prefix
        (the #273 outage class). Popping the known-removed key — and only that
        key — preserves rolling-upgrade hydration without weakening the
        forbid-extra guard for genuinely unknown fields.
        """
        if not isinstance(data, dict):
            return data
        cleaned: dict[str, object] = dict(cast("dict[str, object]", data))
        cleaned.pop("nodeResources", None)
        cleaned.pop("node_resources", None)
        return cleaned

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
