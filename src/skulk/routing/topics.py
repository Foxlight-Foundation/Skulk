from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from skulk.routing.connection_message import ConnectionMessage
from skulk.shared.election import ElectionMessage
from skulk.shared.types.chunks import DataChunk
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.utils.pydantic_ext import CamelCaseModel


class PublishPolicy(str, Enum):
    Never = "Never"
    """Never publish to the network - this is a local message"""
    Minimal = "Minimal"
    """Only publish when there is no local receiver for this type of message"""
    Always = "Always"
    """Always publish to the network"""


@dataclass  # (frozen=True)
class TypedTopic[T: CamelCaseModel]:
    topic: str
    publish_policy: PublishPolicy

    model_type: type[
        T
    ]  # This can be worked around with evil type hacking, see https://stackoverflow.com/a/71720366 - I don't think it's necessary here.

    # Optional per-message routing-key suffix for the Zenoh data plane (#279
    # Phase 2). When set, the Router publishes to the Zenoh key
    # ``f"{topic}/{suffix}"`` so a message reaches only the subscriber declared
    # on that suffix (its own node id) instead of every node, killing the
    # cluster-wide fan-out. ``None`` (the default, and every topic on gossipsub)
    # keys by the bare topic, preserving broadcast semantics.
    routing_key: "Callable[[T], str | None] | None" = None

    @staticmethod
    def serialize(t: T) -> bytes:
        return t.model_dump_json().encode("utf-8")

    def deserialize(self, b: bytes) -> T:
        return self.model_type.model_validate_json(b.decode("utf-8"))


GLOBAL_EVENTS = TypedTopic("global_events", PublishPolicy.Always, GlobalForwarderEvent)
LOCAL_EVENTS = TypedTopic("local_events", PublishPolicy.Always, LocalForwarderEvent)
COMMANDS = TypedTopic("commands", PublishPolicy.Always, ForwarderCommand)
ELECTION_MESSAGES = TypedTopic(
    "election_messages", PublishPolicy.Always, ElectionMessage
)
CONNECTION_MESSAGES = TypedTopic(
    "connection_messages", PublishPolicy.Never, ConnectionMessage
)
DOWNLOAD_COMMANDS = TypedTopic(
    "download_commands", PublishPolicy.Always, ForwarderDownloadCommand
)
STATE_SYNC_MESSAGES = TypedTopic(
    "state_sync_messages", PublishPolicy.Always, StateSyncMessage
)
# Telemetry plane (#279): per-node live readings gossiped last-write-wins,
# off the event log. Slice 1 carries NodeResources only.
TELEMETRY = TypedTopic("telemetry", PublishPolicy.Always, NodeTelemetry)
# Data plane (#279 Phase 2): per-token generation output chunks streamed
# directly from the serving rank-0 worker to the owning API node, off the event
# log entirely. The master never sees these — no indexing, no disk, no
# cluster-wide rebroadcast. Only API nodes consume them (demux by command_id).
def _data_owner_key(chunk: DataChunk) -> str | None:
    """Route a data-plane chunk to its owning API node (#279 Phase 2).

    Returns the owner node id as the Zenoh key suffix, so the chunk is published
    to ``data/<owner_node>`` and reaches only the API node subscribed on its own
    id. Returns ``None`` when ``owner_node`` is unset (a pre-Phase-2 serving task
    or the gossipsub path), which falls back to keying by the bare topic.
    """
    return None if chunk.owner_node is None else str(chunk.owner_node)


DATA = TypedTopic(
    "data", PublishPolicy.Always, DataChunk, routing_key=_data_owner_key
)
