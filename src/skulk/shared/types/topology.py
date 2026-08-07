from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import Field

from skulk.shared.types.common import NodeId
from skulk.shared.types.multiaddr import Multiaddr
from skulk.utils.pydantic_ext import FrozenModel


@dataclass(frozen=True)
class Cycle:
    node_ids: list[NodeId]

    def __len__(self) -> int:
        return self.node_ids.__len__()

    def __iter__(self) -> Iterator[NodeId]:
        return self.node_ids.__iter__()


class RDMAConnection(FrozenModel):
    source_rdma_iface: str
    sink_rdma_iface: str


class SocketConnection(FrozenModel):
    sink_multiaddr: Multiaddr
    session: bool = Field(
        default=False,
        description=(
            "True for an edge recorded from an authenticated libp2p session "
            "rather than an address probe. Session edges prove CONNECTIVITY "
            "(cycles, dashboard) but their annotated address is the "
            "connection's observed remote endpoint, which for a NAT'd or "
            "proxied member is not a dialable listener; transport host "
            "selection skips them."
        ),
    )

    def __hash__(self):
        return hash((self.sink_multiaddr.ip_address, self.session))


class Connection(FrozenModel):
    source: NodeId
    sink: NodeId
    edge: RDMAConnection | SocketConnection
