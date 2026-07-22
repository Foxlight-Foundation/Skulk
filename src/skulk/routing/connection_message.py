from skulk_pyo3_bindings import PyFromSwarm

from skulk.shared.types.common import NodeId
from skulk.utils.pydantic_ext import CamelCaseModel

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(CamelCaseModel):
    node_id: NodeId
    connected: bool
    # The connection's observed remote endpoint (#662). Consumers record the
    # authenticated libp2p session as a first-class topology path, so a
    # member whose advertised addresses are all unreachable (a NAT'd or
    # proxied remote node) still contributes a truthful topology edge.
    # ``None`` on disconnect updates. This message is process-local (router
    # to in-process consumers), not gossiped, so the additive fields carry
    # no wire impact.
    remote_ip: str | None = None
    remote_tcp_port: int | None = None

    @classmethod
    def from_update(cls, update: PyFromSwarm.Connection) -> "ConnectionMessage":
        return cls(
            node_id=NodeId(update.peer_id),
            connected=update.connected,
            remote_ip=update.remote_ip or None,
            remote_tcp_port=update.remote_tcp_port or None,
        )
