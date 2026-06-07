from skulk_pyo3_bindings import PyFromSwarm

from skulk.shared.types.common import NodeId
from skulk.utils.pydantic_ext import CamelCaseModel

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(CamelCaseModel):
    node_id: NodeId
    connected: bool

    @classmethod
    def from_update(cls, update: PyFromSwarm.Connection) -> "ConnectionMessage":
        return cls(node_id=NodeId(update.peer_id), connected=update.connected)
