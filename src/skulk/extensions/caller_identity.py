"""Process-local original callers produced by Core request authentication.

An identity is neither an execution grant nor a serializable bearer credential.
The host must still resolve a signed destination and apply current call policy.
Installed author applications receive only the separately admitted native context.
"""

from dataclasses import dataclass
from typing import Literal, final

from skulk.shared.types.common import NodeId


@final
@dataclass(frozen=True, slots=True, repr=False)
class AuthenticatedOwner:
    """One direct owner realm authenticated by this node's actual HTTP boundary.

    The direct dashboard currently establishes local owner authority, rather
    than an individual human identity. Keep its fixed principal separate from
    the issuing node, a capability executor, and a provider account. This value
    remains process-local; it does not authorize inference or delegated effects.

    Attributes:
        issuer_node_id: Actual server node that authenticated the request.
        identifier: Fixed owner realm; never read from a request or prompt.
        kind: Original-principal category, independent of the executing app.
    """

    issuer_node_id: NodeId
    identifier: Literal["direct-owner"] = "direct-owner"
    kind: Literal["operator"] = "operator"
