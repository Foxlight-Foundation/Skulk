"""Offline HTTP/peer model references with identical independent trust decisions."""

from foxlight_model_preservation.handoffs import read_reference
from foxlight_model_preservation.i2p_http import i2p_http_endpoint
from foxlight_model_preservation.invitations import (
    Invitation as HTTPInvitation,
)
from foxlight_model_preservation.invitations import (
    accept_invitation_keys as accept_http_keys,
)
from foxlight_model_preservation.invitations import (
    authenticate_invitation as authenticate_http,
)
from foxlight_model_preservation.models import Manifest, PreservationError, TrustProfile
from foxlight_model_preservation.peer import PeerSource
from foxlight_model_preservation.peer_invitations import (
    PeerInvitation,
    accept_peer_invitation_keys,
    authenticate_peer_invitation,
)

type ModelInvitation = HTTPInvitation | PeerInvitation


def read_model_reference(data: bytes) -> ModelInvitation:
    """Read a strictly versioned complete public reference without network IO."""
    reference = read_reference(data)
    if isinstance(reference, PeerSource):
        raise PreservationError("source handoff must be reviewed as a source")
    return reference


def authenticate_model_reference(
    invitation: ModelInvitation, trust: TrustProfile
) -> Manifest:
    """Require independently selected exact-copy trust, regardless of transport."""
    return (
        authenticate_peer_invitation(invitation, trust)
        if isinstance(invitation, PeerInvitation)
        else authenticate_http(invitation, trust)
    )


def accept_model_reference_keys(
    invitation: ModelInvitation, keys: tuple[str, ...]
) -> TrustProfile:
    """Retain copy-scoped candidate acceptance without registry or code grants."""
    return (
        accept_peer_invitation_keys(invitation, keys)
        if isinstance(invitation, PeerInvitation)
        else accept_http_keys(invitation, keys)
    )


def i2p_model_mirrors(values: tuple[str, ...]) -> tuple[str, ...]:
    """Select only canonical I2P hints offline; locations never establish model trust."""
    accepted: list[str] = []
    for value in values:
        if len(value) > 2048:
            continue
        try:
            checked = i2p_http_endpoint(value)
        except ValueError:
            continue
        if checked not in accepted:
            accepted.append(checked)
    return tuple(accepted)
