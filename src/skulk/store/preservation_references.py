"""Offline HTTP/peer model references with identical independent trust decisions."""

from foxlight_model_preservation.invitations import (
    Invitation as HTTPInvitation,
)
from foxlight_model_preservation.invitations import (
    accept_invitation_keys as accept_http_keys,
)
from foxlight_model_preservation.invitations import (
    authenticate_invitation as authenticate_http,
)
from foxlight_model_preservation.invitations import (
    read_invitation as read_http,
)
from foxlight_model_preservation.models import Manifest, TrustProfile
from foxlight_model_preservation.peer_invitations import (
    PeerInvitation,
    accept_peer_invitation_keys,
    authenticate_peer_invitation,
    read_peer_invitation,
)
from foxlight_model_preservation.signing import strict_json

type ModelInvitation = HTTPInvitation | PeerInvitation


def read_model_reference(data: bytes) -> ModelInvitation:
    """Read a strictly versioned complete public reference without network IO."""
    value = strict_json(data)
    if (
        isinstance(value, dict)
        and value.get("schema_version")
        == "foxlight.model-preservation.peer-invitation.v1"
    ):
        return read_peer_invitation(data)
    return read_http(data)


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
