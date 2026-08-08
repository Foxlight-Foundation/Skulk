"""Stable identity and authorization foundations for remote operators.

The operator subsystem is intentionally separate from Skulk's event-sourced
inference state. Its durable identities and authorization journal have a
different consistency, confidentiality, and recovery contract.
"""

from skulk.operator.authority import (
    AuthorityCommitConflictError,
    AuthorityKeyProvider,
    AuthorityRecord,
    EncryptedAuthorityStore,
)
from skulk.operator.identity import (
    ClusterIdentityMaterial,
    ClusterPublicIdentity,
    NodeInstallationIdentity,
    OperatorIdentityRepository,
    create_cluster_identity,
)

__all__ = [
    "AuthorityCommitConflictError",
    "AuthorityKeyProvider",
    "AuthorityRecord",
    "ClusterIdentityMaterial",
    "ClusterPublicIdentity",
    "EncryptedAuthorityStore",
    "NodeInstallationIdentity",
    "OperatorIdentityRepository",
    "create_cluster_identity",
]
