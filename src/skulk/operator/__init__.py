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
from skulk.operator.replication import (
    AuthorityAppliedCommit,
    AuthorityCertificateError,
    AuthorityCommitDescriptor,
    AuthorityCommitPosition,
    AuthorityMember,
    AuthorityMembership,
    AuthorityQuorumCertificate,
    AuthorityVote,
    apply_quorum_certified_payload,
    authority_bootstrap_position,
    authority_commit_digest,
    authority_membership_digest,
    authority_payload_digest,
    create_authority_member,
    create_authority_vote,
    verify_quorum_certificate,
)

__all__ = [
    "AuthorityAppliedCommit",
    "AuthorityCertificateError",
    "AuthorityCommitConflictError",
    "AuthorityCommitDescriptor",
    "AuthorityCommitPosition",
    "AuthorityKeyProvider",
    "AuthorityMember",
    "AuthorityMembership",
    "AuthorityQuorumCertificate",
    "AuthorityRecord",
    "AuthorityVote",
    "ClusterIdentityMaterial",
    "ClusterPublicIdentity",
    "EncryptedAuthorityStore",
    "NodeInstallationIdentity",
    "OperatorIdentityRepository",
    "apply_quorum_certified_payload",
    "authority_bootstrap_position",
    "authority_commit_digest",
    "authority_membership_digest",
    "authority_payload_digest",
    "create_authority_member",
    "create_authority_vote",
    "create_cluster_identity",
    "verify_quorum_certificate",
]
