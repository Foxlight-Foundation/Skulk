"""Compatibility helpers for cross-node operational diagnostics."""

from collections.abc import Sequence

from skulk.shared.types.diagnostics import (
    ClusterDiagnosticsVersionStatus,
    NodeDiagnostics,
    NodeDiagnosticsVersionStatus,
    NodeRuntimeDiagnostics,
)

_UNKNOWN_BUILD_IDENTIFIERS = frozenset({"", "unknown", "none", "null"})


def parse_peer_node_diagnostics(payload: object) -> NodeDiagnostics:
    """Parse a peer diagnostics payload while ignoring additive fields.

    Operational diagnostics are deliberately a tolerant wire boundary: a newer
    peer may add counters that an older collector does not know. Correctness-
    bearing events, commands, and state snapshots continue to use strict model
    validation and do not call this helper.

    Args:
        payload: JSON-compatible response body from a peer diagnostics endpoint.

    Returns:
        The known portion of the peer's diagnostics bundle.
    """

    return NodeDiagnostics.model_validate(payload, extra="ignore")


def compare_diagnostics_builds(
    reference: NodeRuntimeDiagnostics,
    candidate: NodeRuntimeDiagnostics,
) -> NodeDiagnosticsVersionStatus:
    """Compare the known package and source identities of two diagnostics bundles.

    Args:
        reference: Runtime identity of the API collecting cluster diagnostics.
        candidate: Runtime identity reported by one peer.

    Returns:
        ``version_mismatch`` for any positive disagreement, ``current`` when the
        available identifiers establish equality, otherwise ``unknown``.
    """

    reference_version = _known_identifier(reference.skulk_version)
    candidate_version = _known_identifier(candidate.skulk_version)
    if (
        reference_version is not None
        and candidate_version is not None
        and reference_version != candidate_version
    ):
        return "version_mismatch"

    reference_commit = _known_identifier(reference.skulk_commit)
    candidate_commit = _known_identifier(candidate.skulk_commit)
    if (
        reference_commit is not None
        and candidate_commit is not None
        and reference_commit != candidate_commit
    ):
        return "version_mismatch"

    # A source checkout reporting a commit must be compared with another known
    # commit. Falling back to an equal package version here would hide the
    # same-version/different-commit deployment window that motivated #293.
    if (reference_commit is None) != (candidate_commit is None):
        return "unknown"
    if reference_commit is not None and candidate_commit is not None:
        return "current"
    if reference_version is not None and candidate_version is not None:
        return "current"
    return "unknown"


def aggregate_diagnostics_version_status(
    statuses: Sequence[NodeDiagnosticsVersionStatus],
) -> ClusterDiagnosticsVersionStatus:
    """Reduce per-node build comparisons to one cluster diagnostics status.

    Args:
        statuses: Per-node comparison results, including the local node.

    Returns:
        ``mixed`` when any mismatch is known, ``consistent`` when every node is
        current, otherwise ``unknown``.
    """

    if any(status == "version_mismatch" for status in statuses):
        return "mixed"
    if statuses and all(status == "current" for status in statuses):
        return "consistent"
    return "unknown"


def _known_identifier(value: str) -> str | None:
    """Normalize one reported build identifier, excluding unknown sentinels."""

    normalized = value.strip().lower()
    return None if normalized in _UNKNOWN_BUILD_IDENTIFIERS else normalized
