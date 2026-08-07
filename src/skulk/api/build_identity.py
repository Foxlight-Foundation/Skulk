"""Build-identity normalization and comparison for operational APIs."""

from collections.abc import Iterable

_UNKNOWN_BUILD_IDENTIFIERS = frozenset({"", "unknown", "none", "null"})
_HEXADECIMAL_DIGITS = frozenset("0123456789abcdef")


def known_build_identifier(value: str) -> str | None:
    """Normalize one reported build identifier, excluding unknown sentinels."""

    normalized = value.strip().lower()
    return None if normalized in _UNKNOWN_BUILD_IDENTIFIERS else normalized


def git_commit_identifiers_match(reference: str, candidate: str) -> bool:
    """Return whether two known Git commit identifiers name the same commit.

    Git may advertise the same object with different unique-prefix lengths, so
    hexadecimal identifiers are equivalent when either is a prefix of the
    other. Non-hexadecimal values require exact equality.

    Args:
        reference: Commit identifier reported by the reference node.
        candidate: Commit identifier reported by another node.

    Returns:
        True when both identifiers are known and identify the same Git object.
    """

    normalized_reference = known_build_identifier(reference)
    normalized_candidate = known_build_identifier(candidate)
    if normalized_reference is None or normalized_candidate is None:
        return False
    if normalized_reference == normalized_candidate:
        return True
    if not (
        set(normalized_reference) <= _HEXADECIMAL_DIGITS
        and set(normalized_candidate) <= _HEXADECIMAL_DIGITS
    ):
        return False
    return normalized_reference.startswith(
        normalized_candidate
    ) or normalized_candidate.startswith(normalized_reference)


def git_commit_identifiers_disagree(values: Iterable[str]) -> bool:
    """Return whether any pair of known Git commit identifiers conflicts.

    Unknown sentinels are omitted so an identity telemetry gap does not create
    a false mismatch.

    Args:
        values: Commit identifiers reported by live nodes.

    Returns:
        True when at least two known identifiers cannot name the same commit.
    """

    known_values = tuple(
        value for value in values if known_build_identifier(value) is not None
    )
    return any(
        not git_commit_identifiers_match(reference, candidate)
        for index, reference in enumerate(known_values)
        for candidate in known_values[index + 1 :]
    )
