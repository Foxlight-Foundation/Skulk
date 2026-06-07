"""Shared Skulk version helpers."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=1)
def get_skulk_version() -> str:
    """Return the installed Skulk package version.

    The distribution is named ``skulk`` since the 2026-06 rename; the
    legacy ``exo`` distribution name is probed as a fallback for
    not-yet-resynced environments.
    """
    for distribution_name in ("skulk", "exo"):
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            continue
    return "unknown"


def get_skulk_version_label() -> str:
    """Return a user-facing Skulk version label."""
    return f"skulk v{get_skulk_version()}"
