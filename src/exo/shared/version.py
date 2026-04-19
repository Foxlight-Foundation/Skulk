"""Shared Skulk version helpers."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=1)
def get_skulk_version() -> str:
    """Return the installed Skulk package version.

    The Python package is still published under the legacy `exo` distribution
    name for compatibility, but the returned version represents the Skulk app
    version shown in the UI and node metadata.
    """
    try:
        return version("exo")
    except PackageNotFoundError:
        return "unknown"


def get_skulk_version_label() -> str:
    """Return a user-facing Skulk version label."""
    return f"skulk v{get_skulk_version()}"
