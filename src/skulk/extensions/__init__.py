"""Skulk extension (plugin) system.

Separately installed packages register a factory in the ``skulk.extensions``
entry-point group; Skulk discovers them at startup and calls their hooks as
first-class citizens of the fabric. See :mod:`skulk.extensions.types` for the
contract and its invariants, and :mod:`skulk.extensions.loader` for
discovery, version gating, and guarded dispatch.
"""

from skulk.extensions.loader import (
    ENTRY_POINT_GROUP,
    LoadedExtensions,
    load_extensions,
    resolve_skulk_version,
)
from skulk.extensions.types import (
    BaseChatMiddleware,
    ChatMiddleware,
    ChatResponseSummary,
    EmbedTexts,
    ExtensionContext,
    SkulkExtension,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "BaseChatMiddleware",
    "ChatMiddleware",
    "ChatResponseSummary",
    "EmbedTexts",
    "ExtensionContext",
    "LoadedExtensions",
    "SkulkExtension",
    "load_extensions",
    "resolve_skulk_version",
]
