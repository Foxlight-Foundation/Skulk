"""Skulk extension (plugin) system.

Separately installed packages register a factory in the ``skulk.extensions``
entry-point group; Skulk discovers them at startup and calls their hooks as
first-class citizens of the fabric. See :mod:`skulk.extensions.types` for the
contract and its invariants, and :mod:`skulk.extensions.loader` for
discovery, version gating, and guarded dispatch.
"""

from skulk.extensions.capabilities import (
    CapabilityDescriptor,
    CapabilityIoMode,
    descriptor_revision,
)
from skulk.extensions.loader import (
    ENTRY_POINT_GROUP,
    LoadedExtensions,
    load_extensions,
    resolve_skulk_version,
)
from skulk.extensions.telemetry import ClusterNodeView, snapshot_cluster
from skulk.extensions.types import (
    AdvertiseCapability,
    BaseChatMiddleware,
    CapabilityProvider,
    ChatMiddleware,
    ChatResponseSummary,
    DescribeNode,
    EmbedTexts,
    ExtensionContext,
    ReadClusterTelemetry,
    SkulkExtension,
    SupportsExtensionStartup,
    WithdrawCapability,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "AdvertiseCapability",
    "BaseChatMiddleware",
    "CapabilityDescriptor",
    "CapabilityIoMode",
    "CapabilityProvider",
    "ChatMiddleware",
    "ChatResponseSummary",
    "ClusterNodeView",
    "DescribeNode",
    "EmbedTexts",
    "ExtensionContext",
    "LoadedExtensions",
    "ReadClusterTelemetry",
    "SkulkExtension",
    "SupportsExtensionStartup",
    "WithdrawCapability",
    "descriptor_revision",
    "load_extensions",
    "resolve_skulk_version",
    "snapshot_cluster",
]
