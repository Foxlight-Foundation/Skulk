"""Skulk extension (plugin) system.

Separately installed packages register a factory in the ``skulk.extensions``
entry-point group; Skulk discovers them at startup and calls their hooks as
first-class citizens of the fabric. See :mod:`skulk.extensions.types` for the
contract and its invariants, and :mod:`skulk.extensions.loader` for
discovery, version gating, and guarded dispatch.
"""

from skulk.extensions.calls import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    MAX_CALL_PAYLOAD_BYTES,
    MAX_CALL_TIMEOUT_SECONDS,
    CapabilityCall,
    CapabilityError,
    CapabilityErrorCode,
    CapabilityResult,
    call_failure,
)
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
    CallCapability,
    CapabilityCallHandler,
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
from skulk.extensions.validation import validate_against_schema

__all__ = [
    "DEFAULT_CALL_TIMEOUT_SECONDS",
    "ENTRY_POINT_GROUP",
    "MAX_CALL_PAYLOAD_BYTES",
    "MAX_CALL_TIMEOUT_SECONDS",
    "AdvertiseCapability",
    "BaseChatMiddleware",
    "CallCapability",
    "CapabilityCall",
    "CapabilityCallHandler",
    "CapabilityDescriptor",
    "CapabilityError",
    "CapabilityErrorCode",
    "CapabilityResult",
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
    "call_failure",
    "descriptor_revision",
    "load_extensions",
    "resolve_skulk_version",
    "snapshot_cluster",
    "validate_against_schema",
]
