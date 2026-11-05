"""Purpose-specific limits for managed declarations and signed runtime metadata.

These paths carry bounded schemas and signatures, independently of ordinary
invocation payloads, settings, credentials, media and control-request limits.
"""

from typing import Final

MAX_MANAGED_DESCRIPTORS: Final = 16
"""Largest declared capability set in one managed application."""

MAX_MANAGED_OWNER_DESCRIPTORS: Final = 128
"""Aggregate owner admission bound shared by discovery, dispatch and callbacks."""

MAX_MANAGED_DESCRIPTOR_BYTES: Final = 262144
"""Largest exact selected descriptor read, within the manifest's own byte cap."""

MAX_MANAGED_DISCOVERY_BYTES: Final = 524288
"""Largest describe or full-contract metadata response envelope."""

MAX_RUNTIME_METADATA_BYTES: Final = 524288
"""Largest signed runtime record across download, review, staging and recovery."""
