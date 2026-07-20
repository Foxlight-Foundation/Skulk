"""Managed engine provisioning (#614 Phase 3): the store pattern for binaries.

Pinned, checksummed prebuilt engine builds fetched on demand so a new user
never builds llama.cpp. Manifest in :mod:`skulk.provisioning.manifest`;
llama-server logic in :mod:`skulk.provisioning.llama_server`.
"""

from skulk.provisioning.llama_server import (
    dormant_llama_server,
    ensure_llama_server,
    managed_llama_server_path,
    provision_llama_server,
    select_variant,
)
from skulk.provisioning.manifest import LLAMA_SERVER_PIN, EngineVariant

__all__ = [
    "LLAMA_SERVER_PIN",
    "EngineVariant",
    "dormant_llama_server",
    "ensure_llama_server",
    "managed_llama_server_path",
    "provision_llama_server",
    "select_variant",
]
