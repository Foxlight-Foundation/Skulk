"""Managed engine provisioning (#614 Phase 3): the store pattern for binaries.

Pinned, checksummed prebuilt engine builds fetched on demand so a new user
never builds llama.cpp. Manifest in :mod:`skulk.provisioning.manifest`;
llama-server logic in :mod:`skulk.provisioning.llama_server`.
"""

from skulk.provisioning.comfy import (
    dormant_comfy,
    ensure_comfy,
    managed_comfy_install,
    provision_comfy,
    select_comfy_variant_chain,
)
from skulk.provisioning.llama_server import (
    dormant_llama_server,
    ensure_llama_server,
    managed_llama_server_path,
    provision_llama_server,
    select_variant,
)
from skulk.provisioning.manifest import COMFY_PIN, LLAMA_SERVER_PIN, EngineVariant

__all__ = [
    "COMFY_PIN",
    "LLAMA_SERVER_PIN",
    "EngineVariant",
    "dormant_comfy",
    "ensure_comfy",
    "managed_comfy_install",
    "provision_comfy",
    "select_comfy_variant_chain",
    "dormant_llama_server",
    "ensure_llama_server",
    "managed_llama_server_path",
    "provision_llama_server",
    "select_variant",
]
