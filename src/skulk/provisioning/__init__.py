"""Managed engine provisioning (#614 Phase 3): the store pattern for binaries.

Pinned, checksummed prebuilt engine builds fetched on demand so a new user
never builds llama.cpp. Manifest in :mod:`skulk.provisioning.manifest`;
llama-server logic in :mod:`skulk.provisioning.llama_server`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skulk.provisioning.audio_cpp import (
        prepare_audio_cpp,
        rehydrate_cached_audio_cpp,
    )
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
    "prepare_audio_cpp",
    "rehydrate_cached_audio_cpp",
]


def __getattr__(name: str) -> object:
    """Load the backing implementation for one existing public facade export.

    Preserve the original objects and static signatures while allowing native
    package helpers to import without loading unrelated resident engines.
    Unknown names raise AttributeError so ordinary submodule imports still work.
    """
    match name:
        case "prepare_audio_cpp" | "rehydrate_cached_audio_cpp":
            from skulk.provisioning.audio_cpp import (
                prepare_audio_cpp,
                rehydrate_cached_audio_cpp,
            )

            return {
                "prepare_audio_cpp": prepare_audio_cpp,
                "rehydrate_cached_audio_cpp": rehydrate_cached_audio_cpp,
            }[name]
        case (
            "dormant_comfy"
            | "ensure_comfy"
            | "managed_comfy_install"
            | "provision_comfy"
            | "select_comfy_variant_chain"
        ):
            from skulk.provisioning.comfy import (
                dormant_comfy,
                ensure_comfy,
                managed_comfy_install,
                provision_comfy,
                select_comfy_variant_chain,
            )

            return {
                "dormant_comfy": dormant_comfy,
                "ensure_comfy": ensure_comfy,
                "managed_comfy_install": managed_comfy_install,
                "provision_comfy": provision_comfy,
                "select_comfy_variant_chain": select_comfy_variant_chain,
            }[name]
        case (
            "dormant_llama_server"
            | "ensure_llama_server"
            | "managed_llama_server_path"
            | "provision_llama_server"
            | "select_variant"
        ):
            from skulk.provisioning.llama_server import (
                dormant_llama_server,
                ensure_llama_server,
                managed_llama_server_path,
                provision_llama_server,
                select_variant,
            )

            return {
                "dormant_llama_server": dormant_llama_server,
                "ensure_llama_server": ensure_llama_server,
                "managed_llama_server_path": managed_llama_server_path,
                "provision_llama_server": provision_llama_server,
                "select_variant": select_variant,
            }[name]
        case "COMFY_PIN" | "LLAMA_SERVER_PIN" | "EngineVariant":
            from skulk.provisioning.manifest import (
                COMFY_PIN,
                LLAMA_SERVER_PIN,
                EngineVariant,
            )

            return {
                "COMFY_PIN": COMFY_PIN,
                "LLAMA_SERVER_PIN": LLAMA_SERVER_PIN,
                "EngineVariant": EngineVariant,
            }[name]
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
