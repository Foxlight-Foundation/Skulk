"""The lazy provisioning facade retains its established public objects."""

from skulk import provisioning
from skulk.provisioning import audio_cpp, comfy, llama_server, manifest


def test_existing_exports_are_the_original_implementation_objects() -> None:
    """Loading one engine through the facade preserves identity and signatures."""
    assert provisioning.prepare_audio_cpp is audio_cpp.prepare_audio_cpp
    assert provisioning.rehydrate_cached_audio_cpp is audio_cpp.rehydrate_cached_audio_cpp
    assert provisioning.dormant_comfy is comfy.dormant_comfy
    assert provisioning.ensure_comfy is comfy.ensure_comfy
    assert provisioning.managed_comfy_install is comfy.managed_comfy_install
    assert provisioning.provision_comfy is comfy.provision_comfy
    assert provisioning.select_comfy_variant_chain is comfy.select_comfy_variant_chain
    assert provisioning.dormant_llama_server is llama_server.dormant_llama_server
    assert provisioning.ensure_llama_server is llama_server.ensure_llama_server
    assert provisioning.managed_llama_server_path is llama_server.managed_llama_server_path
    assert provisioning.provision_llama_server is llama_server.provision_llama_server
    assert provisioning.select_variant is llama_server.select_variant
    assert provisioning.COMFY_PIN is manifest.COMFY_PIN
    assert provisioning.LLAMA_SERVER_PIN is manifest.LLAMA_SERVER_PIN
    assert provisioning.EngineVariant is manifest.EngineVariant
