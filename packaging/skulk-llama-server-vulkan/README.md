# skulk-llama-server-vulkan

The pip-installable Vulkan `llama-server` build for [Skulk](https://github.com/Foxlight-Foundation/Skulk)'s served GGUF engine (Linux x86_64).

The wheel carries the Foxlight-built `llama-server` and `ggml-rpc-server` binaries, compiled from the pinned upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) release with the Vulkan backend, plus the Khronos Vulkan loader (Apache-2.0) bundled alongside, so it has no Python dependencies at all. The system's one prerequisite is the GPU driver's Vulkan ICD: `mesa-vulkan-drivers` on AMD, the NVIDIA driver's ICD on bare-metal NVIDIA.

```bash
uv pip install skulk-llama-server-vulkan
llama-server-vulkan --list-devices
```

Skulk's engine provisioning discovers the installed wheel automatically and wires it as the node's served engine. Version scheme: `0.<llama.cpp build>.<packaging revision>`, in lockstep with the CUDA sibling wheel and Skulk's engine pin. Built from source in the Skulk repository's `engine-wheel` workflow and published via PyPI trusted publishing.
