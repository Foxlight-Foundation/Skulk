# skulk-llama-server-vulkan

The pip-installable Vulkan `llama-server` build for [Skulk](https://github.com/Foxlight-Foundation/Skulk)'s served GGUF engine (Linux x86_64).

The wheel carries the Foxlight-built `llama-server` and `ggml-rpc-server` binaries, compiled from the pinned upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) release with the Vulkan backend, plus the Khronos Vulkan loader (Apache-2.0) bundled alongside, so it has no Python dependencies at all. The system's one prerequisite is the GPU driver's Vulkan ICD: `mesa-vulkan-drivers` on AMD, the NVIDIA driver's ICD on bare-metal NVIDIA.

```bash
uv pip install --extra-index-url https://wheels.foxlight.ai/simple/ \
  skulk-llama-server-vulkan
llama-server-vulkan --list-devices
```

The index flag is required: new versions publish only to the Foxlight index, so a plain `uv pip install` resolves the last version mirrored to PyPI before 2026-08-30 rather than the current pinned build.

Skulk's engine provisioning discovers the installed wheel automatically and wires it as the node's served engine. Version scheme: `0.<llama.cpp build>.<packaging revision>`, in lockstep with the CUDA sibling wheel and Skulk's engine pin. Built from source in the Skulk repository's `engine-wheel` workflow and published to the Foxlight package index at `wheels.foxlight.ai`, which is the sole channel for new versions (releases published to PyPI before 2026-08-30 remain available but are not updated).

This wheel is the engine path for AMD GPU nodes and the fallback for bare-metal NVIDIA nodes when the CUDA wheel is unavailable; how it fits into Skulk's install and provisioning flow is documented in the [Build & Runtime Paths guide](https://foxlight-foundation.github.io/Skulk/build-and-runtime/).

The bundled binaries derive from llama.cpp (MIT) and the bundled `libvulkan` is the Khronos Vulkan Loader (Apache-2.0); both license texts ship in the wheel under `skulk_llama_server_vulkan/licenses/`.
