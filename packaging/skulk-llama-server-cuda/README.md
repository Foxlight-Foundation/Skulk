# skulk-llama-server-cuda

The pip-installable CUDA `llama-server` build for [Skulk](https://github.com/Foxlight-Foundation/Skulk)'s served GGUF engine (Linux x86_64).

The wheel carries the Foxlight-built `llama-server` and `ggml-rpc-server` binaries, compiled from the pinned upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) release with CUDA enabled (upstream publishes no Linux CUDA prebuilt). The CUDA runtime is not rehosted here: it resolves from NVIDIA's official PyPI wheels (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`), which install as ordinary dependencies. The `llama-server-cuda` entry point puts those libraries on the loader path and execs the real binary, forwarding all arguments.

```bash
uv pip install skulk-llama-server-cuda
llama-server-cuda --list-devices
```

Skulk's engine provisioning discovers the installed wheel automatically and wires it as the node's served engine; no configuration is needed. A machine additionally needs the NVIDIA driver (anything where `nvidia-smi` works), which only NVIDIA can ship.

Version scheme: `0.<llama.cpp build>.<packaging revision>`; `0.10068.0` is the first packaging of upstream `b10068`. Built and published by the `engine-wheel` workflow in the Skulk repository via PyPI trusted publishing.
