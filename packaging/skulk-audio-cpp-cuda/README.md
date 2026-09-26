# Skulk audio.cpp CUDA engine

This separately installed wheel contains audio.cpp v0.8.2 at source commit
`4d88768fbcae4e6eb3352c6ab1422dabb7d90b58`, compiled for only ACE-Step
and MiniMax Music 3 with CPU and CUDA backends. It carries the two pinned
loader model specs and incorporated licenses. Model weights remain separate.

The extracted wheel uses the host CUDA 12 user-space libraries: `libcudart.so.12`,
`libcublas.so.12`, `libcublasLt.so.12`, and `libnccl.so.2`, plus the NVIDIA
driver's `libcuda.so.1`. Install them in the system loader path before mounting
a CUDA music model. Skulk verifies the pinned binary with `--version` and
`--list-devices` and leaves the CUDA lane unavailable with a loader diagnostic
if a library is missing. The CUDA package is compiled for GB10 SM 12.1;
Skulk requires that exact live hardware class and a signed claim naming it.
A wheel build or successful CPU probe does not establish usable CUDA music generation.
Skulk publishes `audio_cpp-cuda` only after a device probe, and a model can
be placed there only with a matching signed support claim.

The wheel index refuses to replace a published filename with different bytes.
Rebuilds require a new package version and updated Skulk SHA-256 pins.
