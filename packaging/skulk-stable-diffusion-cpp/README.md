# Native image engine payloads

`scripts/build_stable_diffusion_cpp_wheel.py` assembles prebuilt Metal, CUDA,
Vulkan, and ROCm packages from the immutable source and recorded lifecycle patch.
No CPU-only package is built. The wheel contains the native server, license
notices, payload digests, and build provenance. It contains no model weights.

The server has no embedded frontend. Its private native API requires an
ephemeral bearer token, bounds queued/retained jobs and encoded bodies, supports
active cancellation, releases terminal jobs after collection, and exits when
its owning runner dies. These changes are carried in `server-lifecycle.patch`.

Use portable compilation (`GGML_NATIVE=OFF`); Metal requires deployment target
15.0 and an explicit portable ARM instruction baseline. GPU builds still have
their internal CPU helper kernels, but Skulk does not advertise CPU serving.
CUDA builds must set `GGML_CUDA_NCCL=OFF`: each image instance uses one GPU, so
the wheel must not acquire a collective-communication library dependency from
the build host. The assembler rejects CUDA builds without this explicit flag.

Candidate descriptors are **unqualified**. They cannot enter the installable
manifest until loader/device, generation, failure/lifecycle, memory, and fresh
installation evidence has passed on the exact target hardware. Publication is
append-only: never overwrite a published package or reuse a version with new bytes.
