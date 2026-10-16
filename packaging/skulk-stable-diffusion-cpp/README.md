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

Candidate builds require representative loader/device, generation, lifecycle,
memory and fresh-install evidence before their exact immutable bytes are promoted.
That evidence verifies the engine package; it does not create a list of GPU models
allowed to install it. Eligibility uses actual ABI, CPU and compiled GPU targets.
Package publication is
append-only: never overwrite a published package or reuse a version with new bytes.

The candidate workflow builds all five OS/CPU/backend targets. Linux compilation
runs in digest-pinned Ubuntu 22.04 compiler images. CUDA x86 embeds SM 80/86/89
and SM 90 PTX; ARM CUDA selects SM 121a with CUDA 13. Neither compiled target
coverage nor a successful CI build establishes a measured model outcome on
another GPU. Compatible untested hardware remains eligible for a runtime attempt.
CUDA runtime/BLAS libraries are statically linked, while the NVIDIA driver stays
external. Vulkan includes its loader, with the vendor driver external. ROCm
candidate assembly includes resolved runtime libraries and the selected gfx942
code objects. Representative package tests establish evidence before package
promotion; model-card publication is governed independently by artifact integrity,
structural validation and provenance.
Component license notices accompany the runtime files. Nested runtime data is
included in the same immutable payload digest manifest as the executable.

Promotion uses a successful candidate run and an explicit filename-to-SHA-256
map, verifies build provenance against the original workflow/commit, and uploads
only the selected verified bytes without rebuilding. A successful build or
publication does not create model/device support claims. Publish remains a
manual workflow dispatch; record its immutable wheel identities in the private
engine ledger after each verified publication.
