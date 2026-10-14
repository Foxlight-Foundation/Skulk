# MLX video engine wheels

This package contains the immutable MIT-licensed MLX LTX pipeline source,
Foxlight's local-job adapter and every hash-locked binary dependency. Weights
remain separate selected model artifacts. No runtime source checkout, package
index resolution or independent model download is allowed.

The Foxlight adapter carries Skulk's Apache 2.0 license. The pinned pipeline
retains its MIT notice, and bundled dependencies retain their original wheel
license files. The aggregate package does not relabel these components as MIT.

The candidate workflow assembles and attests exact bytes. Promotion selects a
successful candidate by digest and verifies its attestation without rebuilding.
Add its descriptor to `provisioning/mlx_video.py` only after publication and
read-back verification. Candidate packages grant no model qualification claims.

The CPython 3.13 wheel requires macOS 15 or later on ARM64 with working Metal.
GPU marketing names and missing benchmarks are not compatibility restrictions.
The engine runs one host per job with transformer block streaming and staged
encode/decode. Real memory admission remains mandatory; aggregate cluster RAM
does not establish that a native video job fits.

Dependency updates require regenerating `requirements.lock` with the macOS 15
deployment target and CPython 3.13, then verifying a complete binary wheelhouse.
Increment the immutable wheel version whenever any packaged bytes change.
