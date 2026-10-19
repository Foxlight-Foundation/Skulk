# Managed vLLM runtime recipes

These wheels contain the complete hashed dependency lock for one isolated
Python 3.12 runtime. They do not contain model weights or rebuild vLLM. The
upstream vLLM wheels are mirrored byte for byte on the append-only Foxlight
index; other dependencies retain their upstream channels and recorded hashes.

`scripts/build_vllm_runtime_wheel.py` deterministically builds each recipe and
verifies/downloads its upstream vLLM wheel. Publication promotes those exact
candidate bytes. Advance the packaging revision for any lock change; never
replace a published wheel. The provisioner verifies the recipe digest, installs
only hashed binary distributions in a separate managed environment, and checks
the dependency versions and GPU before making the install available.

The lock inputs are vLLM 0.32.0's cu129 (x86_64/aarch64) and rocm723 (x86_64)
distributions plus `ninja==1.13.2`, resolved for Python 3.12. CUDA installation
uses the cu129 Torch channel. ROCm uses the release's own complete wheel channel.
The exact locks, rather than a later resolver result, define the runtime.

Upstream vLLM inputs live under `https://wheels.foxlight.ai/runtime-inputs/vllm/`,
selected directly by each hashed recipe. They are deliberately excluded from the
generic PEP 503 index so CUDA and ROCm distributions cannot be confused by a
package resolver. The Foxlight recipe packages remain indexed normally.
