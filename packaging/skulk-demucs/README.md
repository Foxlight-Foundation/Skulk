# Native source-separation adapter

This initiative-only component implements a supervised, per-job HTDemucs
process. It is **not yet a provisionable engine or public API**. There is no
model-mount/bootstrap dispatch or ready capacity advertisement for it. Studio's
stem control remains disabled until task, resource admission, addressed input /
four-result delivery, persistence and installed acceptance are connected.

The base Skulk installation does not install this runtime or its requirements.
`requirements.in` describes the candidate dependency policy. The `locks/` CPU
profiles close every transitive wheel by version and SHA-256 for macOS ARM64,
Linux x86_64 and Linux ARM64. `python.json` pins the complete standalone CPython
3.12.13 archive for each target, including its original release URL, size and
digest. Never install these dependencies into the resident Core or Studio
environment. Publication and Core provisioning remain separate gates.

`source.json` binds the upstream inference library. This owned adapter uses
`demucs.hf`'s exact safetensors loader and `demucs.apply`, with its own bounded
libsndfile WAV I/O. Its dependencies are pinned separately from upstream's general
audio CLI. Hub downloading remains outside the native runtime. The package builder
bundles the exact library source and licenses with this adapter; users do not
need a Git checkout. Candidate CI builds and installs a sealed package without
development import paths before exercising the actual model.

## Reproducible candidate packages

The build controller uses Core's Python 3.13, separately from the bundled native
Python 3.12.13. All intake, build and installation output must live **outside the
source checkout**: the producer refuses dirty or uncommitted source inputs.

1. Fetch the exact author commit in `source.json` into a clean source directory.
2. Run `fetch_wheels.py <profile> --output <intake>` with one of `macos-arm64`,
   `linux-amd64` or `linux-arm64`. It downloads only objects matching the existing
   target hash lock and interpreter pin. Changed cached objects, incomplete
   inventories and foreign archives fail; the command never installs or publishes.
3. Run `build_wheel.py` with explicit `--repository`, `--source`, `--wrapper`,
   `--wheelhouse`, `--lock`, `--python-archive`, `--target` and `--output` paths.
   `--wrapper` must be this checkout's `packaging/skulk-demucs/skulk_demucs`.
   The producer checks the target ABI and actual dependency metadata closure,
   dereferences the canonical contract into regular bytes and includes both
   component notices and every dependency's original wheel. SoundFile must
   include its platform libsndfile library. Weights remain a separate download.
4. Repeat the build into another fresh output directory and compare complete
   wheel bytes. Stored ZIP members, fixed timestamps and sorted records make the
   candidate deterministic. Its adjacent JSON records complete size/SHA-256,
   payload identity, platform, interpreter and exact source revisions.
5. Run `verify_package.py --wheel <candidate> --sha256 <expected> --output <new-root>`
   on the target OS/CPU. It verifies the entire package, manifest and member
   inventory, safely extracts the bounded interpreter archive, creates a private
   environment and installs with no index, no cache and the complete hash lock.
   No developer source path or pre-cached managed interpreter is required.
   Expanded import code and native libraries are checked against the sealed
   wheel bytes, including rejection of missing or injected files. Installed
   RECORDs are not trusted as integrity evidence. The output root is never
   reused or overwritten; failed validation remains available for inspection.

The verifier is a **candidate validation tool**, not the production engine-cache
installer. Its completion receipt does not authorize a backend or support claim.
Normal cache reuse/invalidation, installation supervision, resource admission,
GPU-device binding, accelerator packages and signed publication are still needed.
CPU package execution alone does not qualify an accelerator or full-length
workload. The workflow retains candidate artifacts; it does not promote them.

## Local protocol

Core's `worker/runner/demucs/process.py` starts the isolated interpreter with
`-I -B -m skulk_demucs --job <owned-job> --parent-pid <owner>` in its own process
session. The private job has exact worker-owned paths and a device/thread/segment
policy; public request metadata cannot contain paths or native options. The
runtime forces offline Hub loading and disables MPS CPU fallback. CUDA/HIP use
FP32 with TF32 disabled. A weightless math probe is not loaded model readiness.

`skulk_demucs/contracts.py` is a repository symlink to the canonical dependency-
light `src/skulk/shared/types/audio_separation.py`. Packaging must dereference
this link into ordinary wheel bytes: native environments never need the Core
source tree. This keeps the two sides' strict JSON validation identical.

Only the exact HTDemucs artifact is accepted:

- Author source `adefossez/demucs` at
  `829d21890023800713e5dd231c67e43ecc06069b`, version 4.1.0.
- Model `adefossez/HTDemucs` at
  `cbc8a9b1a87023b7fd74e7b3412e6321c0eab003`.
- `955717e8.safetensors`, 84,025,440 bytes, SHA-256
  `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`.

The adapter checks the complete hash and constructor before the safetensors
loader. It never calls upstream `get_model`, Hub model management or its legacy
pickle fallback. The source is verified as finite uncompressed mono/stereo WAV,
1–120 seconds, <=64 MiB encoded and <=128 MiB decoded. SciPy's pinned polyphase
FIR resamples to 44.1 kHz before mono is duplicated for the stereo model. Its
coefficient workspace is linear in the reduced rates, avoiding Julius's large
phase bank for valid coprime rates such as 44,101 Hz. Time zero is retained; the
duration uses the nearest integer frame count. At most one final fractional-frame
sample is removed from SciPy's ceiling-length result; missing audio is never
padded. Julius remains an upstream Demucs internal dependency.

Input normalization/denormalization follows the pinned upstream tensor API,
with one reproducible seeded shift, split inference and overlap 0.25. The four
estimated sources are drums, bass, other and vocals. They are not recovered
original master tracks. Each output is PCM24 stereo WAV at 44.1 kHz and the same
frame count. One common attenuation prevents PCM clipping without independently
rescaling stem balance. This export cannot restore source precision or detail.

Completion is a bounded JSON file plus four independently checked WAVs; no
samples or encoded audio enter the JSON protocol. Each WAV is <=64 MiB, the
complete set <=256 MiB. The owner verifies every size/digest/geometry, retaining
no result for missing/corrupt/partial output. Progress is bounded and replaceable.
Cancellation, resource refusal, process crash, invalid output and deadline
terminate the complete owned group before partial-file removal. If process
cleanup fails, ownership/files remain for recovery and the failure propagates.
The native process also kills its own group if the owner disappears.

## Validation

Core contract/media/process tests use `pytest` under the normal Core environment.
The process tests use a tiny standard-library fake runtime and real subprocess
trees; they do not establish Demucs model quality or accelerator qualification.

Within a separately prepared candidate native environment, make this package
importable and run `python packaging/skulk-demucs/verify_runtime.py` plus a strict
type check against that interpreter. This exercises real libsndfile/torch APIs,
shared gain, finite aligned output, original protection, exact-model refusal and
coprime-rate resampling memory/signal regressions.
Real-model execution and listening evidence remain separate from these analytic
tests. `scripts/qualify_demucs_native.py` exercises the actual isolated CLI and
Core supervisor using explicit local weights, with a mono/odd-frame resampling
fixture or an explicit original WAV. The three-platform candidate workflow builds
twice, compares complete bytes, installs the actual package offline, runs native
codec checks and fetches only the immutable test weight object separately for
the supervised model smoke. It retains exact package identities and measurements.
This is packaged CPU execution evidence, separate from normal provisioned-engine
and installed Studio acceptance. The smoke requests a two-second **split chunk**.
The pinned HTDemucs model retains `use_train_segment=True` and pads each forward
pass to its 7.8-second training window, even for that shorter chunk. Shorter
splits are not a lower-memory model mode. Admission must account for the trained
forward window plus the full source and four-result buffers; this short-source
smoke does not qualify the complete production workload's memory requirement.
Process RSS and CUDA/HIP allocator peaks are separately named measurements,
not total device or host memory requirements. Native pipeline timing excludes
interpreter/import startup and parent verification; the separate `supervision.json`
measurement includes the complete supervised process wall time.
