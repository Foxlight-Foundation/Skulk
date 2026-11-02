# Native source-separation adapter

This initiative-only component implements a supervised, per-job HTDemucs
process. It is **not yet a provisionable engine or public API**. There is no
model-mount/bootstrap dispatch or ready capacity advertisement for it. Studio's
stem control remains disabled until task, resource admission, addressed input /
four-result delivery, persistence and installed acceptance are connected.

The base Skulk installation does not install this runtime or its requirements.
`requirements.in` describes the candidate dependency policy; complete immutable
offline wheelhouses, platform hash locks, provenance and provisioning are still
publication gates. Never install it into the resident Core or Studio environment.

`source.json` binds the upstream inference library. This owned adapter uses
`demucs.hf`'s exact safetensors loader and `demucs.apply`, with its own bounded
libsndfile WAV I/O. Its dependencies are pinned separately from upstream's general
audio CLI. Hub downloading remains outside the native runtime. The package builder
must bundle the exact library source and licenses with this adapter; users do not
need a Git checkout. Candidate CI uses staged source paths and proves execution,
while immutable packaged installation remains a separate gate.

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
1–120 seconds, <=64 MiB encoded and <=128 MiB decoded. Mono is explicitly
duplicated for the stereo model. Julius resamples to 44.1 kHz using the nearest
integer frame count; no guessed duration, trim, offset or silence is appended.

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
shared gain, finite aligned output, original protection and exact-model refusal.
Real-model execution and listening evidence remain separate from these analytic
tests. `scripts/qualify_demucs_native.py` exercises the actual isolated CLI and
Core supervisor using explicit local weights, with a mono/odd-frame resampling
fixture or an explicit original WAV. The three-platform candidate workflow fetches
only the immutable test weight object separately, runs that smoke and retains its
measurements. It uses a development import path, so it is not normal packaged
installation evidence. Process RSS and CUDA/HIP allocator peaks are separately named measurements,
not total device or host memory requirements. Native pipeline timing excludes
interpreter/import startup and parent verification; the separate `supervision.json`
measurement includes the complete supervised process wall time.
