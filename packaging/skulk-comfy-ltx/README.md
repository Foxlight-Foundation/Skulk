# Trusted LTX nodes for Comfy

This Foxlight wheel adds explicit LTX sampler and workflow nodes to the pinned
Comfy runtime. It is an extension of the existing Comfy engine. Its manifest binds
the exact source, node features and payload; weights remain separate selected
artifacts. The numeric sampler adaptation retains the upstream LTX license and
attribution, while Foxlight's integration code retains Apache 2.0 licensing.

Only published, attested and read-back-verified wheel descriptors may create
installation offers. New LTX placements bind the selected composite engine build
and use the shared `InstallEngine` path. A previous running instance retains its
original build and cannot acquire newer features by a telemetry refresh.

The interpreter and base engine remain isolated. Only the verified package folder
is whitelisted; other custom nodes and remote API nodes remain disabled. No source
checkout, request-time package resolution or model download occurs in this wheel.

Candidate source and tensor checks do not qualify a model or hardware class.

The issued `0.1.0` wheel provides Res2s, retake and extension on the existing CUDA
and ROCm lanes. Its complete archive is 48,584 bytes, SHA256
`3ff6a292521de798de59d171474a8ec041e741d3fcd4f59b0a51aa4c7e8879e3`.
The descriptor was registered after attestation verification and full owned-URL
read-back. Later features require a new version; issued bytes are immutable.

The solver runs on the reference's token-major video and audio layouts, then
restores Comfy's channel-first packed states at each prediction and callback.
Step-level SDE noise uses the pinned HQ pipeline's default seed (-1, with 9999
for its substeps); the requested render seed still owns initial noise and the
model call. The numerical check evaluates the pinned upstream patchifiers and
solver independently, including fractional masks and separate modality clocks.
Promotion also compares every package source and license member to the reviewed
checkout; an internally consistent wheel from a different branch is rejected.

Temporal editing keeps video and audio on their own causal VAE clocks. Retake
masks only the requested interval; extension freezes the complete source prefix
and restores its full-resolution context before spatial refinement. The output
nodes restore unchanged source RGB frames and normalized stereo PCM before
container encoding. This is a pre-mux preservation contract: lossy video/audio
codecs may change the final decoded values. A missing source audio track supplies
silent context; preserving audible output requires an actual track.

Source PCM is encoded through a shallow view of the selected audio VAE with
generic spatial cropping disabled. The pinned Comfy wrapper otherwise crops
audio's time axis to 4096-sample boundaries, which can discard enough context
to reject an ordinary retake. The view shares weights and managed placement;
the cached VAE and the normalized source waveform remain unchanged.

Run `uv run pytest packaging/skulk-comfy-ltx/tests` alongside the package's
strict type project. The numerical receipt also compares temporal masks to the
actual immutable upstream patchifiers and region-conditioning implementation.
