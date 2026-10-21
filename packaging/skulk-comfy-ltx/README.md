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

The issued `0.1.4` wheel adds a validated Union IC-LoRA bridge. It requires the actual loaded adapter to declare reference downscale factor two and a complete aligned guide clip, then delegates to pinned native Comfy guide creation and crop semantics. Prepared guides bypass annotation; RGB inputs use explicitly selected native Comfy Canny, SDPose/RT-DETR or DA3 chains. The publication checker evaluates the actual packed bridge against hash-bound native guide/dilation/crop methods on bounded tensors. This source contract does not qualify model weights or hardware.

Its complete archive is 65,552 bytes, SHA256
`bcafacf142ec1d782f192e5f7f0305e91203f589017db6f89d6424c17534f909`,
with payload identity
`9cb0a7990934537f50daaea9573aaf3f35860bf06e95486159236c5a8e35962c`.
The successful hosted candidate was promoted without rebuilding after verification
of its source-bound attestation and packed numerical contracts. Complete owned-URL
read-back through the ordinary downloader matches the attested archive and
reviewed source. All four earlier archives and index entries are unchanged. New
placements prefer this compatible issued version; retained instances keep their
original builds. GPU and model qualification remain separate.

The issued `0.1.3` wheel adds `prompt_relay` with combined Gemma4 encoding and clone-scoped video text-attention routing. It measures actual text columns and surviving query coordinates, uses the fixed native-compatible soft Gaussian profile, and preserves global/register/guide access. Audio keeps the combined prompt without temporal routing. Every sampling stage receives its own restorable model patch. The 32 MiB query-chunk budget avoids a full-video bias; original projections, all-head norms, position handling, gates and GPU kernels remain delegated to Comfy. Forward-scoped state clears after success, failure or cancellation; every query chunk honors Comfy's normal interrupt flag.

Its complete archive is 63,228 bytes, SHA256
`dade727b886fbcf3980f11c991ac35c70f1f5f98896c63ecf6a7cd06dac27b5e`,
with payload identity
`6b6a6b719952f292252fe78c1d02b851ad334cca8defc6dc2fd5878f49b4a7fd`.
The hosted candidate, reviewed source members, source-bound attestation, full
owned-URL/index read-back and ordinary engine downloader were verified before
registration. Earlier issued archives are unchanged. At its registration, new
placements preferred this compatible issued version; workflow and hardware
qualification remain separate.

The issued `0.1.2` wheel adds the DFR mask/keyframe bridge to Res2s and editing.
It is 51,797 bytes, SHA256
`6be37e297325d9ae7da2b185c0034f98e945745c6b9989f58507505f6ab7d842`,
with payload identity
`91edc941109ab83734c55a65c34eb47a6040ef502dd868457c41fa867b901184`.
The original hosted candidate's attestation, reviewed package/license bytes,
complete owned-URL download and ordinary downloader were verified before its
descriptor was registered. At that registration, new placements preferred it; hardware/model acceptance
remains separately scoped.

The earlier issued `0.1.1` wheel provides Res2s, retake and extension on the existing CUDA
and ROCm lanes, including full source-audio encoding for editing. Its complete
archive is 48,855 bytes, SHA256
`0b6c28318b7f0aeaded710dea7862cfb1700cda3dc421354f62de65c9f78ea4a`,
with payload identity
`df3189f3232051717b249cec0087d728f0808463183f866ee32e0b4414cd3b16`.
The descriptor was registered after attestation verification, full owned-URL
read-back and verification with the ordinary engine downloader. New placements
prefer the latest compatible issued overlay even when an older one is cached. The issued `0.1.0` descriptor
remains available for retained instances: 48,584 bytes, archive SHA256
`3ff6a292521de798de59d171474a8ec041e741d3fcd4f59b0a51aa4c7e8879e3`.
Later changes require a new version; issued bytes are immutable.

The `0.1.2` wheel implements a `dfr` feature and the
`SkulkLTXAddGeneratedKeyframes` bridge. It preserves dense upscaled video while
restoring a five-dimensional sampling mask; real image/guide masks are cloned
unchanged. Default grid positions retain stock behavior. If anchors occupy every
candidate, one detail frame uses the largest free interior gap (earliest on ties).
A fully occupied canvas is rejected rather than dropping detailing. Actual token
creation and conditioning remain delegated to the pinned Comfy node.

Only the exact issued descriptor enables text-only DFR; candidate source alone
creates no engine offer. Existing image-conditioned stock-node DFR and issued
0.1.0/0.1.1 identities remain intact. Candidate validation reproduces the original
dense-video failure against hash-bound pinned occupancy helpers, then checks
correct masks, preserved anchors and the short-clip fallback on bounded tensors.

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
