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

Version 0.1.0 is published in the Foxlight engine channel. Its complete wheel
SHA-256 is `e2ced0b7c5b176c9285b047eb908c5a63b1322c22b0ce5d8a07e504bf02bfdee`;
its runtime payload identity is
`839dcad5cb57cc6cf1a4215cd3dbd9422d97a2b2d21963dbbc1bf7d8bf14bc89`.
Candidate run `38631488818` binds the source at `48e11c3d`; promotion run
`38631848725` retained those exact bytes. Published-wheel installation and
offline reuse are separate from model fit and physical render qualification.

The CPython 3.13 wheel requires macOS 15 or later on ARM64 with working Metal.
GPU marketing names and missing benchmarks are not compatibility restrictions.
The engine runs one host per job with transformer block streaming and staged
encode/decode. Real memory admission remains mandatory; aggregate cluster RAM
does not establish that a native video job fits.

Dependency updates require regenerating `requirements.lock` with the macOS 15
deployment target and CPython 3.13, then verifying a complete binary wheelhouse.
Increment the immutable wheel version whenever any packaged bytes change.

Advanced source workflows use a new immutable wheel version. The descriptor and probe carry exact `audio_conditioning`, `retake`, `extend` and `prompt_relay` features. The builder bundles `worker/runner/video_input.py` as `skulk_mlx_video.media` without importing Skulk into the engine. Source preparation streams bounded PyAV decode and never loads model weights. The advanced adapter binds upstream ffprobe calls to verified worker source facts, preserves the requested waveform for retake, and removes audio for silent A2V output. Timed scenes receive deterministic latent-grid allocation. These implementation controls remain distinct from model fit and feature qualification.

The advanced v0.2.0 wheel was attested at initiative commit `a0dbfa74` by candidate run `38655354771` and promoted unchanged by run `38655556829`. Its complete SHA-256 is `151d2f57a8b4a0b9f849057ed2412fb4173733f9f59a41365dcbf9e715c93941`, payload identity `89249f099dbc74780bc6a75a1c1fc22571d271e10d9bdec46230790fc1f677a9`, and size 119,958,178 bytes. Owned URL/index read-back, a fresh real published-URL install, archive-backed integrity, Metal, offline reuse and restart probe passed. New on-demand installations select v0.2.0; the immutable v0.1.0 descriptor remains for existing exact placements. This establishes distribution and protocol availability, while model-fit and render qualification remain separate.

The published v0.2.1 wheel corrects render-wide peak accounting and decode progress. Candidate `38666657955`, attested at reviewed source `528a5383` and accepted in initiative merge `b6716241`, was promoted unchanged by `38668395794`. Complete SHA-256 is `a08f846ddd88469e0ed5aa897596f5d584362739e06a10a800fd02db64766fd3`, payload identity `58523bb6c166dc864730afbd75c047ee8d0a040be9066534f2901931f073d6d6`, and size 119,959,343 bytes. Live URL/index read-back and fresh local/physical-host integrity, Metal, offline reuse and restart probes passed. A physical 512×512/49-frame render reported the whole-render peak and decode event, with video byte-identical to the v0.2.0 baseline. This distribution and scoped engine evidence does not establish all-workflow or all-hardware qualification. New installs prefer v0.2.1; earlier exact descriptors remain available.

The standalone `video_observability.py` helper quiets the pinned decoder's private profiler, preserving earlier allocator peaks, and observes the actual decode/save call. Core request admission can supply the existing pinned upstream decode-budget control after environment sanitization. That parent-side resource setting leaves this immutable wheel and its local JSON protocol unchanged. Sampling, decoder and pipeline selections remain explicit; resource admission does not silently replace them.

Core integrity verification hashes archive members in bounded chunks and retains digests rather than expanded library bytes. Bundled dependency archives are copied to temporary files before inspection so nested ZIP seeks cannot allocate an entire wheel. These temporary files close after verification; installed-code, dependency, isolation and unexpected-file checks remain in force. Verification must leave headroom for the following render.
