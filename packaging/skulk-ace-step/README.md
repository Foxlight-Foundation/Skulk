# Native ACE-Step package inputs

These are candidate inputs for a separately installed native ACE-Step engine.
They are independent of the qualified audio.cpp packages and model weights.

- Native library source: exact revision in `source.json`.
- Interpreter: the independently verified Python 3.12.13 standalone archive
  catalog in `python.json`, already used by the native CPU packaging substrate.
- Dependencies: exact common versions retained from the accepted native M0
  experiments, with the pinned source's Torch/Torchaudio 2.10 CPU family.
- Binary-only, Python 3.12 dependency resolution succeeded for macOS ARM64,
  Linux x86_64 and Linux ARM64. Each lock retains distribution SHA-256 hashes
  and package-index attribution. Linux uses the explicit PyTorch CPU backend.

Reproduce a CPU lock with `uv pip compile requirements.in --python-version 3.12
--python-platform <target> --torch-backend cpu --only-binary :all:
--generate-hashes --emit-index-annotation --no-header --output-file <lock>`.
Targets are `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, and
`aarch64-unknown-linux-gnu`. Use the public PyPI default index for common packages.

The `archives/` inventories now select one exact compatible wheel per locked
distribution and the standalone interpreter object for each CPU target. All 53
actual wheels per target were independently rehashed and checked for matching ABI
and complete transitive `Requires-Dist` closure, including SoundFile’s native
library. Torch/Torchaudio use the explicit CPU index on all targets; macOS
versions have no `+cpu` suffix, so that suffix cannot select their index.

The owned adapter, private resident CLI and deterministic producer are implemented
in this directory. Fresh package execution, Core runner integration and full-model
qualification are distinct validation steps. The archive inputs establish
selection and metadata closure only. They do not publish a package,
advertise a ready backend or qualify a model on any architecture.
Keep model weights as separate, explicitly selected immutable downloads. The
runtime must own its reviewed model implementation; repository Python, UI,
auto-download and language-model planner startup are outside its execution path.

## Native lifetime and contracts

The private CLI accepts `--serve <configuration.json> --parent-pid <owner>`.
The worker must start it in a dedicated session with the actual supervising PID.
Parent death, a protocol error or any inference failure retires the entire native
process group. An upstream timeout can leave a daemon inference thread active;
the process must never admit another request or delete that thread's partial
files. The owner drains the process before cleaning its private lifetime.

Core's canonical `shared/types/native_music.py` is linked into this source tree
and copied as ordinary bytes into the eventual package. It defines strict
generation, Add a part, Rework and Continue requests, contiguous resident
sequences, progress/ready/completed/failed frames and verified WAV facts. Each
control record is bounded to 256 KiB; stdout is reserved for those records while
native diagnostics go to stderr. No media or caller-selected native options
travel through this protocol. There is one active generation per loaded model.

Before importing model constructors, the wrapper copies and verifies the full
three-component model inventory and its three runtime-owned implementation
files into an exclusive private view. Its typed acquisition hook prevents the
upstream initializer from fetching unrelated default/Turbo weights. Hub caches
are private and offline; automatic compilation, quantization, CPU offload,
alternate MLX implementations and language-model planning are disabled.

The requested device executes a small math probe. Readiness then inspects all
three loaded components rather than echoing configuration. The pinned source
uses FP32 on CPU, MPS and ROCm for its diffusion model; CUDA uses BF16 when
available and FP16 otherwise. The MPS autoencoder independently uses FP16.
Readiness and completed results retain each component's observed policy. These
are module observations, not claims about every operator's accumulation type.

## Source and output policy

The adapter uses the direct base-model policy: 50 steps, guidance 7, repaint
strength 0.5, explicit resolved seed in both native seed paths, one output, no
CoT/planner, no DCW or automatic loudness normalization. Add a part uses native
`lego`; Rework and Continue use `repaint`. Continue supplies the canonical
context with a silent extension and targets only that tail.

Original WAV bytes remain unchanged. Conditioning uses stereo 48 kHz FLOAT WAV
with an explicit ceiling frame map for resampling. If float input or filter
overshoot exceeds unit headroom, one recorded common attenuation prevents the
native source loader from silently clipping. The wrapper observes the real
native source processor and compares its complete tensor with those exact
canonical samples. Completion binds original and conditioning digests plus the
conditioning gain. Generation cannot contain source context.

The wrapper owns `result.wav` and `result.json`, written exclusively with 0600
permissions under the request's private directory. Results are native-rate
stereo PCM24 WAV, capped at 120 seconds and 64 MiB; edited results must have the
exact requested frame count. Only necessary output headroom attenuation is
applied and reported. WAV facts and digests come from the actual encoded file.
Files and directory entries are synced before completion is emitted. Core must
independently verify the result and use its acknowledged media transport before
accepting custody. Native server-local output paths are never trusted.

The dedicated runtime checks exercise real pinned constructors/source preprocessing,
real codecs/resampling/tensors, complete data-copy ordering, process-group exits,
strict control admission, failure retirement and result custody. Diffusion is
controlled in wrapper tests. A full-size model run, package installation and
hardware qualification remain separate required evidence.

## Build and verify

`fetch_archives.py <profile> --output <outside-checkout>` collects only the exact
committed object URLs, lengths and digests. It does not resolve a new dependency
set from an index. Profiles are `macos-arm64`, `linux-amd64` and `linux-arm64`.
`build_ace_step.py` requires clean committed adapter/upstream repositories, the
selected committed profile lock, and the exact wheelhouse/interpreter archive.
It copies only `source-files.json`'s selection, records its explicit namespace
override and seals the three runtime-owned model implementation identities.
Large inputs stream through bounded buffers and are rehashed while copied.
Independent builds use sorted regular members, fixed timestamps and stored ZIP
entries. Outputs and intake caches must remain outside the source checkout.

Build-time checks run in Core's Python 3.13 environment:

```sh
uv run pytest packaging/skulk-ace-step/producer_tests
uv run basedpyright --project packaging/skulk-ace-step/pyrightconfig.build.json --pythonpath <Core-3.13-python>
```

Native checks live in `runtime_checks/check_*.py` so ordinary Core pytest never
imports model libraries through the wrong interpreter. After fresh offline
installation with `scripts/stage_ace_step_candidate.py`, run:

```sh
<installed-native-python> -I -B packaging/skulk-ace-step/verify_runtime.py
uv run basedpyright --project packaging/skulk-ace-step/pyrightconfig.json --pythonpath <installed-native-python>
```

The verifier requires dependencies and the native adapter to resolve inside that
private installation. Its child-process checks use that same installed package,
not a repository wrapper. `.github/workflows/ace-step-native-boundary.yml` builds
twice, compares complete bytes and executes fresh offline packages on all three
OS/CPU targets. This workflow does not download music weights or qualify music
generation, accelerator performance or installed Core admission.

The disabled `enable_normalization` option is the optional native postprocessor.
Pinned native decode still attenuates peaks above one before returning its tensor.
`output.gain` describes only additional PCM export attenuation; it is not a full
gain trace from diffusion latents. Qualification must measure actual host/GPU
working sets, including native decode's temporary CPU fallback, and verify
fractional-second source/extension geometry with real model weights.
