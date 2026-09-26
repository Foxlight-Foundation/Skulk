---
id: audio-cpp-gb10-qualification
title: audio.cpp GB10 CUDA qualification
---

This record captures the technical qualification for exact `music.generate`
claims for the two music cards below on the `nvidia:sm-12.1` hardware class.
The class describes GPU compute compatibility, not an individual Skulk node.
Both cards passed human listening and production-signed activation on this
exact build and hardware class. Other CUDA builds and hardware classes need
their own qualification.

| Contract | Qualified value |
| --- | --- |
| Engine source | audio.cpp v0.8.2, revision `4d88768fbcae4e6eb3352c6ab1422dabb7d90b58` |
| Engine lane | `audio_cpp-cuda`, Linux `arm64`, GB10 SM 12.1 |
| Wheel | `skulk_audio_cpp_cuda-0.8.2.post1-py3-none-manylinux_2_35_aarch64.whl` |
| Wheel SHA-256 | `39d9f4e2f037ac808ba3588119e3d11a2437e3eb2d85e8bf5eecf63cc3c4c772` |
| Executable SHA-256 | `21db727c1f2ec030b93cabffbd09e3c522ceb77fa2ae1092cd96a0f4b0120c9c` |
| Node build identity | `audio.cpp@sha256:21db727c1f2ec030b93cabffbd09e3c522ceb77fa2ae1092cd96a0f4b0120c9c` |
| Package provenance | [Attested build](https://github.com/Foxlight-Foundation/Skulk/actions/runs/36219388508) and [immutable publication](https://github.com/Foxlight-Foundation/Skulk/actions/runs/36223461058) |

Qualification used an isolated Skulk process, a test-signed catalog, and real
GB10 hardware while the normal Skulk service kept its existing workloads.
The engine and model caches began empty. Each normal mount selected the
published wheel, verified its executable and model specs, downloaded the
card's immutable artifact bundle, and reached runner readiness on
`audio_cpp-cuda` with the build identity above. This tested signed discovery,
engine preparation, placement, model loading, and the `/v1/music` output path
together.

| Card | Immutable model revision | First delivered WAV |
| --- | --- | --- |
| `audio-cpp/ACE-Step1.5-Turbo-BF16` | `a776907b362419343f4b9996bdd899619efcf3f8` | 10.00 seconds, stereo 48 kHz, 1,920,044 bytes, SHA-256 `a5573d1fecf4a22ee7d82d0c8b65f9067dd7708fa87f37c6adb42e17fb622273` |
| `audio-cpp/MiniMax-Music3-GGUF-Q4` | `9634a1e1364f94f1ac85a38c114ef105c678f824` | 9.98 seconds, stereo 44.1 kHz, 1,761,324 bytes, SHA-256 `1e0c0021a95f61c1bac978c7ab133f62d4a2d6ee21f88b49a757996a0a8e8035` |

Both WAVs matched the API's digest and metadata, were below the 64 MiB limit,
and contained non-silent PCM samples. A MiniMax request without
lyrics returned HTTP 400. Active and queued cancellation reached terminal
`cancelled` for both families, and their runners returned to ready. Parallel
ACE and MiniMax requests completed with digest-matched WAV content while both
models were mounted.
These signal measurements did not establish subjective music quality; the
separate listening set below closed that gate.

Both models then completed concurrent, paced stability runs on the same isolated node.
Every result was a stereo WAV with the expected sample rate and a matching
API digest. Sampled PCM16 amplitude was non-silent in every job.

| Card | Completed jobs | Elapsed paced run | Job latency | Minimum sampled PCM16 RMS |
| --- | ---: | ---: | ---: | ---: |
| ACE-Step 1.5 Turbo BF16 | 8 | 1,845.2 seconds | 9.1–15.2 seconds; 9.1-second median | 4,590 |
| MiniMax Music 3 Q4 | 8 | 2,077.9 seconds | 39.3–42.3 seconds; 39.3-second median | 1,221 |

Each job was followed by a 220-second idle interval. These runs establish
repeated generation over time but do not satisfy the release plan's continuous
30-minute generation gate. A subsequent continuous run closed that technical
gate. Both models were mounted concurrently and admitted back-to-back work on
the same isolated reader. There were no intentional pauses between requests;
every job completed with valid, digest-matched, non-silent WAV output.

| Card | Jobs / wall time | Busy fraction / longest request gap | Latency, min–median–max | Minimum sampled PCM16 RMS | Server RSS, beginning / high-water / ending |
| --- | --- | --- | --- | ---: | --- |
| ACE-Step 1.5 Turbo BF16 | 93 / 1,808.78 seconds | 99.93% / 0.029 seconds | 11.05 / 20.08 / 34.12 seconds | 2,159 | 284,508 / 1,458,764 / 1,082,408 KiB |
| MiniMax Music 3 Q4 | 31 / 1,804.79 seconds | 99.97% / 0.033 seconds | 49.19 / 58.20 / 63.21 seconds | 1,373 | 269,736 / 2,520,672 / 920,608 KiB |

The middle RSS value is the Linux process high-water mark; the sampled RSS
peaks were 1,353,332 KiB for ACE-Step and 2,520,672 KiB for MiniMax. A separate per-process NVIDIA GPU-memory
sampler observed 14,455–15,729 MiB for ACE-Step and 1,803–8,403 MiB for
MiniMax across 325 successful five-second samples. Sampling began about three
minutes after generation began, so these are observed ranges rather than
lifetime GPU-memory peaks. Both processes remained alive through the run;
the normal Skulk service retained its three existing placements.

In the earlier paced runs, the largest sampled resident set of either mounted
audio.cpp server was 1,200,404 KiB during the ACE run and 1,259,520 KiB
during the MiniMax run.
Those peaks are not attributed to an individual model. Across 70 samples,
aggregate CUDA memory used by the shared host peaked at 86.48 GiB, while
host-available RAM stayed at or above 62.96 GiB. These aggregate readings
include workloads in the unchanged normal Skulk service. GPU busy reached
0.95 during music generation.

After the paced runs, the isolated API restarted at the final reader code with
the signed catalog server unavailable. Previously completed jobs for both
cards retained their metadata and WAV bytes with identical SHA-256 digests.
Both cards remained discoverable and remounted from the verified engine and
model caches; the live node still reported the exact CUDA build and
`nvidia:sm-12.1` class. Both cards generated fresh digest-matched, non-silent
WAVs with the catalog server unavailable.
An additional restart with Skulk's `--offline` mode kept both cards and the
verified engine ready, and both remounted models generated valid WAVs without
a registry or package download.
Killing the isolated API during an active MiniMax job and restarting offline
marked that job `failed` with no content; previously completed WAVs remained
available with their original digests. The three preexisting placements in
the normal Skulk service remained present throughout qualification.

The GB10 CUDA memory pool is shared with host RAM. On this hardware NVML
reported memory as unsupported, while CUDA measured a 130,595,991,552-byte
pool and a live free figure. Skulk used the lower of CUDA free memory, host
free memory less 16 GiB of OS headroom, and 75% of total host RAM for music
admission. A failed CUDA memory query leaves the lane without a usable memory
budget.

The exact claim requires the immutable card, `music.generate`, the build
identity above, and `nvidia:sm-12.1`. A generic NVIDIA class or another
audio.cpp executable does not inherit this result. The package needs host
CUDA 12 runtime, cuBLAS, NCCL, and NVIDIA driver libraries; the binary probe
reports missing libraries before advertising the lane.

## Music quality and production-signed activation

Each card generated eight reviewed outputs: four different music prompts and
lyric settings at two seeds, including longer requests. ACE-Step produced two
25-second results; MiniMax produced one 22.18-second result and one additional
24.96-second result with longer lyrics. One MiniMax request targeted 25
seconds but ended at 9.15 seconds, consistent with the API's generation-target
contract. All reviewed outputs were accepted as music. The listener preferred
ACE-Step's quality in this set; acceptance does not imply equal quality.

The registry then published exact `supported` claims for `music.generate`,
`audio_cpp-cuda`, the executable build above, and `nvidia:sm-12.1`:

| Card | Support claim |
| --- | --- |
| ACE-Step 1.5 Turbo BF16 | `support_y5ukbrwmdmmcl53pgwfqzaj5rpe2c7gzlgchg6csxd2nz4o56paq` |
| MiniMax Music 3 Q4 | `support_lnm5obsbwkld4icrzsi7ccviwvxlj2l6vpa45dz4lyvmz2cvir2a` |

A stock TUF reader verified signed snapshot
`snapshot_141_309aa14cac11c76a45238f5c` with both claims. The normal
three-node Skulk service discovered both cards, prepared the pinned engine
package on demand, advertised the matching live build, and mounted each card
to `RunnerReady` on the GB10. A MiniMax placement explicitly selected this
CUDA hardware class because unconstrained placement also had a supported
Metal option. Both ordinary `/v1/music` requests returned non-silent stereo
WAV content matching their job metadata and SHA-256 digests:

| Card | Actual duration | Sample rate | Bytes | WAV SHA-256 |
| --- | ---: | ---: | ---: | --- |
| ACE-Step 1.5 Turbo BF16 | 10.00 s | 48 kHz | 1,920,044 | `c6d0b4b0b07256e8444e25805034fb11e78dd492b0b0a1bda3e4f980698412a3` |
| MiniMax Music 3 Q4 | 9.985 s | 44.1 kHz | 1,761,324 | `00680569c79bebef0f8d2a93dd61dd7d959336adea577e4661148b89ff731f1a` |

The test jobs and placements were removed. The three nodes finished healthy
with their two original runners ready. This qualification is for these exact
cards, build, and hardware class; it does not extend to every Linux arm64 CPU
or NVIDIA CUDA combination.
