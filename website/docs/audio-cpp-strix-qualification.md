---
id: audio-cpp-strix-qualification
title: audio.cpp Strix Vulkan qualification
---

This record supports the `music.generate` engine support claims for the two
exact music cards below on the `amd:pci-1002-1586` hardware class. The class
identifies a GPU type, not an individual Skulk node.

| Contract | Qualified value |
| --- | --- |
| Engine source | audio.cpp v0.8.2, revision `4d88768fbcae4e6eb3352c6ab1422dabb7d90b58` |
| Engine lane | `audio_cpp-vulkan`, Linux `amd64` |
| Wheel | `skulk_audio_cpp_vulkan-0.8.2.post1-py3-none-manylinux_2_35_x86_64.whl` |
| Wheel SHA-256 | `72f0a600cff38d65640254e24c7c7b4271effbb49bf9e948ecdebcb2bb210930` |
| Executable SHA-256 | `aacd5af30a4e7c4f7bcc02e392c46a3e9f73dd07c992488fbf53c74469c2f9e0` |
| Node build identity | `audio.cpp@sha256:aacd5af30a4e7c4f7bcc02e392c46a3e9f73dd07c992488fbf53c74469c2f9e0` |
| Package provenance | [Build](https://github.com/Foxlight-Foundation/Skulk/actions/runs/36213472081) and [immutable package publication](https://github.com/Foxlight-Foundation/Skulk/actions/runs/36214442698), with artifact attestation verified |

The qualification used an isolated Skulk process and signed test catalog on
real AMD Strix Halo hardware. Engine and model caches began empty. For each
card, normal placement prepared the pinned package, downloaded the exact model
bundle, reached runner readiness, and served `POST /v1/music` plus WAV content
retrieval. The API digest matched the downloaded bytes. Both outputs were
non-silent when measured as PCM, and the Vulkan render device was observed
during generation.

| Card | Immutable model revision | Qualification result |
| --- | --- | --- |
| `audio-cpp/ACE-Step1.5-Turbo-BF16` | `a776907b362419343f4b9996bdd899619efcf3f8` | Stereo 48 kHz WAV; eight completed digest-verified, non-silent jobs in a paced 30-minute stability run; generation latency 60.74–74.87 seconds |
| `audio-cpp/MiniMax-Music3-GGUF-Q4` | `9634a1e1364f94f1ac85a38c114ef105c678f824` | Stereo 44.1 kHz WAV; eight completed digest-verified, non-silent jobs in a paced 30-minute stability run; generation latency 58.74–62.79 seconds; missing lyrics rejected |

The paced runs admitted work for both cards concurrently. They included idle
intervals and do not satisfy the release plan's 30-minute continuous generation
gate. A subsequent run kept both mounted models generating concurrently with
no intentional pause between requests. Every job completed with a valid,
digest-matched stereo WAV under the 64 MiB output limit.

| Card | Jobs / wall time | Busy fraction / longest request gap | Latency, min–median–max | Minimum sampled PCM16 RMS | Server RSS, beginning / high-water / ending |
| --- | --- | --- | --- | ---: | --- |
| ACE-Step 1.5 Turbo BF16 | 28 / 1,805.67 seconds | 99.98% / 0.067 seconds | 57.21 / 66.23 / 94.42 seconds | 3,044 | 124,096 / 5,526,504 / 4,019,448 KiB |
| MiniMax Music 3 Q4 | 29 / 1,803.93 seconds | 99.99% / 0.010 seconds | 58.27 / 62.24 / 64.22 seconds | 71.63 | 93,892 / 1,941,020 / 340,952 KiB |

The middle RSS value is the Linux process high-water mark, with the larger
MiniMax mark observed by the independent sampler. Across 369 five-second
samples, the model processes' DRM fdinfo reported up to 15,137,104 KiB and
6,769,036 KiB of resident Vulkan memory respectively. These are sampled GPU maxima, not
lifetime peaks. One MiniMax result was unusually quiet (full-wave RMS 73.55
PCM16, peak 565; SHA-256
`b9acbb53fe0b1bf77c440ed0b6ec128ebed06f32d218587f4100d83f4261bb5b`).
It met the signal gate but needs listening review before making a quality
judgment. Both servers remained alive, and removing the temporary mounts left
the original runners ready, all three nodes healthy, and configuration unchanged.

Cancellation during active generation terminated and replaced the affected
model server through runner supervision. Queued cancellation did not interrupt
the active job. An API restart preserved completed result bytes and digests; both models remounted
from cached artifacts and generated again. A subsequent isolated run with the
current candidate reader also mounted and generated from both models.

The earlier paced run did not record beginning and ending RSS for each model;
the continuous run above supplies that evidence.
Human listening of an exact output from each card is still needed for the
release plan's subjective quality gate; PCM measurements establish only that
the WAVs are non-silent.
These claims apply only to the exact cards, build identity, capability, and
hardware class listed here. Other audio.cpp builds and GPUs need their own
qualification.
