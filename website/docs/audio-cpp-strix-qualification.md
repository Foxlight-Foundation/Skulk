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
| `audio-cpp/ACE-Step1.5-Turbo-BF16` | `a776907b362419343f4b9996bdd899619efcf3f8` | Stereo 48 kHz WAV; eight completed digest-verified, non-silent soak jobs; 30-minute soak; generation latency 60.74–74.87 seconds |
| `audio-cpp/MiniMax-Music3-GGUF-Q4` | `9634a1e1364f94f1ac85a38c114ef105c678f824` | Stereo 44.1 kHz WAV; eight completed digest-verified, non-silent soak jobs; 30-minute soak; generation latency 58.74–62.79 seconds; missing lyrics rejected |

The soak admitted work for both cards concurrently. Cancellation during active
generation terminated and replaced the affected model server through runner
supervision. Queued cancellation did not interrupt the active job. An API
restart preserved completed result bytes and digests; both models remounted
from cached artifacts and generated again. A subsequent isolated run with the
current candidate reader also mounted and generated from both models.

Measured peak server resident memory was 5,254,484 KiB for ACE-Step and
511,356 KiB for MiniMax. Sampled GPU memory use was 9,973,000 KiB and
1,650,852 KiB respectively; these GPU samples are not peak measurements.
These claims apply only to the exact cards, build identity, capability, and
hardware class listed here. Other audio.cpp builds and GPUs need their own
qualification.
