---
id: audio-cpp-strix-qualification
title: audio.cpp Strix Vulkan qualification
---

This record preserves the original functional qualification of the two exact
music cards below on the `amd:pci-1002-1586` hardware class. The class
identifies a GPU type, not an individual Skulk node.

**MiniMax quality correction (2026-09-27):** Later listening found that the
MiniMax build recorded here can begin musically and then collapse into a held
note or noise. Its non-silent WAV and stability results below do not establish
acceptable music quality, and this build should not be used as evidence for a
MiniMax `supported` claim. The pinned audio.cpp MiniMax depth decoder resets
its fallback random generator to the same request seed for every autoregressive
frame on non-CUDA backends. A source patch that also mixes in the frame's
sample index removed the repeated-token lock in controlled Strix runs, including
a 59.9-second clip accepted by listening. The historical wheel below does not
contain it. The replacement build's qualification and signed activation are
recorded below. ACE-Step's original results were unaffected by this MiniMax
defect.

**Sampler-corrected candidate:** [CI run 36294070711](https://github.com/Foxlight-Foundation/Skulk/actions/runs/36294070711)
built `skulk_audio_cpp_vulkan-0.8.2.post3-py3-none-manylinux_2_35_x86_64.whl`
with wheel SHA-256
`12297a15a1b7fd5fea6deb1ea3f431b9fa2a4efc5e695b3377ccbaf46e2980ca`
and executable SHA-256
`a0fe05ce8f120126a6997cfdeea7f28ca46f4ee853586ea94349509d3a333ab3`.
The exact attested wheel loaded on Strix Vulkan and returned a 59.919-second
structured vocal WAV and a different-seed instrumental WAV that ended naturally
at 35.352 seconds. Both passed human listening as very good. These were direct
engine-server requests. [Publication run 36297920867](https://github.com/Foxlight-Foundation/Skulk/actions/runs/36297920867)
verified the source run, wheel digest, and attestation, then published those
exact wheel bytes to the engine-package index.

An isolated Skulk reader with a test-signed catalog then prepared the published
wheel on ordinary placement, verified its executable build identity, and
mounted both exact music cards. MiniMax and ACE-Step each completed
`POST /v1/music` jobs with bounded WAV content and matching output digests.
MiniMax also rejected missing lyrics. A queued cancellation left active work
running; an active cancellation stopped that model server, and a subsequent
job completed on its supervised replacement. An ACE-Step job overlapped two
completed MiniMax generations.

The isolated MiniMax reader then completed **34 back-to-back ten-second jobs in
1,849.11 seconds** with no failed job or intentional idle interval. All 34
WAVs had distinct SHA-256 digests, valid stereo 44.1 kHz PCM16 headers, output
metadata matching the delivered bytes, and measured signal. The maximum gap
between requests was 0.014 seconds; the minimum full-wave PCM16 RMS was
682.61. Latency was 53.20–54.20–58.21 seconds (minimum, median, maximum).
Across 394 five-second samples of that MiniMax server, the process high-water
RSS was 1,956,500 KiB and its maximum reported resident Vulkan memory on one
DRM file descriptor was 6,761,408 KiB. These are sampled measurements on one
64 GiB Strix host, not universal capacity requirements.

An induced crash of only the isolated MiniMax server failed its in-flight job,
supervision started a different server process, and a fresh job returned a
digest-matched WAV. After removing both test mounts, the isolated reader was
restarted offline with engine auto-provisioning disabled and the test catalog
server stopped. It recovered a completed job's original WAV and remounted
both music cards from verified cached engine and model artifacts. Each card
then completed a fresh offline generation with content matching its digest.
The normal three-node service retained its original process, code revision,
configuration digest, two placements, and empty diagnostics warnings after
the isolated test. The temporary reader and catalog were removed and the
fleet lease released.

These results establish the new build's MiniMax functional and stability
path, alongside the two accepted listening samples. ACE-Step also loaded,
generated, and ran concurrently through this build. A separate direct-server
check of the same wheel and SHA-256-verified ACE-Step artifact completed
**62 back-to-back ten-second jobs in 1,800.146 seconds** with no failed job.
All 62 stereo 48 kHz PCM16 WAVs had distinct digests and measured signal;
the minimum full-wave PCM16 RMS was 549.36. The maximum inter-request gap was
0.002 seconds and latency was 23.14–30.30–34.00 seconds
(minimum, median, maximum). Its server high-water RSS was 5,549,056 KiB and
maximum sampled resident Vulkan memory on one DRM fd was 15,542,144 KiB
across 374 five-second samples. A separate 45-second ACE-Step output had
SHA-256 `b9b897f1f99bfdab4150db47f70a1c7a647f5567f8f9be68e49133b89e01dc6e`;
human listening found it acceptable through the end.

**Production-signed activation (2026-09-27):** The compatible Skulk reader
merged to `dev` at `4024b93112119973e1c1dca41ab264b26c42a9dd` and was
deployed to all three participating nodes before new support claims were
published. The deployed registry required no code update or seed-card
re-import. A stock TUF client verified
`snapshot_145_8193e4a1f00f1d6db83ecd9c`, containing 176 unchanged cards
and engine-support matrix version 145. Its exact `music.generate` decisions
for hardware class `amd:pci-1002-1586` are:

| Card and build | Signed decision |
| --- | --- |
| MiniMax Music 3 Q4, corrected Vulkan build `audio.cpp@sha256:a0fe05ce8f120126a6997cfdeea7f28ca46f4ee853586ea94349509d3a333ab3` | `supported`, claim `support_pnrn6wewdqks4n3n3vscge4wl5j2xpvsxv6bmbaemnjpolcb4oya` |
| ACE-Step 1.5 Turbo BF16, same corrected Vulkan build | `supported`, claim `support_bvkdzihcvhgaopa5ztzj2fp4lsmbqfemlxgwi2cl5ds7ezc7kvhq` |
| MiniMax Music 3 Q4, older Vulkan build `audio.cpp@sha256:aacd5af30a4e7c4f7bcc02e392c46a3e9f73dd07c992488fbf53c74469c2f9e0` | `unsupported`, claim `support_nzjbyvlci2dpob64vk2htnyskvahszwrieztwkybslmmpzwidbea` superseding the historical `supported` claim |

Ordinary signed placement mounted both cards through `audio_cpp-vulkan` with
the corrected executable hash. Concurrent `POST /v1/music` jobs completed:
MiniMax returned a 29.953741-second stereo 44.1 kHz PCM16 WAV of 5,283,884
bytes (SHA-256 `a37547034dd440c37badc3a480b911d8c22415064c4c61a466c211d474b3f67b`);
ACE-Step returned a 10-second stereo 48 kHz PCM16 WAV of 1,920,044 bytes
(SHA-256 `0baeb8d2d6ae3ca7b09064e9ede7318dab728e958d9fb80dc05c71ce5331b905`).
Each downloaded content digest and WAV geometry matched its completed job
metadata. MiniMax rejected omitted lyrics with HTTP 400. Test jobs and
placements were removed; all three nodes remained healthy, on the same
commit, and with the two original placements ready. This is a
configured-fleet regression, not a fresh-install qualification.

## MiniMax F32 accumulation qualification (2026-09-28)

The sampler defect is separate from a measured Vulkan precision issue: some
large MiniMax matrix multiplications on this hardware use 16-bit accumulation.
The post4 package retains the sampler correction and requests F32 accumulation
for MiniMax's Vulkan matrix multiplications. It preserves the upstream F16 KV
cache. Other model families and CPU, Metal, and CUDA execution are unchanged.

| Identity | Qualified value |
| --- | --- |
| Wheel | `skulk_audio_cpp_vulkan-0.8.2.post4-py3-none-manylinux_2_35_x86_64.whl` |
| Wheel SHA-256 / size | `a077d3627b96430a8359c707e608296955501bc8042c7b1efe49f3e407fa1071` / 23,905,021 bytes |
| Executable SHA-256 | `886e9d4fce8eb1a4eef5363e5bbe61c086ee417765681b7eced7ef2a1b336440` |
| Source | audio.cpp `4d88768fbcae4e6eb3352c6ab1422dabb7d90b58`, plus the pinned sampler and MiniMax Vulkan precision patches in [Skulk #1076](https://github.com/Foxlight-Foundation/Skulk/pull/1076) |
| Attested build | [CI run 36376118682](https://github.com/Foxlight-Foundation/Skulk/actions/runs/36376118682), source `1fbf6c59533fb828ecfcceba2d9bb27582c2095a`; its complete tree matches reviewed head `fcb7b3460ef27c90899d0a6446dd5f76332b70c0` |
| Hardware class / lane | `amd:pci-1002-1586` / `audio_cpp-vulkan` |

Both clean standard controls reproduced previously accepted WAVs byte for byte.
The listener preferred F32 in both randomized, loudness-matched comparisons:
one vocal request and one instrumental request. This establishes preference
for those two requests, not a universal improvement for every song.

The exact post4 executable reproduced the accepted 60-second vocal F32 sample
byte for byte: SHA-256
`1767b5daa14cd3707660fb4a76131a5c135f22db772497cc0f54da184f35038b`.
Three fresh 60-second targets passed listening: the acoustic folk vocal was
exceptionally good; electronic and chamber-orchestra requests were acceptable
music. The latter two produced unwanted wordless vocals despite `No vocals.`
in their prompts and `[Instrumental]` in their lyrics. Instrumental conditioning
is therefore not reliable in these samples. Actual durations were 59.919093,
43.583855, and 31.079909 seconds respectively. MiniMax's duration target remains
a generation budget, not an exact-length promise.

ACE-Step reproduced its accepted 45-second folk output byte for byte, SHA-256
`b9b897f1f99bfdab4150db47f70a1c7a647f5567f8f9be68e49133b89e01dc6e`.
Both models passed the real Skulk runner's queued and active cancellation,
single-active-generation admission, crash recovery, replacement generation,
and reload/shutdown checks. Neither left partial output after cancellation.
These used an isolated verified-cache reader and real sidecars, separate from
ordinary production-signed placement.

| Exact-package gate | MiniMax Q4 | ACE-Step Turbo BF16 |
| --- | --- | --- |
| Continuous generation | 34 requests / 1,819.266 s | 51 requests / 1,817.715 s |
| Median generation-call latency | 53.417 s | 35.262 s |
| Sampled process RSS maximum | 876,064 KiB | 4,852,228 KiB |
| Sampled resident Vulkan memory, one DRM descriptor | 8,755,704 KiB | 16,072,680 KiB |
| Output | Distinct, non-silent stereo 44.1 kHz PCM16 WAVs; no full-scale samples | Distinct, non-silent stereo 48 kHz PCM16 WAVs; 10,565 full-scale samples / 48,960,000 samples (0.022%) |

Memory samples remained bounded during these separate 30-minute runs. They do
not establish longer endurance or universal hardware capacity requirements.
Three matched 30-second MiniMax seeds, with fresh servers and equal warmup,
measured median wall time of 194.725 s for post3 and 190.688 s for post4
(2.074% lower). This workload showed no speed penalty; other workloads may
differ. All qualification results were valid WAVs below the 64 MiB limit.

The new reader pin selects these exact tested wheel bytes. Signed support still
requires separate claims for this executable, each exact card, `music.generate`,
and the hardware class above. Package availability alone does not grant
placement. Existing post3 qualification remains valid for its own build.

### Historical post1 functional results

The following post1 measurements predate the MiniMax listening failure.
They document load and WAV mechanics; the post1 MiniMax build is now
`unsupported` for `music.generate` on this hardware class.

| Contract | Original post1 value |
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

| Card | Immutable model revision | Original functional result |
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
The later MiniMax listening failure supersedes the original non-silent signal
gate. Any replacement claims apply only to the exact cards, build identity,
capability, and hardware class tested. Other audio.cpp builds and GPUs need
their own qualification.
