---
id: inference
title: Inference and Media
description: Choose engines and serve text, vision, embeddings, images, speech, video and music.
---

Skulk connects model discovery, verified artifacts, placement and inference across
a cluster. A model's capabilities determine which requests it accepts; its engine
and the live nodes determine where it can run. A machine with a GPU is not
automatically compatible with every model or workload.

## Choose the workload

| Workload | Serving surface | Prerequisites |
| --- | --- | --- |
| Text and tool-calling chat | `/v1/chat/completions`, Responses, Ollama and Claude-compatible adapters | A placed text model; tool and reasoning behavior follows its capability profile. |
| Image understanding | Image content in chat messages | A vision-capable card and a compatible runner, including any pinned vision companion. |
| Embeddings | `/v1/embeddings` | A placed embedding model. |
| Image generation and editing | `/v1/images/generations`, `/v1/images/edits` | Image models enabled and a placed model for the requested task. |
| Speech synthesis | `/v1/audio/speech` | A placed TTS card; streaming and reference voices depend on card support. |
| Speech recognition and translation | `/v1/audio/transcriptions`, `/v1/audio/translations` | A placed STT card; translation requires declared translation support. |
| Realtime speech | `/v1/realtime` WebSocket and Fabric speech chain | Ready realtime STT capacity; optional VAD, chat and TTS compose the voice loop. |
| Video with optional synchronized audio | `/v1/videos` jobs | Video models enabled, a compatible video card and ComfyUI capacity. |
| Text-to-music | `/v1/music` jobs | A mounted `TextToMusic` card, a verified audio.cpp engine build, and an exact signed support claim for its hardware class. |

See the [API guide](api-guide.md) for request schemas, streaming events, limits and
examples. Compatibility adapters expose supported subsets of their upstream APIs;
an OpenAI-compatible base URL does not imply that every OpenAI feature exists.

## From a card to a ready instance

1. Find a model in the catalog or add a custom model. Signed registry cards bind
   artifact identity and behavior; custom cards are operator-managed.
2. Inspect its capabilities and placement previews. Verify the proposed nodes,
   engine, memory and context capacity. `/models/requirements` is an advisory
   sizing read, not a reservation.
3. Launch the placement through the dashboard or instance API. The store or
   download path verifies the selected artifacts and stages local files.
4. Wait for the assigned runner to become ready, then send an ordinary request.

Ordinary inference does not place an unknown or unmounted model automatically.
A cached download is not a loaded runner, and a ready runner is not evidence that
an arbitrary workload fits its memory budget. Context limits, output length,
concurrency and media size all affect resource use.

Model compatibility combines card declarations, signed support evidence for exact
engine builds and artifacts, live node engine/hardware observations, and Skulk's
runner limitations. A model family name alone cannot establish support. See
[Model capabilities](model-capabilities.md), [Model cards](model-cards.md) and
[Model store](model-store.md).

## Engine and hardware choices

| Engine | Role |
| --- | --- |
| `mlx` | Apple Silicon text and vision, with dedicated MLX embedding and image runners; supported models can use pipeline or tensor sharding. |
| `llama_cpp` | In-process GGUF text on a single node with a compatible CPU/GPU backend. |
| `llama_server` | Managed external llama-server for GGUF text, including supported native speculative decoding and driver/donor RPC placements. |
| `vllm` | External GPU text serving with continuous batching and paged attention on supported CUDA/ROCm nodes. |
| `mlx_audio` | Single-node MLX Audio TTS and STT. |
| `comfy` | Managed headless ComfyUI for card-defined video and audio-video generation. |
| `audio_cpp` | Separately installed, single-node audio.cpp server for card-defined music generation. |

Multi-node support is engine- and model-specific. MLX sharding does not combine
Apple and Linux GPU memory into one universal execution pool. GGUF RPC placements
use a driver and compatible donors, while ordinary in-process llama.cpp remains
single-node. Consult [architecture](architecture.md), [vLLM](vllm-engine.md),
[NVIDIA nodes](nvidia-cuda-nodes.md) and [AMD Strix Halo nodes](amd-strix-halo-nodes.md)
for the relevant serving path. Speculative decoding requires the carded companion
and engine support; see [Speculative decoding](speculative-decoding.md).

## Vision and image workflows

For image understanding, send image content with a chat request to a vision-capable
model. Skulk transfers bounded input media directly to the ranks selected for the
task, verifies it, and waits for their acknowledgements before generation. Input
bytes do not enter replicated cluster state or the event log.

Image synthesis and editing use a separate model task and dedicated endpoints.
Enable their catalog on the relevant nodes with `SKULK_ENABLE_IMAGE_MODELS=true`.
Generation takes JSON; editing takes multipart image upload. Responses can contain
inline base64 bytes or a URL to content stored on the accepting API node. Optional
SSE partial images expose progress. Use the same origin for stored-image reads,
and save outputs you need to retain.

## Video jobs

Enable video models on the relevant nodes with `SKULK_ENABLE_VIDEO_MODELS=true`.
Skulk can provision a pinned ComfyUI runtime on supported NVIDIA and AMD lanes;
operator-managed installations can provide `SKULK_COMFY_BIN` and `SKULK_COMFY_ROOT`.
The model card defines supported modes: text-to-video (`t2va`), first/last-frame
conditioning (`fl2va`), or reference images, clips and audio (`ref2va`). It also
defines durations, frame grids, canvas limits, output audio and available adapters.
Select only modes and references supported by that card. A card may also pin guide
preprocessor weights hosted in other repositories (a pose estimator and the person
detector it crops with, a depth estimator); Skulk downloads them with the card at
their pinned revisions, and each names the license its repository declares.

Submit `/v1/videos`, retain the returned job ID, and poll its status. Download the
completed MP4 or thumbnail from that API node. Completion requires both the
terminal manifest and verified output bytes; progress alone is not success. Cancel
or delete through the job API when work is no longer needed. Output retention is
bounded: completed content expires after 24 hours and storage pressure may evict
older completed jobs sooner. Download durable results promptly.

Jobs and output storage belong to the accepting API node. After its restart,
retained completed artifacts are verified before serving and interrupted jobs are
marked failed. Do not treat a different API node as a transparent job replica.

`test_video` is a separate deterministic test engine enabled by
`SKULK_TEST_VIDEO_ENGINE`; its synthetic clips exercise the job and transport
lifecycle and are not model-generated video.

## Music jobs and engine preparation

Skulk ships without the audio.cpp engine package. When you mount a music card,
the API chooses a hardware-eligible node, asks that worker to fetch and verify
the pinned engine package, then waits for fresh node resources before ordinary
placement. Apple Silicon macOS, Linux `amd64`, and Linux `arm64` have CPU-capable
packages. Linux `amd64` has a separate Vulkan package, and Linux `arm64` has a
CUDA package compiled for NVIDIA GB10 compute 12.1. A GPU package is selected
only when a signed claim covers the exact card, engine build, and hardware class.
The GB10 CUDA package requires CUDA 12 runtime, cuBLAS, and NCCL libraries in
the host loader path; its probe reports missing libraries before the lane can
be advertised. The package contains the server, model specs, and licenses;
model weights download separately. An offline node can use a verified cached package
but cannot fetch one. A failed preparation returns a mount error and leaves the
node serving its other workloads. `SKULK_AUDIO_CPP_BIN`,
`SKULK_AUDIO_CPP_VULKAN_BIN`, and `SKULK_AUDIO_CPP_CUDA_BIN` can point to
separate operator-installed builds, each of which must pass the same source
and device probes. Preparing a GPU wheel does not replace another music
instance's executable or build identity.
For AMD, node resources include a PCI chip class such as
`amd:pci-1002-1586`, allowing a claim to cover the qualified hardware class
without identifying a specific node.
On GB10, the NVML memory reading may be unavailable. Skulk uses CUDA's live
device-memory reading and admits the shared CPU/GPU pool only within host-RAM
headroom and its unified-memory working-set limit. If neither memory query
succeeds, GPU music placement waits for usable capacity instead of guessing.

`SKULK_AUDIO_CPP_SPECS_DIR` points to its v0.8.2 model specs when they are not
beside the binary in the package layout. Skulk checks both required spec
digests before advertising the engine or loading a model. Both override paths
must be absolute. On restart, a
previously installed package is restored from its verified cache without a
download, including on offline nodes.

`SKULK_AUDIO_CPP_BACKENDS` can restrict advertised compute lanes. Check node
capability conflicts and engine build inventory when an override or native
library is incompatible. A CPU-capable package never implies that a given
model is qualified on that machine.
The Linux wheels require glibc 2.35 or newer, `libstdc++.so.6`, and
`libgomp.so.1`; the node probe reports a missing loader dependency instead of
advertising the engine. They do not embed GPU driver libraries.

The selected model runs in one supervised loopback server with one active
generation. Submit `/v1/music`, poll the returned ID, and download its WAV
from the accepting API node. The card controls lyric requirements and duration
bounds. `seconds` is a target or budget; MiniMax may produce a different actual
duration, which the completed job reports. WAV results are limited to 64 MiB
and retained for up to 24 hours. Cancel or delete a job through the music API;
download any result you need to keep. See [Music generation](api-guide.md#music-generation).

## Speech and application integration

Speech uses the same placement and model-store lifecycle as text, with distinct
cards and runners. Voice catalogs, bundled reference profiles, uploaded reference
audio, streaming and realtime support are model-specific. Start with
[Speech providers and realtime transcription](speech-fabric-realtime.md).

Applications can use the HTTP APIs directly. Extensions can call typed Fabric
providers, including `tts`, `stt`, `stt.realtime` and `vad`, with admission,
cancellation and streaming contracts. [Extensions](extensions.md) describes that
integration boundary; [Talk to Skulk](steward.md) describes the resident cluster
conversation built on these capabilities.
