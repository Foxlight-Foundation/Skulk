"""Typed bridges from pinned Comfy inputs to the owned preservation primitives."""

import math
from collections.abc import Mapping
from typing import final

from comfy_api.latest import io
from comfy_extras import nodes_audio
from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode
from torch import Tensor

from skulk_comfy_ltx.editing import (
    AudioVae,
    AudioWave,
    Operation,
    edit_latent,
    normalize_source_audio,
    preserve_audio,
    preserve_video,
    uncropped_audio_vae,
)


def _operation_inputs() -> list[object]:
    return [io.Combo.Input("operation", options=["retake", "extend"]), io.Int.Input("fps", default=24, min=1, max=1000), io.Float.Input("start_seconds", default=0.0, min=0.0, max=120.0), io.Float.Input("end_seconds", default=0.0, min=0.0, max=121.0)]


def _samples(latent: Mapping[str, object]) -> Tensor:
    samples = latent.get("samples")
    if not isinstance(samples, Tensor):
        raise ValueError("editing requires a single decoded modality latent")
    return samples


@final
class SkulkLTXEditVideoLatent(io.ComfyNode):
    """Bind source video to temporal sampling masks at the current stage resolution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose explicit source and target latents without arbitrary file access."""
        return io.Schema(node_id="SkulkLTXEditVideoLatent", display_name="Skulk LTX temporal video context", category="model/latent/ltxv", inputs=[io.Latent.Input("source"), io.Latent.Input("target"), *_operation_inputs()], outputs=[io.Latent.Output()])

    @classmethod
    def execute(cls, source: Mapping[str, object], target: Mapping[str, object], operation: Operation, fps: int, start_seconds: float, end_seconds: float) -> io.NodeOutput:
        """Copy the source context and replace only the requested temporal mask."""
        samples, mask = edit_latent(_samples(source), _samples(target), modality="video", operation=operation, fps=fps, start_seconds=start_seconds, end_seconds=end_seconds)
        return io.NodeOutput({**target, "samples": samples, "noise_mask": mask})


@final
class SkulkLTXEditAudioLatent(io.ComfyNode):
    """Normalize source PCM, encode it through the selected VAE, and bind its mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Keep waveform and VAE clocks explicit, including absent source tracks."""
        return io.Schema(node_id="SkulkLTXEditAudioLatent", display_name="Skulk LTX temporal audio context", category="model/latent/ltxv", inputs=[io.Audio.Input("audio"), io.Latent.Input("target"), io.Vae.Input("audio_vae"), io.Int.Input("source_frames", default=49, min=1, max=120001), io.Int.Input("sample_rate", default=48000, min=8000, max=192000), io.Boolean.Input("regenerate_audio", default=True), *_operation_inputs()], outputs=[io.Latent.Output(), io.Audio.Output()])

    @classmethod
    def execute(cls, audio: AudioWave | None, target: Mapping[str, object], audio_vae: AudioVae, source_frames: int, sample_rate: int, regenerate_audio: bool, operation: Operation, fps: int, start_seconds: float, end_seconds: float) -> io.NodeOutput:
        """Encode normalized source context once and retain that waveform for output."""
        # The pinned core owns this locked resampler and already uses it for
        # LTX VAE input. Reuse that boundary instead of adding an engine dependency.
        source = normalize_source_audio(audio, frames=source_frames, fps=fps, sample_rate=sample_rate, resample=nodes_audio.torchaudio.functional.resample)
        clock = audio_vae.first_stage_model
        if not all(math.isfinite(value) and value > 0 for value in (clock.sample_rate, clock.mel_hop_length, clock.latents_per_second)):
            raise ValueError("the selected audio VAE lacks a valid causal clock")
        factor = clock.sample_rate / (clock.mel_hop_length * clock.latents_per_second)
        downsample = round(factor)
        if not 1 <= downsample <= 16 or not math.isclose(factor, downsample, rel_tol=1e-6):
            raise ValueError("the selected audio VAE has an unsupported causal downsample")
        encoded = LTXVAudioVAEEncode.execute(source, uncropped_audio_vae(audio_vae)).result[0]["samples"]
        samples, mask = edit_latent(encoded, _samples(target), modality="audio", operation=operation, fps=fps, start_seconds=start_seconds, end_seconds=end_seconds, regenerate_audio=regenerate_audio, sample_rate=clock.sample_rate, hop_length=clock.mel_hop_length, downsample_factor=downsample)
        return io.NodeOutput({**target, "samples": samples, "noise_mask": mask}, source)


@final
class SkulkLTXPreserveVideo(io.ComfyNode):
    """Restore unchanged source frames after VAE decode and before container encoding."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Accept decoded source and generated images on one matching frame clock."""
        return io.Schema(node_id="SkulkLTXPreserveVideo", display_name="Skulk LTX source frame preservation", category="video", inputs=[io.Image.Input("source"), io.Image.Input("generated"), *_operation_inputs()], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, source: Tensor, generated: Tensor, operation: Operation, fps: int, start_seconds: float, end_seconds: float) -> io.NodeOutput:
        """Keep the input cache intact while copying the unchanged source regions."""
        return io.NodeOutput(preserve_video(source, generated, operation=operation, fps=fps, start_seconds=start_seconds, end_seconds=end_seconds))


@final
class SkulkLTXPreserveAudio(io.ComfyNode):
    """Restore unchanged normalized PCM independently of the video's latent clock."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Bind actual output frames and the requested audio regeneration policy."""
        return io.Schema(node_id="SkulkLTXPreserveAudio", display_name="Skulk LTX source PCM preservation", category="audio", inputs=[io.Audio.Input("source"), io.Audio.Input("generated"), io.Int.Input("output_frames", default=49, min=1, max=120001), io.Boolean.Input("regenerate_audio", default=True), *_operation_inputs()], outputs=[io.Audio.Output()])

    @classmethod
    def execute(cls, source: AudioWave, generated: AudioWave, output_frames: int, regenerate_audio: bool, operation: Operation, fps: int, start_seconds: float, end_seconds: float) -> io.NodeOutput:
        """Splice only the requested region or tail before the muxer's AAC conversion."""
        return io.NodeOutput(preserve_audio(source, generated, operation=operation, output_frames=output_frames, fps=fps, start_seconds=start_seconds, end_seconds=end_seconds, regenerate_audio=regenerate_audio))
