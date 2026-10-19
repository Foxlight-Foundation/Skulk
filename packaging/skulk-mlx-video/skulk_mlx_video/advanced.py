"""Exact pinned source-media workflows; no alternate engine or host binaries."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from skulk_mlx_video.schema import Job

if TYPE_CHECKING:
    from ltx_pipelines_mlx._base import BasePipeline


def relay_lengths(weights: tuple[int, ...], latent_frames: int) -> list[int]:
    """Distribute relative scene lengths on the latent grid, reserving each a frame."""
    if not weights or any(weight < 1 for weight in weights) or latent_frames < len(weights):
        raise ValueError("each prompt-relay scene requires one complete latent frame")
    remaining = latent_frames - len(weights)
    total = sum(weights)
    lengths = [1 + remaining * weight // total for weight in weights]
    order = sorted(range(len(weights)), key=lambda index: (-(remaining * weights[index] % total), index))
    for index in order[:latent_frames - sum(lengths)]:
        lengths[index] += 1
    return lengths


def create_source_pipeline(job: Job) -> BasePipeline:
    """Select the requested pinned native source workflow and require measured facts."""
    from ltx_core_mlx.utils.ffmpeg import VideoInfo
    from ltx_pipelines_mlx import extend_distilled, retake
    from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline

    source = job.source_media
    if source is None:
        raise ValueError("source generation requires worker-prepared media")
    if job.audio_conditioning is not None:
        if source.frames or not source.has_audio or source.audio_sample_rate != 48000 or source.audio_channels != 2:
            raise ValueError("source audio was not prepared as bounded 48 kHz stereo PCM")
        return A2VidDistilledPipeline(job.model_dir, gemma_model_id=job.model_dir, low_memory=True, low_ram_streaming=True)
    if job.editing is None or job.transformer_variant is None:
        raise ValueError("source editing requires an explicit transformer and timeline operation")

    def measured_probe(path: str) -> VideoInfo:
        # Upstream expects ffprobe, which our wheel does not ship. Bind only
        # this exact verified clip to the worker's fully decoded measurements;
        # other paths cannot acquire guessed metadata or invoke a host binary.
        if path != source.path:
            raise ValueError("editing attempted to probe an unverified source")
        return VideoInfo(width=job.width, height=job.height, num_frames=source.frames, fps=float(job.fps), duration=source.frames / job.fps, has_audio=source.has_audio)

    retake.probe_video_info = measured_probe
    extend_distilled.probe_video_info = measured_probe
    if job.pipeline == "extend" and job.transformer_variant == "distilled":
        if not 25 <= source.frames < job.frame_count:
            raise ValueError("distilled extension requires at least 25 source frames and a longer output")
        return extend_distilled.ExtendDistilledPipeline(job.model_dir, gemma_model_id=job.model_dir, low_memory=True, low_ram_streaming=True)
    if job.pipeline == "retake" and source.frames != job.frame_count:
        raise ValueError("retake source frame count must equal the requested output")
    if source.frames < 9 or (job.pipeline == "extend" and source.frames >= job.frame_count):
        raise ValueError("editing requires a complete source latent grid and a longer output for extension")
    return retake.RetakePipeline(job.model_dir, gemma_model_id=job.model_dir, low_memory=True, low_ram_streaming=True, distilled=job.transformer_variant == "distilled")


def render_source_job(pipeline: BasePipeline, job: Job, ffmpeg: Path) -> None:
    """Render the verified source with exact masks and preserve requested audio behavior."""
    from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
    from ltx_pipelines_mlx.extend_distilled import ExtendDistilledPipeline
    from ltx_pipelines_mlx.retake import RetakePipeline
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    source = job.source_media
    assert source is not None
    output = Path(job.output_dir) / "output.mp4"
    preserve_source = False
    if isinstance(pipeline, A2VidDistilledPipeline):
        pipeline.generate_and_save(job.prompt, str(output), audio_path=source.path, height=job.height, width=job.width, num_frames=job.frame_count, frame_rate=float(job.fps), seed=job.seed, images=[ImageConditioningInput(item.path, item.frame, 1.0, 18 if item.frame == 0 else None) for item in job.images] or None)
    elif isinstance(pipeline, ExtendDistilledPipeline):
        video, audio = pipeline.extend_from_video(job.prompt, source.path, (job.frame_count - source.frames) // 8, seed=job.seed)
        pipeline._decode_and_save_video(video, audio, str(output), frame_rate=float(job.fps), seed=job.seed)
    elif isinstance(pipeline, RetakePipeline):
        editing = job.editing
        assert editing is not None
        if editing.operation == "retake":
            assert editing.start_seconds is not None and editing.end_seconds is not None
            start = math.floor(editing.start_seconds * job.fps / 8)
            end = min((source.frames - 1) // 8 + 1, math.ceil(editing.end_seconds * job.fps / 8))
            if not 0 <= start < end:
                raise ValueError("the retake region contains no complete latent-frame interval")
            video, audio = pipeline.retake_from_video(job.prompt, source.path, start, end, seed=job.seed, num_steps=job.base_steps, regenerate_audio=editing.regenerate_audio, negative_prompt=job.negative_prompt)
            preserve_source = job.audio and not editing.regenerate_audio
        else:
            video, audio = pipeline.extend_from_video(job.prompt, source.path, (job.frame_count - source.frames) // 8, seed=job.seed, num_steps=job.base_steps, negative_prompt=job.negative_prompt)
        pipeline._decode_and_save_video(video, audio, str(output), frame_rate=float(job.fps), seed=job.seed)
    else:
        raise ValueError("this engine cannot execute the selected source workflow")
    if not job.audio or preserve_source:
        # A2V's upstream save always includes source audio. An explicit silent
        # request must remove it; retake preservation must bypass VAE roundtrip.
        target = output.with_name("_remux.mp4")
        command = [str(ffmpeg), "-v", "error", "-i", str(output)]
        if preserve_source:
            command.extend(["-i", source.path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", str(job.frame_count / job.fps)])
        else:
            command.extend(["-map", "0:v:0", "-c:v", "copy", "-an"])
        subprocess.run([*command, "-y", str(target)], check=True, timeout=30)
        target.replace(output)
