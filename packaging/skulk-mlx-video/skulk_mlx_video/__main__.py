"""Weightless GPU probing and bounded local jobs over the pinned native pipelines."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, TypedDict

from skulk_mlx_video.schema import BuildMetadata, Job
from skulk_mlx_video.telemetry import observe_video_decode

if TYPE_CHECKING:
    import mlx.core as mx
    from ltx_core_mlx.components.guiders import MultiModalGuiderParams
    from ltx_core_mlx.conditioning.prompt_relay import PromptRelayInput
    from ltx_pipelines_mlx._base import OnStep
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput


class GenerateOptions(TypedDict, total=False):
    """Only keyword arguments implemented by the pinned generation methods."""

    frame_rate: float
    seed: int
    images: list[ImageConditioningInput] | None
    generated_keyframes: list[int]
    enable_teacache: bool
    stage1_steps: int
    negative_prompt: str | None
    video_guider_params: MultiModalGuiderParams
    audio_guider_params: MultiModalGuiderParams
    prompt_relay: PromptRelayInput


class Arguments(argparse.Namespace):
    """Typed local CLI arguments; this process exposes no caller-facing server."""

    probe: bool
    job: Path | None
    parent_pid: int | None


def emit(record: Mapping[str, str | int | float | bool | tuple[str, ...]], *, stream: TextIO | None = None) -> None:
    """Send one small JSON record on the private subprocess protocol."""
    print(json.dumps(record, separators=(",", ":")), file=stream or sys.stdout, flush=True)


def gpu_probe() -> str:
    """Require a real Metal operation and packaged dependencies without loading weights."""
    import mlx.core as mx
    from ltx_pipelines_mlx.distilled import DistilledPipeline
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

    importlib.import_module("av")
    if not mx.metal.is_available():
        raise RuntimeError("the native video engine requires an available Metal GPU")
    mx.set_default_device(mx.gpu)
    mx.eval(mx.ones((2, 2)) @ mx.ones((2, 2)))
    del DistilledPipeline, TI2VidTwoStagesHQPipeline
    metadata = BuildMetadata.model_validate_json((Path(__file__).parent / "build.json").read_bytes())
    return f"mlx_video@sha256:{metadata.payload_sha256}"


def preflight(model: Path, job: Job) -> None:
    """Refuse an incomplete or older pack before any fallback loader runs."""
    if not model.is_dir():
        raise ValueError("the exact staged model directory is missing")
    required = {
        "text_encoder.safetensors", "text_encoder_config.json", "connector.safetensors",
        "tokenizer.json", "tokenizer_config.json", "embedded_config.json",
        "chat_template.jinja", "quantize_config.json", "vae_encoder_conv.safetensors",
        "vae_decoder_conv.safetensors", "audio_vae.safetensors", "vocoder.safetensors",
    }
    required.add("transformer-dev.safetensors" if job.transformer_variant == "dev" or job.pipeline in {"dev_two_stage", "hq_two_stage"} else "transformer-distilled.safetensors")
    if job.pipeline != "retake" and not (job.pipeline == "extend" and job.transformer_variant == "dev"):
        required.update({"spatial_upscaler_x2_v1_0.safetensors", "spatial_upscaler_x2_v1_0_config.json"})
    if job.pipeline in {"dev_two_stage", "hq_two_stage"}:
        required.update({"transformer-dev.safetensors", "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"})
    if job.decoder == "diffusion":
        required.add("vae_decoder_av.safetensors")
    if job.duration == "auto":
        required.add("duration_head.safetensors")
    missing = sorted(name for name in required if not (model / name).is_file())
    if missing:
        raise ValueError(f"the selected LTX-2.5 pack is incomplete: {', '.join(missing)}")
    from ltx_pipelines_mlx.utils.generation import is_ltx25_pack

    if not is_ltx25_pack(model):
        raise ValueError("native LTX rendering requires an exact LTX-2.5 pack")
    if job.pipeline == "dfr":
        raise ValueError("this native build does not implement the selected pipeline")
    if job.codec != "h264":
        raise ValueError("this native build implements H.264 MP4 output")


def render(job: Job) -> None:
    """Render selected local components and report actual steps, Metal peak and media."""
    import imageio_ffmpeg
    import mlx.core as mx
    from ltx_core_mlx.components.guiders import MultiModalGuiderParams
    from ltx_core_mlx.conditioning.prompt_relay import PromptRelayInput
    from ltx_pipelines_mlx._base import BasePipeline
    from ltx_pipelines_mlx.distilled import DistilledPipeline
    from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput
    from ltx_pipelines_mlx.utils.types import AutoDuration

    from skulk_mlx_video.advanced import (
        create_source_pipeline,
        relay_lengths,
        render_source_job,
    )

    model = Path(job.model_dir)
    preflight(model, job)
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    executables = list((Path(imageio_ffmpeg.__file__).parent / "binaries").glob("ffmpeg-*"))
    if len(executables) != 1 or not executables[0].is_file():
        raise RuntimeError("the bundled media encoder is missing")
    ffmpeg = executables[0]
    binaries = output / "bin"
    binaries.mkdir()
    (binaries / "ffmpeg").symlink_to(ffmpeg)
    os.environ["PATH"] = str(binaries) + os.pathsep + os.environ.get("PATH", "")
    mx.reset_peak_memory()
    started = time.monotonic()
    steps_observed = 0
    sampling_seconds = 0.0
    stage_started: dict[int, float] = {}
    protocol_stdout = sys.stdout

    def hook(latent_frames: int, latent_height: int, latent_width: int, *, stage: int | None = None) -> OnStep:
        del latent_frames, latent_height, latent_width
        name = stage or 1
        stage_started.setdefault(name, time.monotonic())
        def on_step(step_index: int, num_steps: int, video_latent: mx.array, sigma: float) -> None:
            nonlocal steps_observed, sampling_seconds
            del video_latent, sigma
            steps_observed += 1
            emit({"stage": "sampling", "step": step_index + 1, "total_steps": num_steps, "sampling_stage": name}, stream=protocol_stdout)
            if step_index + 1 == num_steps:
                sampling_seconds += time.monotonic() - stage_started[name]
        return on_step

    emit({"stage": "encoding"})
    pipeline: BasePipeline
    with contextlib.redirect_stdout(sys.stderr):
        if job.audio_conditioning is not None or job.editing is not None:
            pipeline = create_source_pipeline(job)
        elif job.pipeline == "distilled_two_stage":
            pipeline = DistilledPipeline(str(model), gemma_model_id=str(model), low_memory=True, low_ram_streaming=True)
        else:
            selected = TI2VidTwoStagesHQPipeline if job.pipeline == "hq_two_stage" else TI2VidTwoStagesPipeline
            pipeline = selected(str(model), gemma_model_id=str(model), low_memory=True, low_ram_streaming=True, distilled_lora_strength=job.refinement_strength)
    # Instance-held callables receive no implicit self. These signatures match
    # pinned upstream hooks and never keep a preview decoder resident in sampling.
    pipeline._stepwise_hook = hook
    pipeline.verbose = False
    pipeline.generate_audio = job.audio
    pipeline.video_decoder = "conv" if job.decoder == "convolutional" else "diffusion"
    pipeline._decode_and_save_video = observe_video_decode(
        pipeline.video_decoder_block,
        pipeline._decode_and_save_video,
        lambda: emit({"stage": "decoding"}, stream=protocol_stdout),
    )
    frames: int | AutoDuration = job.frame_count
    if job.duration == "auto":
        frames = AutoDuration(min_seconds=float(job.minimum_seconds), max_seconds=float(job.maximum_seconds))
    images = [ImageConditioningInput(item.path, item.frame, 1.0, 18 if item.frame == 0 else None) for item in job.images]
    interval = job.generated_keyframe_interval
    generated = list(range(interval, job.frame_count - 1, interval)) if interval is not None else []
    if job.duration == "auto" and interval is not None:
        original_resolve = pipeline._resolve_num_frames
        spacing = interval
        def resolve_frames(num_frames: int | AutoDuration, *, video_encoding: mx.array, audio_encoding: mx.array, frame_rate: float) -> int:
            actual = original_resolve(num_frames, video_encoding=video_encoding, audio_encoding=audio_encoding, frame_rate=frame_rate)
            # Resolution precedes upstream normalization of these shared positions.
            generated[:] = range(spacing, actual - 1, spacing)
            return actual
        pipeline._resolve_num_frames = resolve_frames
    options: GenerateOptions = {"frame_rate": float(job.fps), "seed": job.seed, "images": images or None, "generated_keyframes": generated, "enable_teacache": False}
    if job.prompt_relay is not None:
        options["prompt_relay"] = PromptRelayInput(local_prompts=[item.prompt for item in job.prompt_relay.segments], segment_lengths=[length for length in relay_lengths(tuple(item.length for item in job.prompt_relay.segments), (job.frame_count - 1) // 8 + 1)])
    if job.pipeline in ("dev_two_stage", "hq_two_stage"):
        guidance = job.guidance
        if guidance is None:
            raise ValueError("guided native generation requires resolved guidance")
        options.update(stage1_steps=job.base_steps, negative_prompt=job.negative_prompt,
            video_guider_params=MultiModalGuiderParams(cfg_scale=guidance.video_cfg, stg_scale=guidance.stg_scale, stg_blocks=[28], modality_scale=guidance.modality_scale, rescale_scale=0.45 if job.pipeline == "hq_two_stage" else 0.0),
            audio_guider_params=MultiModalGuiderParams(cfg_scale=guidance.audio_cfg, stg_scale=guidance.stg_scale, stg_blocks=[28], modality_scale=guidance.modality_scale, rescale_scale=1.0 if job.pipeline == "hq_two_stage" else 0.0))
    with contextlib.redirect_stdout(sys.stderr):
        if job.audio_conditioning is not None or job.editing is not None:
            render_source_job(pipeline, job, ffmpeg)
        else:
            video_latent, audio_latent = pipeline.generate_two_stage(job.prompt, job.height, job.width, frames, **options)
            pipeline._decode_and_save_video(video_latent, audio_latent, str(output / "output.mp4"), frame_rate=float(job.fps), seed=job.seed)
    emit({"stage": "muxing"})
    subprocess.run([str(ffmpeg), "-v", "error", "-i", str(output / "output.mp4"), "-frames:v", "1", "-q:v", "3", str(output / "thumbnail.jpg")], check=True, timeout=30)
    (output / "result.json").write_text(json.dumps({"steps": steps_observed, "sampling_seconds": sampling_seconds, "total_generation_time": time.monotonic() - started, "peak_memory_bytes": mx.get_peak_memory()}))
    (binaries / "ffmpeg").unlink()
    binaries.rmdir()


def main() -> None:
    """Probe or execute a strict local job and reap children if the runner dies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--parent-pid", type=int)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    parent = arguments.parent_pid
    if parent is not None:
        def reap() -> None:
            while os.getppid() == parent:
                time.sleep(0.25)
            os.killpg(os.getpgrp(), signal.SIGKILL)
        threading.Thread(target=reap, daemon=True).start()
    build = gpu_probe()
    if arguments.probe:
        metadata = BuildMetadata.model_validate_json((Path(__file__).parent / "build.json").read_bytes())
        emit({"build_identity": build, "compute": "metal", "ready": True, "features": metadata.features})
        return
    if arguments.job is None or arguments.job.stat().st_size > 128 * 1024:
        raise ValueError("a bounded local job is required")
    render(Job.model_validate_json(arguments.job.read_bytes()))


if __name__ == "__main__":
    main()
