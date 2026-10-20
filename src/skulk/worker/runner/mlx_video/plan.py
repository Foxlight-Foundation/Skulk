"""Resolve native video controls and expose only exact card-selected local files."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal, final

from pydantic import Field

from skulk.download.download_utils import artifact_install_directory, build_model_path
from skulk.shared.models.model_cards import ModelCard, VideoCompanionKind
from skulk.shared.types.common import ModelId
from skulk.shared.types.video import (
    VideoAudioConditioning,
    VideoCodecName,
    VideoDecoder,
    VideoEditing,
    VideoGenerationTaskParams,
    VideoGuidance,
    VideoPipeline,
    VideoPromptRelay,
)
from skulk.utils.pydantic_ext import FrozenModel
from skulk.worker.runner.comfy.graph import bind_references
from skulk.worker.runner.video_input import VerifiedMediaInput
from skulk.worker.runner.video_plan import RenderPlan, plan_render


@final
class NativeImageAnchor(FrozenModel):
    """One verified image and its requested pixel-frame position."""

    path: str = Field(description="Bounded worker-staged image path.")
    frame: int = Field(description="Pixel-frame position; -1 resolves after automatic duration.")


@final
class NativeVideoJob(FrozenModel):
    """Bounded JSON protocol delivered only to the verified native interpreter."""

    model_dir: str = Field(description="Process-local view of the exact card-selected pack members.")
    output_dir: str = Field(description="Worker-owned directory for this command's media and completion record.")
    prompt: str
    width: int
    height: int
    fps: int
    frame_count: int
    seed: int
    audio: bool
    pipeline: VideoPipeline
    decoder: VideoDecoder
    codec: VideoCodecName
    base_steps: int = Field(description="Intervals in stage one; refinement uses the pinned three-interval table.")
    negative_prompt: str | None
    guidance: VideoGuidance | None
    generated_keyframe_interval: int | None = Field(description="Pixel-frame spacing for internal keyframes, resolved after predicted duration.")
    duration: Literal["fixed", "auto"]
    minimum_seconds: int = Field(description="Lower requested/card duration bound before frame-grid alignment.")
    maximum_seconds: int = Field(description="Upper requested allocation/duration bound before frame-grid alignment.")
    refinement_strength: float = Field(description="Exact selected distilled-refinement companion strength for guided pipelines.")
    images: tuple[NativeImageAnchor, ...] = Field(description="Verified worker-local first, last and timed frame anchors.")
    audio_conditioning: VideoAudioConditioning | None = Field(default=None, description="Source waveform conditioning controls.")
    editing: VideoEditing | None = Field(default=None, description="Source clip editing controls.")
    prompt_relay: VideoPromptRelay | None = Field(default=None, description="Relative timed local scene prompts.")
    source_media: VerifiedMediaInput | None = Field(default=None, description="Bounded decoder facts supplied by worker preparation.")
    transformer_variant: Literal["distilled", "dev"] | None = Field(default=None, description="Editing transformer choice; absent on the immutable baseline protocol.")


def selected_model_view(card: ModelCard, destination: Path) -> Path:
    """Expose exact selected artifact members so upstream cannot pick other files.

    Missing components fail before generation. Weight bytes stay in the managed
    model store, while this process-local view is removed on runner shutdown.
    """
    bundle = card.artifact_bundle
    if bundle is None:
        raise ValueError("native video requires an immutable selected artifact bundle")
    if any(len(PurePosixPath(entry.path).parts) != 1 for entry in bundle.files):
        raise ValueError("native video components must have unique pack-root names")
    load_directory = build_model_path(ModelId(card.model_id), card.source_revision, bundle.root)
    root = artifact_install_directory(load_directory, bundle.root)
    destination.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    for entry in bundle.files:
        # An upstream pack's resolvers search its root. Nested or duplicate
        # components are ambiguous and require an explicit packaging conversion.
        name = PurePosixPath(entry.path)
        if len(name.parts) != 1 or name.name in names or name.name.startswith("."):
            raise ValueError("native video components must have unique pack-root names")
        names.add(name.name)
        source = root / name.name
        if not source.is_file() or source.stat().st_size != entry.size_bytes:
            raise ValueError(f"selected native video component is missing or truncated: {name}")
        target = destination / name.name
        target.symlink_to(source.resolve())
    return destination


def plan_native_job(params: VideoGenerationTaskParams, card: ModelCard, model_dir: Path, output_dir: Path, input_dir: Path) -> tuple[NativeVideoJob, RenderPlan]:
    """Apply common family validation, then resolve the native engine's exact controls."""
    video = card.video
    if card.family != "ltx-2.5" or video is None:
        raise ValueError("this native engine serves LTX-2.5 video packs")
    params.validate_family(card.family, video)
    if params.implied_mode() not in video.modes:
        raise ValueError("the selected native artifact does not implement this mode")
    pipeline = params.pipeline or (params.editing.operation if params.editing is not None else "distilled_two_stage" if video.transformer_variant == "distilled" else "dev_two_stage")
    if pipeline == "dfr":
        raise ValueError("DFR needs an exact LTX-2.5 detailing adapter; this build does not implement it")
    if pipeline == "hq_two_stage" and params.sampler is not None:
        raise ValueError("native HQ uses its pinned Res2s integrator; an explicit Comfy sampler cannot be honored")
    codec = params.codec or "h264"
    if codec != "h264":
        raise ValueError("this native build implements H.264 MP4 output")
    render = plan_render(params, video.model_copy(update={"canvas_multiple": 64, **({"default_steps": 15} if pipeline == "hq_two_stage" else {})}))
    if min(render.width, render.height) < 128:
        raise ValueError("LTX two-stage generation requires edges of at least 128 pixels")
    base_steps = params.steps or (8 if video.transformer_variant == "distilled" else 15 if pipeline == "hq_two_stage" else 30)
    anchors: list[NativeImageAnchor] = []
    for binding in bind_references(params.references, input_dir):
        if binding.spec.kind != "image":
            continue
        role = binding.spec.role
        frame = -1 if role == "last_frame" else round((binding.spec.at_seconds or 0) * video.fps) if role == "keyframe" else 0
        anchors.append(NativeImageAnchor(path=str(input_dir / binding.input_name), frame=frame))
    refinement_strength = next((item.strength if item.strength is not None else 1.0 for item in video.companions if item.kind is VideoCompanionKind.Lora and item.name == "distilled_refinement" and item.supports_mode(params.implied_mode())), 1.0)
    return NativeVideoJob(
        model_dir=str(model_dir), output_dir=str(output_dir), prompt=params.prompt,
        width=render.width, height=render.height, fps=render.fps, frame_count=render.frame_count,
        seed=render.seed, audio=render.audio, pipeline=pipeline, decoder=params.decoder or "convolutional",
        codec=codec, base_steps=base_steps, negative_prompt=params.negative_prompt,
        guidance=(params.guidance or VideoGuidance(modality_scale=3.0 if pipeline == "hq_two_stage" else 1.0)) if pipeline in ("dev_two_stage", "hq_two_stage") else None,
        generated_keyframe_interval=params.generated_keyframe_interval, duration=params.duration,
        minimum_seconds=params.duration_min_seconds or video.min_seconds, maximum_seconds=params.seconds,
        refinement_strength=refinement_strength, images=tuple(anchors),
        audio_conditioning=params.audio_conditioning, editing=params.editing,
        prompt_relay=params.prompt_relay,
        transformer_variant=video.transformer_variant if params.editing is not None else None,
    ), render
