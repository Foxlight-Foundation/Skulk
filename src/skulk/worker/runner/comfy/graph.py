"""Bind a card and a request onto a ComfyUI graph of MiniMax H3 nodes.

Everything here is pure: the same card, request, and reference set produce
the same API-format prompt (``{node_id: {"class_type", "inputs"}}``). The
graph mirrors ComfyUI's own MiniMax H3 workflow templates node for node
(``UNETLoader``, ``CLIPLoader(type=minimax)``, two ``VAELoader``,
``MiniMaxH3ImageToVideo`` or ``MiniMaxH3ReferenceToVideo``,
``LoraLoaderModelOnly``, ``MiniMaxH3SigmaShift``, ``KSamplerSelect
res_multistep``, ``BasicScheduler simple``, ``SamplerCustomAdvanced``,
``VAEDecode``, ``VAEDecodeAudio``, ``CreateVideo``, ``SaveVideo``) so a clip
rendered through Skulk matches one rendered in the ComfyUI frontend with the
same inputs. Node identifiers are stable strings so progress can be mapped
back onto Skulk's coarse stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from skulk.shared.models.model_cards import (
    ModelCard,
    VideoCompanionConfig,
    VideoCompanionKind,
    VideoMode,
)
from skulk.shared.types.video import (
    VIDEO_OUTPUT_FILENAME,
    VIDEO_THUMBNAIL_FILENAME,
    VideoGenerationTaskParams,
    VideoReferenceSpec,
    VideoStage,
)
from skulk.worker.runner.video_plan import RenderPlan, plan_render

ComfyPrompt = dict[str, dict[str, Any]]
"""ComfyUI's API prompt format: node id to ``class_type`` plus ``inputs``."""

SAMPLER_NAME: Final = "res_multistep"
SCHEDULER_NAME: Final = "simple"
OUTPUT_PREFIX_STEM: Final = PurePosixPath(VIDEO_OUTPUT_FILENAME).stem
THUMBNAIL_PREFIX_STEM: Final = PurePosixPath(VIDEO_THUMBNAIL_FILENAME).stem

NODE_UNET: Final = "unet"
NODE_LORA: Final = "lora"
NODE_SHIFT: Final = "shift"
NODE_CLIP: Final = "clip"
NODE_VIDEO_VAE: Final = "video_vae"
NODE_AUDIO_VAE: Final = "audio_vae"
NODE_CONDITION: Final = "condition"
NODE_NOISE: Final = "noise"
NODE_SAMPLER_SELECT: Final = "sampler_select"
NODE_SCHEDULER: Final = "scheduler"
NODE_GUIDER: Final = "guider"
NODE_SAMPLER: Final = "sampler"
NODE_DECODE_VIDEO: Final = "decode_video"
NODE_DECODE_AUDIO: Final = "decode_audio"
NODE_CREATE_VIDEO: Final = "create_video"
NODE_SAVE_VIDEO: Final = "save_video"
NODE_THUMBNAIL_FRAME: Final = "thumbnail_frame"
NODE_SAVE_THUMBNAIL: Final = "save_thumbnail"

_STAGE_BY_NODE: Final[dict[str, VideoStage]] = {
    NODE_UNET: "encoding",
    NODE_LORA: "encoding",
    NODE_SHIFT: "encoding",
    NODE_CLIP: "encoding",
    NODE_VIDEO_VAE: "encoding",
    NODE_AUDIO_VAE: "encoding",
    NODE_CONDITION: "encoding",
    NODE_NOISE: "sampling",
    NODE_SAMPLER_SELECT: "sampling",
    NODE_SCHEDULER: "sampling",
    NODE_GUIDER: "sampling",
    NODE_SAMPLER: "sampling",
    NODE_DECODE_VIDEO: "decoding",
    NODE_DECODE_AUDIO: "decoding",
    NODE_CREATE_VIDEO: "muxing",
    NODE_SAVE_VIDEO: "muxing",
    NODE_THUMBNAIL_FRAME: "muxing",
    NODE_SAVE_THUMBNAIL: "muxing",
}

_COMPONENT_TRANSFORMER: Final = "transformer"
_COMPONENT_TEXT_ENCODER: Final = "text_encoder"
_COMPONENT_VIDEO_VAE: Final = "video_vae"
_COMPONENT_AUDIO_VAE: Final = "audio_vae"

MODEL_FOLDERS: Final[tuple[str, ...]] = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "embeddings",
    "model_patches",
)
"""ComfyUI folder keys the engine exposes from a staged artifact directory."""


def stage_for_node(node_id: str) -> VideoStage | None:
    """Map an executing graph node onto Skulk's coarse render stage."""
    if node_id in _STAGE_BY_NODE:
        return _STAGE_BY_NODE[node_id]
    if node_id.startswith(("ref_", "load_", "guide_")):
        return "encoding"
    return None


@dataclass(frozen=True, slots=True)
class ComfyModelFiles:
    """The card's weight files by the ComfyUI loader that reads them.

    Values are the file names ComfyUI lists under each model folder (the
    path relative to the folder, which for the H3 repacks is the bare
    file name).
    """

    diffusion_model: str
    text_encoder: str
    video_vae: str
    audio_vae: str | None


def _bundle_paths(card: ModelCard) -> tuple[PurePosixPath, ...]:
    if card.artifact_bundle is None:
        raise ValueError(f"{card.model_id} declares no artifact bundle; the comfy engine needs exact files")
    return tuple(PurePosixPath(file.path) for file in card.artifact_bundle.files)


def _component_file(card: ModelCard, component_name: str, *, required: bool = True) -> str | None:
    """Pick the one bundle file a component loads.

    A component names its directory; the bundle lists the files. When the
    directory holds one file that file is it. When several components share a
    directory (the H3 repacks keep both VAEs under ``vae/``) the file whose
    name carries the component's own qualifier (``video`` or ``audio``) is
    chosen, and anything still ambiguous is refused rather than guessed.
    """
    component = next((c for c in (card.components or ()) if c.component_name == component_name), None)
    if component is None:
        if required:
            raise ValueError(f"{card.model_id} declares no {component_name!r} component")
        return None
    directory = PurePosixPath(component.component_path.rstrip("/"))
    candidates = [path for path in _bundle_paths(card) if path.parent == directory]
    if len(candidates) > 1:
        qualifiers = [token for token in component_name.split("_") if token not in ("vae", "model", "encoder")]
        narrowed = [
            path for path in candidates if any(token in path.name.lower() for token in qualifiers)
        ]
        if len(narrowed) == 1:
            candidates = narrowed
    if len(candidates) != 1:
        raise ValueError(
            f"{card.model_id}: {component_name!r} under {directory}/ matches "
            f"{len(candidates)} bundle files; the card must pin exactly one"
        )
    return candidates[0].relative_to(directory).as_posix()


def resolve_model_files(card: ModelCard) -> ComfyModelFiles:
    """Resolve the loader file names from the card's components and bundle."""
    video = card.video
    diffusion = _component_file(card, _COMPONENT_TRANSFORMER)
    text_encoder = _component_file(card, _COMPONENT_TEXT_ENCODER)
    video_vae = _component_file(card, _COMPONENT_VIDEO_VAE)
    audio_vae = _component_file(
        card, _COMPONENT_AUDIO_VAE, required=video is not None and video.audio_output
    )
    assert diffusion is not None and text_encoder is not None and video_vae is not None
    return ComfyModelFiles(
        diffusion_model=diffusion,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
    )


def companion_file(companion: VideoCompanionConfig, card: ModelCard) -> str:
    """The name ComfyUI lists a companion under, relative to its model folder.

    Companions in the card's own repository sit inside the staged artifact
    directory, so the name is the path below the folder ComfyUI reads for
    that kind (``loras/``, ``model_patches/``, ``embeddings/``).
    """
    if companion.repo is not None and companion.repo != card.model_id:
        raise ValueError(
            f"companion {companion.name!r} lives in {companion.repo}; externally hosted "
            "companions are not wired into the comfy engine"
        )
    path = PurePosixPath(companion.path)
    folder = {
        VideoCompanionKind.Lora: "loras",
        VideoCompanionKind.ModelPatch: "model_patches",
        VideoCompanionKind.Embedding: "embeddings",
    }.get(companion.kind)
    if folder is None or not path.parts or path.parts[0] != folder:
        raise ValueError(
            f"companion {companion.name!r} ({companion.kind.value}) at {companion.path!r} is not "
            f"under the folder ComfyUI reads for that kind"
        )
    return path.relative_to(folder).as_posix()


@dataclass(frozen=True, slots=True)
class AdapterChoice:
    """A low-rank adapter selected for one render plus the sigma shifts it wants."""

    file: str
    strength: float
    steps: int | None


def select_adapter(
    params: VideoGenerationTaskParams, card: ModelCard, mode: VideoMode
) -> VideoCompanionConfig | None:
    """Pick the request's named adapter, or none.

    An unnamed request renders at the card's full step schedule: the turbo
    adapters trade quality for speed and the caller opts in by name. A named
    adapter must exist on the card, be a LoRA, and apply to the mode.
    """
    if params.lora is None:
        return None
    assert card.video is not None
    for companion in card.video.companions:
        if companion.name != params.lora:
            continue
        if companion.kind is not VideoCompanionKind.Lora:
            raise ValueError(f"companion {params.lora!r} is a {companion.kind.value}, not a lora")
        if companion.modes and mode not in companion.modes:
            raise ValueError(f"lora {params.lora!r} does not apply to mode {mode.value}")
        return companion
    raise ValueError(f"{card.model_id} has no lora companion named {params.lora!r}")


@dataclass(frozen=True, slots=True)
class ComfyRenderPlan:
    """A render plan plus the engine choices ComfyUI needs to build the graph."""

    plan: RenderPlan
    mode: VideoMode
    files: ComfyModelFiles
    adapter: AdapterChoice | None
    video_shift: float | None
    audio_shift: float | None

    @property
    def steps(self) -> int:
        """Sampling steps for this render."""
        return self.plan.steps


def plan_comfy_render(params: VideoGenerationTaskParams, card: ModelCard) -> ComfyRenderPlan:
    """Resolve a request against the card for the ComfyUI engine.

    Steps come from the request, else the selected adapter's trained count,
    else the card default; sigma shifts come from the adapter when it declares
    them, else the card. Audio is rendered only when the request asks and the
    card produces it.
    """
    video = card.video
    if video is None:
        raise ValueError(f"{card.model_id} has no [video] section")
    mode = params.implied_mode()
    if mode not in video.modes:
        raise ValueError(f"{card.model_id} does not serve mode {mode.value}")
    companion = select_adapter(params, card, mode)
    adapter = None
    video_shift = video.video_shift
    audio_shift = video.audio_shift
    if companion is not None:
        adapter = AdapterChoice(
            file=companion_file(companion, card),
            strength=params.lora_strength if params.lora_strength is not None else (companion.strength or 1.0),
            steps=companion.steps,
        )
        video_shift = companion.video_shift if companion.video_shift is not None else video_shift
        audio_shift = companion.audio_shift if companion.audio_shift is not None else audio_shift
    base = plan_render(params, video)
    if params.steps is None and adapter is not None and adapter.steps is not None:
        base = RenderPlan(
            width=base.width,
            height=base.height,
            fps=base.fps,
            frame_count=base.frame_count,
            steps=adapter.steps,
            seed=base.seed,
            audio=base.audio,
            sample_rate=base.sample_rate,
            channels=base.channels,
        )
    return ComfyRenderPlan(
        plan=base,
        mode=mode,
        files=resolve_model_files(card),
        adapter=adapter,
        video_shift=video_shift,
        audio_shift=audio_shift,
    )


def _node(class_type: str, title: str, **inputs: object) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": dict(inputs), "_meta": {"title": title}}


def _link(node_id: str, output: int = 0) -> list[Any]:
    return [node_id, output]


@dataclass(frozen=True, slots=True)
class ReferenceBinding:
    """One attachment as ComfyUI will load it: role, kind, and input-relative name."""

    spec: VideoReferenceSpec
    input_name: str
    """Path relative to ComfyUI's input directory."""


def bind_references(
    references: tuple[VideoReferenceSpec, ...], input_dir: Path
) -> tuple[ReferenceBinding, ...]:
    """Express each verified attachment relative to ComfyUI's input directory.

    The worker writes attachments beneath the node's video input directory
    and ComfyUI's ``--input-directory`` points at that same root, so loader
    nodes name files by their path below it and no copy is made.
    """
    bindings: list[ReferenceBinding] = []
    root = input_dir.resolve()
    for spec in references:
        if spec.local_path is None:
            raise ValueError(f"reference slot {spec.slot} has no local file on this node")
        local = Path(spec.local_path).resolve()
        if not local.is_relative_to(root):
            raise ValueError(
                f"reference slot {spec.slot} at {local} is outside the video input directory {root}"
            )
        bindings.append(ReferenceBinding(spec=spec, input_name=local.relative_to(root).as_posix()))
    return tuple(bindings)


def build_prompt(
    render: ComfyRenderPlan,
    params: VideoGenerationTaskParams,
    references: tuple[ReferenceBinding, ...],
    output_subdir: str,
) -> ComfyPrompt:
    """Build the API-format prompt for one render.

    ``output_subdir`` is the directory below ComfyUI's output root the
    ``SaveVideo`` and ``SaveImage`` nodes write into (the command id), so the
    runner can find the container and thumbnail without a lookup.
    """
    plan = render.plan
    files = render.files
    prompt: ComfyPrompt = {}
    prompt[NODE_UNET] = _node("UNETLoader", "Load H3", unet_name=files.diffusion_model, weight_dtype="default")
    model_ref = _link(NODE_UNET)
    if render.adapter is not None:
        prompt[NODE_LORA] = _node(
            "LoraLoaderModelOnly",
            "Turbo adapter",
            model=model_ref,
            lora_name=render.adapter.file,
            strength_model=render.adapter.strength,
        )
        model_ref = _link(NODE_LORA)
    if render.video_shift is not None or render.audio_shift is not None:
        shift_inputs: dict[str, object] = {"model": model_ref}
        if render.video_shift is not None:
            shift_inputs["shift_video"] = render.video_shift
        if render.audio_shift is not None:
            shift_inputs["shift_audio"] = render.audio_shift
        prompt[NODE_SHIFT] = _node("MiniMaxH3SigmaShift", "Sigma shift", **shift_inputs)
        model_ref = _link(NODE_SHIFT)
    prompt[NODE_CLIP] = _node(
        "CLIPLoader", "Load text encoder", clip_name=files.text_encoder, type="minimax", device="default"
    )
    prompt[NODE_VIDEO_VAE] = _node("VAELoader", "Load video VAE", vae_name=files.video_vae)
    # The audio VAE encodes reference soundtracks as well as decoding the
    # output track, so a request that attaches audio (or a clip, which
    # carries its own) needs it even when it asks for a silent output.
    references_carry_audio = any(binding.spec.kind in ("audio", "video") for binding in references)
    if (plan.audio or references_carry_audio) and files.audio_vae is not None:
        prompt[NODE_AUDIO_VAE] = _node("VAELoader", "Load audio VAE", vae_name=files.audio_vae)

    if render.mode is VideoMode.ReferenceToAudioVideo:
        conditioning = _add_reference_condition(prompt, render, params, references)
    else:
        conditioning = _add_keyframe_condition(prompt, render, params, references)

    prompt[NODE_NOISE] = _node("RandomNoise", "Seed", noise_seed=plan.seed)
    prompt[NODE_SAMPLER_SELECT] = _node("KSamplerSelect", "Sampler", sampler_name=SAMPLER_NAME)
    prompt[NODE_SCHEDULER] = _node(
        "BasicScheduler", "Schedule", model=model_ref, scheduler=SCHEDULER_NAME, steps=plan.steps, denoise=1.0
    )
    prompt[NODE_GUIDER] = _node("BasicGuider", "Guider", model=model_ref, conditioning=_link(conditioning, 0))
    prompt[NODE_SAMPLER] = _node(
        "SamplerCustomAdvanced",
        "Sample",
        noise=_link(NODE_NOISE),
        guider=_link(NODE_GUIDER),
        sampler=_link(NODE_SAMPLER_SELECT),
        sigmas=_link(NODE_SCHEDULER),
        latent_image=_link(NODE_CONDITION, 1),
    )
    prompt[NODE_DECODE_VIDEO] = _node(
        "VAEDecode", "Decode video", samples=_link(NODE_SAMPLER, 0), vae=_link(NODE_VIDEO_VAE)
    )
    create_inputs: dict[str, object] = {"images": _link(NODE_DECODE_VIDEO), "fps": float(plan.fps)}
    if plan.audio:
        prompt[NODE_DECODE_AUDIO] = _node(
            "VAEDecodeAudio", "Decode audio", samples=_link(NODE_SAMPLER, 0), vae=_link(NODE_AUDIO_VAE)
        )
        create_inputs["audio"] = _link(NODE_DECODE_AUDIO)
    prompt[NODE_CREATE_VIDEO] = _node("CreateVideo", "Mux", **create_inputs)
    prompt[NODE_SAVE_VIDEO] = _node(
        "SaveVideo",
        "Save clip",
        video=_link(NODE_CREATE_VIDEO),
        filename_prefix=f"{output_subdir}/{OUTPUT_PREFIX_STEM}",
        format="mp4",
        **{"format.codec": "h264"},
    )
    prompt[NODE_THUMBNAIL_FRAME] = _node(
        "ImageFromBatch", "First frame", image=_link(NODE_DECODE_VIDEO), batch_index=0, length=1
    )
    prompt[NODE_SAVE_THUMBNAIL] = _node(
        "SaveImage",
        "Save thumbnail",
        images=_link(NODE_THUMBNAIL_FRAME),
        filename_prefix=f"{output_subdir}/{THUMBNAIL_PREFIX_STEM}",
    )
    return prompt


def _add_keyframe_condition(
    prompt: ComfyPrompt,
    render: ComfyRenderPlan,
    params: VideoGenerationTaskParams,
    references: tuple[ReferenceBinding, ...],
) -> str:
    """Add the text-and-keyframe condition node; return the conditioning node id."""
    plan = render.plan
    inputs: dict[str, object] = {
        "clip": _link(NODE_CLIP),
        "vae": _link(NODE_VIDEO_VAE),
        "prompt": params.prompt,
        "width": plan.width,
        "height": plan.height,
        "length": plan.frame_count,
    }
    for binding in references:
        role = binding.spec.role
        if role not in ("first_frame", "last_frame"):
            raise ValueError(
                f"mode {render.mode.value} accepts keyframe attachments only; slot "
                f"{binding.spec.slot} has role {role}"
            )
        node_id = f"load_{role}"
        prompt[node_id] = _node("LoadImage", f"Load {role.replace('_', ' ')}", image=binding.input_name)
        inputs[role] = _link(node_id, 0)
    prompt[NODE_CONDITION] = _node("MiniMaxH3ImageToVideo", "Condition", **inputs)
    return NODE_CONDITION


def _add_reference_condition(
    prompt: ComfyPrompt,
    render: ComfyRenderPlan,
    params: VideoGenerationTaskParams,
    references: tuple[ReferenceBinding, ...],
) -> str:
    """Add the reference condition node plus keyframe guides; return the conditioning node id.

    ``MiniMaxH3ReferenceToVideo`` has no keyframe inputs, so a first or last
    frame sent alongside numbered references (the documented Ref2VA request
    shape) is anchored afterwards with ``MiniMaxH3AddGuide`` at frame 0 or
    the last frame, chained onto the conditioning.
    """
    plan = render.plan
    inputs: dict[str, object] = {
        "clip": _link(NODE_CLIP),
        "vae": _link(NODE_VIDEO_VAE),
        "prompt": params.prompt,
        "width": plan.width,
        "height": plan.height,
        "length": plan.frame_count,
        "ref_image_size": "match",
    }
    if NODE_AUDIO_VAE in prompt:
        inputs["audio_vae"] = _link(NODE_AUDIO_VAE)
    counts = {"image": 0, "video": 0, "audio": 0}
    keyframes: list[ReferenceBinding] = []
    for binding in references:
        spec = binding.spec
        if spec.role != "reference":
            keyframes.append(binding)
            continue
        index = counts[spec.kind]
        counts[spec.kind] = index + 1
        if spec.kind == "image":
            node_id = f"ref_image_{index}"
            prompt[node_id] = _node("LoadImage", f"Reference image {index + 1}", image=binding.input_name)
            inputs[f"ref_images.ref_image_{index}"] = _link(node_id, 0)
        elif spec.kind == "video":
            load_id = f"ref_video_{index}"
            split_id = f"ref_video_components_{index}"
            prompt[load_id] = _node("LoadVideo", f"Reference video {index + 1}", file=binding.input_name)
            prompt[split_id] = _node("GetVideoComponents", f"Reference video {index + 1} frames", video=_link(load_id))
            inputs[f"ref_videos.ref_video_{index}"] = _link(split_id, 0)
            inputs[f"ref_video_audios.ref_video_audio_{index}"] = _link(split_id, 1)
        else:
            node_id = f"ref_audio_{index}"
            prompt[node_id] = _node("LoadAudio", f"Reference audio {index + 1}", audio=binding.input_name)
            inputs[f"ref_audios.ref_audio_{index}"] = _link(node_id, 0)
    prompt[NODE_CONDITION] = _node("MiniMaxH3ReferenceToVideo", "Condition", **inputs)
    conditioning = NODE_CONDITION
    for binding in keyframes:
        role = binding.spec.role
        load_id = f"load_{role}"
        guide_id = f"guide_{role}"
        prompt[load_id] = _node("LoadImage", f"Load {role.replace('_', ' ')}", image=binding.input_name)
        prompt[guide_id] = _node(
            "MiniMaxH3AddGuide",
            f"Anchor {role.replace('_', ' ')}",
            positive=_link(conditioning, 0),
            vae=_link(NODE_VIDEO_VAE),
            latent=_link(NODE_CONDITION, 1),
            image=_link(load_id, 0),
            frame_idx=0 if role == "first_frame" else -1,
        )
        conditioning = guide_id
    return conditioning


def extra_model_paths_yaml(model_dir: Path) -> str:
    """The ``extra_model_paths.yaml`` that exposes a staged artifact to ComfyUI.

    Only folders present in the artifact are listed; ComfyUI treats each as
    an additional search root for that model kind, so the loader file names
    in the graph resolve without copying weights into the checkout.
    """
    lines = ["skulk:", f"  base_path: {model_dir.resolve().as_posix()}"]
    for folder in MODEL_FOLDERS:
        if (model_dir / folder).is_dir():
            lines.append(f"  {folder}: {folder}")
    return "\n".join(lines) + "\n"

