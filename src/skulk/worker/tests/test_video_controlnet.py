# pyright: reportPrivateUsage=false, reportAny=false
"""The card's ControlNet: control, mask and source attachments, bound and recorded."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from skulk.shared.models.model_cards import (
    ModelCard,
    VideoCompanionConfig,
    VideoCompanionKind,
    VideoMode,
)
from skulk.shared.types.video import VideoReferenceSpec, derivable_guides
from skulk.worker.runner.comfy.graph import (
    NODE_CONDITION,
    NODE_CONTROL,
    NODE_CONTROL_PATCH,
    NODE_GUIDER,
    NODE_SCHEDULER,
    NODE_SHIFT,
    ComfyPrompt,
    ComfyRenderPlan,
    bind_references,
    build_prompt,
    plan_comfy_render,
)
from skulk.worker.tests.test_video_comfy_engine import (
    FL2VA_ID,
    _card,
    _params,
    _reference,
)

PATCH_FILE = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"


def _attachments(
    tmp_path: Path, *roles: tuple[str, str, str]
) -> list[VideoReferenceSpec]:
    references: list[VideoReferenceSpec] = []
    for slot, (role, kind, media_type) in enumerate(roles):
        path = tmp_path / f"{role}-{slot}.bin"
        path.write_bytes(b"x")
        references.append(_reference(slot, kind, role, media_type, path))
    return references


def _card_with(model_patch_modes: list[str] | None) -> ModelCard:
    """The keyframe card, its ControlNet removed or restricted to some modes."""
    card = _card(FL2VA_ID)
    assert card.video is not None
    companions: list[VideoCompanionConfig] = []
    for companion in card.video.companions:
        if companion.kind is VideoCompanionKind.ModelPatch:
            if model_patch_modes is None:
                continue
            companion = companion.model_copy(
                update={"modes": tuple(VideoMode(mode) for mode in model_patch_modes)}
            )
        companions.append(companion)
    video = card.video.model_copy(update={"companions": tuple(companions)})
    return card.model_copy(update={"video": video})


def test_a_control_clip_patches_the_model_after_the_shift(tmp_path: Path) -> None:
    card = _card(FL2VA_ID)
    references = _attachments(tmp_path, ("control", "video", "video/mp4"))
    params = _params(FL2VA_ID, references=tuple(references), total_input_chunks=1, reference_bytes=1)
    render = plan_comfy_render(params, card)
    assert render.mode is VideoMode.TextToAudioVideo
    prompt = build_prompt(render, params, bind_references(params.references, tmp_path), "cmd")
    assert prompt[NODE_CONTROL_PATCH]["class_type"] == "ModelPatchLoader"
    assert prompt[NODE_CONTROL_PATCH]["inputs"]["name"] == PATCH_FILE
    control = prompt[NODE_CONTROL]
    assert control["class_type"] == "MiniMaxH3FunControlNetApply"
    inputs = cast("dict[str, Any]", control["inputs"])
    # The card's strength over the whole schedule, read on the shifted model.
    assert (inputs["strength"], inputs["start_percent"], inputs["end_percent"]) == (1.0, 0.0, 1.0)
    assert inputs["model"] == [NODE_SHIFT, 0]
    assert inputs["model_patch"] == [NODE_CONTROL_PATCH, 0]
    # The clip is ordinary footage: the default guide drawn from it is the pose.
    assert prompt[inputs["control_video"][0]]["class_type"] == "SDPoseDrawKeypoints"
    assert "mask" not in inputs and "source_video" not in inputs
    # The sampler walks the patched model.
    assert prompt[NODE_SCHEDULER]["inputs"]["model"] == [NODE_CONTROL, 0]
    assert prompt[NODE_GUIDER]["inputs"]["model"] == [NODE_CONTROL, 0]
    engine = render.engine_settings()
    assert engine.control_inputs == ("control",)
    assert (engine.control_strength, engine.control_start, engine.control_end) == (1.0, 0.0, 1.0)
    assert engine.control_kind == "pose"


def _guided(
    tmp_path: Path, card: ModelCard, **overrides: object
) -> tuple[ComfyPrompt, ComfyRenderPlan]:
    references = _attachments(tmp_path, ("control", "video", "video/mp4"))
    params = _params(
        FL2VA_ID, references=tuple(references), total_input_chunks=1, reference_bytes=1, **overrides
    )
    render = plan_comfy_render(params, card)
    prompt = build_prompt(render, params, bind_references(params.references, tmp_path), "cmd")
    return prompt, render


def test_the_pose_guide_is_comfys_own_sdpose_chain(tmp_path: Path) -> None:
    """Detect people, estimate whole-body keypoints, draw them: the template's subgraph."""
    prompt, _ = _guided(tmp_path, _card(FL2VA_ID), control_kind="pose")
    resize = prompt["derive_resize"]["inputs"]
    assert prompt[resize["input"][0]]["class_type"] == "GetVideoComponents"
    assert (resize["resize_type"], resize["resize_type.longer_size"], resize["scale_method"]) == (
        "scale longer dimension",
        1024,
        "lanczos",
    )
    assert prompt["derive_pose_model"]["inputs"]["ckpt_name"] == "sdpose_wholebody_fp16.safetensors"
    assert prompt["derive_detector"]["inputs"]["unet_name"] == "rt_detr_v4-x-hgnet_fp16.safetensors"
    people = prompt["derive_people"]["inputs"]
    assert (people["threshold"], people["class_name"], people["max_detections"]) == (0.5, "person", 2)
    keypoints = prompt["derive_keypoints"]["inputs"]
    assert keypoints["model"] == ["derive_pose_model", 0] and keypoints["vae"] == ["derive_pose_model", 2]
    assert keypoints["bboxes"] == ["derive_people", 0] and keypoints["image"] == ["derive_resize", 0]
    drawn = prompt["derive_pose"]["inputs"]
    assert all(drawn[flag] for flag in ("draw_body", "draw_hands", "draw_face", "draw_feet", "draw_head"))
    assert (drawn["stick_width"], drawn["face_point_size"], drawn["score_threshold"]) == (4, 2, 0.51)
    assert prompt[NODE_CONTROL]["inputs"]["control_video"] == ["derive_pose", 0]


def test_the_depth_guide_runs_depth_anything_3(tmp_path: Path) -> None:
    prompt, render = _guided(tmp_path, _card(FL2VA_ID), control_kind="depth")
    assert prompt["derive_depth_model"]["inputs"]["model_name"] == "depth_anything_3_mono_large.safetensors"
    geometry = prompt["derive_depth_geometry"]["inputs"]
    assert (geometry["resolution"], geometry["resize_method"], geometry["mode"]) == (
        504,
        "lower_bound_resize",
        "mono",
    )
    rendered = prompt["derive_depth"]["inputs"]
    # Dynamic inputs travel as dotted keys in the API format.
    assert (rendered["output"], rendered["output.normalization"], rendered["output.apply_sky_clip"]) == (
        "depth",
        "v2_style",
        False,
    )
    assert prompt[NODE_CONTROL]["inputs"]["control_video"] == ["derive_depth", 0]
    assert render.engine_settings().control_kind == "depth"


def _card_without_preprocessors() -> ModelCard:
    card = _card(FL2VA_ID)
    assert card.video is not None
    kept = tuple(c for c in card.video.companions if c.kind is not VideoCompanionKind.Preprocessor)
    return card.model_copy(update={"video": card.video.model_copy(update={"companions": kept})})


def test_edges_need_no_weights_and_other_guides_do(tmp_path: Path) -> None:
    """A card without preprocessor weights still derives edges, and says so."""
    card = _card_without_preprocessors()
    prompt, _ = _guided(tmp_path, card, control_kind="edges")
    edges = prompt["derive_edges"]["inputs"]
    assert (edges["low_threshold"], edges["high_threshold"]) == (0.4, 0.8)
    assert prompt[edges["image"][0]]["class_type"] == "GetVideoComponents"
    with pytest.raises(ValueError, match="cannot derive a pose guide for mode t2va; it derives edges"):
        _guided(tmp_path, card)


def test_the_guides_a_card_derives_follow_its_weights_and_controlnet() -> None:
    card = _card(FL2VA_ID)
    assert card.video is not None
    assert derivable_guides(card.video, VideoMode.TextToAudioVideo) == ("pose", "depth", "edges")
    stripped = _card_without_preprocessors().video
    assert stripped is not None
    assert derivable_guides(stripped, VideoMode.TextToAudioVideo) == ("edges",)
    no_patch = _card_with(None).video
    assert no_patch is not None
    assert derivable_guides(no_patch, VideoMode.TextToAudioVideo) == ()


def test_a_guide_kind_needs_a_control_clip(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="control_kind names what to derive"):
        _params(FL2VA_ID, control_kind="depth")
    mask = _attachments(tmp_path, ("mask", "image", "image/png"))
    with pytest.raises(ValidationError, match="control_kind names what to derive"):
        _params(FL2VA_ID, references=tuple(mask), total_input_chunks=1, reference_bytes=1, control_kind="pose")


def test_a_control_clip_is_never_read_as_a_keyframe(tmp_path: Path) -> None:
    """The condition node sees the first frame, never the control clip."""
    card = _card(FL2VA_ID)
    references = _attachments(
        tmp_path,
        ("first_frame", "image", "image/png"),
        ("control", "video", "video/mp4"),
    )
    params = _params(FL2VA_ID, references=tuple(references), total_input_chunks=2, reference_bytes=2)
    render = plan_comfy_render(params, card)
    assert render.mode is VideoMode.FramesToAudioVideo
    prompt = build_prompt(render, params, bind_references(params.references, tmp_path), "cmd")
    first = cast("list[str]", prompt[NODE_CONDITION]["inputs"]["first_frame"])
    assert prompt[first[0]]["inputs"]["image"] == "first_frame-0.bin"
    assert "last_frame" not in prompt[NODE_CONDITION]["inputs"]
    loaders = [
        cast("dict[str, Any]", node["inputs"]).get("file")
        for node in prompt.values()
        if node["class_type"] == "LoadVideo"
    ]
    assert loaders == ["control-1.bin"]


def test_masked_regeneration_reads_the_mask_and_the_clip_behind_it(tmp_path: Path) -> None:
    card = _card(FL2VA_ID)
    references = _attachments(
        tmp_path,
        ("mask", "video", "video/mp4"),
        ("source", "video", "video/mp4"),
    )
    params = _params(
        FL2VA_ID,
        references=tuple(references),
        total_input_chunks=2,
        reference_bytes=2,
        control_strength=0.7,
        control_start=0.2,
        control_end=0.8,
    )
    render = plan_comfy_render(params, card)
    prompt = build_prompt(render, params, bind_references(params.references, tmp_path), "cmd")
    inputs = cast("dict[str, Any]", prompt[NODE_CONTROL]["inputs"])
    assert (inputs["strength"], inputs["start_percent"], inputs["end_percent"]) == (0.7, 0.2, 0.8)
    mask = prompt[inputs["mask"][0]]
    assert (mask["class_type"], mask["inputs"]["channel"]) == ("ImageToMask", "red")
    assert prompt[mask["inputs"]["image"][0]]["class_type"] == "GetVideoComponents"
    assert prompt[inputs["source_video"][0]]["class_type"] == "GetVideoComponents"
    assert "control_video" not in inputs
    assert render.engine_settings().control_inputs == ("mask", "source")


def test_a_still_mask_loads_as_an_image(tmp_path: Path) -> None:
    card = _card(FL2VA_ID)
    references = _attachments(tmp_path, ("mask", "image", "image/png"))
    params = _params(FL2VA_ID, references=tuple(references), total_input_chunks=1, reference_bytes=1)
    prompt = build_prompt(
        plan_comfy_render(params, card), params, bind_references(params.references, tmp_path), "cmd"
    )
    mask = prompt[prompt[NODE_CONTROL]["inputs"]["mask"][0]]
    assert prompt[mask["inputs"]["image"][0]]["class_type"] == "LoadImage"


def test_no_control_input_leaves_the_graph_as_it_was() -> None:
    card = _card(FL2VA_ID)
    params = _params(FL2VA_ID)
    render = plan_comfy_render(params, card)
    prompt = build_prompt(render, params, (), "cmd")
    assert NODE_CONTROL not in prompt and NODE_CONTROL_PATCH not in prompt
    assert render.control is None and render.engine_settings().control_inputs == ()


def test_a_card_without_a_controlnet_for_the_mode_refuses(tmp_path: Path) -> None:
    references = _attachments(tmp_path, ("control", "video", "video/mp4"))
    params = _params(FL2VA_ID, references=tuple(references), total_input_chunks=1, reference_bytes=1)
    with pytest.raises(ValueError, match="carries no ControlNet"):
        plan_comfy_render(params, _card_with(None))
    # A ControlNet the card declares only for another mode is not this one's.
    with pytest.raises(ValueError, match="carries no ControlNet for mode t2va"):
        plan_comfy_render(params, _card_with(["fl2va"]))
    assert plan_comfy_render(params, _card_with(["t2va"])).control is not None


def test_control_attachments_follow_their_rules(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    with pytest.raises(ValidationError, match="an image or a video"):
        _reference(0, "audio", "control", "audio/wav", audio)
    two = _attachments(
        tmp_path,
        ("control", "video", "video/mp4"),
        ("control", "video", "video/mp4"),
    )
    with pytest.raises(ValidationError, match="at most one control"):
        _params(FL2VA_ID, references=tuple(two), total_input_chunks=2, reference_bytes=2)
    source = _attachments(tmp_path, ("source", "video", "video/mp4"))
    with pytest.raises(ValidationError, match="only behind a mask"):
        _params(FL2VA_ID, references=tuple(source), total_input_chunks=1, reference_bytes=1)
    with pytest.raises(ValidationError, match="need a control or mask"):
        _params(FL2VA_ID, control_strength=0.5)
    control = _attachments(tmp_path, ("control", "video", "video/mp4"))
    with pytest.raises(ValidationError, match="control_start must come before"):
        _params(
            FL2VA_ID,
            references=tuple(control),
            total_input_chunks=1,
            reference_bytes=1,
            control_start=0.6,
            control_end=0.4,
        )
