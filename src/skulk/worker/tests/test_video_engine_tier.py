# pyright: reportPrivateUsage=false
"""Engine-tier settings on a video request: resolved, bound, and recorded."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from skulk.shared.models.model_cards import VideoCompanionKind, VideoMode
from skulk.shared.types.video import VIDEO_SAMPLERS, video_sampler_refusal
from skulk.worker.runner.comfy.graph import (
    NODE_CONDITION,
    NODE_SAMPLER_SELECT,
    NODE_SAVE_VIDEO,
    NODE_SCHEDULER,
    NODE_SHIFT,
    bind_references,
    build_prompt,
    plan_comfy_render,
)
from skulk.worker.tests.test_video_comfy_engine import (
    FL2VA_ID,
    REF2VA_ID,
    _card,
    _params,
    _reference,
)


def test_defaults_are_the_official_template() -> None:
    card = _card(FL2VA_ID)
    params = _params(FL2VA_ID)
    render = plan_comfy_render(params, card)
    prompt = build_prompt(render, params, (), "cmd")
    assert prompt[NODE_SAMPLER_SELECT]["inputs"]["sampler_name"] == "res_multistep"
    assert prompt[NODE_SCHEDULER]["inputs"]["scheduler"] == "simple"
    assert prompt[NODE_SAVE_VIDEO]["inputs"]["format.codec"] == "h264"
    assert prompt[NODE_CONDITION]["inputs"]["prompt"] == "a fox at dusk"
    engine = render.engine_settings()
    assert (engine.sampler, engine.scheduler, engine.codec) == (
        "res_multistep",
        "simple",
        "h264",
    )
    assert engine.reference_fidelity is None and engine.styles == ()
    assert (engine.video_shift, engine.audio_shift) == (12.0, 3.0)


def test_request_settings_reach_the_graph_and_the_record() -> None:
    card = _card(FL2VA_ID)
    params = _params(
        FL2VA_ID,
        lora="turbo_fl2v_4step_768p",
        sampler="dpmpp_2m",
        scheduler="beta",
        video_shift=9.0,
        audio_shift=2.0,
        codec="av1",
    )
    render = plan_comfy_render(params, card)
    prompt = build_prompt(render, params, (), "cmd")
    assert prompt[NODE_SAMPLER_SELECT]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert prompt[NODE_SCHEDULER]["inputs"]["scheduler"] == "beta"
    # The request's shifts win over the adapter's trained 6.0.
    assert prompt[NODE_SHIFT]["inputs"]["shift_video"] == 9.0
    assert prompt[NODE_SHIFT]["inputs"]["shift_audio"] == 2.0
    assert prompt[NODE_SAVE_VIDEO]["inputs"]["format.codec"] == "av1"
    engine = render.engine_settings()
    assert engine.adapter == "turbo_fl2v_4step_768p" and engine.adapter_strength == 1.0
    assert (engine.steps, engine.video_shift, engine.audio_shift) == (4, 9.0, 2.0)


def test_styles_bind_as_embedding_tokens_before_the_prompt() -> None:
    card = _card(FL2VA_ID)
    assert card.video is not None
    names = [
        companion.name
        for companion in card.video.companions
        if companion.kind is VideoCompanionKind.Embedding
    ][:2]
    assert len(names) == 2
    params = _params(FL2VA_ID, styles=names)
    render = plan_comfy_render(params, card)
    text = cast(
        "str", build_prompt(render, params, (), "cmd")[NODE_CONDITION]["inputs"]["prompt"]
    )
    assert text == f"embedding:{names[0]} embedding:{names[1]} a fox at dusk"
    assert render.engine_settings().styles == tuple(names)
    with pytest.raises(ValueError, match="no style embedding named 'nope'"):
        plan_comfy_render(_params(FL2VA_ID, styles=["nope"]), card)


def test_reference_fidelity_sizes_ref2va_images(tmp_path: Path) -> None:
    card = _card(REF2VA_ID)
    input_dir = tmp_path / "video_input"
    (input_dir / "cmd").mkdir(parents=True)
    (input_dir / "cmd" / "0.png").write_bytes(b"x")
    refs = (_reference(0, "image", "reference", "image/png", input_dir / "cmd" / "0.png"),)
    params = _params(
        REF2VA_ID,
        references=refs,
        reference_bytes=1,
        total_input_chunks=1,
        reference_fidelity="max",
    )
    render = plan_comfy_render(params, card)
    prompt = build_prompt(render, params, bind_references(refs, input_dir), "cmd")
    assert prompt[NODE_CONDITION]["inputs"]["ref_image_size"] == "max"
    assert render.engine_settings().reference_fidelity == "max"


def test_reference_fidelity_is_refused_outside_ref2va() -> None:
    with pytest.raises(ValidationError, match="ref2va only"):
        _params(FL2VA_ID, reference_fidelity="max")
    with pytest.raises(ValidationError, match="ref2va only"):
        _params(FL2VA_ID, mode=VideoMode.TextToAudioVideo, reference_fidelity="match")


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("dpm_fast", "chooses its own step count"),
        ("dpm_adaptive", "chooses its own step count"),
        ("euler_cfg_pp", "classifier-free guidance"),
        ("res_multistep_cfg_pp", "classifier-free guidance"),
        ("cfgpp_ud10_ab", "classifier-free guidance"),
        ("made_up", "not offered by the pinned engine"),
    ],
)
def test_unusable_samplers_are_refused_by_name(name: str, reason: str) -> None:
    refusal = video_sampler_refusal(name)
    assert refusal is not None and reason in refusal
    with pytest.raises(ValidationError, match=reason):
        _params(FL2VA_ID, sampler=name)


def test_every_offered_sampler_is_accepted() -> None:
    assert "res_multistep" in VIDEO_SAMPLERS
    for name in VIDEO_SAMPLERS:
        assert video_sampler_refusal(name) is None
        assert _params(FL2VA_ID, sampler=name).sampler == name


@pytest.mark.parametrize("field", ["video_shift", "audio_shift"])
def test_shifts_stay_inside_the_node_range(field: str) -> None:
    with pytest.raises(ValidationError):
        _params(FL2VA_ID, **{field: 0.0})
    with pytest.raises(ValidationError):
        _params(FL2VA_ID, **{field: 100.5})


def test_styles_are_unique_and_bounded() -> None:
    with pytest.raises(ValidationError, match="applied once"):
        _params(FL2VA_ID, styles=["a", "a"])
    with pytest.raises(ValidationError):
        _params(FL2VA_ID, styles=["a", "b", "c", "d", "e"])
