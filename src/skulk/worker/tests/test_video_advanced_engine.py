"""Advanced requests retain exact source behavior and installed-build routing."""

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest

from skulk.api.types.api import VideoCapabilitySection, VideoCreateRequest
from skulk.provisioning import mlx_video
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.video_support import video_request_supported
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.common import ModelId
from skulk.shared.types.video import VideoGenerationTaskParams, VideoReferenceSpec
from skulk.worker.runner.comfy.graph import (
    ReferenceBinding,
    build_prompt,
    plan_comfy_render,
)
from skulk.worker.runner.mlx_video.plan import plan_native_job
from skulk.worker.runner.mlx_video.process import serialize_native_job


def _card(native: bool = True) -> ModelCard:
    card = asyncio.run(load_fixture_card(ModelId("dgrauet/LTX-2.5-distilled-mlx-q4" if native else "Lightricks/LTX-2.5-distilled-comfy-int8")))
    assert card is not None
    return card


def _params(**changes: object) -> VideoGenerationTaskParams:
    return VideoGenerationTaskParams.model_validate({"model": str(_card().model_id), "prompt": "A fox walking", "seconds": 2, "size": "768x512", **changes})


def _reference(kind: Literal["audio", "video"], root: Path | None = None) -> VideoReferenceSpec:
    return VideoReferenceSpec(slot=0, kind=kind, role="reference" if kind == "audio" else "source", media_type="audio/wav" if kind == "audio" else "video/mp4", size_bytes=1, sha256="0" * 64, local_path=str(root / "source") if root is not None else None)


def _published(monkeypatch: pytest.MonkeyPatch) -> str:
    wheel = mlx_video.MlxVideoWheel(filename="skulk_mlx_video-0.2.0-cp313-cp313-macosx_15_0_arm64.whl", sha256="1" * 64, payload_sha256="2" * 64, size_bytes=1, features=("audio_conditioning", "retake", "extend", "prompt_relay"))
    monkeypatch.setattr(mlx_video, "PUBLISHED_WHEELS", (wheel,))
    return wheel.build_identity


@pytest.mark.parametrize("audible", [True, False])
def test_comfy_freezes_original_audio_in_both_stages_and_honors_mute(audible: bool) -> None:
    source = _reference("audio")
    params = _params(audio=audible, audio_conditioning={"start_seconds": 0.5}, references=(source,), reference_bytes=1, total_input_chunks=1)
    graph = build_prompt(plan_comfy_render(params, _card(False)), params, (ReferenceBinding(spec=source, input_name="prepared.wav"),), "command")
    assert graph["ltx_source_audio"]["class_type"] == "LoadAudio"
    assert graph["ltx_base_av"]["inputs"]["audio_latent"] == ["ltx_frozen_source_audio", 0]
    assert graph["ltx_refine_av"]["inputs"]["audio_latent"] == ["ltx_frozen_source_audio", 0]
    assert "decode_audio" not in graph
    assert not any(node["class_type"] == "LoadImage" for node in graph.values())
    if audible:
        assert graph["create_video"]["inputs"]["audio"] == ["ltx_source_audio", 0]
    else:
        assert "audio" not in graph["create_video"]["inputs"]


@pytest.mark.parametrize("editing", [{"operation": "retake", "start_seconds": 0.0, "end_seconds": 1.0}, {"operation": "extend"}])
def test_native_edit_plan_does_not_treat_source_clip_as_an_image(tmp_path: Path, editing: dict[str, object]) -> None:
    source = _reference("video", tmp_path)
    params = _params(editing=editing, references=(source,), reference_bytes=1, total_input_chunks=1)
    job, _ = plan_native_job(params, _card(), tmp_path, tmp_path, tmp_path)
    assert job.pipeline == editing["operation"]
    assert job.images == () and job.guidance is None
    assert job.transformer_variant == "distilled" and job.base_steps == 8


@pytest.mark.parametrize("controls", [
    {"audio_conditioning": {}, "duration": "auto"},
    {"audio_conditioning": {}, "generated_keyframe_interval": 24},
    {"audio_conditioning": {}, "prompt_relay": {"segments": [{"prompt": "day", "length": 1}, {"prompt": "night", "length": 1}]}},
])
def test_source_workflow_combinations_are_refused_before_loading(controls: dict[str, object]) -> None:
    params = _params(**controls, references=(_reference("audio"),), reference_bytes=1, total_input_chunks=1)
    card = _card()
    assert card.video is not None
    with pytest.raises(ValueError, match="does not combine"):
        params.validate_family(card.family, card.video)


@pytest.mark.parametrize("editing", [
    {"operation": "retake"},
    {"operation": "retake", "start_seconds": 1.0, "end_seconds": 0.0},
    {"operation": "extend", "regenerate_audio": False},
    {"operation": "extend", "start_seconds": 0.0},
])
def test_invalid_source_timeline_controls_do_not_enter_task_protocol(editing: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _params(editing=editing)


def test_advanced_metadata_tracks_published_builds_without_hardware_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlx_video, "PUBLISHED_WHEELS", ())
    before = VideoCapabilitySection.from_model_card(_card())
    assert before is not None and not before.editing and not before.audio_conditioning and not before.prompt_relay
    build = _published(monkeypatch)
    after = VideoCapabilitySection.from_model_card(_card())
    assert after is not None and after.editing == ["retake", "extend"] and after.audio_conditioning and after.prompt_relay
    assert after.pipeline_modes["retake"] == ["ref2va"]
    assert after.reference_fidelities == [] and after.default_reference_fidelity is None
    params = _params(prompt_relay={"segments": [{"prompt": "day", "length": 1}, {"prompt": "night", "length": 2}]})
    assert video_request_supported(_card(), params, "mlx_video-metal", build)
    assert not video_request_supported(_card(), params, "mlx_video-metal", "old-build")
    assert not video_request_supported(_card(), params, "comfy-cuda", build)


def test_old_native_protocol_keeps_new_optional_fields_absent(tmp_path: Path) -> None:
    job, _ = plan_native_job(_params(), _card(), tmp_path, tmp_path, tmp_path)
    message = serialize_native_job(job)
    assert all(f'"{name}"' not in message for name in ("audio_conditioning", "editing", "prompt_relay", "source_media", "transformer_variant"))


def test_multipart_scene_controls_use_strict_json_objects() -> None:
    request = VideoCreateRequest.model_validate({"model": "model", "prompt": "fox", "audio_conditioning": '{"start_seconds":0.5}'})
    assert request.audio_conditioning is not None and request.audio_conditioning.start_seconds == 0.5
    with pytest.raises(ValueError):
        VideoCreateRequest.model_validate({"model": "model", "prompt": "fox", "editing": '{"operation":"extend","unused":true}'})


def test_comfy_runner_limit_does_not_shrink_intrinsic_model_truth() -> None:
    card = _card(False)
    assert card.video is not None
    assert card.video.editing == ("retake", "extend") and card.video.prompt_relay
    metadata = VideoCapabilitySection.from_model_card(card)
    assert metadata is not None and not metadata.editing and not metadata.prompt_relay
    source = _reference("video")
    params = _params(editing={"operation": "extend"}, references=(source,), reference_bytes=1, total_input_chunks=1)
    with pytest.raises(ValueError, match="temporal"):
        plan_comfy_render(params, card)


@pytest.mark.parametrize("pipeline", ["dev_two_stage", "hq_two_stage"])
@pytest.mark.parametrize("guidance", [{}, {"video_cfg": 4.0}])
def test_guided_protocol_keeps_required_default_fields(tmp_path: Path, pipeline: str, guidance: dict[str, float]) -> None:
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-dev-comfy-int8")))
    assert card is not None
    params = _params(pipeline=pipeline, guidance=guidance)
    job, _ = plan_native_job(params, card, tmp_path, tmp_path, tmp_path)
    expected = {"video_cfg": guidance.get("video_cfg", 3.0), "audio_cfg": 7.0, "stg_scale": 0.0, "modality_scale": 1.0}
    assert f'"guidance":{json.dumps(expected, separators=(",", ":"))}' in serialize_native_job(job)


@pytest.mark.parametrize("seconds", [1, 20, 120])
def test_retake_can_select_the_complete_aligned_clip(seconds: int) -> None:
    card = _card()
    assert card.video is not None
    video = card.video.model_copy(update={"max_seconds": 120})
    actual_seconds = video.frame_count_for_seconds(seconds) / video.fps
    params = _params(seconds=seconds, editing={"operation": "retake", "start_seconds": 0.0, "end_seconds": actual_seconds}, references=(_reference("video"),), reference_bytes=1, total_input_chunks=1)
    params.validate_family(card.family, video)
    assert params.editing is not None and params.editing.end_seconds == actual_seconds
    with pytest.raises(ValueError, match="exceeds"):
        _params(seconds=seconds, editing={"operation": "retake", "start_seconds": 0.0, "end_seconds": actual_seconds + 0.01}, references=(_reference("video"),), reference_bytes=1, total_input_chunks=1).validate_family(card.family, video)
