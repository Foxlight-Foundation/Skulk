"""Dynamic duration never permits an unverified or out-of-bounds manifest."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from skulk.worker.runner.video_plan import RenderPlan
from skulk.worker.runner.video_probe import probe_video_output


def _plan() -> RenderPlan:
    return RenderPlan(width=768, height=512, fps=24, frame_count=121, steps=8, seed=1, audio=True, sample_rate=48000, channels=2)


def _probe_payload(**changes: object) -> bytes:
    video: dict[str, object] = {"codec_type": "video", "codec_name": "h264", "width": 768, "height": 512, "nb_frames": "73", "r_frame_rate": "24/1"}
    video.update(changes)
    return json.dumps({"streams": [video, {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}], "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"}}).encode()


def _response(payload: bytes) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=payload)
    return run


def test_dynamic_duration_records_actual_bounded_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _response(_probe_payload()))
    result = probe_video_output(Path("output.mp4"), _plan(), codec="h264", minimum_frames=25)
    assert result.frame_count == 73 and result.seconds == 73 / 24


@pytest.mark.parametrize("changes", [{"nb_frames": "129"}, {"nb_frames": "72"}, {"nb_frames": "17"}, {"nb_frames": None}, {"r_frame_rate": "30/1"}, {"r_frame_rate": "24/0"}, {"codec_name": "av1"}, {"width": 1024}])
def test_invalid_engine_output_cannot_be_reported_as_success(changes: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _response(_probe_payload(**changes)))
    with pytest.raises(ValueError):
        probe_video_output(Path("output.mp4"), _plan(), codec="h264", minimum_frames=25)


def test_fixed_duration_requires_exact_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _response(_probe_payload()))
    with pytest.raises(ValueError, match="duration bounds"):
        probe_video_output(Path("output.mp4"), _plan(), codec="h264")
