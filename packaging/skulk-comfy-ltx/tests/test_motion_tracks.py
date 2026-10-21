"""Compare actual pixels with the pinned upstream renderer and reject bad media."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import pytest
import torch
from skulk_comfy_ltx.motion_tracks import parse_motion_tracks, render_motion_tracks


class _Case(TypedDict):
    width: int
    height: int
    frame_count: int
    tracks: list[list[dict[str, float]]]
    sha256: str


class _Reference(TypedDict):
    cases: list[_Case]


_REFERENCE = cast(_Reference, json.loads((Path(__file__).parent / "fixtures/motion-track-reference.json").read_text()))


@pytest.mark.parametrize("case", _REFERENCE["cases"])
def test_pixels_match_pinned_upstream_including_overlaps_and_trail_expiry(case: _Case) -> None:
    """Golden bytes came from the unmodified upstream full-batch rasterizer."""
    raw = json.dumps({"version": 1, "frame_count": case["frame_count"], "tracks": case["tracks"]}).encode()
    calls: list[bool] = []
    result = render_motion_tracks(raw, case["width"], case["height"], case["frame_count"], torch.device("cpu"), check_cancelled=lambda: calls.append(True))
    assert result.dtype == torch.float16
    assert hashlib.sha256(result.numpy().tobytes()).hexdigest() == case["sha256"]
    assert len(calls) == case["frame_count"]


@pytest.mark.parametrize("changes", [{"version": True}, {"version": 2}, {"frame_count": 10}, {"tracks": []}, {"tracks": [[{"x": 0.1, "y": 0.2}]]}, {"tracks": [[{"x": float("nan"), "y": 0.2}] * 9]}, {"tracks": [[{"x": True, "y": 0.2}] * 9]}, {"tracks": [[{"x": 0.1, "y": 1.1}] * 9]}])
def test_invalid_tracks_refused_before_tensor_allocation(changes: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed paths cannot consume a frame batch or silently hold positions."""
    def refuse_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("invalid data allocated a tensor")

    monkeypatch.setattr(torch, "empty", refuse_allocation)
    raw = json.dumps({"version": 1, "frame_count": 9, "tracks": [[{"x": 0.1, "y": 0.2}] * 9], **changes}).encode()
    with pytest.raises(ValueError):
        render_motion_tracks(raw, 128, 128, 9, torch.device("cpu"), check_cancelled=lambda: None)


def test_guide_budget_and_input_size_are_checked_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guide's workload bound is independent of accelerator qualification."""
    def refuse_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("over-budget data allocated a tensor")

    monkeypatch.setattr(torch, "empty", refuse_allocation)
    raw = json.dumps({"version": 1, "frame_count": 9, "tracks": [[{"x": 0.1, "y": 0.2}] * 9]}).encode()
    with pytest.raises(ValueError, match="decoded-pixel budget"):
        render_motion_tracks(raw, 4096, 4096, 9, torch.device("cpu"), check_cancelled=lambda: None)
    with pytest.raises(ValueError, match="bound"):
        parse_motion_tracks(b" " * (4 * 1024 * 1024 + 1), 9)


def test_track_render_cancellation_does_not_finish_a_long_batch() -> None:
    """A request's normal cancellation check runs before each expensive frame."""
    raw = json.dumps({"version": 1, "frame_count": 9, "tracks": [[{"x": 0.1, "y": 0.2}] * 9]}).encode()
    calls: list[bool] = []

    def cancel() -> None:
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("owned request cancelled")

    with pytest.raises(RuntimeError, match="owned request cancelled"):
        render_motion_tracks(raw, 128, 128, 9, torch.device("cpu"), check_cancelled=cancel)
    assert len(calls) == 3
