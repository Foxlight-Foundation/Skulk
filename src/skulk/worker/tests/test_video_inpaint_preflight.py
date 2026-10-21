"""Refuse invalid source/mask graphs before Comfy allocates complete batches."""

from dataclasses import replace
from pathlib import Path

import pytest

from skulk.shared.types.video import VideoReferenceSpec
from skulk.shared.types.video_inpainting import VideoInpainting
from skulk.worker.runner.comfy.graph import ReferenceBinding
from skulk.worker.runner.comfy.inpaint_input import (
    MAX_INPAINT_PIXEL_BYTES,
    inpaint_allocation_bytes,
    prepare_inpaint_media,
)
from skulk.worker.runner.video_input import MediaInputRequest, VerifiedMediaInput
from skulk.worker.runner.video_plan import RenderPlan


def _bindings(*, video_mask: bool = False, opening: bool = False) -> tuple[ReferenceBinding, ...]:
    source = VideoReferenceSpec(slot=0, kind="video", role="source", media_type="video/mp4", size_bytes=1, sha256="0" * 64)
    mask = VideoReferenceSpec(slot=1, kind="video" if video_mask else "image", role="mask", media_type="video/mp4" if video_mask else "image/png", size_bytes=1, sha256="1" * 64)
    items = (source, mask)
    if opening:
        items = (*items, VideoReferenceSpec(slot=2, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="2" * 64))
    return tuple(ReferenceBinding(spec=item, input_name=f"{item.slot}") for item in items)


def _plan() -> RenderPlan:
    return RenderPlan(width=512, height=384, frame_count=25, fps=24, steps=8, seed=1, audio=True, sample_rate=48000, channels=2)


def _facts(**changes: object) -> VerifiedMediaInput:
    return VerifiedMediaInput.model_validate({"path": "measured", "width": 320, "height": 192, "frames": 25, "has_audio": True, "audio_channels": 2, **changes})


@pytest.mark.parametrize("video_mask,opening", [(False, False), (True, False), (False, True)])
def test_preflight_uses_real_source_geometry_and_keeps_the_complete_requested_grid(video_mask: bool, opening: bool) -> None:
    """A bigger output is permitted only when the measured source really fits."""
    calls: list[MediaInputRequest] = []

    def prepare(request: MediaInputRequest) -> VerifiedMediaInput:
        calls.append(request)
        assert request.width == 512 and request.height == 384
        assert request.maximum_frames == 25 and request.exact_frames
        slot = Path(request.source).name
        return _facts(frames=25 if slot == "0" or video_mask and slot == "1" else 1, has_audio=slot == "0", width=512 if slot == "2" else 320, height=384 if slot == "2" else 192)

    facts = prepare_inpaint_media(_plan(), VideoInpainting(left=128, top=128), _bindings(video_mask=video_mask, opening=opening), Path("input"), prepare=prepare)
    assert facts is not None and (facts.source_width, facts.source_height, facts.frames) == (320, 192, 25)
    assert facts.mask_frames == (25 if video_mask else 1)
    assert facts.allocation_bytes < MAX_INPAINT_PIXEL_BYTES
    assert [item.operation for item in calls] == ["inpaint", "inpaint" if video_mask else "image", *(["image"] if opening else [])]


@pytest.mark.parametrize("slot,changes,reason", [(0, {"width": None}, "measured"), (0, {"frames": 24}, "complete"), (0, {"width": 500}, "without cropping"), (0, {"audio_channels": 6}, "mono or stereo"), (1, {"width": 319}, "mask must match"), (1, {"frames": 24}, "mask must match"), (1, {"has_audio": True}, "audio track"), (2, {"width": 320}, "opening image")])
def test_preflight_stops_at_the_first_invalid_input(slot: int, changes: dict[str, object], reason: str) -> None:
    """Malformed decoder results never become graph geometry or hidden resizes."""
    calls: list[int] = []

    def prepare(request: MediaInputRequest) -> VerifiedMediaInput:
        current = int(Path(request.source).name)
        calls.append(current)
        values: dict[str, object] = {"frames": 25 if current == 0 else 1, "has_audio": current == 0, "width": 512 if current == 2 else 320, "height": 384 if current == 2 else 192}
        if current == slot:
            values.update(changes)
        return _facts(**values)

    with pytest.raises(ValueError, match=reason):
        prepare_inpaint_media(_plan(), VideoInpainting(left=32), _bindings(opening=True), Path("input"), prepare=prepare)
    assert calls == list(range(slot + 1))


def test_cancellation_during_mask_preparation_creates_no_graph() -> None:
    """The caller observes normal cancellation instead of partial source facts."""
    def prepare(request: MediaInputRequest) -> VerifiedMediaInput | None:
        return _facts() if Path(request.source).name == "0" else None

    assert prepare_inpaint_media(_plan(), VideoInpainting(), _bindings(), Path("input"), prepare=prepare) is None


def test_combined_retained_budget_refuses_separately_small_inputs() -> None:
    """Per-input pixel limits alone miss cached stages on a padded output canvas."""
    def prepare(request: MediaInputRequest) -> VerifiedMediaInput:
        return _facts(frames=25 if Path(request.source).name == "0" else 1, has_audio=False)

    large = replace(_plan(), width=4096, height=4096)
    with pytest.raises(ValueError, match="combined decoded graph buffers"):
        prepare_inpaint_media(large, VideoInpainting(), _bindings(), Path("input"), prepare=prepare)
    ordinary = inpaint_allocation_bytes(1024, 576, 121, source_width=1024, source_height=576, mask_frames=121, fps=24, opening=True)
    assert ordinary < MAX_INPAINT_PIXEL_BYTES
