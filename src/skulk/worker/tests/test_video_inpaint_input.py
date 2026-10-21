"""Measured inpainting geometry must precede full Comfy pixel allocation."""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType, TracebackType

import pytest
from PIL import Image

from skulk.worker.runner import video_input
from skulk.worker.runner.video_input import MediaInputRequest, VerifiedMediaInput


@dataclass(frozen=True)
class _Frame:
    width: int
    height: int
    pts: int | None
    time_base: Fraction | None = Fraction(1, 24)
    is_corrupt: bool = False


@dataclass(frozen=True)
class _Codec:
    width: int
    height: int


@dataclass(frozen=True)
class _Stream:
    codec_context: _Codec
    average_rate: Fraction = Fraction(24)
    frames: int = 0


@dataclass(frozen=True)
class _Streams:
    video: tuple[_Stream, ...]
    audio: tuple[object, ...] = ()

    def __len__(self) -> int:
        return len(self.video) + len(self.audio)


@dataclass(frozen=True)
class _Container:
    streams: _Streams
    frames: tuple[_Frame, ...]

    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, _kind: type[BaseException] | None, _error: BaseException | None, _traceback: TracebackType | None) -> None:
        pass

    def decode(self, *, video: int) -> Iterator[_Frame]:
        assert video == 0
        yield from self.frames


def _decoder(monkeypatch: pytest.MonkeyPatch, frames: tuple[_Frame, ...], *, width: int = 320, height: int = 192, fps: int = 24, declared_frames: int = 0) -> None:
    decoder = ModuleType("av")
    container = _Container(_Streams((_Stream(_Codec(width, height), Fraction(fps), declared_frames),)), frames)

    def open_source(_source: str) -> _Container:
        return container

    def resampler(**_kwargs: object) -> None:
        raise AssertionError("still/video measurement used an audio resampler")

    monkeypatch.setattr(decoder, "open", open_source, raising=False)
    monkeypatch.setattr(decoder, "AudioResampler", resampler, raising=False)
    original = video_input.importlib.import_module

    def import_decoder(name: str, package: str | None = None) -> ModuleType:
        return decoder if name == "av" else original(name, package)

    monkeypatch.setattr(video_input.importlib, "import_module", import_decoder)


def _request(operation: str = "inpaint") -> MediaInputRequest:
    return MediaInputRequest.model_validate({"operation": operation, "source": "verified-source", "duration_seconds": 25 / 24, "width": 512, "height": 384, "fps": 24, "maximum_frames": 25, "exact_frames": True})


def test_inpaint_measures_smaller_source_instead_of_claiming_output_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """An outpaint canvas supplies a bound; it cannot declare the source dimensions."""
    _decoder(monkeypatch, tuple(_Frame(320, 192, index) for index in range(25)))
    result = video_input.verify_video_input(_request())
    assert (result.width, result.height, result.frames, result.has_audio) == (320, 192, 25, False)
    with pytest.raises(ValueError, match="size and frame rate"):
        video_input.verify_video_input(_request("video"))


@pytest.mark.parametrize("bad_frame", [_Frame(319, 192, 12), _Frame(320, 192, None), _Frame(320, 192, 19), _Frame(320, 192, 12, is_corrupt=True)])
def test_measurement_checks_every_frame_geometry_and_timeline(monkeypatch: pytest.MonkeyPatch, bad_frame: _Frame) -> None:
    """Valid headers cannot hide a mid-clip resize, corruption or discontinuity."""
    frames = tuple(bad_frame if index == 12 else _Frame(320, 192, index) for index in range(25))
    _decoder(monkeypatch, frames)
    with pytest.raises(ValueError, match="frames|timeline"):
        video_input.verify_video_input(_request())


@pytest.mark.parametrize("count", [24, 26])
def test_measurement_never_truncates_or_holds_a_frame(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """A mask/source clip must cover the complete requested trained timeline."""
    _decoder(monkeypatch, tuple(_Frame(320, 192, index) for index in range(count)))
    with pytest.raises(ValueError, match="duration|frame grid"):
        video_input.verify_video_input(_request())


def test_still_mask_reports_decoded_geometry_and_refuses_animation(tmp_path: Path) -> None:
    """One source-sized mask may broadcast; a disguised animation may not."""
    path = tmp_path / "mask.png"
    with Image.new("RGB", (64, 96), "white") as image:
        image.save(path)
    request = _request("image").model_copy(update={"source": str(path)})
    result = video_input.verify_image_input(request)
    assert (result.width, result.height, result.frames, result.has_audio) == (64, 96, 1, False)
    with Image.new("RGB", (64, 96), "black") as first, Image.new("RGB", (64, 96), "white") as second:
        first.save(path, save_all=True, append_images=[second], duration=100)
    with pytest.raises(ValueError, match="animated"):
        video_input.verify_image_input(request)


def test_opening_uses_comfys_displayed_exif_dimensions(tmp_path: Path) -> None:
    """Phone-image rotation must not become an implicit full-canvas resize."""
    path = tmp_path / "opening.jpg"
    with Image.new("RGB", (384, 512), "blue") as image:
        orientation = Image.Exif()
        orientation[274] = 6
        image.save(path, exif=orientation)
    result = video_input.verify_image_input(_request("image").model_copy(update={"source": str(path)}))
    assert (result.width, result.height) == (512, 384)


def test_source_header_bound_precedes_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Huge coded dimensions must fail before a full image buffer is decoded."""
    path = tmp_path / "too-large.png"
    with Image.new("RGB", (768, 768), "white") as image:
        image.save(path)

    def no_load(_image: Image.Image) -> None:
        raise AssertionError("out-of-bounds image was decoded")

    monkeypatch.setattr(Image.Image, "load", no_load)
    with pytest.raises(ValueError, match="canvas|budget"):
        video_input.verify_image_input(_request("image").model_copy(update={"source": str(path)}))


def test_legacy_reply_shape_stays_compatible_and_new_geometry_is_private() -> None:
    """A new Core reader accepts old replies, and legacy calls emit no new keys."""
    old = '{"path":"verified-source","frames":25,"has_audio":false,"audio_sample_rate":null,"audio_channels":null}'
    legacy = VerifiedMediaInput.model_validate_json(old)
    assert legacy.width is None and legacy.height is None
    assert video_input.serialize_media_input_result(_request("video"), legacy) == old
    measured = legacy.model_copy(update={"width": 320, "height": 192})
    assert video_input.serialize_media_input_result(_request(), measured) == old[:-1] + ',"width":320,"height":192}'
