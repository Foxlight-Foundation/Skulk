"""Whole-render memory accounting and the real decode boundary."""

from dataclasses import dataclass, field
from typing import final

import pytest

from skulk.worker.runner.video_observability import observe_video_decode


@final
@dataclass
class _Decoder:
    verbose: bool = True
    peak_bytes: int = 16_000_000_000
    events: list[str] = field(default_factory=list)

    def write(
        self,
        video_latent: bytes,
        audio_latent: bytes,
        output_path: str,
        *,
        frame_rate: float,
        seed: int = 0,
        keyframes: tuple[int, ...] | None = None,
    ) -> str:
        """Model upstream's verbose-only reset before a smaller decode peak."""
        assert (video_latent, audio_latent, output_path, frame_rate, seed, keyframes) == (
            b"video", b"audio", "output.mp4", 24.0, 7, (3, 7)
        )
        if self.verbose:
            self.peak_bytes = 0
        self.peak_bytes = max(self.peak_bytes, 6_000_000_000)
        self.events.append("write")
        return output_path


def test_decode_keeps_encoder_peak_and_forwards_the_exact_workload() -> None:
    """A smaller decoder cannot erase earlier memory or change the requested clip."""
    decoder = _Decoder()
    write = observe_video_decode(decoder, decoder.write, lambda: decoder.events.append("decoding"))
    result = write(b"video", b"audio", "output.mp4", frame_rate=24.0, seed=7, keyframes=(3, 7))
    assert result == "output.mp4"
    assert decoder.peak_bytes == 16_000_000_000
    assert decoder.events == ["decoding", "write"]


def test_decode_failure_is_announced_and_propagates_without_retry() -> None:
    """The process owner receives a real decoder failure after the phase change."""
    decoder = _Decoder()
    def fail(
        video_latent: bytes,
        audio_latent: bytes,
        output_path: str,
        *,
        frame_rate: float,
        seed: int = 0,
        keyframes: tuple[int, ...] | None = None,
    ) -> str:
        del video_latent, audio_latent, output_path, frame_rate, seed, keyframes
        decoder.events.append("failed")
        raise RuntimeError("decode failed")
    write = observe_video_decode(decoder, fail, lambda: decoder.events.append("decoding"))
    with pytest.raises(RuntimeError, match="decode failed"):
        write(b"video", b"audio", "output.mp4", frame_rate=24.0)
    assert decoder.events == ["decoding", "failed"]
    assert decoder.peak_bytes == 16_000_000_000
