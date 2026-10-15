"""Decode observation shared with the standalone native video wheel."""

from collections.abc import Callable
from typing import Protocol


class VerboseVideoDecoder(Protocol):
    """A decoder whose verbose mode owns a separate allocator measurement."""

    verbose: bool


class VideoDecodeWriter[Latent, Keyframes](Protocol):
    """The pinned pipeline's typed decode-and-save callable."""

    def __call__(
        self,
        video_latent: Latent,
        audio_latent: Latent,
        output_path: str,
        *,
        frame_rate: float,
        seed: int = 0,
        keyframes: Keyframes | None = None,
    ) -> str:
        """Decode both latents to output_path, forwarding rate, seed and anchors."""
        ...


def observe_video_decode[Latent, Keyframes](
    decoder: VerboseVideoDecoder,
    write_video: VideoDecodeWriter[Latent, Keyframes],
    on_decode: Callable[[], None],
) -> VideoDecodeWriter[Latent, Keyframes]:
    """Return a writer that announces decode and preserves the render-wide peak.

    Disable the decoder's private verbose profiler, whose counter reset would
    erase encoder/sampling peaks. The returned callable forwards every latent,
    output path, frame rate, seed and keyframe unchanged, announcing decode before
    the actual operation and propagating failures to its process owner.
    """
    # Upstream's verbose decoder resets MLX's process-global peak, even when
    # the pipeline facade is quiet. A headless job measures all its stages.
    decoder.verbose = False

    def decode(
        video_latent: Latent,
        audio_latent: Latent,
        output_path: str,
        *,
        frame_rate: float,
        seed: int = 0,
        keyframes: Keyframes | None = None,
    ) -> str:
        on_decode()
        return write_video(
            video_latent,
            audio_latent,
            output_path,
            frame_rate=frame_rate,
            seed=seed,
            keyframes=keyframes,
        )

    return decode
