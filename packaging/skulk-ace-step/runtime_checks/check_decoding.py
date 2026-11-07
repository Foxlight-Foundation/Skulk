"""Exercise pinned decode overlap/discard assembly with a visible latent clock.

These fixtures run the actual upstream tiler around a tiny observable codec;
they establish window/clock behavior, not model quality or GPU memory fit.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from skulk_ace_step.upstream_bindings import (
    install_bounded_source_decoding,
    native_api,
)


@dataclass(frozen=True)
class DecodedAudio:
    """Pinned codec result with a visible 1,920-sample latent clock."""

    sample: torch.Tensor


class RecordingDecoder(torch.nn.Module):
    """Record actual decode windows and expand their complete latent clocks."""

    dtype: torch.dtype = torch.float32

    def __init__(self, *, stochastic: bool = False) -> None:
        """Create one independent codec observation without model weights."""
        super().__init__()
        self.input_frames: list[int] = []
        self.stochastic = stochastic
        self.fail = False

    def decode(self, latents: torch.Tensor) -> DecodedAudio:
        """Expose tiling/trimming errors without replacing upstream assembly."""
        self.input_frames.append(latents.shape[-1])
        if self.fail:
            raise RuntimeError("controlled decoder allocation failure")
        samples = latents[:, :1].repeat_interleave(1920, dim=-1).expand(-1, 2, -1)
        if self.stochastic:
            samples = samples + torch.randn_like(samples)
        return DecodedAudio(samples)


class NativeDecodingTests(unittest.TestCase):
    """Run real pinned handlers around deterministic, deliberately tiny codecs."""

    def test_every_boundary_preserves_complete_clock_and_silent_tail(self) -> None:
        """Both output assembly paths retain all frames through the 120s ceiling."""
        for frames in (125, 256, 257, 375, 750, 1500, 3000):
            for offload in (False, True):
                with self.subTest(frames=frames, offload=offload):
                    values = torch.arange(frames, dtype=torch.float32)
                    values[375:] = 0
                    latents = values.reshape(1, 1, -1).expand(1, 64, -1)
                    handler = native_api().handler()
                    decoder = RecordingDecoder()
                    handler.vae = decoder
                    install_bounded_source_decoding(handler)
                    actual = handler.tiled_decode(latents, offload_wav_to_cpu=offload)
                    expected = values.repeat_interleave(1920).reshape(1, 1, -1)
                    torch.testing.assert_close(
                        actual, expected.expand(1, 2, -1), rtol=0, atol=0
                    )
                    self.assertLessEqual(max(decoder.input_frames), 256)
                    self.assertEqual(len(decoder.input_frames) == 1, frames <= 256)

    def test_only_one_handler_changes_when_global_gpu_capacity_is_large(self) -> None:
        """A large host cannot enlarge this instance's decoder workspace."""
        bounded, other = native_api().handler(), native_api().handler()
        original_class_decoder = type(other).tiled_decode
        bounded.device = other.device = "cuda"
        first, second = RecordingDecoder(), RecordingDecoder()
        bounded.vae, other.vae = first, second
        latents = torch.zeros((1, 64, 750), dtype=torch.float32)
        install_bounded_source_decoding(bounded)
        with patch(
            "acestep.core.generation.handler.memory_utils.get_effective_free_vram_gb",
            return_value=60.0,
        ):
            bounded.tiled_decode(latents)
            other.tiled_decode(latents)
        self.assertLessEqual(max(first.input_frames), 256)
        self.assertGreater(max(second.input_frames), 256)
        self.assertIs(type(other).tiled_decode, original_class_decoder)

    def test_short_seeded_decode_matches_unmodified_direct_path(self) -> None:
        """A short decode keeps the same native operation and request RNG."""
        handler = native_api().handler()
        handler.vae = RecordingDecoder(stochastic=True)
        original = handler.tiled_decode
        latents = torch.zeros((1, 64, 125), dtype=torch.float32)
        saved_state = torch.default_generator.get_state()
        try:
            torch.default_generator.manual_seed(12025)
            expected = original(latents)
            install_bounded_source_decoding(handler)
            torch.default_generator.manual_seed(12025)
            actual = handler.tiled_decode(latents)
        finally:
            torch.default_generator.set_state(saved_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shorter_explicit_window_and_overlap_keep_complete_geometry(self) -> None:
        """The adapter never enlarges an existing smaller decode policy."""
        for offload in (False, True):
            with self.subTest(offload=offload):
                handler = native_api().handler()
                decoder = RecordingDecoder()
                handler.vae = decoder
                values = torch.arange(375, dtype=torch.float32)
                install_bounded_source_decoding(handler)
                actual = handler.tiled_decode(
                    values.reshape(1, 1, -1).expand(1, 64, -1),
                    chunk_size=128,
                    overlap=16,
                    offload_wav_to_cpu=offload,
                )
                self.assertLessEqual(max(decoder.input_frames), 128)
                torch.testing.assert_close(
                    actual,
                    values.repeat_interleave(1920).reshape(1, 1, -1).expand(1, 2, -1),
                    rtol=0,
                    atol=0,
                )

    def test_native_mps_smaller_window_remains_authoritative(self) -> None:
        """The pinned MPS Conv1d restriction still applies beneath the bound."""
        handler = native_api().handler()
        handler.device = "mps"
        decoder = RecordingDecoder()
        handler.vae = decoder
        install_bounded_source_decoding(handler)
        actual = handler.tiled_decode(torch.zeros((1, 64, 125)))
        self.assertEqual(actual.shape, (1, 2, 125 * 1920))
        self.assertLessEqual(max(decoder.input_frames), 32)

    def test_codec_failure_propagates_through_existing_supervision(self) -> None:
        """The bound cannot conceal failure behind a new adapter fallback."""
        handler = native_api().handler()
        decoder = RecordingDecoder()
        decoder.fail = True
        handler.vae = decoder
        install_bounded_source_decoding(handler)
        with self.assertRaisesRegex(RuntimeError, "controlled decoder allocation"):
            handler.tiled_decode(torch.zeros((1, 64, 750)))

    def test_clock_mismatch_fails_before_replacing_the_callable(self) -> None:
        """A future source-clock change must not silently reuse latent constants."""
        handler = native_api().handler()
        original = handler.tiled_decode
        handler.sample_rate = 44100
        with self.assertRaisesRegex(ValueError, "48kHz clock"):
            install_bounded_source_decoding(handler)
        self.assertEqual(handler.tiled_decode, original)
